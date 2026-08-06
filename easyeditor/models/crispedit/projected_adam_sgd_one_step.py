import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.optim import SGD


class ProjectedSGD(SGD):
    """
    Apply the one-shot soft K-FAC solution from section 3.2.

    For a weight matrix W and gradient G, the preconditioner solves the
    generalized-basis form of

        B_e dW A_e + lambda B_c dW A_c = -G.

    ``step`` is deliberately restricted to one call with unit learning rate
    and no SGD state. Under those conditions, SGD's subtraction supplies the
    minus sign in the closed-form update instead of turning the solution into
    a repeatedly applied gradient preconditioner.
    """

    _PRECOMPUTED_BASIS_KEYS = (
        ("soft_q_a", "soft_q_b", "soft_eig_a", "soft_eig_b"),
    )

    _FACTOR_KEY_SETS = (
        (("edit_A", "edit_B"), ("cap_A", "cap_B")),
    )

    def __init__(
        self,
        params,
        projection_cache_map: Optional[Dict] = None,
        lr=1.0,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        maximize=False,
        additional_projection_cache_map: Optional[Dict] = None,
        soft_lambda: float = 1.0,
        factor_damping: float = 1e-5,
        cache_generalized_basis: bool = True,
        debug_factor_stats: bool = True,
        factor_stats_quantiles: Tuple[float, ...] = (
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ),
        factor_stats_sample_size: int = 0,
        factor_stats_json_path: Optional[str] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )

        defaults = {
            "projection_cache_map": projection_cache_map or {},
            "additional_projection_cache_map": additional_projection_cache_map or {},
            "soft_lambda": float(soft_lambda),
        }
        for group in self.param_groups:
            group.update(defaults)

        self._validate_exact_groups()
        self._exact_step_applied = False
        self.factor_damping = max(float(factor_damping), 0.0)
        self.cache_generalized_basis = bool(cache_generalized_basis)
        self.debug_factor_stats = bool(debug_factor_stats)
        self.factor_stats_quantiles = tuple(
            q
            for q in (float(q) for q in factor_stats_quantiles)
            if 0.0 <= q <= 1.0
        )
        self.factor_stats_sample_size = max(int(factor_stats_sample_size), 0)
        repo_root = Path(__file__).resolve().parents[3]
        self.factor_stats_json_path = Path(
            factor_stats_json_path or repo_root / "projected_sgd_factor_stats.json"
        )
        self._factor_stats_recorded = set()
        self._factor_stats_records = {
            "quantiles": list(self.factor_stats_quantiles),
            "sample_size": self.factor_stats_sample_size,
            "layers": [],
        }

    def _validate_exact_groups(self):
        for group in self.param_groups:
            if float(group.get("lr", 0.0)) != 1.0:
                print(f"[lr] -> {float(group.get('lr', 0.0))}")
            if float(group.get("momentum", 0.0)) != 0.0:
                raise ValueError("The section 3.2 exact update requires momentum=0.")
            if float(group.get("dampening", 0.0)) != 0.0:
                raise ValueError("The section 3.2 exact update requires dampening=0.")
            if float(group.get("weight_decay", 0.0)) != 0.0:
                raise ValueError("The section 3.2 exact update requires weight_decay=0.")
            if bool(group.get("nesterov", False)):
                raise ValueError("The section 3.2 exact update does not support Nesterov momentum.")
            if bool(group.get("maximize", False)):
                raise ValueError("The section 3.2 exact update requires maximize=False.")

    def _validate_exact_gradients(self):
        for group in self.param_groups:
            cache_map = group.get("projection_cache_map", {}) or {}
            additional_cache_map = (
                group.get("additional_projection_cache_map", {}) or {}
            )
            if additional_cache_map:
                raise RuntimeError(
                    "The one-shot section 3.2 validation does not support a sequential "
                    "additional projection cache."
            )

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    raise RuntimeError(
                        "A trainable matrix has no accumulated full-data gradient; "
                        "the section 3.2 solve would be incomplete."
                    )
                if grad.is_sparse or grad.ndim != 2:
                    raise RuntimeError(
                        "The one-shot soft K-FAC solve only supports dense matrix gradients."
                    )
                if p not in cache_map:
                    raise RuntimeError(
                        "Missing soft K-FAC cache for a trainable matrix; refusing to "
                        "fall back to an unprojected SGD update."
                    )

    def reset_cache_old(self, new_projection_cache_map):
        self.reset_cache(new_projection_cache_map)

    def reset_cache(self, new_projection_cache_map):
        new_projection_cache_map = new_projection_cache_map or {}
        for group in self.param_groups:
            group["projection_cache_map"] = new_projection_cache_map
            self._project_momentum(group, new_projection_cache_map)

    def reset_additional_cache(self, additional_projection_cache_map):
        additional_projection_cache_map = additional_projection_cache_map or {}
        for group in self.param_groups:
            group["additional_projection_cache_map"] = additional_projection_cache_map
            self._project_momentum(group, additional_projection_cache_map)

    def _project_momentum(self, group, cache_map: Dict):
        for p in group["params"]:
            if p not in self.state or p not in cache_map:
                continue

            momentum_buffer = self.state[p].get("momentum_buffer")
            if momentum_buffer is None or momentum_buffer.ndim != 2:
                continue

            projected = self._soft_kfac_precondition(
                momentum_buffer,
                cache_map[p],
                soft_lambda=group.get("soft_lambda", 1.0),
            )
            if projected is not None:
                momentum_buffer.copy_(projected)

    @staticmethod
    def _tensor(cache: Dict, key: str, like: torch.Tensor, dtype: torch.dtype):
        value = cache.get(key, None)
        if value is None:
            return None
        return value.to(device=like.device, dtype=dtype)

    @staticmethod
    def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _check_square(matrix: torch.Tensor, name: str):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be a square matrix, got {tuple(matrix.shape)}.")

    @staticmethod
    def _check_basis_shape(
        tensor: torch.Tensor,
        q_a: torch.Tensor,
        q_b: torch.Tensor,
        eig_a: torch.Tensor,
        eig_b: torch.Tensor,
    ):
        out_dim, in_dim = tensor.shape
        if q_a.ndim != 2 or q_b.ndim != 2:
            raise ValueError(
                f"Generalized bases must be matrices, got q_a={tuple(q_a.shape)}, "
                f"q_b={tuple(q_b.shape)}."
            )
        if q_a.shape[0] != in_dim or q_b.shape[0] != out_dim:
            raise ValueError(
                "Generalized basis dimension mismatch: "
                f"tensor={tuple(tensor.shape)}, q_a={tuple(q_a.shape)}, "
                f"q_b={tuple(q_b.shape)}."
            )
        if eig_a.numel() != q_a.shape[1] or eig_b.numel() != q_b.shape[1]:
            raise ValueError(
                "Generalized eigenvalue dimension mismatch: "
                f"eig_a={tuple(eig_a.shape)}, q_a={tuple(q_a.shape)}, "
                f"eig_b={tuple(eig_b.shape)}, q_b={tuple(q_b.shape)}."
            )

    def _generalized_basis(
        self,
        edit_factor: torch.Tensor,
        cap_factor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_square(edit_factor, "edit_factor")
        self._check_square(cap_factor, "cap_factor")

        if edit_factor.shape != cap_factor.shape:
            raise ValueError(
                "K-FAC factor shape mismatch: "
                f"edit={tuple(edit_factor.shape)}, cap={tuple(cap_factor.shape)}."
            )

        edit_factor = self._symmetrize(edit_factor)
        cap_factor = self._symmetrize(cap_factor)

        n = edit_factor.shape[0]
        trace_scale = edit_factor.diagonal().abs().mean().clamp(min=1e-12)
        eps = self.factor_damping * trace_scale
        edit_factor_reg = edit_factor + eps * torch.eye(n, device=edit_factor.device, dtype=edit_factor.dtype)

        # 3) 条件数统计：阻尼前后对比，诊断 edit_factor 病态程度与阻尼效果。
        #    若 cond(后) 仍很大，说明 factor_damping 偏小，Cholesky 可能不稳。
        eigvals = torch.linalg.eigvalsh(edit_factor)
        cond = eigvals.max() / eigvals.clamp(min=1e-20).min()
        eigvals_reg = torch.linalg.eigvalsh(edit_factor_reg)
        cond_reg = eigvals_reg.max() / eigvals_reg.clamp(min=1e-20).min()
        print(f"条件数(阻尼前): {cond.item():.6f}  条件数(阻尼后): {cond_reg.item():.6f}  "
              f"eig[min,max]=[{eigvals.clamp(min=1e-20).min().item():.3e}, {eigvals.max().item():.3e}]  "
              f"eps={eps.item():.3e}")

        L = torch.linalg.cholesky(edit_factor_reg) 
        tmp = torch.linalg.solve_triangular(L, cap_factor, upper=False)        # 解 L @ tmp = cap_factor
        whitened_cap = torch.linalg.solve_triangular(L, tmp.T, upper=False).T
        cap_eigs, cap_vecs = torch.linalg.eigh(whitened_cap)
        q = torch.linalg.solve_triangular(L.transpose(-1, -2), cap_vecs, upper=True)

        cap_eigs = torch.clamp(cap_eigs, min=0.0)
        return q.contiguous(), cap_eigs.contiguous()

    def _basis_from_precomputed(
        self,
        tensor: torch.Tensor,
        cache: Dict,
        dtype: torch.dtype,
    ):
        for q_a_key, q_b_key, eig_a_key, eig_b_key in self._PRECOMPUTED_BASIS_KEYS:
            if all(key in cache for key in (q_a_key, q_b_key, eig_a_key, eig_b_key)):
                q_a = self._tensor(cache, q_a_key, tensor, dtype)
                q_b = self._tensor(cache, q_b_key, tensor, dtype)
                eig_a = self._tensor(cache, eig_a_key, tensor, dtype)
                eig_b = self._tensor(cache, eig_b_key, tensor, dtype)
                return q_a, q_b, eig_a.flatten(), eig_b.flatten()
        return None

    def _factors_from_cache(self, tensor: torch.Tensor, cache: Dict, dtype: torch.dtype):
        for edit_keys, cap_keys in self._FACTOR_KEY_SETS:
            if all(key in cache for key in (*edit_keys, *cap_keys)):
                edit_a = self._tensor(cache, edit_keys[0], tensor, dtype)
                edit_b = self._tensor(cache, edit_keys[1], tensor, dtype)
                cap_a = self._tensor(cache, cap_keys[0], tensor, dtype)
                cap_b = self._tensor(cache, cap_keys[1], tensor, dtype)
                return edit_a, edit_b, cap_a, cap_b
        return None

    def _basis_from_factors(self, tensor: torch.Tensor, cache: Dict, dtype: torch.dtype):
        factors = self._factors_from_cache(tensor, cache, dtype)
        if factors is None:
            return None

        edit_a, edit_b, cap_a, cap_b = factors
        q_a, eig_a = self._generalized_basis(edit_a, cap_a)
        q_b, eig_b = self._generalized_basis(edit_b, cap_b)

        if self.cache_generalized_basis:
            cache["soft_q_a"] = q_a.detach().cpu()
            cache["soft_q_b"] = q_b.detach().cpu()
            cache["soft_eig_a"] = eig_a.detach().cpu()
            cache["soft_eig_b"] = eig_b.detach().cpu()

        return q_a, q_b, eig_a, eig_b

    def _get_generalized_basis(self, tensor: torch.Tensor, cache: Dict, dtype: torch.dtype):
        precomputed = self._basis_from_precomputed(tensor, cache, dtype)
        if precomputed is not None:
            return precomputed
        return self._basis_from_factors(tensor, cache, dtype)

    @staticmethod
    def _exact_quantiles(values: torch.Tensor, quantiles: Tuple[float, ...]) -> Dict:
        num_values = values.numel()
        if num_values == 0 or len(quantiles) == 0:
            return {}

        kth_cache = {}

        def kth_value(zero_based_idx: int) -> float:
            zero_based_idx = max(0, min(int(zero_based_idx), num_values - 1))
            if zero_based_idx not in kth_cache:
                kth_cache[zero_based_idx] = torch.kthvalue(
                    values,
                    zero_based_idx + 1,
                ).values.item()
            return kth_cache[zero_based_idx]

        result = {}
        for quantile in quantiles:
            position = float(quantile) * (num_values - 1)
            lower_idx = int(position)
            upper_idx = min(lower_idx + 1, num_values - 1)
            lower_value = kth_value(lower_idx)
            upper_value = kth_value(upper_idx)
            weight = position - lower_idx
            result[f"{quantile:.4g}"] = lower_value + (upper_value - lower_value) * weight
        return result

    def _distribution_stats(self, values: torch.Tensor) -> Dict:
        flat = values.detach().reshape(-1)
        total = flat.numel()
        if total == 0:
            return {
                "numel": 0,
                "min": None,
                "max": None,
                "quantile_source": "empty",
                "quantiles": {},
            }

        flat = flat.float()
        finite_mask = torch.isfinite(flat)
        finite_flat = flat if finite_mask.all().item() else flat[finite_mask]
        if finite_flat.numel() == 0:
            return {
                "numel": total,
                "finite_numel": 0,
                "min": None,
                "max": None,
                "quantile_source": "non_finite",
                "quantiles": {},
            }

        min_value = finite_flat.min().item()
        max_value = finite_flat.max().item()

        quantile_source = "exact"
        quantile_values = finite_flat
        if (
            self.factor_stats_sample_size > 0
            and finite_flat.numel() > self.factor_stats_sample_size
        ):
            sample_idx = torch.arange(
                self.factor_stats_sample_size,
                device=finite_flat.device,
                dtype=torch.long,
            )
            if self.factor_stats_sample_size > 1:
                sample_idx = (
                    sample_idx
                    * (finite_flat.numel() - 1)
                    // (self.factor_stats_sample_size - 1)
                )
            quantile_values = finite_flat[sample_idx]
            quantile_source = (
                f"sample={self.factor_stats_sample_size}/{finite_flat.numel()}"
            )

        if len(self.factor_stats_quantiles) == 0:
            quantiles = {}
        else:
            finite_values = quantile_values.detach().cpu().contiguous()
            quantiles = self._exact_quantiles(
                finite_values,
                self.factor_stats_quantiles,
            )

        return {
            "numel": total,
            "finite_numel": finite_flat.numel(),
            "min": min_value,
            "max": max_value,
            "quantile_source": quantile_source,
            "quantiles": quantiles,
        }

    def _save_factor_stats(self):
        self.factor_stats_json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.factor_stats_json_path.open("w", encoding="utf-8") as handle:
            json.dump(self._factor_stats_records, handle, indent=2, sort_keys=True)

    def _maybe_record_factor_stats(
        self,
        cache: Dict,
        eig_a: torch.Tensor,
        eig_b: torch.Tensor,
        joint_eigs: torch.Tensor,
        soft_lambda: float,
    ):
        if not self.debug_factor_stats:
            return

        cache_key = id(cache)
        if cache_key in self._factor_stats_recorded:
            return
        self._factor_stats_recorded.add(cache_key)

        layer_name = cache.get("layer_name", "<unknown>")
        self._factor_stats_records["layers"].append({
            "layer_name": str(layer_name),
            "lambda": float(soft_lambda),
            "a": self._distribution_stats(eig_a),
            "b": self._distribution_stats(eig_b),
            "ba": self._distribution_stats(joint_eigs),
        })
        self._save_factor_stats()

    def _soft_kfac_precondition(
        self,
        tensor: torch.Tensor,
        cache: Optional[Dict],
        soft_lambda: float,
    ):
        if cache is None or tensor.ndim != 2:
            return None

        compute_dtype = (
            torch.float32
            if tensor.dtype in (torch.float16, torch.bfloat16)
            else tensor.dtype
        )
        source = tensor.to(dtype=compute_dtype)

        basis = self._get_generalized_basis(source, cache, compute_dtype)
        if basis is None:
            return None

        q_a, q_b, eig_a, eig_b = basis
        self._check_basis_shape(source, q_a, q_b, eig_a, eig_b)

        coeffs = q_b.T @ source @ q_a
        joint_eigs = torch.outer(
            torch.clamp(eig_b.flatten(), min=0.0),
            torch.clamp(eig_a.flatten(), min=0.0),
        ).to(device=source.device, dtype=source.dtype)
        self._maybe_record_factor_stats(cache, eig_a, eig_b, joint_eigs, soft_lambda)
        denom = 1.0 + float(soft_lambda) * joint_eigs
        preconditioned = q_b @ (coeffs / denom.clamp(min=1e-12)) @ q_a.T
        return preconditioned.to(dtype=tensor.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError(
                "The one-shot section 3.2 update requires gradients to be accumulated "
                "before step(); optimizer closures are not supported."
            )
        if self._exact_step_applied:
            raise RuntimeError(
                "ProjectedSGD.step() may only be called once for the section 3.2 "
                "closed-form update."
            )

        self._validate_exact_groups()
        self._validate_exact_gradients()

        for group in self.param_groups:
            cache_map = group.get("projection_cache_map", {}) or {}
            additional_cache_map = (
                group.get("additional_projection_cache_map", {}) or {}
            )
            soft_lambda = group.get("soft_lambda", 1.0)

            for p in group["params"]:
                grad = p.grad
                if grad is None or grad.is_sparse or grad.ndim != 2:
                    continue

                grad_proj = None
                if p in cache_map:
                    grad_proj = self._soft_kfac_precondition(
                        grad,
                        cache_map[p],
                        soft_lambda=soft_lambda,
                    )

                if p in additional_cache_map:
                    print("[additional_cache_map] is not null")
                    source_grad = grad_proj if grad_proj is not None else grad
                    additional_proj = self._soft_kfac_precondition(
                        source_grad,
                        additional_cache_map[p],
                        soft_lambda=soft_lambda,
                    )
                    if additional_proj is not None:
                        grad_proj = additional_proj

                if grad_proj is None:
                    raise RuntimeError(
                        "Soft K-FAC cache is incomplete; refusing to apply a raw SGD update."
                    )
                grad.copy_(grad_proj)

        super().step()
        self._exact_step_applied = True
        return None
