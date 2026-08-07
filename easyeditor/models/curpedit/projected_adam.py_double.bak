from typing import Dict, Optional

import torch
from torch.optim import Adam


class ProjectedAdam(Adam):
    def __init__(
        self,
        params,
        projection_cache_map: Optional[Dict] = None,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        additional_projection_cache_map: Optional[Dict] = None,
        use_second_projection: bool = True,
        newton_damping: float = 1e-3,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

        defaults = {
            "projection_cache_map": projection_cache_map or {},
            "additional_projection_cache_map": additional_projection_cache_map or {},
            "use_second_projection": use_second_projection,
        }
        for group in self.param_groups:
            group.update(defaults)

        self.newton_damping = max(float(newton_damping), 0.0)

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

            state = self.state[p]
            exp_avg = state.get("exp_avg", None)
            if exp_avg is None or exp_avg.ndim != 2:
                continue

            projected = self._project_tensor(
                exp_avg,
                cache_map[p],
                apply_task=group.get("use_second_projection", True),
            )
            if projected is not None:
                exp_avg.copy_(projected)

    @staticmethod
    def _tensor(cache: Dict, key: str, like: torch.Tensor):
        value = cache.get(key, None)
        if value is None:
            return None
        return value.to(device=like.device, dtype=like.dtype)

    @staticmethod
    def _check_basis_shape(tensor: torch.Tensor, mask_a: torch.Tensor, mask_b: torch.Tensor):
        out_dim, in_dim = tensor.shape
        if mask_a.shape[0] != in_dim or mask_b.shape[0] != out_dim:
            raise ValueError(
                "Projection basis dimension mismatch: "
                f"tensor={tuple(tensor.shape)}, "
                f"mask_a={tuple(mask_a.shape)}, mask_b={tuple(mask_b.shape)}."
            )

    def _newton_project(self, tensor: torch.Tensor, cache: Dict, prefix: str = ""):
        mask_a = self._tensor(cache, f"{prefix}mask_a", tensor)
        mask_b = self._tensor(cache, f"{prefix}mask_b", tensor)
        eig_a = self._tensor(cache, f"{prefix}eig_a", tensor)
        eig_b = self._tensor(cache, f"{prefix}eig_b", tensor)
        if mask_a is None or mask_b is None or eig_a is None or eig_b is None:
            return None
        if mask_a.numel() == 0 or mask_b.numel() == 0:
            return tensor

        self._check_basis_shape(tensor, mask_a, mask_b)
        coeffs = mask_b.T @ tensor @ mask_a
        joint_eigs = torch.outer(
            torch.clamp(eig_b.flatten(), min=0.0),
            torch.clamp(eig_a.flatten(), min=0.0),
        ).to(device=tensor.device, dtype=tensor.dtype)

        if joint_eigs.numel() == 0 or joint_eigs.max() <= 0:
            return tensor

        damping = self.newton_damping * joint_eigs.max().clamp(min=1e-12)
        strength = joint_eigs / (joint_eigs + damping)
        protected = mask_b @ (coeffs * strength) @ mask_a.T
        return tensor - protected

    def _hard_project(self, tensor: torch.Tensor, cache: Dict, prefix: str = ""):
        mask_a = self._tensor(cache, f"{prefix}mask_a", tensor)
        mask_b = self._tensor(cache, f"{prefix}mask_b", tensor)
        if mask_a is None or mask_b is None:
            return None
        if mask_a.numel() == 0 or mask_b.numel() == 0:
            return tensor

        self._check_basis_shape(tensor, mask_a, mask_b)
        I_out = torch.eye(mask_b.shape[0], device=tensor.device, dtype=tensor.dtype)
        I_in = torch.eye(mask_a.shape[0], device=tensor.device, dtype=tensor.dtype)
        return (I_out - mask_b @ mask_b.T) @ tensor @ (I_in - mask_a @ mask_a.T)

    def _project_tensor(
        self,
        tensor: torch.Tensor,
        cache: Optional[Dict],
        apply_task: bool = True,
    ):
        if cache is None or tensor.ndim != 2:
            return None
        '''
        # 先牛顿,再硬投影
        if apply_task:
            task_projected = self._newton_project(tensor, cache, prefix="task_")
            if task_projected is not None:
                result = task_projected

        result = self._hard_project(result, cache)
        if result is None:
            result = tensor
        '''
        # 先硬投影，再牛顿
        
        result = self._hard_project(tensor, cache)
        #result = self._newton_project(tensor, cache)
        if result is None:
            result = tensor

        if apply_task:
            print("[_project_tensor]")
            task_projected = self._newton_project(result, cache, prefix="task_")
            if task_projected is not None:
                result = task_projected
        
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            cache_map = group.get("projection_cache_map", {}) or {}
            additional_cache_map = group.get("additional_projection_cache_map", {}) or {}
            use_second_projection = group.get("use_second_projection", True)

            for p in group["params"]:
                if p.grad is None or p.grad.ndim != 2:
                    continue

                grad_proj = None
                if p in cache_map:
                    cache = cache_map[p]
                    has_task_projection = all(
                        key in cache
                        for key in ("task_mask_a", "task_mask_b", "task_eig_a", "task_eig_b")
                    )
                    grad_proj = self._project_tensor(
                        p.grad,
                        cache,
                        apply_task=use_second_projection,
                    )
                    print(
                        "[ProjectedAdam.step] primary projection done "
                        f"projected={grad_proj is not None}"
                    )

                if use_second_projection and p in additional_cache_map:
                    source_grad = grad_proj if grad_proj is not None else p.grad
                    print(
                        "[ProjectedAdam.step] additional newton projection "
                        f"shape={tuple(source_grad.shape)}"
                    )
                    task_projected = self._newton_project(source_grad, additional_cache_map[p])
                    if task_projected is not None:
                        grad_proj = task_projected
                    print(
                        "[ProjectedAdam.step] additional newton projection done "
                        f"projected={task_projected is not None}"
                    )

                if grad_proj is not None:
                    p.grad.copy_(grad_proj)

        return super().step(closure)
