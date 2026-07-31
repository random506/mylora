import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from dotenv import load_dotenv
from torch.optim import Adam

# 加载 .env 环境变量（STATS_DIR 等），供后续梯度范数统计的保存路径使用。
load_dotenv()


class ProjectedAdam(Adam):
    """
    第 3.2 节 K-FAC 软约束更新对应的 Adam 包装器。

    对于一个权重块 W 及其梯度 G，K-FAC 软约束 Newton 系统为

        B_e dW A_e + lambda B_c dW A_c = -G.

    本实现会构造广义基 Q_A、Q_B，使其满足

        Q_A.T A_e Q_A = I, Q_A.T A_c Q_A = diag(a)
        Q_B.T B_e Q_B = I, Q_B.T B_c Q_B = diag(b)

    令 R_A = Q_A^{-T}, R_B = Q_B^{-T} 为对偶基，然后用下式过滤原始梯度

        Q_B [(R_B.T G R_A) / (1 + lambda * outer(b, a))] Q_A.T.

    当 lambda=0 时该变换精确重建 G；lambda>0 时按广义特征值过滤方向。
    过滤结果还会受 max_relative_change 限制，再交给 Adam 执行常规下降步骤。

    ----------------------------------------------------------------
    设计要点：
    1. 本类继承 Adam，不重写 Adam 的动量/二阶矩逻辑，只在 step() 里把
       原始梯度 p.grad 原地替换为「K-FAC 软预条件后的梯度」，再交给父类 step。
    2. 广义基 Q_A/Q_B 可以从缓存里直接读（precomputed），也可以从 A/B 因子
       现算（_generalized_basis）。现算结果可选地回写缓存避免重复计算。
    3. 因子（edit_A/edit_B = 编辑任务协方差, cap_A/cap_B = 保留知识协方差）
       由上游 utils.py 的 calculate_projection_caches_from_cov_caches 预先算好，
       通过 projection_cache_map 传入，本类不负责统计协方差。
    """

    # 预计算广义基的缓存键名。Q 用于重建，dual_Q=Q^{-T} 用于提取坐标。
    _PRECOMPUTED_BASIS_KEYS = (
        (
            "soft_q_a",
            "soft_q_b",
            "soft_dual_q_a",
            "soft_dual_q_b",
            "soft_eig_a",
            "soft_eig_b",
        ),
    )

    # A/B 因子的缓存键名集合。第一组是编辑因子(edit_*)，第二组是保留因子(cap_*)。
    # 只要 cache 同时含这 4 个键，就能用 _generalized_basis 现算广义基。
    _FACTOR_KEY_SETS = (
        (("edit_A", "edit_B"), ("cap_A", "cap_B")),
    )

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
        soft_lambda: float = 1.0,            # 软约束强度 λ，越大则高能力方向梯度收缩越强
        factor_damping: float = 1e-5,        # edit_factor 阻尼系数，保证 Cholesky 正定
        cache_generalized_basis: bool = True,# 是否把算出的广义基回写缓存
        max_relative_change: Optional[float] = 0.2,
        debug_factor_stats: bool = True,     # 是否记录特征值分布统计到 JSON
        factor_stats_quantiles: Tuple[float, ...] = (
            0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99,
        ),
        factor_stats_sample_size: int = 0,   # 0=精确分位数; >0=对大张量做确定性下采样
        factor_stats_json_path: Optional[str] = None,
        debug_grad_norm_stats: bool = True,  # 是否记录投影前后梯度范数
        grad_norm_stats_json_path: Optional[str] = None,
    ):
        # 先调用 Adam 父类初始化（lr/betas/eps/weight_decay/amsgrad）。
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

        # 把 K-FAC 相关配置写入每个 param_group，供 step() 时按组读取。
        defaults = {
            "projection_cache_map": projection_cache_map or {},
            "additional_projection_cache_map": additional_projection_cache_map or {},
            "soft_lambda": float(soft_lambda),
        }
        for group in self.param_groups:
            group.update(defaults)

        # ---- 广义基 / 阻尼相关 ----
        self.factor_damping = max(float(factor_damping), 0.0)
        self.cache_generalized_basis = bool(cache_generalized_basis)
        self.max_relative_change = (
            None
            if max_relative_change is None
            else max(float(max_relative_change), 0.0)
        )

        # ---- 特征值分布统计相关 ----
        self.debug_factor_stats = bool(debug_factor_stats)
        # 过滤合法分位数 [0,1]。
        self.factor_stats_quantiles = tuple(
            q
            for q in (float(q) for q in factor_stats_quantiles)
            if 0.0 <= q <= 1.0
        )
        # 0 means exact quantiles over all finite values; a positive value enables
        # deterministic subsampling for very large tensors.
        self.factor_stats_sample_size = max(int(factor_stats_sample_size), 0)
        # 统计 JSON 默认存到仓库根目录。
        repo_root = Path(__file__).resolve().parents[3]
        self.factor_stats_json_path = Path(
            factor_stats_json_path or repo_root / "projected_adam_factor_stats.json"
        )
        # 用 id(cache) 去重，每个 cache 只记录一次特征值统计。
        self._factor_stats_recorded = set()
        self._factor_stats_records = {
            "quantiles": list(self.factor_stats_quantiles),
            "sample_size": self.factor_stats_sample_size,
            "layers": [],
        }

        # ---- 梯度范数统计相关 ----
        # 保存路径与 utils.py 一致（STATS_DIR），文件名不同，便于和离线脚本对照。
        self.debug_grad_norm_stats = bool(debug_grad_norm_stats)
        _stats_dir = os.getenv("STATS_DIR")
        _grad_norm_dir = Path(_stats_dir) if _stats_dir else repo_root
        self.grad_norm_stats_json_path = Path(
            grad_norm_stats_json_path or _grad_norm_dir / "projected_adam_grad_norm_stats.json"
        )
        self._grad_norm_step = 0
        self._grad_norm_records = {"steps": []}

    # ======================================================================
    # 缓存切换：顺序编辑时，每个 edit batch 会换新的协方差缓存。
    # 切换后既要把新缓存存进 param_group，也要把已累积的 Adam 动量 exp_avg
    # 按新基重新投影，避免动量停留在旧基里造成方向错配。
    # ======================================================================

    def reset_cache_old(self, new_projection_cache_map):
        # 旧接口别名，向后兼容。
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
        # 对每个有动量且有缓存的参数，把 exp_avg 通过软 K-FAC 预条件投影一遍。
        for p in group["params"]:
            if p not in self.state or p not in cache_map:
                continue

            state = self.state[p]
            exp_avg = state.get("exp_avg", None)
            # 只有 2D 权重才有 K-FAC A/B 因子；1D(bias/标量)跳过。
            if exp_avg is None or exp_avg.ndim != 2:
                continue

            projected = self._soft_kfac_precondition(
                exp_avg,
                cache_map[p],
                soft_lambda=group.get("soft_lambda", 1.0),
            )
            if projected is not None:
                exp_avg.copy_(projected)

    # ======================================================================
    # 小工具
    # ======================================================================

    @staticmethod
    def _tensor(cache: Dict, key: str, like: torch.Tensor, dtype: torch.dtype):
        # 从 cache 取张量并搬到指定 device/dtype；缺失返回 None。
        value = cache.get(key, None)
        if value is None:
            return None
        return value.to(device=like.device, dtype=dtype)

    @staticmethod
    def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
        # 对称化：M -> (M+M^T)/2，消除数值上的轻微不对称，保证后续 eigh/cholesky 稳定。
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _check_square(matrix: torch.Tensor, name: str):
        # 校验为方阵，否则广义特征分解无定义。
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be a square matrix, got {tuple(matrix.shape)}.")

    @staticmethod
    def _check_basis_shape(
        tensor: torch.Tensor,
        q_a: torch.Tensor,
        q_b: torch.Tensor,
        dual_q_a: torch.Tensor,
        dual_q_b: torch.Tensor,
        eig_a: torch.Tensor,
        eig_b: torch.Tensor,
    ):
        # 校验广义基与权重张量形状自洽：
        #   q_a 列数 = 输入维特征值数, q_b 列数 = 输出维特征值数；
        #   q_a 行数 = 输入维, q_b 行数 = 输出维。
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
        if dual_q_a.shape != q_a.shape or dual_q_b.shape != q_b.shape:
            raise ValueError(
                "Dual basis dimension mismatch: "
                f"q_a={tuple(q_a.shape)}, dual_q_a={tuple(dual_q_a.shape)}, "
                f"q_b={tuple(q_b.shape)}, dual_q_b={tuple(dual_q_b.shape)}."
            )
        if eig_a.numel() != q_a.shape[1] or eig_b.numel() != q_b.shape[1]:
            raise ValueError(
                "Generalized eigenvalue dimension mismatch: "
                f"eig_a={tuple(eig_a.shape)}, q_a={tuple(q_a.shape)}, "
                f"eig_b={tuple(eig_b.shape)}, q_b={tuple(q_b.shape)}."
            )

    # ======================================================================
    # 核心：广义特征分解 _generalized_basis
    #
    # 目标：给定 edit_factor (=A_e, 编辑任务协方差) 和 cap_factor (=A_c, 保留
    # 知识协方差)，求广义基 Q 与特征值 a，使得
    #     Q^T A_e Q = I,   Q^T A_c Q = diag(a)
    # 数学做法（Cholesky 路线）：
    #   1. A_e 加阻尼 eps*I 保证正定：A_e_reg = A_e + eps*I
    #   2. Cholesky: A_e_reg = L L^T
    #   3. 白化 A_c:  whitened = L^{-1} A_c L^{-T}   （在 A_e=I 的基下看 A_c）
    #   4. 对 whitened 做普通 eigh:  (a, cap_vecs)
    #   5. 广义基 q = L^{-T} cap_vecs
    # 可验证: q^T A_e q = cap_vecs^T L^{-1} (L L^T) L^{-T} cap_vecs = cap_vecs^T cap_vecs = I
    #         q^T A_c q = cap_vecs^T (L^{-1} A_c L^{-T}) cap_vecs = diag(a)  ✓
    #
    # 关键优化：不显式构造 L_inv，而是用三角回代(solve_triangular)直接把
    # L^{-1} 作用到 cap_factor / cap_vecs 上，更稳、更省、不物化逆矩阵。
    # ======================================================================

    # 原实现（基于 eigh 特征分解求 inv_sqrt），保留备查：
    # def _generalized_basis(
    #     self,
    #     edit_factor: torch.Tensor,
    #     cap_factor: torch.Tensor,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     self._check_square(edit_factor, "edit_factor")
    #     self._check_square(cap_factor, "cap_factor")
    #     if edit_factor.shape != cap_factor.shape:
    #         raise ValueError(
    #             "K-FAC factor shape mismatch: "
    #             f"edit={tuple(edit_factor.shape)}, cap={tuple(cap_factor.shape)}."
    #         )
    #
    #     edit_factor = self._symmetrize(edit_factor)
    #     cap_factor = self._symmetrize(cap_factor)
    #
    #     edit_eigs, edit_vecs = torch.linalg.eigh(edit_factor)
    #     scale = edit_eigs.abs().max().clamp(min=1.0)
    #     floor = self.factor_damping * scale
    #     edit_eigs = torch.clamp(edit_eigs, min=floor)
    #
    #     inv_sqrt = edit_vecs @ torch.diag(torch.rsqrt(edit_eigs)) @ edit_vecs.T
    #     whitened_cap = self._symmetrize(inv_sqrt @ cap_factor @ inv_sqrt)
    #     cap_eigs, cap_vecs = torch.linalg.eigh(whitened_cap)
    #
    #     q = inv_sqrt @ cap_vecs
    #     cap_eigs = torch.clamp(cap_eigs, min=0.0)
    #     return q.contiguous(), cap_eigs.contiguous()

    def _generalized_basis(self, edit_factor, cap_factor):
        self._check_square(edit_factor, "edit_factor")
        self._check_square(cap_factor, "cap_factor")

        # 1) 对称化，消除数值不对称。
        edit_factor = self._symmetrize(edit_factor)
        cap_factor = self._symmetrize(cap_factor)

        # 2) 阻尼：eps = factor_damping * |对角元|均值，相对谱尺度自适应。
        #    eps*I 抬高近零/负特征值，保证 A_e_reg 正定可 Cholesky。
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

        # 4) Cholesky 分解：A_e_reg = L L^T (L 下三角)。
        L = torch.linalg.cholesky(edit_factor_reg)
        # 原实现：显式构造 L_inv 再做稠密矩阵乘（保留备查）
        # L_inv = torch.linalg.solve_triangular(L, torch.eye(n, device=L.device, dtype=L.dtype), upper=False)
        #
        # whitened_cap = self._symmetrize(L_inv @ cap_factor @ L_inv.T)
        # q = L_inv.T @ cap_vecs

        # 5) 白化 cap_factor：求 whitened_cap = L^{-1} A_c L^{-T}，用两次三角回代
        #    避免显式构造 L_inv（数值更稳、省一个 n×n 逆矩阵）。
        #    step a: tmp = L^{-1} A_c        (解 L·tmp = A_c)
        #    step b: whitened = (L^{-1} tmp^T)^T = tmp · L^{-T} = L^{-1} A_c L^{-T}
        tmp = torch.linalg.solve_triangular(L, cap_factor, upper=False)        # 解 L @ tmp = cap_factor
        whitened_cap = torch.linalg.solve_triangular(L, tmp.T, upper=False).T  # 解 L & X.T = tmp.T  =>  L^{-1} cap_factor L^{-T}
        # 6) 对白化后的矩阵做标准对称特征分解，得到 cap 特征值 a 与特征向量。
        cap_eigs, cap_vecs = torch.linalg.eigh(whitened_cap)

        # 7) 广义基 q = L^{-T} cap_vecs，同样用三角回代（解 L^T·q = cap_vecs，
        #    L^T 为上三角，故 upper=True），避免 L_inv。
        q = torch.linalg.solve_triangular(L.transpose(-1, -2), cap_vecs, upper=True)
        # The dual basis R = Q^{-T} = L V extracts coordinates without applying
        # the inverse edit curvature a second time. It satisfies R^T Q = I.
        dual_q = L @ cap_vecs
        # 特征值截断到非负（数值上可能出微小负值）。
        cap_eigs = torch.clamp(cap_eigs, min=0.0)

        # 8) 有限性兜底：若出现 NaN/Inf，说明因子病态，直接报错而非静默产出坏结果。
        if (
            not torch.isfinite(q).all()
            or not torch.isfinite(dual_q).all()
            or not torch.isfinite(cap_eigs).all()
        ):
            raise RuntimeError("generalized basis produced non-finite values, check factor conditioning")

        return q.contiguous(), dual_q.contiguous(), cap_eigs.contiguous()

    # ======================================================================
    # 广义基的获取：优先用缓存里现成的，否则从 A/B 因子现算并回写缓存。
    # ======================================================================

    def _basis_from_precomputed(
        self,
        tensor: torch.Tensor,
        cache: Dict,
        dtype: torch.dtype,
    ):
        # 若 cache 里已有 (soft_q_a, soft_q_b, soft_eig_a, soft_eig_b)，直接取出。
        for (
            q_a_key,
            q_b_key,
            dual_q_a_key,
            dual_q_b_key,
            eig_a_key,
            eig_b_key,
        ) in self._PRECOMPUTED_BASIS_KEYS:
            keys = (
                q_a_key,
                q_b_key,
                dual_q_a_key,
                dual_q_b_key,
                eig_a_key,
                eig_b_key,
            )
            if all(key in cache for key in keys):
                q_a = self._tensor(cache, q_a_key, tensor, dtype)
                q_b = self._tensor(cache, q_b_key, tensor, dtype)
                dual_q_a = self._tensor(cache, dual_q_a_key, tensor, dtype)
                dual_q_b = self._tensor(cache, dual_q_b_key, tensor, dtype)
                eig_a = self._tensor(cache, eig_a_key, tensor, dtype)
                eig_b = self._tensor(cache, eig_b_key, tensor, dtype)
                return (
                    q_a,
                    q_b,
                    dual_q_a,
                    dual_q_b,
                    eig_a.flatten(),
                    eig_b.flatten(),
                )
        return None

    def _factors_from_cache(self, tensor: torch.Tensor, cache: Dict, dtype: torch.dtype):
        # 从 cache 取 (edit_A, edit_B, cap_A, cap_B) 四个 K-FAC 因子。
        for edit_keys, cap_keys in self._FACTOR_KEY_SETS:
            if all(key in cache for key in (*edit_keys, *cap_keys)):
                edit_a = self._tensor(cache, edit_keys[0], tensor, dtype)
                edit_b = self._tensor(cache, edit_keys[1], tensor, dtype)
                cap_a = self._tensor(cache, cap_keys[0], tensor, dtype)
                cap_b = self._tensor(cache, cap_keys[1], tensor, dtype)
                return edit_a, edit_b, cap_a, cap_b
        return None

    def _basis_from_factors(self, tensor: torch.Tensor, cache: Dict, dtype: torch.dtype):
        # 从因子现算广义基：对 A 侧 (edit_A, cap_A) 和 B 侧 (edit_B, cap_B) 各做一次。
        factors = self._factors_from_cache(tensor, cache, dtype)
        if factors is None:
            return None

        edit_a, edit_b, cap_a, cap_b = factors
        q_a, dual_q_a, eig_a = self._generalized_basis(edit_a, cap_a)
        q_b, dual_q_b, eig_b = self._generalized_basis(edit_b, cap_b)

        # 回写缓存，后续 step 直接走 _basis_from_precomputed，省去重复特征分解。
        if self.cache_generalized_basis:
            cache["soft_q_a"] = q_a.detach().cpu()
            cache["soft_q_b"] = q_b.detach().cpu()
            cache["soft_dual_q_a"] = dual_q_a.detach().cpu()
            cache["soft_dual_q_b"] = dual_q_b.detach().cpu()
            cache["soft_eig_a"] = eig_a.detach().cpu()
            cache["soft_eig_b"] = eig_b.detach().cpu()

        return q_a, q_b, dual_q_a, dual_q_b, eig_a, eig_b

    def _get_generalized_basis(self, tensor: torch.Tensor, cache: Dict, dtype: torch.dtype):
        # 调度：先查预计算基，没有再从因子现算。
        precomputed = self._basis_from_precomputed(tensor, cache, dtype)
        if precomputed is not None:
            return precomputed
        return self._basis_from_factors(tensor, cache, dtype)

    # ======================================================================
    # 特征值分布统计（debug 用，写 JSON）
    # ======================================================================

    @staticmethod
    def _exact_quantiles(values: torch.Tensor, quantiles: Tuple[float, ...]) -> Dict:
        # 用 kthvalue 精确求分位数，并用 dict 缓存已算过的 kth 值避免重复排序。
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

        # 对每个分位点做线性插值（介于两个 kth 值之间）。
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
        # 计算一个张量的分布统计：numel/min/max/分位数，自动剔除 NaN/Inf。
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

        # 大张量可选确定性下采样以控开销；默认 0 = 精确。
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
        # 把累积的特征值统计写 JSON。
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
        # 每个 cache 只记录一次（按 id 去重），记录 a/b/ba 三组特征值分布。
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

    # ======================================================================
    # 梯度范数统计（投影前后对比，写 JSON）
    # ======================================================================

    def _save_grad_norm_stats(self):
        self.grad_norm_stats_json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.grad_norm_stats_json_path.open("w", encoding="utf-8") as handle:
            json.dump(self._grad_norm_records, handle, indent=2, sort_keys=True)

    def _limit_relative_change(
        self,
        source: torch.Tensor,
        filtered: torch.Tensor,
    ) -> torch.Tensor:
        if self.max_relative_change is None:
            return filtered
        if not torch.isfinite(filtered).all():
            raise RuntimeError("spectral filtering produced non-finite values")

        delta = filtered - source
        delta_norm = delta.norm()
        if delta_norm == 0:
            return filtered

        source_norm = source.norm()
        if source_norm == 0:
            return source

        max_delta_norm = self.max_relative_change * source_norm
        scale = torch.clamp(max_delta_norm / delta_norm, max=1.0)
        return source + scale * delta

    def _record_grad_norm(
        self,
        cache: Optional[Dict],
        grad_before: torch.Tensor,
        grad_after: Optional[torch.Tensor],
    ):
        # 记录投影前后梯度的 L2 范数、范数比、余弦相似度，用于诊断预条件强度。
        if not self.debug_grad_norm_stats:
            return

        before_flat = grad_before.detach().float().reshape(-1)
        before_norm = before_flat.norm().item()
        if grad_after is None:
            after_norm = None
            norm_ratio = None
            change_norm = None
            relative_change = None
            cosine = None
        else:
            after_flat = grad_after.detach().float().reshape(-1)
            after_norm = after_flat.norm().item()
            norm_ratio = (after_norm / before_norm) if before_norm > 0 else None
            change_norm = (after_flat - before_flat).norm().item()
            relative_change = (
                change_norm / before_norm if before_norm > 0 else None
            )
            denom = before_norm * after_norm
            # cosine = <g_before, g_after> / (||g_before|| * ||g_after||)
            cosine = (
                torch.dot(before_flat, after_flat).item() / denom
                if denom > 0 else None
            )

        layer_name = cache.get("layer_name", "<unknown>") if cache else "<unknown>"
        self._grad_norm_records["steps"].append({
            "step": self._grad_norm_step,
            "layer_name": str(layer_name),
            "grad_norm_before": before_norm,
            "grad_norm_after": after_norm,
            "norm_ratio": norm_ratio,
            "grad_change_norm": change_norm,
            "relative_change": relative_change,
            "cosine_sim": cosine,
        })
        self._save_grad_norm_stats()

    # ======================================================================
    # 核心：K-FAC 软预条件 _soft_kfac_precondition
    #
    # 把输入张量（梯度或动量）按广义基变换到 (a,b) 谱空间，用
    #   1 / (1 + λ * b_i * a_j)
    # 做逐元素收缩（高能力曲率方向收缩更强，保护旧知识），再变回原空间。
    # 这是保持 lambda=0 恒等性的广义谱过滤，不再等价于原始 Newton 逆预条件。
    # ======================================================================

    def _soft_kfac_precondition(
        self,
        tensor: torch.Tensor,
        cache: Optional[Dict],
        soft_lambda: float,
    ):
        # 仅对 2D 张量且 cache 存在时生效；否则返回 None 表示「不改」。
        if cache is None or tensor.ndim != 2:
            return None

        # 半精度(fp16/bf16)下做特征分解不安全，统一升到 fp32 计算。
        compute_dtype = (
            torch.float32
            if tensor.dtype in (torch.float16, torch.bfloat16)
            else tensor.dtype
        )
        source = tensor.to(dtype=compute_dtype)

        basis = self._get_generalized_basis(source, cache, compute_dtype)
        if basis is None:
            return None

        q_a, q_b, dual_q_a, dual_q_b, eig_a, eig_b = basis
        self._check_basis_shape(
            source,
            q_a,
            q_b,
            dual_q_a,
            dual_q_b,
            eig_a,
            eig_b,
        )

        # R = Q^{-T} is the dual basis. Using R for analysis and Q for
        # synthesis makes the unfiltered transform exactly reconstruct source.
        coeffs = dual_q_b.T @ source @ dual_q_a
        # 联合特征值外积 b_i * a_j，截断到非负。
        joint_eigs = torch.outer(
            torch.clamp(eig_b.flatten(), min=0.0),
            torch.clamp(eig_a.flatten(), min=0.0),
        ).to(device=source.device, dtype=source.dtype)
        self._maybe_record_factor_stats(cache, eig_a, eig_b, joint_eigs, soft_lambda)
        # 软收缩分母：高能力方向 (b_i*a_j 大) 收缩更强。
        denom = 1.0 + float(soft_lambda) * joint_eigs
        # 预条件：谱空间逐元素除，再变回原空间。
        filtered = q_b @ (coeffs / denom.clamp(min=1e-12)) @ q_a.T
        # 对比！！！！！！！
        #preconditioned = self._limit_relative_change(source, filtered)
        return filtered.to(dtype=tensor.dtype)

    # ======================================================================
    # step：在 Adam 更新前，把每个矩阵权重的梯度原地替换为预条件梯度。
    # ======================================================================

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            # Adam 保持可选 closure 约定：如果调用方提供 closure，
            # 就在启用梯度的情况下重新计算 loss。
            with torch.enable_grad():
                loss = closure()

        # 梯度范数统计的步数计数器。
        self._grad_norm_step += 1

        for group in self.param_groups:
            # 主缓存是每个参数对应的主要 K-FAC 软约束缓存。
            # 按 PDF 中的记号，它应包含 H_e 和 H_c 因子，
            # 或者包含预先计算好的广义基。
            cache_map = group.get("projection_cache_map", {}) or {}

            # 额外缓存会在主缓存之后应用。
            # 当第二个保护来源需要进一步预条件化梯度时使用它。
            additional_cache_map = group.get("additional_projection_cache_map", {}) or {}

            # soft_lambda 对应 1 / (1 + lambda * b_i * a_j) 中的 lambda。
            # 该值越大，高能力曲率方向上的梯度收缩越强。
            soft_lambda = group.get("soft_lambda", 1.0)

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim != 2:
                    # 只有矩阵权重才有 K-FAC A/B 因子。
                    # bias、标量以及没有梯度的参数会退回普通 Adam 行为。
                    continue

                grad_proj = None
                if p in cache_map:
                    # 第一分支：使用主缓存执行第 3.2 节的 K-FAC 软求解。
                    # 如果缓存不完整，辅助函数会返回 None，
                    # 此处不会修改该参数的梯度。
                    # 记录投影前梯度，用于范数统计。
                    _grad_before = p.grad.detach()
                    grad_proj = self._soft_kfac_precondition(
                        p.grad,
                        cache_map[p],
                        soft_lambda=soft_lambda,
                    )
                    self._record_grad_norm(cache_map[p], _grad_before, grad_proj)

                # 当前不会进入
                if p in additional_cache_map:
                    # 第二分支：可选地再执行一次软求解。
                    # 如果主分支已经生成梯度，就基于该结果继续；
                    # 否则从原始梯度开始。
                    source_grad = grad_proj if grad_proj is not None else p.grad
                    additional_proj = self._soft_kfac_precondition(
                        source_grad,
                        additional_cache_map[p],
                        soft_lambda=soft_lambda,
                    )
                    if additional_proj is not None:
                        # 只有额外缓存可用时才覆盖结果。
                        # 如果第二个缓存缺少必要因子，就保留主分支结果。
                        grad_proj = additional_proj

                if grad_proj is not None:
                    # Adam 会在 super().step() 内部读取 p.grad。
                    # 因此这里原地写回预条件化后的梯度，
                    # 让 Adam 的动量从投影/软求解后的梯度更新。
                    p.grad.copy_(grad_proj)
        return super().step(closure)
