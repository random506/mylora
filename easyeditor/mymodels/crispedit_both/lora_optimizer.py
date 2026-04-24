"""
lora_optimizer.py  —  crispedit_both
=====================================
复用自 mymodels/limit_lora/lora_optimizer.py。

GlobalAwareProjectedLoRAOptimizer：
  - 安全方向（低曲率，mask=1）：正常传递数据梯度
  - 危险方向（高曲率，mask=0）：用权重衰减拉力替代数据梯度（Subspace Weight Decay）

这是"Both"方案中的梯度侧约束。
参数侧约束由 CurvatureLora（前向投影）负责。
"""

import torch
from torch.optim import Adam
from typing import Dict, Optional


class GlobalAwareProjectedLoRAOptimizer(Adam):
    """
    带有全局意识的子空间投影优化器。
    在安全方向（低曲率）：正常使用数据梯度。
    在危险方向（高曲率）：将梯度替换为全局拉力（Subspace Weight Decay），防止动量漂移。
    """

    def __init__(
        self,
        params,
        projection_cache_map: Dict,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        projection_mode: str = "marginal_AB",
        subspace_penalty: float = 0.1,
    ):
        defaults = dict(
            projection_cache_map={},
            projection_mode=projection_mode,
            subspace_penalty=subspace_penalty,
        )
        super().__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad,
        )
        for group in self.param_groups:
            group.update(defaults)
        self._preload_cache(projection_cache_map)

    # ------------------------------------------------------------------
    # 缓存预加载
    # ------------------------------------------------------------------

    def _preload_cache(self, projection_cache_map: Dict):
        """
        将投影矩阵预先搬到各参数所在设备（GPU）并缓存。
        避免每次 step() 重复 CPU→GPU 数据搬运。
        """
        all_params = {}
        for group in self.param_groups:
            for p in group["params"]:
                all_params[id(p)] = p

        preloaded = {}
        for param, cache in projection_cache_map.items():
            if id(param) not in all_params:
                continue
            p = all_params[id(param)]
            dev, dtype = p.device, p.dtype

            new_cache = {"param_type": cache.get("param_type", "unknown")}
            if "Ua" in cache:
                new_cache["Ua"] = cache["Ua"].to(device=dev, dtype=dtype)
            if "mask_a" in cache:
                new_cache["mask_a"] = cache["mask_a"].to(device=dev, dtype=dtype)
            if "Ub" in cache:
                new_cache["Ub"] = cache["Ub"].to(device=dev, dtype=dtype)
            if "mask_b" in cache:
                new_cache["mask_b"] = cache["mask_b"].to(device=dev, dtype=dtype)
            preloaded[param] = new_cache

        for group in self.param_groups:
            group["projection_cache_map"] = preloaded

        print(f"[GlobalAwareOpt] 已预加载 {len(preloaded)} 个参数的投影矩阵到 GPU")

    def reset_cache(self, new_projection_cache_map: Dict):
        """
        连续编辑时更新投影缓存，并将一阶矩投影到新子空间。
        """
        for group in self.param_groups:
            old_cache_map = group.get("projection_cache_map", {})
            for p in group["params"]:
                if p not in self.state or p not in old_cache_map:
                    continue
                cache = old_cache_map[p]
                state = self.state[p]
                if "exp_avg" not in state:
                    continue
                m = state["exp_avg"]
                param_type = cache.get("param_type", "unknown")
                m_proj = self._project_grad_with_global_awareness(
                    p, m, cache, param_type,
                    group["projection_mode"], group["subspace_penalty"]
                )
                if m_proj is not None:
                    m.copy_(m_proj)

        self._preload_cache(new_projection_cache_map)

    # ------------------------------------------------------------------
    # 核心：全局意识梯度投影
    # ------------------------------------------------------------------

    def _project_grad_with_global_awareness(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        cache: Dict,
        param_type: str,
        mode: str,
        subspace_penalty: float,
    ) -> Optional[torch.Tensor]:
        """
        lora_A (r, d_in):
          - 安全方向（mask_a=1）：保留数据梯度
          - 危险方向（mask_a=0）：用 subspace_penalty * W 替代（权重拉回0）

        lora_B (d_out, r):
          - 安全方向（mask_b=1）：保留数据梯度
          - 危险方向（mask_b=0）：用 subspace_penalty * W 替代
        """
        if param_type == "lora_A":
            if mode not in ("marginal_A", "marginal_AB"):
                return None
            if "Ua" not in cache or "mask_a" not in cache:
                return None

            Ua     = cache["Ua"]      # (d_in, d_in)，已在 GPU
            mask_a = cache["mask_a"]  # (d_in,) float，1=安全，0=危险

            # 转到 Ua 基底
            grad_basis   = grad   @ Ua    # (r, d_in)
            weight_basis = param.data @ Ua  # (r, d_in)

            # 安全方向：保留数据梯度
            grad_safe   = grad_basis * mask_a.unsqueeze(0)
            # 危险方向：用权重衰减拉力代替
            danger_mask = (1.0 - mask_a).unsqueeze(0)
            grad_danger = subspace_penalty * weight_basis * danger_mask

            grad_proj = (grad_safe + grad_danger) @ Ua.T   # (r, d_in)
            return grad_proj

        elif param_type == "lora_B":
            if mode not in ("marginal_B", "marginal_AB"):
                return None
            if "Ub" not in cache or "mask_b" not in cache:
                return None

            Ub     = cache["Ub"]      # (d_out, d_out)，已在 GPU
            mask_b = cache["mask_b"]  # (d_out,) float

            # 转到 Ub 基底
            grad_basis   = Ub.T @ grad        # (d_out, r)
            weight_basis = Ub.T @ param.data  # (d_out, r)

            grad_safe   = grad_basis * mask_b.unsqueeze(1)
            danger_mask = (1.0 - mask_b).unsqueeze(1)
            grad_danger = subspace_penalty * weight_basis * danger_mask

            grad_proj = Ub @ (grad_safe + grad_danger)   # (d_out, r)
            return grad_proj

        return None

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            cache_map = group.get("projection_cache_map", {})
            mode      = group.get("projection_mode", "marginal_AB")
            penalty   = group.get("subspace_penalty", 0.1)

            for p in group["params"]:
                if p.grad is None or p not in cache_map:
                    continue

                cache      = cache_map[p]
                param_type = cache.get("param_type", "unknown")
                grad_proj  = self._project_grad_with_global_awareness(
                    p, p.grad, cache, param_type, mode, penalty
                )
                if grad_proj is not None:
                    p.grad.copy_(grad_proj)

        # 修复：传 None 避免 closure 被二次调用导致梯度覆盖
        return super().step(closure=None)
