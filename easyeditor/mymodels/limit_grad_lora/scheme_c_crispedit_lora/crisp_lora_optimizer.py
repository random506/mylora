"""
方案C核心：CrispEdit-LoRA联合优化器
=====================================================
将方案A的边缘化投影与方案B的KFac初始化合并为一个完整框架。

额外功能（继承CrispEdit）：
  - 支持连续编辑时的协方差缓存动态更新
  - 支持双重投影（预训练数据 + 新编辑数据的联合协方差）
  - reset_cache后自动同步动量缓冲区
"""

import torch
from torch.optim import Adam
from typing import Dict, Optional


class CrispLoRAOptimizer(Adam):
    """
    CrispEdit-LoRA联合优化器。

    支持：
    1. 单投影：仅使用预训练数据的协方差（primary_cache_map）
    2. 双重投影：同时使用预训练数据 + 编辑请求数据的协方差
       先投影primary -> 再投影additional

    projection_cache_map格式（与方案A相同）：
        {
            param: {
                'Ua': Tensor[d_in, d_in],
                'Ub': Tensor[d_out, d_out],
                'mask_a': Tensor[d_in],    # bool，lora_A用
                'mask_b': Tensor[d_out],   # bool，lora_B用
                'param_type': str,          # 'lora_A' 或 'lora_B'
            }
        }
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
    ):
        defaults = dict(
            projection_cache_map=projection_cache_map,
            projection_mode=projection_mode,
            additional_projection_cache_map=None,
        )
        super().__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad,
        )
        for group in self.param_groups:
            group.update(defaults)

    def reset_cache(self, new_cache_map: Dict):
        """更新主投影缓存并同步动量缓冲区"""
        for group in self.param_groups:
            group["projection_cache_map"] = new_cache_map
            for p in group["params"]:
                if p not in self.state or p not in new_cache_map:
                    continue
                cache = new_cache_map[p]
                state = self.state[p]
                if "exp_avg" not in state:
                    continue
                m = state["exp_avg"]
                m_proj = self._project(m, cache, group["projection_mode"])
                if m_proj is not None:
                    m.copy_(m_proj)

    def reset_additional_cache(self, additional_cache_map: Optional[Dict]):
        """更新辅助投影缓存（用于双重投影模式）"""
        for group in self.param_groups:
            group["additional_projection_cache_map"] = additional_cache_map
            if additional_cache_map is None:
                continue
            for p in group["params"]:
                if p not in self.state or p not in additional_cache_map:
                    continue
                cache = additional_cache_map[p]
                state = self.state[p]
                if "exp_avg" not in state:
                    continue
                m = state["exp_avg"]
                m_proj = self._project(m, cache, group["projection_mode"])
                if m_proj is not None:
                    m.copy_(m_proj)

    def _project(
        self,
        grad: torch.Tensor,
        cache: Dict,
        mode: str,
    ) -> Optional[torch.Tensor]:
        """
        对梯度执行边缘化子空间投影。
        （与方案A的 _project_grad 逻辑相同，但整合到此类中）
        """
        dev, dtype = grad.device, grad.dtype
        param_type = cache.get("param_type", "unknown")

        if param_type == "lora_A":
            if mode not in ("marginal_A", "marginal_AB"):
                return None
            if "Ua" not in cache or "mask_a" not in cache:
                return None
            Ua = cache["Ua"].to(device=dev, dtype=dtype)
            mask_a = cache["mask_a"].to(device=dev, dtype=dtype)
            # grad: (r, d_in) -> 在Ua空间中，仅保留mask_a=True的方向
            grad_proj = (grad @ Ua) * mask_a.unsqueeze(0)
            grad_proj = grad_proj @ Ua.T
            return grad_proj

        elif param_type == "lora_B":
            if mode not in ("marginal_B", "marginal_AB"):
                return None
            if "Ub" not in cache or "mask_b" not in cache:
                return None
            Ub = cache["Ub"].to(device=dev, dtype=dtype)
            mask_b = cache["mask_b"].to(device=dev, dtype=dtype)
            # grad: (d_out, r)
            grad_in_basis = Ub.T @ grad
            grad_masked = grad_in_basis * mask_b.unsqueeze(1)
            grad_proj = Ub @ grad_masked
            return grad_proj

        return None

    @torch.no_grad()
    def step(self, closure=None):
        """执行带双重投影约束的Adam更新步"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            primary_cache = group.get("projection_cache_map", {})
            additional_cache = group.get("additional_projection_cache_map")
            mode = group.get("projection_mode", "marginal_AB")

            for p in group["params"]:
                if p.grad is None:
                    continue

                # 主投影
                if primary_cache and p in primary_cache:
                    g_proj = self._project(p.grad, primary_cache[p], mode)
                    if g_proj is not None:
                        p.grad.copy_(g_proj)

                # 辅助投影（双重投影）
                if additional_cache and p in additional_cache:
                    g_proj2 = self._project(p.grad, additional_cache[p], mode)
                    if g_proj2 is not None:
                        p.grad.copy_(g_proj2)

        return super().step(closure)
