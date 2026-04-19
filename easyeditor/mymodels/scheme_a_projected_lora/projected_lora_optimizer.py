"""
方案A核心组件：ProjectedLoRAOptimizer
=====================================================
针对LoRA低秩矩阵（A: r×d_in, B: d_out×r）的KFac边缘化投影优化器。

问题：CrispEdit原始投影公式
    grad_proj = Ub @ ((Ub.T @ grad @ Ua) * M.T) @ Ua.T
作用在全量权重矩阵(d_out × d_in)上，无法直接作用于LoRA的低秩矩阵。

解决：边缘化投影（Marginalized Projection）
  - 对 lora_A (r × d_in)：仅使用右投影 Ua (d_in × d_in)
      grad_A_proj = grad_A @ Ua @ diag(mask_a) @ Ua.T
    其中 mask_a[j] = True 若 Sa[j] < threshold_a（低能量方向）

  - 对 lora_B (d_out × r)：仅使用左投影 Ub (d_out × d_out)
      grad_B_proj = Ub @ diag(mask_b) @ Ub.T @ grad_B
    其中 mask_b[i] = True 若 Sb[i] < threshold_b（低能量方向）

性能优化：
  - 投影矩阵（Ua/Ub/mask）在 __init__ 时预先搬到参数所在设备并缓存，
    避免每次 step() 都重复做 CPU→GPU 的数据搬运（D2H/H2D 是主要瓶颈）。
  - 预先将 bool mask 转为 float，避免每步类型转换。
"""

import torch
from torch.optim import Adam
from typing import Dict, Optional


class ProjectedLoRAOptimizer(Adam):
    """
    LoRA参数专用的KFac边缘化投影Adam优化器。

    projection_cache_map 格式（初始化时传入，Ua/Ub 可在 CPU）：
        {
            param_tensor: {
                'Ua': Tensor[d_in, d_in],   # 输入协方差特征向量
                'Ub': Tensor[d_out, d_out], # 输出协方差特征向量
                'mask_a': Tensor[d_in],     # 输入方向低能量掩码 (bool)
                'mask_b': Tensor[d_out],    # 输出方向低能量掩码 (bool)
                'param_type': str,          # 'lora_A' 或 'lora_B'
            }
        }
    初始化后内部会自动将矩阵搬到对应参数的设备上缓存，step() 中不再重复搬运。
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
        """
        Args:
            params: 待优化参数（仅LoRA参数）
            projection_cache_map: 参数→投影缓存的映射
            projection_mode: "marginal_A" | "marginal_B" | "marginal_AB"
        """
        # 先用空 cache 初始化父类，之后再填入预处理后的缓存
        defaults = dict(
            projection_cache_map={},
            projection_mode=projection_mode,
        )
        super().__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad,
        )
        for group in self.param_groups:
            group.update(defaults)

        # 预处理：将投影矩阵搬到参数所在设备并转为正确dtype，缓存起来
        # 避免每次 step() 都重复做 CPU→GPU 的数据搬运
        self._preload_cache(projection_cache_map)

    def _preload_cache(self, projection_cache_map: Dict):
        """
        将投影矩阵预先搬到各参数所在的设备（通常是 GPU）并缓存。
        这是性能关键：Ub(14336×14336) 每步搬运会带来极大延迟。
        同时将 bool mask 预先转为 float，避免 step() 中重复类型转换。
        """
        # 收集所有参数对象，建立 id→param 的映射
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
                # (d_in, d_in) → GPU，转为参数的 dtype（通常 bfloat16）
                new_cache["Ua"] = cache["Ua"].to(device=dev, dtype=dtype)
            if "mask_a" in cache:
                # bool mask → float，方便直接做逐元素乘法
                new_cache["mask_a"] = cache["mask_a"].to(device=dev, dtype=dtype)
            if "Ub" in cache:
                # (d_out, d_out) → GPU，这是最大的矩阵，搬运最耗时
                new_cache["Ub"] = cache["Ub"].to(device=dev, dtype=dtype)
            if "mask_b" in cache:
                new_cache["mask_b"] = cache["mask_b"].to(device=dev, dtype=dtype)

            preloaded[param] = new_cache

        # 写回 param_groups
        for group in self.param_groups:
            group["projection_cache_map"] = preloaded

        print(f"[ProjectedLoRAOptimizer] 已预加载 {len(preloaded)} 个参数的投影矩阵到 GPU")

    def reset_cache(self, new_projection_cache_map: Dict):
        """
        更新投影缓存，并同步动量缓冲区到新的子空间。
        在连续编辑场景下，每次更新投影后调用此方法。
        重新触发预加载，保证矩阵始终在 GPU 上。
        """
        # 先同步动量（用旧 cache 里已在 GPU 上的矩阵）
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
                m_proj = self._project_grad(m, cache, param_type, group["projection_mode"])
                if m_proj is not None:
                    m.copy_(m_proj)

        # 重新预加载新 cache
        self._preload_cache(new_projection_cache_map)

    def _project_grad(
        self,
        grad: torch.Tensor,
        cache: Dict,
        param_type: str,
        mode: str,
    ) -> Optional[torch.Tensor]:
        """
        对梯度执行边缘化投影。
        注意：此处 cache 中的矩阵已经在 GPU 上（由 _preload_cache 保证），
        不再需要 .to() 搬运，直接做矩阵乘法。

        Args:
            grad: 原始梯度（已在 GPU）
            cache: 投影缓存（Ua/Ub/mask 已预加载到 GPU）
            param_type: 'lora_A' 或 'lora_B'
            mode: 投影模式

        Returns:
            投影后的梯度，如果不需要投影则返回 None
        """
        if param_type == "lora_A":
            # lora_A shape: (r, d_in)
            # 右投影：grad @ Ua → mask → @ Ua.T
            # 等价于只保留 Ua 列中 mask_a=True 对应的低能量方向
            if mode not in ("marginal_A", "marginal_AB"):
                return None
            if "Ua" not in cache or "mask_a" not in cache:
                return None
            Ua     = cache["Ua"]      # (d_in, d_in)，已在 GPU
            mask_a = cache["mask_a"]  # (d_in,)，float，已在 GPU
            grad_proj = (grad @ Ua) * mask_a.unsqueeze(0)  # (r, d_in)
            grad_proj = grad_proj @ Ua.T                    # (r, d_in)
            return grad_proj

        elif param_type == "lora_B":
            # lora_B shape: (d_out, r)
            # 左投影：Ub.T @ grad → mask → Ub @
            # 等价于只保留 Ub 列中 mask_b=True 对应的低能量方向
            if mode not in ("marginal_B", "marginal_AB"):
                return None
            if "Ub" not in cache or "mask_b" not in cache:
                return None
            Ub     = cache["Ub"]      # (d_out, d_out)，已在 GPU
            mask_b = cache["mask_b"]  # (d_out,)，float，已在 GPU
            grad_in_basis = Ub.T @ grad                       # (d_out, r)
            grad_masked   = grad_in_basis * mask_b.unsqueeze(1)  # (d_out, r)
            grad_proj     = Ub @ grad_masked                  # (d_out, r)
            return grad_proj

        else:
            return None

    @torch.no_grad()
    def step(self, closure=None):
        """
        重写 step：在 Adam 更新前对 LoRA 参数梯度施加边缘化 KFac 投影。
        投影矩阵已预加载到 GPU，此处直接做矩阵乘法，无 CPU→GPU 搬运开销。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            cache_map = group.get("projection_cache_map", {})
            mode = group.get("projection_mode", "marginal_AB")
            if not cache_map:
                continue

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p not in cache_map:
                    continue

                cache = cache_map[p]
                param_type = cache.get("param_type", "unknown")
                grad_proj = self._project_grad(p.grad, cache, param_type, mode)
                if grad_proj is not None:
                    p.grad.copy_(grad_proj)

        return super().step(closure)
