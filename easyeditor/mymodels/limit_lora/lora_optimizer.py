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
        subspace_penalty: float = 0.1, # 新增：危险子空间的权重惩罚系数
    ):
        defaults = dict(
            projection_cache_map={},
            projection_mode=projection_mode,
            subspace_penalty=subspace_penalty,
        )
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            group.update(defaults)
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

    def _project_grad_with_global_awareness(
        self,
        param: torch.Tensor, # 新增：传入参数本身以获取全局状态
        grad: torch.Tensor,
        cache: Dict,
        param_type: str,
        mode: str,
        subspace_penalty: float
    ) -> Optional[torch.Tensor]:
        
        if param_type == "lora_A":
            if mode not in ("marginal_A", "marginal_AB"): return None
            Ua = cache["Ua"]
            mask_a = cache["mask_a"] # float tensor: 1.0 为安全, 0.0 为危险
            
            # 1. 转换到 Ua 基底下
            grad_basis = grad @ Ua
            weight_basis = param.data @ Ua  # 【全局意识】获取当前绝对权重
            
            # 2. 安全方向：保留数据梯度
            grad_safe = grad_basis * mask_a.unsqueeze(0)
            
            # 3. 危险方向：不要设为0，而是产生一个将其拉回0的梯度 (L2惩罚的导数)
            danger_mask = (1.0 - mask_a).unsqueeze(0)
            grad_danger = subspace_penalty * weight_basis * danger_mask
            
            # 4. 合并并转回原空间
            grad_proj = (grad_safe + grad_danger) @ Ua.T
            return grad_proj

        elif param_type == "lora_B":
            if mode not in ("marginal_B", "marginal_AB"): return None
            Ub = cache["Ub"]
            mask_b = cache["mask_b"]
            
            grad_basis = Ub.T @ grad
            weight_basis = Ub.T @ param.data # 【全局意识】
            
            grad_safe = grad_basis * mask_b.unsqueeze(1)
            danger_mask = (1.0 - mask_b).unsqueeze(1)
            grad_danger = subspace_penalty * weight_basis * danger_mask
            
            grad_proj = Ub @ (grad_safe + grad_danger)
            return grad_proj

        return None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            cache_map = group.get("projection_cache_map", {})
            mode = group.get("projection_mode", "marginal_AB")
            penalty = group.get("subspace_penalty", 0.1)

            for p in group["params"]:
                if p.grad is None or p not in cache_map:
                    continue

                cache = cache_map[p]
                param_type = cache.get("param_type", "unknown")
                # 传入 p 本身
                grad_proj = self._project_grad_with_global_awareness(
                    p, p.grad, cache, param_type, mode, penalty
                )
                if grad_proj is not None:
                    p.grad.copy_(grad_proj)

        return super().step(closure)