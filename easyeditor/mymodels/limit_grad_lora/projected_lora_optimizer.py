import torch
from torch.optim import Adam
from typing import Dict, Optional


class ProjectedLoRAOptimizer(Adam):
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
                new_cache["Ua"] = cache["Ua"].to(device=dev, dtype=dtype)
            if "mask_a" in cache:
                new_cache["mask_a"] = cache["mask_a"].to(device=dev, dtype=dtype)
            if "Ub" in cache:
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
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            cache_map = group.get("projection_cache_map", {})
            mode = group.get("projection_mode", "marginal_AB")
            if not cache_map:
                print("cache_map is null!")
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
