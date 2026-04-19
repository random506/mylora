# 核心组件：KFac 边缘化投影优化器
from .projected_lora_optimizer import ProjectedLoRAOptimizer

# 工具函数：KFac 统计、投影缓存构建、模型包装
from .utils import (
    build_lora_projection_cache,
    map_proj_cache_to_lora_params,
    wrap_model_and_build_projected_optimizer,
    apply_scheme_a_to_model,
    compute_marginal_masks,
    get_rank_and_threshold_by_energy_ratio,
)


__all__ = [
    # 方案A
    "ProjectedLoRAOptimizer",
    "build_lora_projection_cache",
    "map_proj_cache_to_lora_params",
    "wrap_model_and_build_projected_optimizer",
    "apply_scheme_a_to_model",
    "compute_marginal_masks",
    "get_rank_and_threshold_by_energy_ratio"
]
