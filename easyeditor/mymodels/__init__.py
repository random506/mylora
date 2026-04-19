"""
mymodels：自定义 LoRA 方案集合
=====================================
包含两大类方案：
  - limit_grad_lora  : 限制梯度方向的 LoRA（SchemeA/C/D）
  - limit_param_lora : 限制参数增量本身的 LoRA（CurvatureLora）

公共超参数：
  - CrispLoRAHyperParams : 统一的超参数数据类
"""

# 公共超参数
from .hparams import CrispLoRAHyperParams

# ── 限制梯度方向的 LoRA ──────────────────────────────────────────────────────
# 方案A：KFac 边缘化投影优化器 + 模型包装
from .limit_grad_lora import (
    ProjectedLoRAOptimizer,
    build_lora_projection_cache,
    map_proj_cache_to_lora_params,
    wrap_model_and_build_projected_optimizer,
    apply_scheme_a_to_model,
    compute_marginal_masks,
    get_rank_and_threshold_by_energy_ratio,
)

# ── 限制参数增量的 LoRA ──────────────────────────────────────────────────────
# CurvatureLora：通过投影子空间约束 LoRA 增量本身
from .limit_param_lora import ( 
    CurvatureLora,
    attach_curvature_lora_variant
)


__all__ = [
    # 超参数
    "CrispLoRAHyperParams",
    # 方案A
    "ProjectedLoRAOptimizer",
    "build_lora_projection_cache",
    "map_proj_cache_to_lora_params",
    "wrap_model_and_build_projected_optimizer",
    "apply_scheme_a_to_model",
    "compute_marginal_masks",
    "get_rank_and_threshold_by_energy_ratio",
    # 调整参数增量
    "CurvatureLora",
    "attach_curvature_lora_variant"
]
