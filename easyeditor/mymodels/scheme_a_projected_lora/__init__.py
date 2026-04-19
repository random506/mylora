"""方案A：LoRA + KFac边缘化投影Adam"""
from .hparams import SchemeAHyperParams
from .projected_lora_optimizer import ProjectedLoRAOptimizer
from .utils import (
    apply_scheme_a_to_model,
    build_lora_projection_cache,
    wrap_model_and_build_projected_optimizer,
)

__all__ = [
    "SchemeAHyperParams",
    "ProjectedLoRAOptimizer",
    "apply_scheme_a_to_model",
    "build_lora_projection_cache",
    "wrap_model_and_build_projected_optimizer",
]
