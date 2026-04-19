"""方案C：CrispEdit-LoRA联合框架（最完整方案）"""
from .hparams import SchemeCHyperParams
from .crisp_lora_optimizer import CrispLoRAOptimizer
from .utils import (
    apply_scheme_c_to_model,
    build_crisp_lora_model_and_optimizer,
    compute_cov_and_proj_cache,
    compute_edit_request_proj_cache,
    recalculate_proj_if_weights_changed,
)

__all__ = [
    "SchemeCHyperParams",
    "CrispLoRAOptimizer",
    "apply_scheme_c_to_model",
    "build_crisp_lora_model_and_optimizer",
    "compute_cov_and_proj_cache",
    "compute_edit_request_proj_cache",
    "recalculate_proj_if_weights_changed",
]
