"""方案D：KFac能量驱动的自适应LoRA秩分配"""
from .hparams import SchemeDHyperParams
from .utils import (
    apply_scheme_d_to_model,
    compute_null_energy_ratio_per_layer,
    allocate_ranks_from_energy,
    wrap_model_with_adaptive_rank_lora,
)

__all__ = [
    "SchemeDHyperParams",
    "apply_scheme_d_to_model",
    "compute_null_energy_ratio_per_layer",
    "allocate_ranks_from_energy",
    "wrap_model_with_adaptive_rank_lora",
]
