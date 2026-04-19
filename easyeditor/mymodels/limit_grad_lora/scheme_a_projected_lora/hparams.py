from dataclasses import dataclass, field
from typing import List, Optional
import yaml

from ...util.hparams import HyperParams


@dataclass
class SchemeAHyperParams(HyperParams):

    # ── 基本信息 ──────────────────────────────────────────────────────────────
    alg_name: str = "SchemeA_ProjectedLoRA"
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    device: int = 0
    model_parallel: bool = False

    # ── 编辑目标层配置 ────────────────────────────────────────────────────────
    layers: List[int] = field(default_factory=lambda: [19, 20, 21, 22, 23])
    rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj"
    layer_module_tmp: str = "model.layers.{}"
    mlp_module_tmp: str = "model.layers.{}.mlp"
    attn_module_tmp: str = "model.layers.{}.self_attn"
    ln_f_module: str = "model.norm"
    lm_head_module: str = "lm_head"

    # ── LoRA配置 ──────────────────────────────────────────────────────────────
    lora_type: str = "lora"           # "lora" 或 "adalora"
    lora_rank: int = 8                # LoRA秩
    lora_alpha: int = 32             # LoRA缩放因子
    lora_dropout: float = 0.1        # LoRA dropout
    target_modules: List[str] = field(default_factory=lambda: [["down_proj"]] )

    # ── 训练超参数 ────────────────────────────────────────────────────────────
    num_steps: int = 70
    lr: float = 5e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    max_length: int = 50
    objective_optimization: str = "target_new"  # "target_new" 或 "prompt_last"

    # ── KFac统计配置（用于计算子空间投影） ─────────────────────────────────────
    mom2_dataset: str = "wikipedia"
    mom2_n_samples: int = 10000
    mom2_dtype: str = "float32"
    energy_threshold: float = 0.7    # 低于此阈值的能量分量视为"安全子空间"


    projection_mode: str = "marginal_AB"

    # 是否对LoRA增量ΔW=B@A整体评估后再反传（experimental）
    project_delta_w: bool = False

    # ── 兼容性字段（方案B/C专用，方案A不使用，yaml 中可能含有这些键） ──────────
    normalize_init: bool = False
    recalculate_cache: bool = False
    recalculate_weight_threshold: float = 0.25
    edit_cache_style: str = "mix"
    edit_n_samples: int = 1000

    # ── 其他 ─────────────────────────────────────────────────────────────────
    kl_factor: float = 0.0
    norm_constraint: bool = False

    # calculate_old_loss() 需要此字段：True 时跳过旧知识损失评估以加速训练
    disable_old_loss_check: bool = True

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"
        with open(hparams_name_or_path, "r") as f:
            config = yaml.safe_load(f)
            config = super().construct_float_from_scientific_notation(config)
        return cls(**config)
