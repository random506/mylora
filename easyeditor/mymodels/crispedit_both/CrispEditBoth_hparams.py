from dataclasses import dataclass, field
from typing import List, Optional
import yaml

from ...util.hparams import HyperParams


@dataclass
class CrispEditBothHyperParams(HyperParams):
    # Method
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    device: int
    alg_name: str
    model_name: str
    objective_optimization: str

    # LoRA config
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    target_modules: List[str]
    lora_type: str = "lora"

    # 投影模式：控制优化器侧对 lora_A/lora_B 分别还是联合投影
    projection_mode: str = "marginal_AB"

    # 危险子空间惩罚系数（优化器侧）
    subspace_penalty: float = 0.05

    # Statistics
    mom2_dataset: str = ""
    mom2_n_samples: int = 1000
    mom2_dtype: str = "float32"
    energy_threshold: float = 0.9

    # Continuous editing
    no_crisp: bool = False
    recalculate_cache: bool = False
    recalculate_weight_threshold: float = 0.01
    edit_n_samples: int = 10
    edit_cache_style: str = "new"
    disable_old_loss_check: bool = True

    # Defaults
    batch_size: int = 64
    max_length: int = 40
    model_parallel: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'CRISPEDIT_BOTH') or print(
            f'CrispEditBothHyperParams cannot load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]}'
        )
        return cls(**config)
