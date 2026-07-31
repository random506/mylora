from dataclasses import MISSING, dataclass, fields
from typing import List, Optional

import yaml

from ...util.hparams import HyperParams


@dataclass
class CrispEditHyperParams(HyperParams):
    # Method
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: bool

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

    # Base statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    energy_threshold: float

    # Task/edit statistics for the soft K-FAC formulation.
    task_mom2_dataset: Optional[str] = None
    task_mom2_n_samples: Optional[int] = None

    # Projection/K-FAC cache paths kept for compatibility with older configs.
    base_kfac_cache_path: Optional[str] = None
    task_kfac_cache_path: Optional[str] = None
    additional_kfac_cache_path: Optional[str] = None
    second_kfac_cache_path: Optional[str] = None
    second_projection_kfac_cache_path: Optional[str] = None
    projection_cache_path: Optional[str] = None
    base_projection_cache_path: Optional[str] = None
    primary_projection_cache_path: Optional[str] = None
    task_projection_cache_path: Optional[str] = None
    additional_projection_cache_path: Optional[str] = None
    second_projection_cache_path: Optional[str] = None

    # Soft K-FAC controls.
    soft_lambda: float = 1.0
    use_second_projection: bool = False
    newton_damping: float = 1e-3

    # Runtime defaults used by run_crispedit.py/crispedit.py.
    batch_size: int = 64
    max_length: int = 40
    model_parallel: bool = False
    edit_n_samples: int = 10
    edit_cache_style: str = "mix"
    recalculate_cache: bool = False
    recalculate_weight_threshold: float = 0.01
    no_crisp: bool = False
    disable_old_loss_check: bool = False
    perform_lora: bool = False
    num_edits: int = 1

    # LoRA defaults. They are unused in the base CrispEdit path.
    lora_type: str = "lora"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: Optional[List[str]] = None

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream) or {}
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config["alg_name"] == "CRISPEDIT") or print(
            f"CrispEditHyperParams can not load from {hparams_name_or_path}, "
            f'alg_name is {config["alg_name"]} '
        )

        known_fields = {field.name: field for field in fields(cls)}
        init_config = {
            key: value for key, value in config.items() if key in known_fields
        }
        missing_required = [
            name
            for name, field in known_fields.items()
            if field.default is MISSING
            and field.default_factory is MISSING
            and name not in init_config
        ]
        if missing_required:
            raise KeyError(
                "Missing required CrispEdit hparams: "
                + ", ".join(sorted(missing_required))
            )

        hparams = cls(**init_config)
        for key, value in config.items():
            if key not in known_fields:
                setattr(hparams, key, value)
        return hparams
