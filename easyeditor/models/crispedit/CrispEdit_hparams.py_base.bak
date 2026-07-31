from dataclasses import dataclass
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
    
    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    energy_threshold: float

    # Projection/KFAC cache paths
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

    # Projection controls
    use_second_projection: bool = True
    newton_damping: float = 1e-3

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

        assert (config and config['alg_name'] == 'CRISPEDIT') or print(f'CrispEditHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
