"""
方案C超参数配置
CrispEdit-LoRA联合框架（最完整方案）：
  将KFac初始化（方案B）与ProjectedLoRAOptimizer（方案A）结合，
  同时支持连续编辑时的动态协方差缓存更新（继承CrispEdit原有机制）。

核心思路：
  1. KFac低能量子空间初始化LoRA A/B矩阵（方案B的精华）
  2. 用ProjectedLoRAOptimizer保证每一步梯度都投影到安全子空间（方案A的精华）
  3. 连续编辑时动态更新协方差缓存 + 重置优化器投影（CrispEdit原有机制）
  4. 支持混合数据（预训练数据 + 新编辑请求数据）计算联合协方差
"""

from dataclasses import dataclass, field
from typing import List
import yaml

from ..util.hparams import HyperParams


@dataclass
class CrispLoRAHyperParams(HyperParams):
    # ── 基本信息 ──────────────────────────────────────────────────────────────
    alg_name: str = "CrispLoRA"
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
    lora_type: str = "lora"
    lora_rank: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["down_proj"])

    # ── 训练超参数 ────────────────────────────────────────────────────────────
    num_steps: int = 25
    lr: float = 5e-4
    weight_decay: float = 0.0
    batch_size: int = 32
    max_length: int = 40
    objective_optimization: str = "target_new"

    # ── KFac统计配置 ─────────────────────────────────────────────────────────
    mom2_dataset: str = "wikipedia"
    mom2_n_samples: int = 1000
    mom2_dtype: str = "float32"
    energy_threshold: float = 0.5

    # ── 方案C特有：联合框架配置 ──────────────────────────────────────────────
    # 是否使用KFac初始化（True=方案B+A，False=仅方案A）
    use_kfac_init: bool = False
    # 是否使用投影优化器（True=方案A约束，False=仅方案B初始化）
    use_projected_optimizer: bool = True
    # 投影模式
    projection_mode: str = "marginal_AB"
    # 是否归一化初始化
    normalize_init: bool = True

    # ── 连续编辑配置（CrispEdit继承） ────────────────────────────────────────
    # 是否在权重显著变化时重新计算协方差缓存
    recalculate_cache: bool = False
    recalculate_weight_threshold: float = 0.1
    # 编辑数据的协方差缓存风格
    # "pretrain_only"：仅使用预训练数据
    # "edit_only"    ：仅使用当前编辑请求数据
    # "mix"          ：混合预训练 + 编辑请求数据
    edit_cache_style: str = "pretrain_only"
    edit_n_samples: int = 10

    # ── 损失监控 ─────────────────────────────────────────────────────────────
    disable_old_loss_check: bool = True

    # ── 其他 ─────────────────────────────────────────────────────────────────
    kl_factor: float = 0.0
    norm_constraint: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"
        with open(hparams_name_or_path, "r") as f:
            config = yaml.safe_load(f)
            config = super().construct_float_from_scientific_notation(config)
        return cls(**config)
