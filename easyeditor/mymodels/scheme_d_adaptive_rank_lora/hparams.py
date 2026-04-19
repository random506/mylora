"""
方案D超参数配置
自适应秩选择（AdaLoRA × CrispEdit）：
  用KFac的"安全子空间能量占比"自动决定各层LoRA秩预算，
  能量越低（越安全）的层分配越多rank，从而在安全的地方做更丰富的编辑。

核心思路：
  1. 统计各层KFac协方差特征值分布
  2. 计算各层"低能量方向占比"（= 安全子空间大小 / 总维度）
  3. 按照该比例分配总rank预算到各层（越安全的层 → 越多rank）
  4. 用AdaLoraConfig的target_r和init_r配置动态秩
  5. 可选：对rank分配结果再施加方案A的边缘化投影

额外特性：
  - 自动可视化各层rank分配（如有wandb）
  - 支持能量比例 vs rank 的线性/平方/log映射
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml

from ...util.hparams import HyperParams


@dataclass
class SchemeDHyperParams(HyperParams):
    """方案D：KFac能量驱动的自适应LoRA秩分配超参数"""

    # ── 基本信息 ──────────────────────────────────────────────────────────────
    alg_name: str = "SchemeD_AdaptiveRankLoRA"
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

    # ── LoRA基本配置 ─────────────────────────────────────────────────────────
    lora_type: str = "adalora"        # 方案D主要使用AdaLoRA
    lora_alpha: int = 16
    lora_dropout: float = 0.05
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
    energy_threshold: float = 0.99

    # ── 方案D特有：自适应秩分配 ──────────────────────────────────────────────
    # 总rank预算（分配到所有目标层）
    total_rank_budget: int = 64

    # 最小/最大单层rank
    min_rank_per_layer: int = 2
    max_rank_per_layer: int = 32

    # rank映射函数：安全比例 → rank
    # "linear"   : rank ∝ safe_ratio（线性）
    # "sqrt"     : rank ∝ sqrt(safe_ratio)（更平衡）
    # "log"      : rank ∝ log(1 + safe_ratio * 9)（对小差异敏感）
    # "uniform"  : 等额分配（忽略KFac，作为对比baseline）
    rank_allocation_strategy: str = "sqrt"

    # 是否在rank分配后打印分配表
    verbose_rank_alloc: bool = True

    # 是否在方案D基础上叠加方案A的边缘化投影优化器
    use_projected_optimizer: bool = False
    projection_mode: str = "marginal_AB"

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
