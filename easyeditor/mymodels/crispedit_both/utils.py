"""
utils.py  —  crispedit_both
=====================================
将 crispedit（全量参数梯度投影）改造为"参数+梯度双重投影"版本：

  参数侧约束（forward）：LeakyCurvatureLora
    - 高曲率方向的激活只允许 leak_rate（可学习，约 0-10%）通过
    - 作用于 LoRA 的 A/B 矩阵输入/输出

  梯度侧约束（optimizer.step）：GlobalAwareProjectedLoRAOptimizer
    - 安全方向：正常传递数据梯度
    - 危险方向：用 subspace_penalty * W 替代（拉回零点）

统一投影缓存格式（build_projection_cache 一次计算，两侧共用）：
    {
        layer_name → {
            "Ua":       (d_in,  d_in),   # 输入侧特征向量矩阵
            "Ub":       (d_out, d_out),  # 输出侧特征向量矩阵
            "mask_a":   (d_in,),  float  # 1=安全(低曲率), 0=危险(高曲率)
            "mask_b":   (d_out,), float
            "U_in_bar": (d_in,  k_in),   # 高曲率列，供 LeakyCurvatureLora
            "U_out_bar":(d_out, k_out),
        }
    }
"""

import os
import torch
from typing import Dict, List, Tuple, Optional
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

from .CurvatureLora import LeakyCurvatureLora
from .lora_optimizer import GlobalAwareProjectedLoRAOptimizer
from .CrispEditBoth_hparams import CrispEditBothHyperParams
from ...models.rome.layer_stats import (
    layer_stats_kfac_one_pass,
    layer_stats_kfac_with_txt_tgt,
    calculate_cache_loss,
    calculate_request_loss,
)

load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")


# ---------------------------------------------------------------------------
# 辅助：模型类型判断
# ---------------------------------------------------------------------------

def _is_llama_or_phi(model_name: str) -> bool:
    lower = model_name.lower()
    return "llama" in lower or "phi" in lower


# ---------------------------------------------------------------------------
# 特征值分析
# ---------------------------------------------------------------------------

def get_rank_and_threshold_by_energy_ratio(eigenvalues: torch.Tensor, percent: float = 0.9):
    total_energy = torch.sum(eigenvalues)
    sorted_eigvals, _ = torch.sort(eigenvalues, descending=True)
    cumulative_energy = torch.cumsum(sorted_eigvals, dim=0)
    energy_ratio = cumulative_energy / total_energy
    rank = torch.searchsorted(energy_ratio, percent).item() + 1
    threshold = sorted_eigvals[rank - 1] if rank - 1 < len(sorted_eigvals) else 0.0
    return rank, threshold


def compute_marginal_masks(
    Sa: torch.Tensor,
    Sb: torch.Tensor,
    energy_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    按边际化能量阈值计算每侧的安全/危险掩码。
    mask = True  → 安全（低曲率），优化器保留数据梯度
    mask = False → 危险（高曲率），优化器用惩罚项替代
    """
    _, threshold_a = get_rank_and_threshold_by_energy_ratio(Sa, percent=energy_threshold)
    _, threshold_b = get_rank_and_threshold_by_energy_ratio(Sb, percent=energy_threshold)
    mask_a = Sa < threshold_a
    mask_b = Sb < threshold_b
    print(
        f"  mask_a: {mask_a.sum().item()}/{len(mask_a)} safe dirs, threshold_a={threshold_a:.6f}\n"
        f"  mask_b: {mask_b.sum().item()}/{len(mask_b)} safe dirs, threshold_b={threshold_b:.6f}"
    )
    return mask_a, mask_b


# ---------------------------------------------------------------------------
# 核心：统一投影缓存构建（一次计算，两侧共用）
# ---------------------------------------------------------------------------

def build_projection_cache(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: CrispEditBothHyperParams,
    force_recompute: bool = False,
) -> Dict[str, Dict]:
    """
    计算各层统一投影缓存，包含参数侧和梯度侧所需的全部矩阵。

    Returns:
        layer_to_proj_cache: {
            layer_name → {
                "Ua":       Tensor(d_in,  d_in),
                "Ub":       Tensor(d_out, d_out),
                "mask_a":   Tensor(d_in,)   float，1=安全
                "mask_b":   Tensor(d_out,)  float，1=安全
                "U_in_bar": Tensor(d_in,  k_in),   高曲率列
                "U_out_bar":Tensor(d_out, k_out),
            }
        }
    """
    print("[CrispEditBoth] 计算各层 KFac 协方差统计...")

    layer_names = [hparams.rewrite_module_tmp.format(l) for l in hparams.layers]

    stats_dict = layer_stats_kfac_one_pass(
        model=model,
        tokenizer=tok,
        layer_names=layer_names,
        stats_dir=STATS_DIR,
        ds_name=hparams.mom2_dataset,
        to_collect=["mom2"],
        sample_size=hparams.mom2_n_samples,
        precision=hparams.mom2_dtype,
        force_recompute=force_recompute,
    )

    layer_to_proj_cache = {}
    for layer_name in layer_names:
        A, B, _ = stats_dict.pop(layer_name)

        # 非 llama/phi 类模型需要交换 A/B（与原 crispedit 保持一致）
        if not _is_llama_or_phi(hparams.model_name):
            A, B = B, A

        A = A.to(dtype=torch.float32)
        B = B.to(dtype=torch.float32)

        Sa, Ua = torch.linalg.eigh(A)  # 升序特征值
        Sb, Ub = torch.linalg.eigh(B)

        print(f"[CrispEditBoth] 层 {layer_name} 边缘化掩码计算:")
        mask_a, mask_b = compute_marginal_masks(Sa, Sb, hparams.energy_threshold)

        # 高曲率列：危险方向的特征向量，供 LeakyCurvatureLora 的软投影
        U_in_bar  = Ua[:, ~mask_a]   # (d_in,  k_in)
        U_out_bar = Ub[:, ~mask_b]   # (d_out, k_out)

        layer_to_proj_cache[layer_name] = {
            "Ua":        Ua.cpu(),
            "Ub":        Ub.cpu(),
            "mask_a":    mask_a.float().cpu(),  # float 方便优化器直接乘法
            "mask_b":    mask_b.float().cpu(),
            "U_in_bar":  U_in_bar.cpu(),
            "U_out_bar": U_out_bar.cpu(),
        }
        del A, B, Sa, Sb, Ua, Ub
        torch.cuda.empty_cache()

    return layer_to_proj_cache


def build_projection_cache_from_request(
    txt, tgt,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: CrispEditBothHyperParams,
) -> Dict[str, Dict]:
    """用编辑请求数据（而非预训练数据）计算协方差，再走同样的特征值分解。"""
    if hparams.no_crisp:
        return None

    cov_stats_dict = layer_stats_kfac_with_txt_tgt(
        model, tok,
        layer_names=[hparams.rewrite_module_tmp.format(l) for l in hparams.layers],
        txt=txt, tgt=tgt,
        precision=hparams.mom2_dtype,
        sample_size=hparams.edit_n_samples,
        to_collect=["mom2"],
        add_pretrain_data=(hparams.edit_cache_style == "mix"),
        pretrain_sample_size=hparams.mom2_n_samples,
    )

    layer_names = [hparams.rewrite_module_tmp.format(l) for l in hparams.layers]
    raw_cov_cache = {}
    for layer_name in layer_names:
        A, B, num_samples = cov_stats_dict.pop(layer_name)
        raw_cov_cache[layer_name] = {
            "A": A.to("cpu", dtype=torch.float32),
            "B": B.to("cpu", dtype=torch.float32),
            "num_samples": num_samples,
        }
        del A, B
        torch.cuda.empty_cache()

    return _cov_cache_to_proj_cache(raw_cov_cache, hparams)


def _cov_cache_to_proj_cache(
    raw_cov_cache: Dict[str, Dict],
    hparams: CrispEditBothHyperParams,
) -> Dict[str, Dict]:
    """从原始协方差矩阵转换为统一投影缓存（内部复用）。"""
    layer_to_proj_cache = {}
    for layer_name, cov in raw_cov_cache.items():
        A = cov["A"].to(dtype=torch.float32)
        B = cov["B"].to(dtype=torch.float32)

        if not _is_llama_or_phi(hparams.model_name):
            A, B = B, A

        Sa, Ua = torch.linalg.eigh(A)
        Sb, Ub = torch.linalg.eigh(B)

        mask_a, mask_b = compute_marginal_masks(Sa, Sb, hparams.energy_threshold)

        layer_to_proj_cache[layer_name] = {
            "Ua":        Ua.cpu(),
            "Ub":        Ub.cpu(),
            "mask_a":    mask_a.float().cpu(),
            "mask_b":    mask_b.float().cpu(),
            "U_in_bar":  Ua[:, ~mask_a].cpu(),
            "U_out_bar": Ub[:, ~mask_b].cpu(),
        }
        del A, B, Sa, Sb, Ua, Ub
        torch.cuda.empty_cache()

    return layer_to_proj_cache


# ---------------------------------------------------------------------------
# 层匹配工具
# ---------------------------------------------------------------------------

def _match_layer_cache(param_name: str, layer_to_proj_cache: Dict) -> Optional[Dict]:
    for layer_name, cache in layer_to_proj_cache.items():
        clean = layer_name.removesuffix(".weight")
        if clean in param_name:
            return cache
    return None


# ---------------------------------------------------------------------------
# 参数侧：注入 LeakyCurvatureLora
# ---------------------------------------------------------------------------

def inject_leaky_curvature_lora(
    peft_model,
    layer_to_proj_cache: Dict[str, Dict],
    adapter_name: str = "default",
) -> None:
    """
    对所有 LoraLayer：
      1. 注册 LeakyCurvatureLora variant（含 leak_rate 和空 buffer）
      2. 从统一缓存写入 U_in_bar / U_out_bar
    """
    count = 0
    for name, module in peft_model.named_modules():
        if not (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "in_features")
            and hasattr(module, "out_features")
        ):
            continue
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            continue

        if not hasattr(module, "lora_variant"):
            module.lora_variant = {}

        module.lora_variant[adapter_name] = LeakyCurvatureLora()
        LeakyCurvatureLora.init(module, adapter_name=adapter_name)

        matched = _match_layer_cache(name, layer_to_proj_cache)
        if matched is None:
            continue

        base_weight = module.base_layer.weight if hasattr(module, "base_layer") else module.weight
        dev, dtype = base_weight.device, base_weight.dtype

        U_in_bar  = matched["U_in_bar"] .to(device=dev, dtype=dtype)
        U_out_bar = matched["U_out_bar"].to(device=dev, dtype=dtype)
        setattr(module, f"U_in_bar_{adapter_name}",  U_in_bar)
        setattr(module, f"U_out_bar_{adapter_name}", U_out_bar)
        count += 1
        print(
            f"[CrispEditBoth] {name}: "
            f"U_in_bar={tuple(U_in_bar.shape)}, U_out_bar={tuple(U_out_bar.shape)}"
        )

    print(f"[CrispEditBoth] 已注入 {count} 个 LeakyCurvatureLora 层")


def update_leaky_curvature_bases(
    peft_model,
    layer_to_proj_cache: Dict[str, Dict],
    adapter_name: str = "default",
) -> None:
    """连续编辑时用新缓存更新 U_in_bar / U_out_bar（不重建 variant）。"""
    for name, module in peft_model.named_modules():
        if not hasattr(module, "lora_variant") or adapter_name not in module.lora_variant:
            continue

        matched = _match_layer_cache(name, layer_to_proj_cache)
        if matched is None:
            continue

        base_weight = module.base_layer.weight if hasattr(module, "base_layer") else module.weight
        dev, dtype = base_weight.device, base_weight.dtype

        setattr(module, f"U_in_bar_{adapter_name}",
                matched["U_in_bar"].to(device=dev, dtype=dtype))
        setattr(module, f"U_out_bar_{adapter_name}",
                matched["U_out_bar"].to(device=dev, dtype=dtype))


# ---------------------------------------------------------------------------
# 梯度侧：建立 param → proj_cache 映射
# ---------------------------------------------------------------------------

def map_proj_cache_to_lora_params(
    peft_model,
    layer_to_proj_cache: Dict[str, Dict],
) -> Dict[torch.nn.Parameter, Dict]:
    """
    将 layer_name → proj_cache 转换为 param_tensor → proj_cache，
    适配 GlobalAwareProjectedLoRAOptimizer。
    lora_A 参数：取 Ua / mask_a
    lora_B 参数：取 Ub / mask_b
    """
    param_to_proj_cache = {}
    for name, param in peft_model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue

        matched = _match_layer_cache(name, layer_to_proj_cache)
        if matched is None:
            continue

        if "lora_A" in name:
            param_to_proj_cache[param] = {
                "Ua":       matched["Ua"],
                "mask_a":   matched["mask_a"],
                "param_type": "lora_A",
            }
        elif "lora_B" in name:
            param_to_proj_cache[param] = {
                "Ub":       matched["Ub"],
                "mask_b":   matched["mask_b"],
                "param_type": "lora_B",
            }

    print(f"[CrispEditBoth] 建立参数→投影映射，共 {len(param_to_proj_cache)} 个 LoRA 参数")
    return param_to_proj_cache


# ---------------------------------------------------------------------------
# 主初始化：包装模型 + 构建双优化器
# ---------------------------------------------------------------------------

def wrap_model_with_both_projection(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: CrispEditBothHyperParams,
    force_recompute: bool = False,
):
    """
    完整初始化流程：
      1. 计算统一投影缓存（KFac 特征分解）
      2. 挂载标准 LoRA
      3. 注入 LeakyCurvatureLora（参数侧约束）
      4. 分离 lora_params / leak_params
      5. 构建 GlobalAwareProjectedLoRAOptimizer（梯度侧约束）+ Adam（leak_rate）

    Returns:
        peft_model, optimizer_lora, optimizer_leak, layer_to_proj_cache
    """
    # 1. 统一投影缓存（在挂 LoRA 之前，基于原始模型权重统计）
    if hparams.no_crisp:
        layer_to_proj_cache = None
    else:
        layer_to_proj_cache = build_projection_cache(model, tok, hparams, force_recompute)

    # 2. 挂载 LoRA
    model.config.use_cache = False
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.lora_rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
        target_modules=hparams.target_modules,
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    # 3. 注入 LeakyCurvatureLora（参数侧）
    if layer_to_proj_cache is not None:
        inject_leaky_curvature_lora(peft_model, layer_to_proj_cache)

    # 4. 分离参数
    lora_params = []
    leak_params = []
    for n, p in peft_model.named_parameters():
        if not p.requires_grad:
            continue
        if "leak_rate" in n:
            leak_params.append(p)
        else:
            lora_params.append(p)

    # 5. 构建优化器
    if layer_to_proj_cache is not None:
        param_to_proj_cache = map_proj_cache_to_lora_params(peft_model, layer_to_proj_cache)
    else:
        param_to_proj_cache = {}

    optimizer_lora = GlobalAwareProjectedLoRAOptimizer(
        lora_params,
        projection_cache_map=param_to_proj_cache,
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
        projection_mode=hparams.projection_mode,
        subspace_penalty=hparams.subspace_penalty,
    )
    optimizer_leak = torch.optim.Adam(leak_params, lr=hparams.lr * 2) if leak_params else None

    print(
        f"[CrispEditBoth] 初始化完成：LoRA rank={hparams.lora_rank}，"
        f"投影模式={hparams.projection_mode}，能量阈值={hparams.energy_threshold}"
    )
    return peft_model, optimizer_lora, optimizer_leak, layer_to_proj_cache


# ---------------------------------------------------------------------------
# 连续编辑：更新双侧投影缓存
# ---------------------------------------------------------------------------

def reset_both_projections(
    peft_model,
    optimizer_lora: GlobalAwareProjectedLoRAOptimizer,
    new_layer_to_proj_cache: Dict[str, Dict],
    adapter_name: str = "default",
) -> None:
    """
    连续编辑时同步更新：
      - 参数侧：更新各层 U_in_bar / U_out_bar buffer
      - 梯度侧：调用 optimizer_lora.reset_cache() 更新预加载矩阵并投影动量
    """
    # 参数侧
    update_leaky_curvature_bases(peft_model, new_layer_to_proj_cache, adapter_name)

    # 梯度侧
    new_param_to_proj_cache = map_proj_cache_to_lora_params(peft_model, new_layer_to_proj_cache)
    optimizer_lora.reset_cache(new_param_to_proj_cache)
