import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from peft.tuners.lora.layer import LoraLayer
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import os

from ..limit_param_lora.CurvatureLora import CurvatureLora
from .CrispEditParam_hparams import CrispEditParamHyperParams
from ...models.rome.layer_stats import (
    layer_stats_kfac,
    layer_stats_kfac_one_pass,
    layer_stats_kfac_with_txt_tgt,
    calculate_cache_loss,
    calculate_request_loss,
)

load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")


# ---------------------------------------------------------------------------
# 特征值分析工具
# ---------------------------------------------------------------------------

def get_rank_and_threshold_by_energy_ratio(eigenvalues: torch.Tensor, percent: float = 0.9):
    """按累积能量比例确定保留的特征值数量及对应阈值。"""
    total_energy = torch.sum(eigenvalues)
    sorted_eigvals, _ = torch.sort(eigenvalues, descending=True)
    cumulative_energy = torch.cumsum(sorted_eigvals, dim=0)
    energy_ratio = cumulative_energy / total_energy
    rank = torch.searchsorted(energy_ratio, percent).item() + 1
    threshold = sorted_eigvals[rank - 1] if rank - 1 < len(sorted_eigvals) else 0.0
    return rank, threshold


def calculate_projection_cache_with_kfac(A: torch.Tensor, B: torch.Tensor, energy_threshold: float = 0.9) -> Dict:
    """
    用 KFAC 协方差矩阵 A（输入侧）和 B（输出侧）计算曲率投影基。

    返回：
        {
            'U_in_bar':  (in_features,  k_in)   输入侧高曲率方向基
            'U_out_bar': (out_features, k_out)  输出侧高曲率方向基
        }

    这里高曲率方向 = 外积特征值大的方向，即 M >= null_threshold 的方向。
    与 projected_adam 的 M（低曲率掩码）方向相反：
        projected_adam: M = (Sa ⊗ Sb) < threshold  → 保留低曲率
        curvature_lora: U_in_bar/U_out_bar 存的是"高曲率列" → forward 里减去它们
    """
    Sa, Ua = torch.linalg.eigh(A)   # A 是输入侧协方差，特征值升序
    Sb, Ub = torch.linalg.eigh(B)   # B 是输出侧协方差

    # 外积能量矩阵，(in_dim, out_dim)
    M_energy = torch.outer(Sa, Sb)
    _, null_threshold = get_rank_and_threshold_by_energy_ratio(
        M_energy.view(-1), percent=energy_threshold
    )

    # 高曲率列 = 该侧特征值超过阈值的列（边际阈值近似）
    high_in_mask  = Sa >= null_threshold.item() if isinstance(null_threshold, torch.Tensor) else Sa >= null_threshold
    high_out_mask = Sb >= null_threshold.item() if isinstance(null_threshold, torch.Tensor) else Sb >= null_threshold

    U_in_bar  = Ua[:, high_in_mask]   # (in_features,  k_in)
    U_out_bar = Ub[:, high_out_mask]  # (out_features, k_out)

    print(
        f"[CrispEditParam] k_in={U_in_bar.shape[1]}/{Ua.shape[1]}, "
        f"k_out={U_out_bar.shape[1]}/{Ub.shape[1]}, threshold={null_threshold:.4g}"
    )
    return {"U_in_bar": U_in_bar, "U_out_bar": U_out_bar}


# ---------------------------------------------------------------------------
# 协方差统计获取
# ---------------------------------------------------------------------------

def get_cov_ab(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: int,
    mom2_dtype: str,
    force_recompute: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """获取单个层的 KFAC 输入/输出协方差矩阵 (A, B)。"""
    A, B = layer_stats_kfac(
        model, tok, layer_name, STATS_DIR, mom2_dataset,
        to_collect=["mom2"],
        sample_size=mom2_n_samples,
        precision=mom2_dtype,
        force_recompute=force_recompute,
    )
    return A, B


def calculate_cov_cache_with_old_data(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: CrispEditParamHyperParams,
    force_recompute: bool = False,
) -> Dict[str, Dict]:
    """用预训练数据（旧数据）一次性计算所有目标层的协方差缓存，返回 {layer_name → {A, B, num_samples}}。"""
    if hparams.no_crisp:
        return None

    layer_name_map = {
        layer_num: hparams.rewrite_module_tmp.format(layer_num)
        for layer_num in hparams.layers
    }
    target_layers = list(layer_name_map.values())

    stats_dict = layer_stats_kfac_one_pass(
        model=model, tokenizer=tok,
        layer_names=target_layers, stats_dir=STATS_DIR,
        ds_name=hparams.mom2_dataset, to_collect=["mom2"],
        sample_size=hparams.mom2_n_samples,
        precision=hparams.mom2_dtype,
        force_recompute=force_recompute,
    )

    layer_to_cov_cache = {}
    for layer_num in hparams.layers:
        layer_name = layer_name_map[layer_num]
        A, B, num_samples = stats_dict.pop(layer_name)
        layer_to_cov_cache[layer_name] = {
            "A": A.to("cpu", dtype=torch.float32),
            "B": B.to("cpu", dtype=torch.float32),
            "num_samples": num_samples,
        }
        del A, B

    return layer_to_cov_cache


def calculate_cov_cache_with_request(
    txt, tgt,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: CrispEditParamHyperParams,
) -> Dict[str, Dict]:
    """用编辑请求样本（txt/tgt）计算各目标层的协方差缓存，支持与预训练数据混合。"""
    if hparams.no_crisp:
        return None

    cov_stats_dict = layer_stats_kfac_with_txt_tgt(
        model, tok,
        layer_names=[hparams.rewrite_module_tmp.format(l) for l in hparams.layers],
        txt=txt, tgt=tgt, precision=hparams.mom2_dtype,
        sample_size=hparams.edit_n_samples, to_collect=["mom2"],
        add_pretrain_data=(hparams.edit_cache_style == "mix"),
        pretrain_sample_size=hparams.mom2_n_samples,
    )

    layer_to_cov_cache = {}
    for layer_num in hparams.layers:
        layer_name = hparams.rewrite_module_tmp.format(layer_num)
        A, B, num_samples = cov_stats_dict.pop(layer_name)
        layer_to_cov_cache[layer_name] = {
            "A": A.to("cpu", dtype=torch.float32),
            "B": B.to("cpu", dtype=torch.float32),
            "num_samples": num_samples,
        }
        del A, B
        torch.cuda.empty_cache()

    return layer_to_cov_cache


# ---------------------------------------------------------------------------
# 协方差缓存合并
# ---------------------------------------------------------------------------

def combine_layer_to_cov_caches(layer_to_cov_caches: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    """将多组协方差缓存按样本数加权平均，合并为一组。"""
    if len(layer_to_cov_caches) == 1:
        return layer_to_cov_caches[0]

    combined = {}
    for layer_name in layer_to_cov_caches[0].keys():
        A_list  = [c[layer_name]["A"] for c in layer_to_cov_caches]
        B_list  = [c[layer_name]["B"] for c in layer_to_cov_caches]
        ns_list = [c[layer_name]["num_samples"] for c in layer_to_cov_caches]
        total   = sum(ns_list)
        combined[layer_name] = {
            "A": sum(A * n for A, n in zip(A_list, ns_list)) / total,
            "B": sum(B * n for B, n in zip(B_list, ns_list)) / total,
            "num_samples": total,
        }
        print(f"Combined samples {ns_list} for {layer_name}")
    return combined


# ---------------------------------------------------------------------------
# CurvatureLora 附加与更新
# ---------------------------------------------------------------------------

def _iter_lora_layers(peft_model, adapter_name: str):
    """遍历 peft_model 中所有注入了指定 adapter 的 LoraLayer。"""
    for name, module in peft_model.named_modules():
        if (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "in_features")
            and hasattr(module, "out_features")
            and adapter_name in module.lora_A
            and adapter_name in module.lora_B
        ):
            yield name, module


def attach_curvature_lora(peft_model, adapter_name: str = "default") -> int:
    """
    对 peft_model 中所有 LoraLayer 注册 CurvatureLora variant 和空投影基 buffer。
    返回挂载层数。
    """
    count = 0
    for _, module in _iter_lora_layers(peft_model, adapter_name):
        if not hasattr(module, "lora_variant"):
            module.lora_variant = {}
        module.lora_variant[adapter_name] = CurvatureLora()
        CurvatureLora.init(module, adapter_name=adapter_name)
        count += 1
    print(f"[CrispEditParam] 已挂载 {count} 个 CurvatureLora 层")
    return count


def update_curvature_bases(
    peft_model,
    layer_to_projection_cache: Dict[str, Dict],
    hparams: CrispEditParamHyperParams,
    adapter_name: str = "default",
):
    """
    将计算好的投影基（U_in_bar / U_out_bar）写入对应 LoraLayer 的 buffer。

    layer_to_projection_cache:
        { layer_name_str → { "U_in_bar": Tensor, "U_out_bar": Tensor } }

    layer_name_str 通过 hparams.rewrite_module_tmp.format(layer_num) 生成，
    例如 "model.layers.{}.mlp.down_proj"。
    这里用模块全名做前缀匹配，找到对应的 LoraLayer。
    """
    updated = 0
    for layer_name, module in _iter_lora_layers(peft_model, adapter_name):
        # 找到该模块对应的 projection cache
        matched_cache = None
        for cache_key, cache_val in layer_to_projection_cache.items():
            if cache_key in layer_name or layer_name.endswith(cache_key):
                matched_cache = cache_val
                break
        if matched_cache is None:
            continue

        dev   = module.lora_A[adapter_name].weight.device
        dtype = module.lora_A[adapter_name].weight.dtype

        U_in_bar  = matched_cache["U_in_bar"].to(device=dev, dtype=dtype)
        U_out_bar = matched_cache["U_out_bar"].to(device=dev, dtype=dtype)

        # 直接替换 buffer（register_buffer 时已注册，这里用 setattr 覆盖）
        setattr(module, f"U_in_bar_{adapter_name}",  U_in_bar)
        setattr(module, f"U_out_bar_{adapter_name}", U_out_bar)
        updated += 1

    print(f"[CrispEditParam] 已更新 {updated} 个层的曲率投影基")


# ---------------------------------------------------------------------------
# 从协方差缓存计算投影基
# ---------------------------------------------------------------------------

def calculate_projection_caches_from_cov_caches(
    model: AutoModelForCausalLM,
    hparams: CrispEditParamHyperParams,
    layer_to_cov_caches: Dict[str, Dict],
    energy_threshold: Optional[float] = None,
) -> Dict[str, Dict]:
    """
    返回 { layer_name → { "U_in_bar": Tensor, "U_out_bar": Tensor } }
    """
    energy_threshold = energy_threshold or hparams.energy_threshold
    layer_to_projection_cache = {}

    for layer_name, cov_cache in layer_to_cov_caches.items():
        A = cov_cache["A"].to(model.device)
        B = cov_cache["B"].to(model.device)

        # crispedit 原版：非 llama 类模型需要交换 A/B
        if hparams.model_name not in ["Llama3-8B", "phi-1.5"]:
            A, B = B, A

        proj_cache = calculate_projection_cache_with_kfac(A, B, energy_threshold)
        layer_to_projection_cache[layer_name] = proj_cache

    return layer_to_projection_cache


# ---------------------------------------------------------------------------
# 模型包装：用 LoRA + CurvatureLora 替代原始权重直接优化
# ---------------------------------------------------------------------------

def wrap_model_with_curvature_lora(
    model: AutoModelForCausalLM,
    hparams: CrispEditParamHyperParams,
    adapter_name: str = "default",
):
    """
    1. 用标准 LoRA 包装模型
    2. 附加 CurvatureLora variant（此时投影基为空，等同普通 LoRA）
    3. 返回 (peft_model, optimizer)

    调用方随后应调用 apply_cov_caches_to_model() 填入投影基。
    """
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

    # 挂载 CurvatureLora，注册空基 buffer
    attach_curvature_lora(peft_model, adapter_name=adapter_name)

    # 只优化 LoRA 参数（投影基是 buffer，不参与优化）
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, peft_model.parameters()),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    return peft_model, opt


def apply_cov_caches_to_model(
    peft_model,
    layer_to_cov_caches: List[Dict[str, Dict]],
    model: AutoModelForCausalLM,
    hparams: CrispEditParamHyperParams,
    adapter_name: str = "default",
):
    """
    合并多组协方差缓存 → 计算投影基 → 写入 peft_model 各层。
    在每次新的编辑请求开始时调用（替代 projected_adam 里的 reset_cache）。
    """
    if hparams.no_crisp or not layer_to_cov_caches:
        return

    combined = combine_layer_to_cov_caches(layer_to_cov_caches)
    layer_to_projection_cache = calculate_projection_caches_from_cov_caches(
        model, hparams, combined
    )
    update_curvature_bases(peft_model, layer_to_projection_cache, hparams, adapter_name)


# ---------------------------------------------------------------------------
# 权重工具（保留与原 crispedit 一致的接口）
# ---------------------------------------------------------------------------

def get_weights(
    model: AutoModelForCausalLM,
    hparams: CrispEditParamHyperParams,
    bias: bool,
    to_cpu: bool = False,
) -> Dict[str, torch.Tensor]:
    """提取目标层的权重参数字典（不含 bias），可选择拷贝到 CPU。"""
    bias = False  # 暂不处理 bias
    return {
        n: (p.detach().cpu().clone() if to_cpu else p)
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n and "bias" not in n
    }


def cache_weights_to_cpu(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """将权重字典中所有张量拷贝到 CPU 并 detach，用于后续变化检测。"""
    return {name: param.detach().cpu().clone() for name, param in weights.items()}


def is_weights_changed(
    current_weights: Dict[str, torch.Tensor],
    cached_weights: Dict[str, torch.Tensor],
    threshold: float,
) -> bool:
    """检测模型权重相对于缓存快照是否发生显著变化（相对范数超过阈值）。"""
    for name, param in current_weights.items():
        cached_param = cached_weights[name]
        change = torch.norm(param.detach().cpu() - cached_param) / (torch.norm(cached_param) + 1e-8)
        if change > threshold:
            print(f"Weight {name} changed by {change:.4f}, exceeding threshold {threshold}.")
            return True
    return False


def recalculate_cov_cache_if_weights_changed(
    model, tok,
    hparams: CrispEditParamHyperParams,
    current_weights_cpu: Dict,
    layer_to_cov_cache: Dict,
) -> Tuple[Dict, Dict, bool]:
    """若权重变化超过阈值则重新计算协方差缓存，返回 (新权重快照, 新协方差缓存, 是否重算)。"""
    if not hparams.recalculate_cache or hparams.no_crisp:
        return current_weights_cpu, layer_to_cov_cache, False

    weights = get_weights(model, hparams, bias=True)
    if not is_weights_changed(weights, current_weights_cpu, hparams.recalculate_weight_threshold):
        return current_weights_cpu, layer_to_cov_cache, False

    del layer_to_cov_cache, weights
    gc.collect()
    torch.cuda.empty_cache()

    layer_to_cov_cache = calculate_cov_cache_with_old_data(model, tok, hparams, force_recompute=True)
    weights = get_weights(model, hparams, bias=True)
    current_weights_cpu = cache_weights_to_cpu(weights)

    return current_weights_cpu, layer_to_cov_cache, True


def update_model_and_tokenizer_with_appropriate_padding_token(model, tokenizer, hparams):
    """为模型和分词器配置合适的 padding token（Qwen 复用 eos，其余模型添加 [PAD]）。"""
    if "Qwen" in hparams.model_name:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
