"""
方案D工具函数：KFac能量驱动的自适应LoRA秩分配
=====================================================

核心算法：
  对每一层计算其"安全子空间能量占比"（null_energy_ratio）：
    null_energy_ratio[l] = sum(eigenvalues < threshold) / total_eigenvalues_sum

  然后用该比例分配总rank预算：
    raw_rank[l] = f(null_energy_ratio[l])   # f是映射函数（linear/sqrt/log）
    final_rank[l] = clip(normalize(raw_rank) * total_budget,
                         min_rank, max_rank)

  直觉：一层的"null_energy_ratio"越高，说明它有更多"空闲"方向可以安全编辑
        → 分配更多rank → 在安全的方向上进行更丰富的知识编辑
"""

import os
import math
import torch
from typing import Dict, List, Tuple, Any
from peft import AdaLoraConfig, LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

from ..scheme_d_adaptive_rank_lora.hparams import SchemeDHyperParams
from ...models.rome.layer_stats import layer_stats_kfac_one_pass

load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")


# ═══════════════════════════════════════════════════════════════════════════
# 安全能量比计算
# ═══════════════════════════════════════════════════════════════════════════

def compute_null_energy_ratio_per_layer(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: SchemeDHyperParams,
    force_recompute: bool = False,
) -> Dict[str, float]:
    """
    计算每一层输入/输出协方差的"低能量空间占比"。

    低能量空间占比（null_energy_ratio）：
      设KFac协方差的特征值为 {λ1, ..., λd}（升序），
      null_threshold 由 energy_threshold 决定（覆盖energy_threshold能量的最小特征值）
      null_energy_ratio = sum(λ_i for λ_i < null_threshold) / sum(λ_i)

    返回值越高 → 该层越多的能量在低活跃方向 → 越"安全"→ 适合分配更多rank

    Returns:
        {layer_name: null_energy_ratio (float, 0~1)}
    """
    print("[SchemeD] 计算各层低能量空间占比...")
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

    layer_to_null_ratio = {}

    for layer_name in layer_names:
        A, B, _ = stats_dict.pop(layer_name)
        if hparams.model_name not in ["Llama3-8B", "phi-1.5"]:
            A, B = B, A
        A = A.to(dtype=torch.float32)
        B = B.to(dtype=torch.float32)

        # 联合计算A和B的能量占比（取平均）
        ratio_A = _compute_single_null_energy_ratio(A, hparams.energy_threshold)
        ratio_B = _compute_single_null_energy_ratio(B, hparams.energy_threshold)
        combined_ratio = (ratio_A + ratio_B) / 2.0

        layer_to_null_ratio[layer_name] = combined_ratio
        print(f"  {layer_name}: ratio_A={ratio_A:.4f}, ratio_B={ratio_B:.4f}, "
              f"combined={combined_ratio:.4f}")

        del A, B
        torch.cuda.empty_cache()

    return layer_to_null_ratio


def _compute_single_null_energy_ratio(
    cov: torch.Tensor,
    energy_threshold: float,
) -> float:
    """
    计算单个协方差矩阵的低能量空间能量占比。

    Args:
        cov: 协方差矩阵 (d, d)
        energy_threshold: 累积能量阈值（如0.99）

    Returns:
        null_energy_ratio: 低能量空间所占能量比例（0~1）
    """
    eigenvalues, _ = torch.linalg.eigh(cov)  # 升序
    eigenvalues = eigenvalues.clamp(min=0)    # 协方差半正定，修正数值误差

    total_energy = eigenvalues.sum()
    if total_energy == 0:
        return 0.0

    # 确定null_threshold（覆盖energy_threshold能量所需的最小特征值）
    sorted_vals, _ = torch.sort(eigenvalues, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=0)
    ratio = cumsum / total_energy
    rank_idx = torch.searchsorted(ratio, energy_threshold).item()
    null_threshold = sorted_vals[min(rank_idx, len(sorted_vals) - 1)].item()

    # 低能量方向所占能量
    null_mask = eigenvalues < null_threshold
    null_energy = eigenvalues[null_mask].sum().item()

    return float(null_energy) / float(total_energy)


# ═══════════════════════════════════════════════════════════════════════════
# Rank分配
# ═══════════════════════════════════════════════════════════════════════════

def allocate_ranks_from_energy(
    layer_to_null_ratio: Dict[str, float],
    hparams: SchemeDHyperParams,
) -> Dict[str, int]:
    """
    根据各层的低能量空间占比，按预算分配LoRA rank。

    分配逻辑：
      1. 对null_ratio应用映射函数f（linear/sqrt/log）
      2. 归一化到总预算
      3. 取整并保证最小值

    Returns:
        {layer_name: allocated_rank (int)}
    """
    strategy = hparams.rank_allocation_strategy
    total_budget = hparams.total_rank_budget
    min_r = hparams.min_rank_per_layer
    max_r = hparams.max_rank_per_layer
    n_layers = len(layer_to_null_ratio)

    if strategy == "uniform":
        # 等额分配
        uniform_rank = max(min_r, total_budget // n_layers)
        ranks = {name: min(uniform_rank, max_r) for name in layer_to_null_ratio}
        if hparams.verbose_rank_alloc:
            _print_rank_table(layer_to_null_ratio, ranks, strategy)
        return ranks

    # 应用映射函数
    mapped = {}
    for name, ratio in layer_to_null_ratio.items():
        if strategy == "linear":
            mapped[name] = ratio
        elif strategy == "sqrt":
            mapped[name] = math.sqrt(ratio)
        elif strategy == "log":
            mapped[name] = math.log(1 + ratio * 9)  # log(1) ~ log(10)
        else:
            raise ValueError(f"未知rank分配策略: {strategy}")

    # 归一化分配
    total_mapped = sum(mapped.values())
    if total_mapped == 0:
        # 全部为0，等额分配
        return {name: max(min_r, total_budget // n_layers)
                for name in layer_to_null_ratio}

    raw_ranks = {
        name: (v / total_mapped) * total_budget
        for name, v in mapped.items()
    }

    # 取整 + clamp
    ranks = {
        name: int(min(max(min_r, round(r)), max_r))
        for name, r in raw_ranks.items()
    }

    # 调整总和（由于取整可能有偏差）
    current_total = sum(ranks.values())
    diff = total_budget - current_total
    if diff != 0:
        # 将差值分配给比例最大的层
        sorted_by_ratio = sorted(layer_to_null_ratio.items(),
                                 key=lambda x: x[1], reverse=True)
        for i in range(abs(diff)):
            name = sorted_by_ratio[i % len(sorted_by_ratio)][0]
            delta = 1 if diff > 0 else -1
            new_r = ranks[name] + delta
            if min_r <= new_r <= max_r:
                ranks[name] = new_r

    if hparams.verbose_rank_alloc:
        _print_rank_table(layer_to_null_ratio, ranks, strategy)

    return ranks


def _print_rank_table(
    layer_to_null_ratio: Dict[str, float],
    ranks: Dict[str, int],
    strategy: str,
):
    """打印rank分配结果表格"""
    print(f"\n{'='*65}")
    print(f"{'[SchemeD] KFac自适应Rank分配结果':^65}")
    print(f"  策略: {strategy}   总预算: {sum(ranks.values())}")
    print(f"{'─'*65}")
    print(f"{'层名':<45}  {'安全比':<8}  {'Rank':>5}")
    print(f"{'─'*65}")
    for name in layer_to_null_ratio:
        ratio = layer_to_null_ratio[name]
        rank = ranks.get(name, 0)
        # 简化层名显示
        short_name = name.split(".")[-3:]
        short_name = ".".join(short_name)
        print(f"  ...{short_name:<42}  {ratio:.4f}  {rank:>5}")
    print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 模型包装（逐层不同rank）
# ═══════════════════════════════════════════════════════════════════════════

def wrap_model_with_adaptive_rank_lora(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: SchemeDHyperParams,
    force_recompute: bool = False,
):
    """
    用KFac自适应秩构建LoRA模型。

    注意：peft的LoraConfig不支持逐层不同rank的直接配置，
    因此我们用以下策略：
      - 使用AdaLoraConfig（支持动态调整rank）
      - 设置 target_r = avg_rank，init_r = max_rank
      - 同时记录各层的rank分配用于后续可视化

    若需要严格逐层不同rank，可用手动注入LoRA层的方式（见注释）。

    Returns:
        peft_model, optimizer, rank_allocation, layer_to_null_ratio
    """
    # 1. 计算各层安全比
    layer_to_null_ratio = compute_null_energy_ratio_per_layer(
        model, tok, hparams, force_recompute
    )

    # 2. 计算rank分配
    layer_name_to_rank = allocate_ranks_from_energy(layer_to_null_ratio, hparams)

    # 3. 计算统计量供AdaLoRA配置
    all_ranks = list(layer_name_to_rank.values())
    avg_rank = max(1, round(sum(all_ranks) / len(all_ranks)))
    max_rank = max(all_ranks)
    min_rank = min(all_ranks)

    print(f"[SchemeD] 自适应Rank统计: avg={avg_rank}, max={max_rank}, min={min_rank}")

    # 4. peft包装
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if hparams.lora_type == "adalora":
        # AdaLoRA：动态调整各层rank
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=avg_rank,          # 目标rank（最终收敛的rank）
            lora_alpha=hparams.lora_alpha,
            lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules,
            target_r=avg_rank,   # AdaLoRA特有：目标秩
            init_r=max_rank,     # AdaLoRA特有：初始秩（从max开始收缩）
        )
    else:
        # 普通LoRA：用平均rank
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=avg_rank,
            lora_alpha=hparams.lora_alpha,
            lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules,
        )

    peft_model = get_peft_model(model, peft_config)
    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()

    # 5. 优化器（可选叠加边缘化投影）
    if hparams.use_projected_optimizer:
        # 复用方案A/C的边缘化投影逻辑
        from ..scheme_a_projected_lora.utils import (
            build_lora_projection_cache,
            map_proj_cache_to_lora_params,
        )
        from ..scheme_a_projected_lora.projected_lora_optimizer import ProjectedLoRAOptimizer

        # 构建新的hparams兼容对象（SchemeA）
        class _TmpHparams:
            model_name = hparams.model_name
            layers = hparams.layers
            rewrite_module_tmp = hparams.rewrite_module_tmp
            mom2_dataset = hparams.mom2_dataset
            mom2_n_samples = hparams.mom2_n_samples
            mom2_dtype = hparams.mom2_dtype
            energy_threshold = hparams.energy_threshold
            projection_mode = hparams.projection_mode

        tmp = _TmpHparams()
        layer_to_proj = build_lora_projection_cache(model, tok, tmp)
        param_to_proj = map_proj_cache_to_lora_params(peft_model, layer_to_proj, tmp)
        optimizer = ProjectedLoRAOptimizer(
            [p for p in peft_model.parameters() if p.requires_grad],
            projection_cache_map=param_to_proj,
            lr=hparams.lr,
            weight_decay=hparams.weight_decay,
            projection_mode=hparams.projection_mode,
        )
        print("[SchemeD] 已叠加方案A的边缘化投影优化器")
    else:
        optimizer = torch.optim.Adam(
            [p for p in peft_model.parameters() if p.requires_grad],
            lr=hparams.lr,
            weight_decay=hparams.weight_decay,
        )

    return peft_model, optimizer, layer_name_to_rank, layer_to_null_ratio


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def apply_scheme_d_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: SchemeDHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    keep_original_weight: bool = False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict]:
    """方案D主入口，适配EasyEditor调用约定"""
    from copy import deepcopy
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    _update_tokenizer(model, tok, hparams)

    peft_model, optimizer, rank_alloc, null_ratios = wrap_model_with_adaptive_rank_lora(
        model, tok, hparams
    )

    device = torch.device(f"cuda:{hparams.device}")
    peft_model = peft_model.to(device)

    texts = [
        r["prompt"].format(r.get("subject", "")) if "{}" in r["prompt"]
        else r["prompt"]
        for r in requests
    ]
    targets = [r["target_new"] for r in requests]

    peft_model.train()
    for step in range(hparams.num_steps):
        total_loss = 0.0
        n_batches = 0
        for txt_batch, tgt_batch in zip(
            _chunks(texts, hparams.batch_size),
            _chunks(targets, hparams.batch_size),
        ):
            optimizer.zero_grad()
            loss = _compute_loss(peft_model, tok, txt_batch, tgt_batch,
                                 device, hparams)
            if loss.item() >= 1e-3:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"[SchemeD] Step {step+1}/{hparams.num_steps}  loss={avg_loss:.4f}")
        if avg_loss < 1e-3:
            print("[SchemeD] 损失收敛，提前结束训练")
            break

    peft_model = peft_model.merge_and_unload()
    return peft_model, weights_copy


def _compute_loss(model, tok, texts, targets, device, hparams):
    mask_token = -100
    full_prompts = [f"{p} {t}" for p, t in zip(texts, targets)]
    prompt_ids = tok(list(texts), return_tensors="pt", padding=True,
                     truncation=True)["input_ids"]
    num_prompt_toks = [(i != tok.pad_token_id).sum().item() for i in prompt_ids]
    tokens = tok(full_prompts, return_tensors="pt", padding=True,
                 truncation=True, max_length=hparams.max_length)
    tokens["labels"] = tokens["input_ids"].clone()
    num_pad_toks = [(i == tok.pad_token_id).sum().item()
                    for i in tokens["labels"]]
    for i in range(len(texts)):
        tokens["labels"][i][num_pad_toks[i]: num_pad_toks[i] + num_prompt_toks[i]] = mask_token
    tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = mask_token
    tokens = {k: v.to(device) for k, v in tokens.items()}
    return model(**tokens).loss


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def _update_tokenizer(model, tokenizer, hparams):
    if "Qwen" in hparams.model_name:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    else:
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            model.config.pad_token_id = tokenizer.pad_token_id
