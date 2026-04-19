"""
方案C工具函数：CrispEdit-LoRA联合框架
=====================================================
整合方案A（边缘化投影优化器）+ 方案B（KFac初始化）+ CrispEdit（动态缓存更新）

流程：
  1. 计算预训练数据协方差 → KFac特征值分解
  2. 用低能量特征向量初始化LoRA A/B
  3. 构建CrispLoRAOptimizer（边缘化投影）
  4. 训练循环中，若权重变化超过阈值，重新计算协方差并更新投影
"""

import gc
import os
import torch
from typing import Dict, List, Tuple, Optional, Any
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

from ..scheme_c_crispedit_lora.hparams import SchemeCHyperParams
from ..scheme_c_crispedit_lora.crisp_lora_optimizer import CrispLoRAOptimizer
from ...models.rome.layer_stats import (
    layer_stats_kfac_one_pass,
    layer_stats_kfac_with_txt_tgt,
)

load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")


# ═══════════════════════════════════════════════════════════════════════════
# 协方差与投影缓存计算
# ═══════════════════════════════════════════════════════════════════════════

def compute_cov_and_proj_cache(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: SchemeCHyperParams,
    force_recompute: bool = False,
) -> Dict[str, Dict]:
    """
    计算预训练数据的协方差 + KFac特征值分解 + 边缘化掩码。

    Returns:
        layer_to_proj: {
            layer_name: {
                'Ua', 'Ub': 特征向量,
                'mask_a', 'mask_b': 低能量掩码,
                'Sa', 'Sb': 特征值（用于初始化选择）,
                'A', 'B': 原始协方差（用于动态更新）,
            }
        }
    """
    print("[SchemeC] 计算预训练数据协方差缓存...")
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

    layer_to_proj = {}
    for layer_name in layer_names:
        A, B, num_samples = stats_dict.pop(layer_name)
        if hparams.model_name not in ["Llama3-8B", "phi-1.5"]:
            A, B = B, A
        A = A.to(dtype=torch.float32)
        B = B.to(dtype=torch.float32)

        Sa, Ua = torch.linalg.eigh(A)
        Sb, Ub = torch.linalg.eigh(B)

        mask_a, mask_b = _compute_marginal_masks(Sa, Sb, hparams.energy_threshold)

        layer_to_proj[layer_name] = {
            "Sa": Sa.cpu(), "Ua": Ua.cpu(),
            "Sb": Sb.cpu(), "Ub": Ub.cpu(),
            "mask_a": mask_a.cpu(),
            "mask_b": mask_b.cpu(),
            "A": A.cpu(),   # 原始协方差备用（动态更新时需要合并）
            "B": B.cpu(),
            "num_samples": num_samples,
        }
        del Sa, Sb, Ua, Ub
        torch.cuda.empty_cache()

    return layer_to_proj


def compute_edit_request_proj_cache(
    txt: List[str],
    tgt: List[str],
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: SchemeCHyperParams,
) -> Dict[str, Dict]:
    """
    用当前编辑请求数据计算协方差 + 边缘化投影缓存。
    用于双重投影模式（edit_cache_style = "edit_only" 或 "mix"）。
    """
    print("[SchemeC] 计算编辑请求数据协方差缓存...")
    layer_names = [hparams.rewrite_module_tmp.format(l) for l in hparams.layers]

    stats_dict = layer_stats_kfac_with_txt_tgt(
        model,
        tok,
        layer_names=layer_names,
        txt=txt,
        tgt=tgt,
        precision=hparams.mom2_dtype,
        sample_size=hparams.edit_n_samples,
        to_collect=["mom2"],
        add_pretrain_data=(hparams.edit_cache_style == "mix"),
        pretrain_sample_size=hparams.mom2_n_samples,
    )

    layer_to_proj = {}
    for layer_name in layer_names:
        A, B, num_samples = stats_dict.pop(layer_name)
        if hparams.model_name not in ["Llama3-8B", "phi-1.5"]:
            A, B = B, A
        A = A.to(dtype=torch.float32)
        B = B.to(dtype=torch.float32)

        Sa, Ua = torch.linalg.eigh(A)
        Sb, Ub = torch.linalg.eigh(B)
        mask_a, mask_b = _compute_marginal_masks(Sa, Sb, hparams.energy_threshold)

        layer_to_proj[layer_name] = {
            "Ua": Ua.cpu(), "Ub": Ub.cpu(),
            "mask_a": mask_a.cpu(), "mask_b": mask_b.cpu(),
        }
        del A, B, Sa, Sb, Ua, Ub
        torch.cuda.empty_cache()

    return layer_to_proj


def _compute_marginal_masks(
    Sa: torch.Tensor,
    Sb: torch.Tensor,
    energy_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算输入/输出方向的低能量一维掩码"""
    def threshold_from_eigenvalues(vals: torch.Tensor, percent: float) -> float:
        sorted_vals, _ = torch.sort(vals, descending=True)
        total = sorted_vals.sum()
        if total == 0:
            return float("inf")
        ratio = torch.cumsum(sorted_vals, dim=0) / total
        rank = torch.searchsorted(ratio, percent).item() + 1
        return sorted_vals[min(rank - 1, len(sorted_vals) - 1)].item()

    thr_a = threshold_from_eigenvalues(Sa, energy_threshold)
    thr_b = threshold_from_eigenvalues(Sb, energy_threshold)

    mask_a = Sa < thr_a
    mask_b = Sb < thr_b

    safe_a = mask_a.sum().item()
    safe_b = mask_b.sum().item()
    print(f"  mask_a: {safe_a}/{len(mask_a)} 安全方向  "
          f"mask_b: {safe_b}/{len(mask_b)} 安全方向")

    return mask_a, mask_b


# ═══════════════════════════════════════════════════════════════════════════
# 参数→投影缓存映射
# ═══════════════════════════════════════════════════════════════════════════

def build_param_to_proj_map(
    peft_model,
    layer_to_proj: Dict[str, Dict],
    hparams: SchemeCHyperParams,
) -> Dict[torch.nn.Parameter, Dict]:
    """将 layer_name → proj 映射转为 param → proj 映射"""
    param_to_proj = {}
    for name, module in peft_model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        matched_layer = None
        for layer_name in layer_to_proj:
            if layer_name.replace(".", ".") in name:
                matched_layer = layer_name
                break
        if matched_layer is None:
            continue
        proj = layer_to_proj[matched_layer]
        for adapter_name in module.lora_A:
            param_A = module.lora_A[adapter_name].weight
            param_B = module.lora_B[adapter_name].weight
            if param_A.requires_grad:
                param_to_proj[param_A] = {
                    "Ua": proj["Ua"],
                    "mask_a": proj["mask_a"],
                    "param_type": "lora_A",
                }
            if param_B.requires_grad:
                param_to_proj[param_B] = {
                    "Ub": proj["Ub"],
                    "mask_b": proj["mask_b"],
                    "param_type": "lora_B",
                }
    print(f"[SchemeC] 参数→投影映射：共 {len(param_to_proj)} 个LoRA参数")
    return param_to_proj


# ═══════════════════════════════════════════════════════════════════════════
# KFac初始化（继承方案B逻辑）
# ═══════════════════════════════════════════════════════════════════════════

def initialize_lora_from_kfac(
    peft_model,
    layer_to_proj: Dict[str, Dict],
    hparams: SchemeCHyperParams,
):
    """用KFac低能量特征向量初始化LoRA A矩阵，B初始化为零"""
    rank = hparams.lora_rank
    print(f"[SchemeC] KFac初始化LoRA参数 (rank={rank})...")

    for name, module in peft_model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        matched_layer = None
        for layer_name in layer_to_proj:
            if layer_name.replace(".", ".") in name:
                matched_layer = layer_name
                break
        if matched_layer is None:
            continue

        proj = layer_to_proj[matched_layer]
        Ua = proj["Ua"]  # (d_in, d_in)
        Ub = proj["Ub"]  # (d_out, d_out)

        for adapter_name in module.lora_A:
            param_A = module.lora_A[adapter_name].weight  # (r, d_in)
            param_B = module.lora_B[adapter_name].weight  # (d_out, r)

            d_in = param_A.shape[1]
            d_out = param_B.shape[0]
            r = min(rank, d_in)

            # 取最低能量的r列（eigh升序，前r列能量最低）
            init_A = Ua[:d_in, :r].T.clone()  # (r, d_in)

            if hparams.normalize_init:
                norms = init_A.norm(dim=1, keepdim=True).clamp(min=1e-8)
                init_A = init_A / norms

            init_B = torch.zeros(d_out, r, dtype=param_B.dtype)

            with torch.no_grad():
                param_A.copy_(init_A.to(dtype=param_A.dtype))
                param_B.copy_(init_B)

            print(f"  ✓ {name}.{adapter_name}: "
                  f"A={tuple(param_A.shape)}, B={tuple(param_B.shape)}")


# ═══════════════════════════════════════════════════════════════════════════
# 主包装函数
# ═══════════════════════════════════════════════════════════════════════════

def build_crisp_lora_model_and_optimizer(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: SchemeCHyperParams,
    force_recompute: bool = False,
):
    """
    构建完整的CrispEdit-LoRA联合框架。

    Returns:
        peft_model: KFac初始化的LoRA模型
        optimizer: CrispLoRAOptimizer（边缘化投影）
        layer_to_proj: 各层投影缓存（用于动态更新）
    """
    # 1. 计算协方差 + 投影缓存
    layer_to_proj = compute_cov_and_proj_cache(model, tok, hparams, force_recompute)

    # 2. peft包装
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
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
    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()

    # 3. KFac初始化（可选）
    if hparams.use_kfac_init:
        initialize_lora_from_kfac(peft_model, layer_to_proj, hparams)

    # 4. 建立参数→投影映射
    param_to_proj = build_param_to_proj_map(peft_model, layer_to_proj, hparams)

    # 5. 构建优化器
    if hparams.use_projected_optimizer:
        optimizer = CrispLoRAOptimizer(
            params=[p for p in peft_model.parameters() if p.requires_grad],
            projection_cache_map=param_to_proj,
            lr=hparams.lr,
            weight_decay=hparams.weight_decay,
            projection_mode=hparams.projection_mode,
        )
    else:
        optimizer = torch.optim.Adam(
            [p for p in peft_model.parameters() if p.requires_grad],
            lr=hparams.lr,
            weight_decay=hparams.weight_decay,
        )

    print(f"[SchemeC] 初始化完成：KFac初始化={hparams.use_kfac_init}，"
          f"投影优化器={hparams.use_projected_optimizer}，"
          f"编辑缓存风格={hparams.edit_cache_style}")

    return peft_model, optimizer, layer_to_proj


def recalculate_proj_if_weights_changed(
    peft_model,
    tok: AutoTokenizer,
    hparams: SchemeCHyperParams,
    optimizer: CrispLoRAOptimizer,
    cached_lora_weights: Dict[str, torch.Tensor],
    layer_to_proj: Dict[str, Dict],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict], bool]:
    """
    检测LoRA权重是否变化超过阈值，若是则重新计算协方差缓存并更新优化器投影。
    继承CrispEdit的动态缓存失效机制，适配LoRA增量参数。

    Returns:
        (new_cached_weights, new_layer_to_proj, updated: bool)
    """
    if not hparams.recalculate_cache:
        return cached_lora_weights, layer_to_proj, False

    # 获取当前LoRA参数（ΔW增量）
    current_lora_weights = {
        n: p.detach().cpu().clone()
        for n, p in peft_model.named_parameters()
        if p.requires_grad
    }

    # 检测变化幅度
    changed = False
    for name, param in current_lora_weights.items():
        if name not in cached_lora_weights:
            continue
        cached = cached_lora_weights[name]
        norm_change = (param - cached).norm() / cached.norm().clamp(min=1e-8)
        if norm_change > hparams.recalculate_weight_threshold:
            print(f"[SchemeC] 参数 {name} 变化 {norm_change:.4f}，触发缓存更新")
            changed = True
            break

    if not changed:
        return cached_lora_weights, layer_to_proj, False

    # 重新计算（使用peft_model的base_model）
    del layer_to_proj
    gc.collect()
    torch.cuda.empty_cache()

    layer_to_proj = compute_cov_and_proj_cache(
        peft_model.get_base_model(), tok, hparams, force_recompute=True
    )
    param_to_proj = build_param_to_proj_map(peft_model, layer_to_proj, hparams)

    if isinstance(optimizer, CrispLoRAOptimizer):
        optimizer.reset_cache(param_to_proj)

    new_cached = {
        n: p.detach().cpu().clone()
        for n, p in peft_model.named_parameters()
        if p.requires_grad
    }
    return new_cached, layer_to_proj, True


# ═══════════════════════════════════════════════════════════════════════════
# 主训练入口
# ═══════════════════════════════════════════════════════════════════════════

def apply_scheme_c_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: SchemeCHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    keep_original_weight: bool = False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict]:
    """方案C主入口，适配EasyEditor调用约定"""
    from copy import deepcopy
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    _update_tokenizer(model, tok, hparams)

    # 构建联合框架
    peft_model, optimizer, layer_to_proj = build_crisp_lora_model_and_optimizer(
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

    # 编辑请求数据的辅助投影缓存（若需要）
    if (isinstance(optimizer, CrispLoRAOptimizer)
            and hparams.edit_cache_style != "pretrain_only"):
        try:
            edit_proj = compute_edit_request_proj_cache(
                texts, targets, peft_model.get_base_model(), tok, hparams
            )
            additional_param_proj = build_param_to_proj_map(
                peft_model, edit_proj, hparams
            )
            optimizer.reset_additional_cache(additional_param_proj)
            print("[SchemeC] 已启用双重投影（预训练 + 编辑请求）")
        except Exception as e:
            print(f"[SchemeC] 辅助投影缓存计算失败，退化为单投影: {e}")

    # 缓存初始LoRA权重（用于动态更新检测）
    cached_lora_weights = {
        n: p.detach().cpu().clone()
        for n, p in peft_model.named_parameters()
        if p.requires_grad
    }

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
        print(f"[SchemeC] Step {step+1}/{hparams.num_steps}  loss={avg_loss:.4f}")

        # 动态检测权重变化，必要时更新投影缓存
        if isinstance(optimizer, CrispLoRAOptimizer):
            cached_lora_weights, layer_to_proj, updated = (
                recalculate_proj_if_weights_changed(
                    peft_model, tok, hparams, optimizer,
                    cached_lora_weights, layer_to_proj
                )
            )
            if updated:
                print(f"[SchemeC] Step {step+1}: 投影缓存已更新")

        if avg_loss < 1e-3:
            print("[SchemeC] 损失收敛，提前结束训练")
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
