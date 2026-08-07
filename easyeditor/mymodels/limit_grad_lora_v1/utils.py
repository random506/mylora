import os
import math
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from peft import LoraConfig, AdaLoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from copy import deepcopy

from .projected_lora_optimizer import ProjectedLoRAOptimizer
from ...models.rome.layer_stats import layer_stats_kfac_one_pass
from ..hparams import CrispLoRAHyperParams
from ...tools import ExperimentTracker

load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")


# ═══════════════════════════════════════════════════════════════════════════
# 工具：判断模型类型
# ═══════════════════════════════════════════════════════════════════════════

def _is_llama_or_phi(model_name: str) -> bool:
    lower = model_name.lower()
    return "llama" in lower or "phi" in lower


def get_topk_indices_by_energy_ratio(eigenvalues: torch.Tensor, percent: float = 0.9):
    """
        根据累积能量比例选择高曲率方向。
        返回：
        k: 需要保留的高曲率方向数量
        idx: 高曲率方向对应的特征值下标
        threshold: 第 k 个特征值阈值
    """
    # 协方差矩阵特征值一定大于等于0
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    sorted_eigvals, sorted_idx = torch.sort(eigenvalues, descending=True)
    total_energy = torch.sum(sorted_eigvals)
    if total_energy <= 0:
        return 0, sorted_idx[:0], torch.tensor(0.0, device=eigenvalues.device)
    cumulative_energy = torch.cumsum(sorted_eigvals, dim=0)
    energy_ratio = cumulative_energy / total_energy
    k = torch.searchsorted(energy_ratio, percent).item() + 1
    idx = sorted_idx[:k]
    threshold = sorted_eigvals[k - 1]
    return k, idx, threshold


def compute_marginal_masks(
    Sa: torch.Tensor,
    Ua: torch.Tensor,
    Sb: torch.Tensor,
    Ub: torch.Tensor,
    energy_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k_in, idx_in, threshold_in = get_topk_indices_by_energy_ratio(Sa, percent=energy_threshold)
    k_out, idx_out, threshold_out = get_topk_indices_by_energy_ratio(Sb, percent=energy_threshold)

    mask_a = Ua[:, idx_in].contiguous()
    mask_b = Ub[:, idx_out].contiguous()

    print(
        f"  mask_a: {k_in}/{Sa.shape[0]} safe dirs, "
        f"threshold_a={threshold_in:.6f}"
    )
    print(
        f"  mask_b: {k_out}/{Sb.shape[0]} safe dirs, "
        f"threshold_b={threshold_out:.6f}"
    )
    return mask_a, mask_b


def build_lora_projection_cache(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: CrispLoRAHyperParams,
    force_recompute: bool = False,
) -> Dict[str, Dict]:
    print("[SchemeA] 计算各层KFac协方差统计...")
    # layer_names 与 CrispEdit 保持一致，key 含 .weight 后缀
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
    for layer_num, layer_name in zip(hparams.layers, layer_names):
        A, B, _ = stats_dict.pop(layer_name)
        if not _is_llama_or_phi(hparams.model_name):
            A, B = B, A

        A = A.to(dtype=torch.float32)
        B = B.to(dtype=torch.float32)

        Sa, Ua = torch.linalg.eigh(A)  
        Sb, Ub = torch.linalg.eigh(B) 

        print(f"[SchemeA] 层 {layer_name} 边缘化掩码计算:")
        mask_a, mask_b = compute_marginal_masks(Sa, Ua,Sb,Ub, hparams.energy_threshold)

        layer_to_proj_cache[layer_name] = {
            "Ua": Ua.cpu(),
            "Ub": Ub.cpu(),
            "mask_a": mask_a.cpu(),
            "mask_b": mask_b.cpu(),
        }
        del A, B, Sa, Sb, Ua, Ub
        torch.cuda.empty_cache()

    return layer_to_proj_cache


def _register_leak_rate_for_layer(module, layer_name_clean: str, base_dtype: torch.dtype) -> nn.Parameter:
    attr = "leak_rate_default"
    if not hasattr(module, attr):
        param = nn.Parameter(torch.tensor([-4.0], dtype=base_dtype))
        module.register_parameter(attr, param)
        print(f"[GradLoRA] {layer_name_clean}: 注册 leak_rate_default (init=-4.0)")
    return getattr(module, attr)


def map_proj_cache_to_lora_params(
    peft_model,
    layer_to_proj_cache: Dict[str, Dict],
) -> Tuple[Dict[torch.nn.Parameter, Dict], List[nn.Parameter]]:
    param_to_proj_cache: Dict[torch.nn.Parameter, Dict] = {}

    # layer_clean → leak_rate_param 的映射，确保同一层 A/B 共享同一个 leak 参数
    layer_to_leak: Dict[str, nn.Parameter] = {}

    for name, param in peft_model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue

        matched_layer = None
        for layer_name in layer_to_proj_cache:
            # 去掉末尾的.weight
            clean = layer_name[:-len(".weight")] if layer_name.endswith(".weight") else layer_name
            if clean in name:
                matched_layer = layer_name
                break

        if matched_layer is None:
            continue

        clean = matched_layer[:-len(".weight")] if matched_layer.endswith(".weight") else matched_layer

        # 找到对应的 LoRA module，注册或复用 leak_rate_default
        if clean not in layer_to_leak:
            # 从 peft_model 中找到该层的 module（用于 register_parameter）
            matched_module = None
            for mod_name, mod in peft_model.named_modules():
                if clean in mod_name and hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
                    matched_module = mod
                    break
            if matched_module is not None:
                base_weight = (
                    matched_module.base_layer.weight
                    if hasattr(matched_module, "base_layer")
                    else matched_module.weight
                )
                leak_param = _register_leak_rate_for_layer(matched_module, clean, base_weight.dtype)
                layer_to_leak[clean] = leak_param
            else:
                # fallback：创建一个游离 Parameter（不挂在 module 上）
                layer_to_leak[clean] = nn.Parameter(torch.tensor([-4.0], dtype=param.dtype))

        leak_rate_param = layer_to_leak[clean]
        cache = layer_to_proj_cache[matched_layer]

        if "lora_A" in name:
            # lora_A: (r, d_in)，右投影，使用 Ua + mask_a
            param_to_proj_cache[param] = {
                "Ua":              cache["Ua"],
                "mask_a":          cache["mask_a"],
                "leak_rate_param": leak_rate_param,
                "param_type":      "lora_A",
            }
        elif "lora_B" in name:
            # lora_B: (d_out, r)，左投影，使用 Ub + mask_b
            param_to_proj_cache[param] = {
                "Ub":              cache["Ub"],
                "mask_b":          cache["mask_b"],
                "leak_rate_param": leak_rate_param,
                "param_type":      "lora_B",
            }

    leak_params = list(layer_to_leak.values())
    print(
        f"[GradLoRA] 建立参数→投影映射，共 {len(param_to_proj_cache)} 个 LoRA 参数，"
        f"{len(leak_params)} 个 leak_rate 参数"
    )
    return param_to_proj_cache, leak_params


def wrap_model_and_build_projected_optimizer(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: CrispLoRAHyperParams,
    force_recompute: bool = False,
):
    # 1. KFac 边缘化投影缓存（挂 LoRA 之前，基于原始模型权重统计）
    print("[GradLoRA] 1、计算 KFac 矩阵...")
    layer_to_proj_cache = build_lora_projection_cache(
        model, tok, hparams, force_recompute
    )

    # 2. 挂载 LoRA 适配器
    print("2、挂载lora")
    model.config.use_cache = False
    model.enable_input_require_grads()

    if hparams.lora_type == "lora":
        ConfigClass = LoraConfig
    elif hparams.lora_type == "adalora":
        ConfigClass = AdaLoraConfig
    else:
        raise ValueError(f"不支持的 lora_type: {hparams.lora_type}")

    peft_config = ConfigClass(
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

    # 3. 建立 param → 投影缓存的映射，同时注册 leak_rate_default 到各层 module
    #    map_proj_cache_to_lora_params 返回 (param_to_proj_cache, leak_params)
    print("3、将之前计算的结果注册到lora的各层")
    param_to_proj_cache, leak_params = map_proj_cache_to_lora_params(
        peft_model, layer_to_proj_cache
    )
    # 使用第二个优化器去优化泄露率参数
    lora_params = [p for p in peft_model.parameters() if p.requires_grad
                   and not any(p is lp for lp in leak_params)]

    # 5. 构建优化器
    #    optimizer_lora: ProjectedLoRAOptimizer，在 step 时自动做梯度软投影
    optimizer_lora = ProjectedLoRAOptimizer(
        params=lora_params,
        projection_cache_map=param_to_proj_cache,
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
        projection_mode=hparams.projection_mode,
        use_leak=hparams.use_leak,
        leak_rate=hparams.leak_rate
    )
    #    optimizer_leak: 普通 Adam，学习率 ×2（与 limit_lora 保持一致）
    optimizer_leak = torch.optim.Adam(leak_params, lr=hparams.lr * 2)

    print(
        f"[GradLoRA] 初始化完成：LoRA rank={hparams.lora_rank}，"
        f"投影模式={hparams.projection_mode}，"
        f"能量阈值={hparams.energy_threshold}，"
        f"leak_params 数量={len(leak_params)}"
    )

    return peft_model, optimizer_lora, optimizer_leak, layer_to_proj_cache



def apply_limit_grad_lora_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: CrispLoRAHyperParams,
    return_orig_weights: bool = False,
    keep_original_weight: bool = False,
    **kwargs,
) -> AutoModelForCausalLM:
    if tok.padding_side != "right":
        tok.padding_side = "right"

    for i, request in enumerate(requests):
        if request["target_new"] and request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]

    peft_model, optimizer_lora, optimizer_leak, _ = wrap_model_and_build_projected_optimizer(
        model, tok, hparams
    )

    device = next(peft_model.parameters()).device

    texts = [
        r["prompt"].format(r.get("subject", "")) if "{}" in r["prompt"] else r["prompt"]
        for r in requests
    ]
    targets = [r["target_new"] for r in requests]

    peft_model.train()
    for step in range(hparams.num_steps):
        total_loss = 0.0
        for txt_batch, tgt_batch in zip(
            _chunks(texts, hparams.batch_size),
            _chunks(targets, hparams.batch_size),
        ):
            optimizer_lora.zero_grad()
            optimizer_leak.zero_grad()

            loss = _compute_loss(peft_model, tok, txt_batch, tgt_batch, device, hparams)

            if loss.item() >= 1e-3:
                loss.backward()
                optimizer_lora.step()
                optimizer_leak.step()

            total_loss += loss.item()

        num_batches = max(1, math.ceil(len(texts) / hparams.batch_size))
        avg_loss = total_loss / num_batches
        ExperimentTracker.log({"LOSS": avg_loss})
        print(f"[GradLoRA] Step {step+1}/{hparams.num_steps}  loss={avg_loss:.4f}")
        if avg_loss < 1e-3:
            print("[GradLoRA] 损失收敛，提前结束训练")
            break

    peft_model = peft_model.merge_and_unload()
    return peft_model


def _compute_loss(
    model,
    tok: AutoTokenizer,
    texts: List[str],
    targets: List[str],
    device: torch.device,
    hparams: CrispLoRAHyperParams,
) -> torch.Tensor:
    """
    计算编辑损失，参照 crispedit.py 的 execute_ft 训练循环：
      - 拼接 prompt + target_new
      - prompt 部分 label 设为 -100（只对 target 计算 loss）
      - padding token 的 label 设为 -100
    """
    inputs_targets = [t + tg for t, tg in zip(texts, targets)]
    encodings = tok(
        inputs_targets,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
    ).to(device)

    labels = encodings["input_ids"].clone()
    # padding 位置不计 loss
    labels[labels == tok.pad_token_id] = -100
    # prompt 部分不计 loss
    for i, prompt in enumerate(texts):
        prompt_len = len(
            tok(prompt, add_special_tokens=True, truncation=True,
                max_length=hparams.max_length)["input_ids"]
        )
        labels[i, :prompt_len] = -100

    return model(**encodings, labels=labels).loss


def _chunks(lst: List, n: int):
    """将列表按大小 n 切分，参照 utils.py 的 chunks 函数"""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]
