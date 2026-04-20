"""
Leaky曲率投影LoRA工具函数
=====================================================
结合 LeakyCurvatureLora（软投影前向）和 GlobalAwareProjectedLoRAOptimizer（梯度投影），
实现带泄漏率的参数增量约束方案。

流程：
  1. 计算 KFac 投影缓存（同 limit_grad_lora）
  2. 挂载 LoRA + 注入 LeakyCurvatureLora variant（设置 U_in_bar/U_out_bar）
  3. 分离 lora_params 和 leak_params，构建双优化器
  4. 执行训练循环，合并权重
"""

import os
import torch
from typing import Dict, List, Tuple, Optional
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

from .LeakyCurvatureLora import LeakyCurvatureLora
from .lora_optimizer import GlobalAwareProjectedLoRAOptimizer
from ...models.rome.layer_stats import layer_stats_kfac_one_pass
from ..hparams import CrispLoRAHyperParams

load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")

def _is_llama_or_phi(model_name: str) -> bool:
    # 不同模型对于A、B矩阵有差异
    lower = model_name.lower()
    return "llama" in lower or "phi" in lower


def get_rank_and_threshold_by_energy_ratio(eigenvalues: torch.Tensor, percent: float = 0.9):
    total_energy = torch.sum(eigenvalues)
    sorted_eigvals, _ = torch.sort(eigenvalues, descending=True)
    cumulative_energy = torch.cumsum(sorted_eigvals, dim=0)
    energy_ratio = cumulative_energy / total_energy
    # 找到阈值
    rank = torch.searchsorted(energy_ratio, percent).item() + 1 
    # 对应的特征值
    threshold = sorted_eigvals[rank - 1] if rank - 1 < len(sorted_eigvals) else 0.0
    return rank, threshold


def compute_marginal_masks(
    Sa: torch.Tensor,
    Sb: torch.Tensor,
    energy_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 计算掩码矩阵，一维
    _, threshold_a = get_rank_and_threshold_by_energy_ratio(Sa, percent=energy_threshold)
    _, threshold_b = get_rank_and_threshold_by_energy_ratio(Sb, percent=energy_threshold)
    mask_a = Sa < threshold_a
    mask_b = Sb < threshold_b
    print(
        f"  mask_a: {mask_a.sum().item()}/{len(mask_a)} safe dirs, "
        f"threshold_a={threshold_a:.6f}"
    )
    print(
        f"  mask_b: {mask_b.sum().item()}/{len(mask_b)} safe dirs, "
        f"threshold_b={threshold_b:.6f}"
    )
    return mask_a, mask_b


def build_leaky_projection_cache(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: CrispLoRAHyperParams,
    force_recompute: bool = False,
) -> Dict[str, Dict]:
    """
    为每一层计算 KFac 边缘化投影缓存（含 Ua/Ub 特征向量和 mask）。

    Returns:
        layer_to_proj_cache: {
            layer_name (含.weight): {
                'Ua': Tensor[d_in, d_in],
                'Ub': Tensor[d_out, d_out],
                'U_in_bar': Tensor[d_in, k_in],   # 高曲率方向，供 LeakyCurvatureLora 使用
                'U_out_bar': Tensor[d_out, k_out],
                'mask_a': Tensor[d_in],             # bool，供 optimizer 使用
                'mask_b': Tensor[d_out],
            }
        }
    """
    print("[LeakyLoRA] 计算各层KFac协方差统计...")

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
    for _, layer_name in zip(hparams.layers, layer_names):
        A, B, _ = stats_dict.pop(layer_name)

        if not _is_llama_or_phi(hparams.model_name):
            A, B = B, A

        A = A.to(dtype=torch.float32)
        B = B.to(dtype=torch.float32)

        Sa, Ua = torch.linalg.eigh(A)  # 升序特征值
        Sb, Ub = torch.linalg.eigh(B)

        print(f"[LeakyLoRA] 层 {layer_name} 边缘化掩码计算:")
        mask_a, mask_b = compute_marginal_masks(Sa, Sb, hparams.energy_threshold)

        # 高曲率方向（danger mask 对应的特征向量列）供 LeakyCurvatureLora 软投影使用
        danger_a = ~mask_a  # (d_in,) bool，True = 高曲率
        danger_b = ~mask_b  # (d_out,) bool
        U_in_bar = Ua[:, danger_a]    # (d_in, k_in)
        U_out_bar = Ub[:, danger_b]   # (d_out, k_out)

        layer_to_proj_cache[layer_name] = {
            "Ua": Ua.cpu(),
            "Ub": Ub.cpu(),
            "mask_a": mask_a.cpu(),
            "mask_b": mask_b.cpu(),
            "U_in_bar": U_in_bar.cpu(),
            "U_out_bar": U_out_bar.cpu(),
        }
        del A, B, Sa, Sb, Ua, Ub
        torch.cuda.empty_cache()

    return layer_to_proj_cache


def inject_leaky_curvature_lora(
    peft_model,
    layer_to_proj_cache: Dict[str, Dict],
    adapter_name: str = "default",
) -> None:
    """
    对 peft_model 中所有 LoRA Linear 层：
      1. 注册 LeakyCurvatureLora variant（含 leak_rate 可训练参数和 U_in_bar/U_out_bar buffer）
      2. 从 layer_to_proj_cache 写入真实的高曲率方向基
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
            continue

        # 注册 variant 并初始化 buffer + leak_rate
        module.lora_variant[adapter_name] = LeakyCurvatureLora()
        LeakyCurvatureLora.init(module, adapter_name=adapter_name)

        # 从 cache 中匹配该层并写入 U_in_bar/U_out_bar
        matched_cache = _match_layer_cache(name, layer_to_proj_cache)
        if matched_cache is not None:
            base_weight = (
                module.base_layer.weight
                if hasattr(module, "base_layer")
                else module.weight
            )
            dev, dtype = base_weight.device, base_weight.dtype

            U_in_bar = matched_cache["U_in_bar"].to(device=dev, dtype=dtype)
            U_out_bar = matched_cache["U_out_bar"].to(device=dev, dtype=dtype)

            # 覆盖 init 时注册的零 buffer
            setattr(module, f"U_in_bar_{adapter_name}", U_in_bar)
            setattr(module, f"U_out_bar_{adapter_name}", U_out_bar)
            count += 1
            print(
                f"[LeakyLoRA] {name}: "
                f"U_in_bar={tuple(U_in_bar.shape)}, "
                f"U_out_bar={tuple(U_out_bar.shape)}"
            )

    print(f"[LeakyLoRA] 已注入 {count} 个 LeakyCurvatureLora 层")


def _match_layer_cache(param_name: str, layer_to_proj_cache: Dict[str, Dict]) -> Optional[Dict]:
    for layer_name, cache in layer_to_proj_cache.items():
        clean = layer_name[:-len(".weight")] if layer_name.endswith(".weight") else layer_name
        if clean in param_name:
            return cache
    return None



def map_proj_cache_to_lora_params(
    peft_model,
    layer_to_proj_cache: Dict[str, Dict],
) -> Dict[torch.nn.Parameter, Dict]:
    """
    将 layer_name → proj_cache 转换为 param_tensor → proj_cache，
    适配 GlobalAwareProjectedLoRAOptimizer。
    """
    param_to_proj_cache = {}

    for name, param in peft_model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue

        matched_layer = None
        for layer_name in layer_to_proj_cache:
            clean = layer_name[:-len(".weight")] if layer_name.endswith(".weight") else layer_name
            if clean in name:
                matched_layer = layer_name
                break

        if matched_layer is None:
            continue

        cache = layer_to_proj_cache[matched_layer]
        if "lora_A" in name:
            param_to_proj_cache[param] = {
                "Ua": cache["Ua"],
                "mask_a": cache["mask_a"],
                "param_type": "lora_A",
            }
        elif "lora_B" in name:
            param_to_proj_cache[param] = {
                "Ub": cache["Ub"],
                "mask_b": cache["mask_b"],
                "param_type": "lora_B",
            }

    print(f"[LeakyLoRA] 建立参数→投影映射，共 {len(param_to_proj_cache)} 个LoRA参数")
    return param_to_proj_cache


def wrap_model_and_build_leaky_optimizers(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: CrispLoRAHyperParams,
    force_recompute: bool = False,
):
    """
    完整初始化流程：
      1. 计算 KFac 投影缓存
      2. 挂载 LoRA 适配器
      3. 注入 LeakyCurvatureLora variant（写入 U_in_bar/U_out_bar）
      4. 分离 lora_params / leak_params
      5. 构建 GlobalAwareProjectedLoRAOptimizer（lora_A/B）+ Adam（leak_rate）

    Returns:
        peft_model: 带 LeakyCurvatureLora 的 peft 模型
        optimizer_lora: GlobalAwareProjectedLoRAOptimizer
        optimizer_leak: Adam，仅优化 leak_rate
        layer_to_proj_cache: 各层投影缓存
    """
    # 1. KFac 投影缓存（在挂 LoRA 之前，基于原始模型权重统计）
    layer_to_proj_cache = build_leaky_projection_cache(
        model, tok, hparams, force_recompute
    )

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

    # 3. 注入 LeakyCurvatureLora variant，写入真实高曲率基
    inject_leaky_curvature_lora(peft_model, layer_to_proj_cache)

    # 4. 分离 lora_params（A/B 矩阵） 和 leak_params（leak_rate）
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
    param_to_proj_cache = map_proj_cache_to_lora_params(peft_model, layer_to_proj_cache)

    optimizer_lora = GlobalAwareProjectedLoRAOptimizer(
        lora_params,
        projection_cache_map=param_to_proj_cache,
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
        projection_mode=hparams.projection_mode,
        subspace_penalty=0.05,
    )
    optimizer_leak = torch.optim.Adam(leak_params, lr=hparams.lr * 2)

    print(
        f"[LeakyLoRA] 初始化完成：LoRA rank={hparams.lora_rank}，"
        f"投影模式={hparams.projection_mode}，"
        f"能量阈值={hparams.energy_threshold}"
    )

    return peft_model, optimizer_lora, optimizer_leak, layer_to_proj_cache


# ═══════════════════════════════════════════════════════════════════════════
# 主训练函数（适配 EasyEditor 调用约定）
# ═══════════════════════════════════════════════════════════════════════════

def apply_leaky_lora_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: CrispLoRAHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    keep_original_weight: bool = False,
    **kwargs,
) -> Tuple[AutoModelForCausalLM, Dict]:
    """
    Leaky曲率LoRA的主入口函数，适配 EasyEditor 调用约定。

    Args:
        requests: 编辑请求列表，每项含 {"prompt": str, "target_new": str}
        hparams:  CrispLoRAHyperParams

    Returns:
        (edited_model, weights_copy)
    """
    from copy import deepcopy

    tracker = kwargs.get("tracker", None)
    device = model.device

    if tok.padding_side != "right":
        tok.padding_side = "right"

    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"] and request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]



    peft_model, optimizer_lora, optimizer_leak, _ = wrap_model_and_build_leaky_optimizers(
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

        avg_loss = total_loss / max(1, len(texts) // hparams.batch_size)
        print(f"[LeakyLoRA] Step {step+1}/{hparams.num_steps}  loss={avg_loss:.4f}")
        if avg_loss < 1e-3:
            print("[LeakyLoRA] 损失收敛，提前结束训练")
            break

    peft_model = peft_model.merge_and_unload()
    return peft_model, weights_copy


def _compute_loss(
    model,
    tok: AutoTokenizer,
    texts: List[str],
    targets: List[str],
    device: torch.device,
    hparams: CrispLoRAHyperParams,
) -> torch.Tensor:
    inputs_targets = [t + tg for t, tg in zip(texts, targets)]
    encodings = tok(
        inputs_targets,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
    ).to(device)

    labels = encodings["input_ids"].clone()
    labels[labels == tok.pad_token_id] = -100
    for i, prompt in enumerate(texts):
        prompt_len = len(
            tok(prompt, add_special_tokens=True, truncation=True,
                max_length=hparams.max_length)["input_ids"]
        )
        labels[i, :prompt_len] = -100

    return model(**encodings, labels=labels).loss


def _chunks(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]
