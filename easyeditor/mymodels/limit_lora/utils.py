"""
Leaky曲率投影LoRA工具函数
=====================================================
结合 LeakyCurvatureLora（软投影前向）和标准 Adam 优化器，
实现带泄漏率的参数增量约束方案。训练完成后对 B@A 做一步硬投影再合并。

流程：
  1. 计算 KFac 投影缓存（含高曲率特征向量 U_in_bar/U_out_bar）
  2. 挂载 LoRA + 注入 LeakyCurvatureLora variant（设置 U_in_bar/U_out_bar）
  3. 分离 lora_params 和 leak_params，构建双优化器（Adam）
  4. 执行训练循环
  5. 训练结束后对 B@A 做硬投影，清除危险方向分量，再合并权重
"""

import os
import math
import torch
from typing import Dict, List, Tuple, Optional
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from copy import deepcopy

from .LeakyCurvatureLora import LeakyCurvatureLora
from ...models.rome.layer_stats import layer_stats_kfac_one_pass
from ..hparams import CrispLoRAHyperParams
from ...tools import *

load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")

def _is_llama_or_phi(model_name: str) -> bool:
    # 不同模型对于A、B矩阵有差异
    lower = model_name.lower()
    return "llama" in lower or "phi" in lower

def get_topk_indices_by_energy_ratio(eigenvalues: torch.Tensor, percent: float = 0.9):
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
        # 软投影部分
        mask_a, mask_b = compute_marginal_masks(Sa, Sb, hparams.energy_threshold)
        danger_a = ~mask_a  # (d_in,) bool，True = 高曲率
        danger_b = ~mask_b  # (d_out,) bool
        U_in_bar = Ua[:, danger_a]    # (d_in, k_in)
        U_out_bar = Ub[:, danger_b]   # (d_out, k_out)

        # 硬投影部分
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
            print("[error] 176")
            continue
        if not hasattr(module, "lora_variant"):
            module.lora_variant = {}  # 初始化容器
            print("初始化容器")

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


def hard_project_ba_and_merge_into_base(
    peft_model,
    layer_to_proj_cache: Dict[str, Dict],
    adapter_name: str = "default",
) -> None:
    """
    训练完成后对 B@A 做一步硬投影，彻底清除危险方向分量，再合并进 base_layer.weight。

    硬投影公式（依次在输出侧和输入侧清除高曲率分量）：
        ΔW       = B @ A * scaling
        ΔW_proj  = ΔW - U_out_bar @ (U_out_bar.T @ ΔW)        # 输出侧硬投影
        ΔW_proj  = ΔW_proj - (ΔW_proj @ U_in_bar) @ U_in_bar.T # 输入侧硬投影

    之后清零 lora_A/lora_B，使 PEFT 自己的 merge_and_unload() 计算 B@A=0，
    不产生额外增量覆盖正确结果。
    """
    count = 0
    for name, module in peft_model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            continue

        matched_cache = _match_layer_cache(name, layer_to_proj_cache)
        if matched_cache is None:
            continue

        base_layer = module.base_layer if hasattr(module, "base_layer") else module
        base_weight = base_layer.weight
        dev, dtype = base_weight.device, base_weight.dtype

        weight_A = module.lora_A[adapter_name].weight  # (r, d_in)
        weight_B = module.lora_B[adapter_name].weight  # (d_out, r)
        scaling  = module.scaling[adapter_name]

        Ua     = matched_cache["Ua"].to(device=dev, dtype=torch.float32)      # (d_in, d_in)
        Ub     = matched_cache["Ub"].to(device=dev, dtype=torch.float32)      # (d_out, d_out)
        mask_a = matched_cache["mask_a"].to(device=dev)                       # (d_in,) bool
        mask_b = matched_cache["mask_b"].to(device=dev)                       # (d_out,) bool
        M = torch.outer(mask_a.float(), mask_b.float())                       # (d_in, d_out)

        with torch.no_grad():
            BA = weight_B.to(dtype=torch.float32) @ weight_A.to(dtype=torch.float32)  # (d_out, d_in)
            # 硬投影：在特征基下按掩码保留安全方向
            BA = Ub @ ((Ub.T @ BA @ Ua) * M.T) @ Ua.T

            delta = (BA * scaling).to(dtype=dtype)
            base_weight.data += delta

            module.lora_A[adapter_name].weight.data.zero_()
            module.lora_B[adapter_name].weight.data.zero_()

        count += 1
        print(f"[LeakyLoRA] 硬投影合并层 {name}：delta norm={delta.norm().item():.6f}")

    print(f"[LeakyLoRA] hard_project_ba_and_merge_into_base 完成，共处理 {count} 层")


def hard_project_ba_inplace(
    peft_model,
    layer_to_proj_cache: Dict[str, Dict],
    adapter_name: str = "default",
) -> None:
    """
    训练完成后对 B@A 做一步硬投影，并将结果通过 SVD 回写进 lora_A / lora_B，
    不需要合并进 base_layer.weight，标准 PEFT forward 即可正确推理。

    流程：
        1. 计算 ΔW = B @ A * scaling
        2. 硬投影：去掉输出/输入侧危险方向分量
        3. SVD 分解：ΔW_proj = U @ S @ Vh，取前 r 个奇异值
        4. 将新的 A' = sqrt(S[:r]) * Vh[:r], B' = U[:, :r] * sqrt(S[:r])
           回写进 lora_A / lora_B，同时将 scaling 设为 1.0（已编码进权重）

    【为什么用 SVD 回写而不是直接清零 + 修改 base_weight】
        - 不合并场景下，base_weight 保持原始预训练权重不变
        - lora_A / lora_B 自身编码完整的投影增量
        - 标准 PEFT forward 无需任何 hook 即可正确推理
    """
    count = 0
    for name, module in peft_model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            continue

        matched_cache = _match_layer_cache(name, layer_to_proj_cache)
        if matched_cache is None:
            continue

        base_layer = module.base_layer if hasattr(module, "base_layer") else module
        dev, dtype = base_layer.weight.device, base_layer.weight.dtype

        weight_A = module.lora_A[adapter_name].weight  # (r, d_in)
        weight_B = module.lora_B[adapter_name].weight  # (d_out, r)
        scaling  = module.scaling[adapter_name]
        r = weight_A.shape[0]

        Ua     = matched_cache["Ua"].to(device=dev, dtype=torch.float32)      # (d_in, d_in)
        Ub     = matched_cache["Ub"].to(device=dev, dtype=torch.float32)      # (d_out, d_out)
        mask_a = matched_cache["mask_a"].to(device=dev)                       # (d_in,) bool
        mask_b = matched_cache["mask_b"].to(device=dev)                       # (d_out,) bool
        M = torch.outer(mask_a.float(), mask_b.float())                       # (d_in, d_out)

        with torch.no_grad():
            BA = weight_B.to(dtype=torch.float32) @ weight_A.to(dtype=torch.float32)  # (d_out, d_in)

            # 硬投影：在特征基下按掩码保留安全方向
            BA = Ub @ ((Ub.T @ BA @ Ua) * M.T) @ Ua.T

            # ΔW_proj 包含 scaling，先乘进去统一处理
            BA_scaled = BA * scaling  # (d_out, d_in)

            # SVD 分解，取前 r 个奇异值重建低秩表示
            U, S, Vh = torch.linalg.svd(BA_scaled, full_matrices=False)  # U:(d_out,k), S:(k,), Vh:(k,d_in), k=min(d_out,d_in)
            S_sqrt = torch.sqrt(S[:r].clamp(min=0.0))

            # 新 A' (r, d_in), 新 B' (d_out, r)，scaling 已编码进权重本身
            new_A = (S_sqrt.unsqueeze(1) * Vh[:r]).to(dtype=dtype)
            new_B = (U[:, :r] * S_sqrt.unsqueeze(0)).to(dtype=dtype)

            weight_A.data.copy_(new_A)
            weight_B.data.copy_(new_B)

            # scaling 已经乘进权重，将 lora_alpha 设为 lora_r 使 scaling=1.0
            module.lora_alpha[adapter_name] = r
            module.scaling[adapter_name]    = 1.0

        count += 1
        orig_norm = (weight_B.to(torch.float32) @ weight_A.to(torch.float32)).norm().item()
        print(f"[LeakyLoRA] 硬投影回写层 {name}：ΔW_proj norm={orig_norm:.6f}")

    print(f"[LeakyLoRA] hard_project_ba_inplace 完成，共处理 {count} 层")


def merge_leaky_lora_into_base(
    peft_model,
    adapter_name: str = "default",
) -> None:
    """
    在调用 PEFT 的 merge_and_unload() 之前，手动将带曲率投影的 delta weight
    预先合并进 base_layer.weight，然后将 lora_A / lora_B 权重清零。

    【为什么需要这一步】
    PEFT 的 merge() 内部调用的是自己的 get_delta_weight()：
        delta = B @ A * scaling
    它完全不知道 LeakyCurvatureLora 的存在，因此合并进去的增量
    没有任何曲率投影保护（即训练时的约束在合并后全部丢失）。

    解决方案：
        1. 用 LeakyCurvatureLora._compute_delta_weight() 计算正确的投影 delta
        2. 手动将 delta 加到 base_layer.weight
        3. 将 lora_A / lora_B 权重清零
    这样 PEFT 自己的 merge() 算出的增量 = B(0) @ A(0) * scaling = 0，
    base_layer.weight 最终包含的就是正确的带投影结果。
    """
    count = 0
    for name, module in peft_model.named_modules():
        # 只处理挂了 LeakyCurvatureLora variant 的层
        if not hasattr(module, "lora_variant"):
            continue
        if adapter_name not in module.lora_variant:
            continue
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            continue

        # 取 base_layer.weight
        base_layer = module.base_layer if hasattr(module, "base_layer") else module
        base_weight = base_layer.weight

        # 用带投影的公式计算正确的 delta
        with torch.no_grad():
            delta = LeakyCurvatureLora._compute_delta_weight(module, adapter_name)
            # delta 的形状已经是 [out_features, in_features]，与 base_weight 一致
            base_weight.data += delta.to(dtype=base_weight.dtype)

            # 清零 lora_A / lora_B，使 PEFT 自己的 merge 计算 B@A=0，不产生额外增量
            module.lora_A[adapter_name].weight.data.zero_()
            module.lora_B[adapter_name].weight.data.zero_()

        count += 1
        print(
            f"[LeakyLoRA] 手动合并层 {name}：delta norm={delta.norm().item():.6f}"
        )

    print(f"[LeakyLoRA] merge_leaky_lora_into_base 完成，共处理 {count} 层")



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
      5. 构建 Adam（lora_A/B）+ Adam（leak_rate）

    Returns:
        peft_model: 带 LeakyCurvatureLora 的 peft 模型
        optimizer_lora: Adam，优化 lora_A/B
        optimizer_leak: Adam，仅优化 leak_rate
        layer_to_proj_cache: 各层投影缓存
    """
    # 1. KFac 投影缓存（在挂 LoRA 之前，基于原始模型权重统计）
    print("1、计算KFAC矩阵")
    layer_to_proj_cache = build_leaky_projection_cache(
        model, tok, hparams, force_recompute
    )

    # 2. 挂载 LoRA
    print("2、挂载LoRA")
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
    # zwz:改进1，使用参数级投影
    #print("3、注入高曲率方向信息")
    #inject_leaky_curvature_lora(peft_model, layer_to_proj_cache)

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
    optimizer_lora = torch.optim.Adam(lora_params, lr=hparams.lr, weight_decay=hparams.weight_decay)
    optimizer_leak = torch.optim.Adam(leak_params, lr=hparams.lr * 2)

    print(
        f"[LeakyLoRA] 初始化完成：LoRA rank={hparams.lora_rank}，"
        f"投影模式={hparams.projection_mode}，"
        f"能量阈值={hparams.energy_threshold}"
    )

    return peft_model, optimizer_lora, optimizer_leak, layer_to_proj_cache


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
    print("进入执行函数 ......")
    tracker = kwargs.get("tracker", None)
    device = model.device

    if tok.padding_side != "right":
        tok.padding_side = "right"

    for i, request in enumerate(requests):
        if request["target_new"] and request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]


    
    peft_model, optimizer_lora, optimizer_leak, layer_to_proj_cache = wrap_model_and_build_leaky_optimizers(
        model, tok, hparams
    )

    texts = [r["prompt"] for r in requests ]
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
        tracker.log({"LOSS":avg_loss})
        print(f"[LeakyLoRA] Step {step+1}/{hparams.num_steps}  loss={avg_loss:.4f}")
        if avg_loss < 1e-3:
            print("[LeakyLoRA] 损失收敛，提前结束训练")
            break

    # 训练结束后对 B@A 做一步硬投影，彻底清除危险方向分量，再合并进 base_layer.weight。
    # 这样 PEFT 自己的 merge_and_unload() 计算 B@A*scaling=0，不会产生额外增量。
    hard_project_ba_and_merge_into_base(peft_model, layer_to_proj_cache, adapter_name="default")
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
