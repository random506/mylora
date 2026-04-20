"""
方案A工具函数
=====================================================
负责：
1. 从KFac协方差矩阵构建LoRA参数专用的边缘化投影缓存
2. 包装模型为LoRA + ProjectedLoRAOptimizer
3. 提供训练主循环

与 easyeditor/models/crispedit/utils.py 的对齐说明：
  - layer_stats_kfac_one_pass 返回的 key 与传入的 layer_names 完全一致，
    因此 rewrite_module_tmp 应与 CrispEdit yaml 保持一致（带 .weight 后缀）。
  - A/B 交换逻辑参照 CrispEdit：用 model_name 关键字判断而非完整路径。
  - A/B 含义：layer_stats_kfac_one_pass 返回 (A=输入协方差, B=输出协方差, n)，
    对于 LLaMA/phi，A 对应 d_in 维度（右投影用 Ua），B 对应 d_out 维度（左投影用 Ub）；
    对于其他模型（Qwen/Mistral 等），A/B 顺序相反，需要交换。
"""

import os
import torch
from typing import Dict, List, Tuple, Optional
from peft import LoraConfig, AdaLoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

from .projected_lora_optimizer import ProjectedLoRAOptimizer
from ...models.rome.layer_stats import layer_stats_kfac_one_pass
from ..hparams import CrispLoRAHyperParams
load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")


# ═══════════════════════════════════════════════════════════════════════════
# 工具：判断模型类型
# ═══════════════════════════════════════════════════════════════════════════

def _is_llama_or_phi(model_name: str) -> bool:
    lower = model_name.lower()
    return "llama" in lower or "phi" in lower


# ═══════════════════════════════════════════════════════════════════════════
# 协方差统计 & 投影构建
# ═══════════════════════════════════════════════════════════════════════════

def get_rank_and_threshold_by_energy_ratio(eigenvalues: torch.Tensor, percent: float = 0.9):
    """
    参照 crispedit/utils.py 的同名函数：
    按累积能量比例确定秩和阈值（降序排列后累积）。
    返回：(rank, threshold)
    """
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
    从KFac特征值（Sa: d_in维, Sb: d_out维）构建边缘化一维掩码。

    与 crispedit/utils.py 的 calculate_projection_cache_with_kfac 对比：
      CrispEdit 原始方法：M = outer(Sa, Sb)，用全量 (d_in × d_out) 矩阵阈值化
      SchemeA 边缘化方法：分别对 Sa/Sb 各自阈值化，得到两个一维 mask
        mask_a[j] = True 若 Sa[j] < threshold_a （输入方向低能量，对 lora_A 安全）
        mask_b[i] = True 若 Sb[i] < threshold_b （输出方向低能量，对 lora_B 安全）
    阈值计算逻辑与 crispedit 的 get_rank_and_threshold_by_energy_ratio 保持一致。

    Args:
        Sa: 输入协方差特征值 (d_in,)，升序排列（eigh 输出）
        Sb: 输出协方差特征值 (d_out,)，升序排列（eigh 输出）
        energy_threshold: 累积能量阈值（0~1），低于此能量的方向视为安全

    Returns:
        mask_a: (d_in,) bool tensor，True=低能量安全方向（供 lora_A 右投影使用）
        mask_b: (d_out,) bool tensor，True=低能量安全方向（供 lora_B 左投影使用）
    """
    _, threshold_a = get_rank_and_threshold_by_energy_ratio(Sa, percent=energy_threshold)
    _, threshold_b = get_rank_and_threshold_by_energy_ratio(Sb, percent=energy_threshold)

    mask_a = Sa < threshold_a  # (d_in,) bool
    mask_b = Sb < threshold_b  # (d_out,) bool

    print(
        f"  mask_a: {mask_a.sum().item()}/{len(mask_a)} safe dirs, "
        f"threshold_a={threshold_a:.6f}"
    )
    print(
        f"  mask_b: {mask_b.sum().item()}/{len(mask_b)} safe dirs, "
        f"threshold_b={threshold_b:.6f}"
    )
    return mask_a, mask_b


def build_lora_projection_cache(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: CrispLoRAHyperParams,
    force_recompute: bool = False,
) -> Dict[str, Dict]:
    """
    为每一层计算 LoRA 参数的 KFac 边缘化投影缓存。

    参照 crispedit/utils.py 的 calculate_cov_cache_with_old_data：
      - 用 layer_stats_kfac_one_pass 一次性收集所有层的 A/B 协方差矩阵
      - layer_names 直接用 hparams.rewrite_module_tmp.format(layer)，
        保证与 stats_dict 返回的 key 完全一致（包括 .weight 后缀）
      - A/B 交换逻辑与 crispedit/utils.py 的 calculate_projection_cache_by_layer 一致

    Returns:
        layer_to_proj_cache: {
            layer_name (含.weight): {
                'Ua': Tensor[d_in, d_in],    # 输入协方差特征向量
                'Ub': Tensor[d_out, d_out],  # 输出协方差特征向量
                'mask_a': Tensor[d_in],      # bool，lora_A 右投影掩码
                'mask_b': Tensor[d_out],     # bool，lora_B 左投影掩码
            }
        }
    """
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

        # A/B 交换逻辑：参照 crispedit/utils.py 的 calculate_projection_cache_by_layer
        # LLaMA/phi：A=输入协方差(d_in×d_in), B=输出协方差(d_out×d_out)，不交换
        # 其他模型（Qwen/Mistral）：layer_stats 返回的 A/B 顺序相反，需要交换
        if not _is_llama_or_phi(hparams.model_name):
            A, B = B, A

        A = A.to(dtype=torch.float32)
        B = B.to(dtype=torch.float32)

        # 特征值分解（eigh 返回升序特征值）
        Sa, Ua = torch.linalg.eigh(A)  # 输入协方差：Sa(d_in,), Ua(d_in,d_in)
        Sb, Ub = torch.linalg.eigh(B)  # 输出协方差：Sb(d_out,), Ub(d_out,d_out)

        print(f"[SchemeA] 层 {layer_name} 边缘化掩码计算:")
        mask_a, mask_b = compute_marginal_masks(Sa, Sb, hparams.energy_threshold)

        layer_to_proj_cache[layer_name] = {
            "Ua": Ua.cpu(),
            "Ub": Ub.cpu(),
            "mask_a": mask_a.cpu(),
            "mask_b": mask_b.cpu(),
        }
        del A, B, Sa, Sb, Ua, Ub
        torch.cuda.empty_cache()

    return layer_to_proj_cache


def map_proj_cache_to_lora_params(
    peft_model,
    layer_to_proj_cache: Dict[str, Dict],
) -> Dict[torch.nn.Parameter, Dict]:
    """
    将 layer_name → proj_cache 的映射转换为 param_tensor → proj_cache 的映射，
    专门适配 peft 模型中 LoRA 参数的命名规则。

    peft 中 LoRA 参数命名格式（以 down_proj 为例）：
      base_model.model.model.layers.19.mlp.down_proj.lora_A.default.weight
      base_model.model.model.layers.19.mlp.down_proj.lora_B.default.weight

    layer_to_proj_cache 的 key 格式（与 rewrite_module_tmp 一致）：
      model.layers.19.mlp.down_proj.weight   ← 带 .weight 后缀

    匹配策略：去掉 cache key 的 .weight 后缀后做子串匹配。
    """
    param_to_proj_cache = {}

    for name, param in peft_model.named_parameters():
        print(f"[SchemeA] name is {name}")
        if not param.requires_grad:
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue

        matched_layer = None
        for layer_name in layer_to_proj_cache:
            print(f"[SchemeA] layer name is {layer_name}")
            # 去掉 .weight 后缀后做子串匹配
            # 例："model.layers.19.mlp.down_proj.weight" → "model.layers.19.mlp.down_proj"
            clean = layer_name[:-len(".weight")] if layer_name.endswith(".weight") else layer_name
            if clean in name:
                matched_layer = layer_name
                break

        if matched_layer is None:
            continue

        cache = layer_to_proj_cache[matched_layer]
        if "lora_A" in name:
            # lora_A: (r, d_in)，右投影：使用 Ua 和 mask_a
            param_to_proj_cache[param] = {
                "Ua": cache["Ua"],
                "mask_a": cache["mask_a"],
                "param_type": "lora_A",
            }
        elif "lora_B" in name:
            # lora_B: (d_out, r)，左投影：使用 Ub 和 mask_b
            param_to_proj_cache[param] = {
                "Ub": cache["Ub"],
                "mask_b": cache["mask_b"],
                "param_type": "lora_B",
            }

    print(f"[SchemeA] 建立参数→投影映射，共 {len(param_to_proj_cache)} 个LoRA参数")
    return param_to_proj_cache


# ═══════════════════════════════════════════════════════════════════════════
# 模型包装 & 优化器构建
# ═══════════════════════════════════════════════════════════════════════════

def wrap_model_and_build_projected_optimizer(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: CrispLoRAHyperParams,
    force_recompute: bool = False,
):
    """
    包装模型为 LoRA + 构建 ProjectedLoRAOptimizer。

    参照 crispedit/utils.py 的 wrap_model_with_lora_and_return_opt，
    额外增加 KFac 投影缓存构建和 ProjectedLoRAOptimizer 初始化。

    Returns:
        peft_model: 添加了 LoRA 适配器的模型
        optimizer:  ProjectedLoRAOptimizer，梯度自动投影到安全子空间
        layer_to_proj_cache: 各层的投影缓存
    """
    # 1. 计算 KFac 边缘化投影缓存（在挂 LoRA 之前，基于原始模型权重统计）
    layer_to_proj_cache = build_lora_projection_cache(
        model, tok, hparams, force_recompute
    )

    # 2. 参照 crispedit/utils.py：挂载 LoRA 适配器
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

    # 3. 建立 param → 投影缓存的映射
    param_to_proj_cache = map_proj_cache_to_lora_params(
        peft_model, layer_to_proj_cache
    )

    # 4. 构建 ProjectedLoRAOptimizer
    # 参照 crispedit/utils.py 的 build_optimizer_with_cov_caches：
    # 仅优化可训练参数（即 LoRA 的 A/B 矩阵），其余参数冻结
    optimizer = ProjectedLoRAOptimizer(
        params=[p for p in peft_model.parameters() if p.requires_grad],
        projection_cache_map=param_to_proj_cache,
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
        projection_mode=hparams.projection_mode,
    )

    print(
        f"[SchemeA] 初始化完成：LoRA rank={hparams.lora_rank}，"
        f"投影模式={hparams.projection_mode}，"
        f"能量阈值={hparams.energy_threshold}"
    )

    return peft_model, optimizer, layer_to_proj_cache


# ═══════════════════════════════════════════════════════════════════════════
# 主训练函数（适配 EasyEditor 调用约定）
# ═══════════════════════════════════════════════════════════════════════════

def apply_scheme_a_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: CrispLoRAHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,   # 保留以兼容 EasyEditor 调用约定，暂未使用
    keep_original_weight: bool = False,  # 保留以兼容 EasyEditor 调用约定，暂未使用
    **kwargs,                            # 保留以兼容 EasyEditor 调用约定，暂未使用
) -> Tuple[AutoModelForCausalLM, Dict]:
    """
    方案A的主入口函数，适配 EasyEditor 调用约定。

    Args:
        requests: 编辑请求列表，每项含 {"prompt": str, "target_new": str}
        hparams: SchemeAHyperParams

    Returns:
        (edited_model, weights_copy)
    """
    from copy import deepcopy
    tracker = kwargs.get("tracker", None)
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    peft_model, optimizer, _ = wrap_model_and_build_projected_optimizer(
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
            optimizer.zero_grad()
            loss = _compute_loss(peft_model, tok, txt_batch, tgt_batch, device, hparams)
            if loss.item() >= 1e-3:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(texts) // hparams.batch_size)
        print(f"[SchemeA] Step {step+1}/{hparams.num_steps}  loss={avg_loss:.4f}")
        if avg_loss < 1e-3:
            print("[SchemeA] 损失收敛，提前结束训练")
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
