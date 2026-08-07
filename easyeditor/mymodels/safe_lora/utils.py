"""
SAFE-LoRA 核心实现
=====================================================
SAFE-LoRA: Spectrum-Adaptive Feature-Enriched LoRA

三层改进架构：
  层次一：连续谱自适应正则化 —— 特征值驱动的连续衰减，替代二元 mask
  层次二：对比安全方向发现 —— 预训练 vs 安全数据 K-FAC 对比，识别安全关键方向
  层次三：ΔW 级联合投影 + 渐进约束收紧 —— 统一投影 + 余弦退火

训练流程：
  1. 计算预训练 K-FAC → 特征值谱 → 连续敏感度
  2. 计算安全数据 K-FAC → 对比分析 → 安全方向分数
  3. 构建 danger 向量（高敏感度 + 低安全分数 → 强约束）
  4. 挂载标准 LoRA
  5. 训练循环：forward → backward → optimizer.step → ΔW 谱投影 → SVD 回写
     渐进收紧：leak 从 leak_max 退火到 leak_min
  6. merge_and_unload()
"""

import os
import math
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

from .SAFELoRA_hparams import SAFELoRAHyperParams
from ...models.rome.layer_stats import (
    layer_stats_kfac_one_pass,
    layer_stats_kfac_with_txt_tgt,
)
from ...tools import *

load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _is_llama_or_phi(model_name: str) -> bool:
    """判断模型类型，决定 A/B 矩阵是否交换（不同架构的 K-FAC A/B 含义不同）。"""
    lower = model_name.lower()
    return "llama" in lower or "phi" in lower


# ═══════════════════════════════════════════════════════════════════════════════
# 层次一：连续谱自适应正则化
# ═══════════════════════════════════════════════════════════════════════════════

def compute_spectrum_sensitivity(S: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    将特征值映射为连续敏感度分数 ∈ [0, 1]。

    核心思想：
      低特征值 = 低曲率 = 对预训练知识不敏感 → 敏感度 → 0（安全，全通）
      高特征值 = 高曲率 = 对预训练知识高度敏感 → 敏感度 → 1（危险，强约束）

    与现有方法的关键区别：
      现有方法用 energy_threshold 做硬截断 → 二元 safe/danger 分类
      本方法保留全谱信息，特征值大小直接反映"危险程度"

    Args:
        S: (d,) 升序特征值，已 clamp 到 ≥ 0
        temperature: 谱温度。1.0=线性映射，<1=更尖锐(接近二元)，>1=更平滑

    Returns:
        sensitivity: (d,) 连续敏感度 ∈ [0, 1]
    """
    S = torch.clamp(S, min=0.0)
    if S.max() - S.min() < 1e-8:
        return torch.zeros_like(S)

    S_norm = (S - S.min()) / (S.max() - S.min())

    # 频谱温度缩放：temperature < 1 时放大差异（更接近二元行为）
    if temperature != 1.0:
        S_norm = S_norm ** (1.0 / max(temperature, 0.01))

    return S_norm


# ═══════════════════════════════════════════════════════════════════════════════
# 层次二：对比安全方向发现
# ═══════════════════════════════════════════════════════════════════════════════

def compute_safety_scores(
    U_pretrain: torch.Tensor,       # (d, d) 预训练特征向量
    A_safety: torch.Tensor,         # (d, d) 安全数据 K-FAC 协方差
    smoothing: float = 0.01,        # 平滑因子
) -> torch.Tensor:
    """
    计算每个特征方向的安全相关分数。

    方法：
      将安全数据的协方差矩阵 A_safety 投影到预训练特征基 U_pretrain 下，
      取对角线元素 = 每个预训练特征方向在安全数据上的激活方差。
      高激活方差 → 该方向对安全生成重要 → 即使高曲率也应放宽约束。

    直觉：
      如果某个"高曲率方向"在安全数据上也高激活，说明改变这个方向
      对安全任务有帮助，不应该被完全约束。

    Args:
        U_pretrain: (d, d) 预训练特征向量矩阵（eigh 结果，列为特征向量）
        A_safety: (d, d) 安全数据协方差矩阵
        smoothing: 平滑因子，避免零样本时除零

    Returns:
        safety_scores: (d,) 每方向的安全相关分数 ∈ [0, 1]
    """
    device = A_safety.device
    dtype = A_safety.dtype

    U_pretrain = U_pretrain.to(device=device, dtype=dtype)

    # A_safety 在预训练特征基下的表示
    A_safety_in_pretrain_basis = U_pretrain.T @ A_safety @ U_pretrain  # (d, d)

    # 对角线 = 每个预训练特征方向在安全数据上的激活方差
    safety_activation = torch.diag(A_safety_in_pretrain_basis)
    safety_activation = torch.clamp(safety_activation, min=0.0)

    # 归一化到 [0, 1]，加平滑防止除零
    max_act = safety_activation.max()
    if max_act > 0:
        safety_scores = safety_activation / (max_act + smoothing)
    else:
        safety_scores = torch.zeros_like(safety_activation)

    return safety_scores


# ═══════════════════════════════════════════════════════════════════════════════
# 连续衰减计算
# ═══════════════════════════════════════════════════════════════════════════════

def compute_danger_vector(
    sensitivity: torch.Tensor,      # (d,) ∈ [0, 1] 曲率敏感度
    safety_score: torch.Tensor,     # (d,) ∈ [0, 1] 安全相关分数
) -> torch.Tensor:
    """
    计算每个方向的"危险度"。

    danger = sensitivity * (1 - safety_score)

    含义：
      - 高敏感度 + 低安全分数 → danger → 1（强约束，几乎完全屏蔽）
      - 高敏感度 + 高安全分数 → danger → 0（放宽约束，因为是安全所需）
      - 低敏感度 + 任意安全分数 → danger → 0（本来就不危险）

    Args:
        sensitivity: 曲率敏感度
        safety_score: 安全相关分数

    Returns:
        danger: (d,) ∈ [0, 1]，数值越高越需要约束
    """
    return sensitivity * (1.0 - safety_score)


def compute_attenuation_from_danger(
    danger: torch.Tensor,           # (d,) ∈ [0, 1]
    leak: float,                    # 当前泄漏率 ∈ [0, 1]
) -> torch.Tensor:
    """
    从 danger 向量和当前 leak 计算衰减因子。

    attenuation = 1.0 - (1.0 - leak) * danger
    ∈ [leak, 1.0]

    含义：
      danger=0 → attenuation=1.0（完全放通，无畏衰减）
      danger=1 → attenuation=leak（仅泄露 leak 比例的信号通过）

    Args:
        danger: 危险度向量
        leak: 当前泄漏率

    Returns:
        attenuation: (d,) 衰减因子向量
    """
    return 1.0 - (1.0 - leak) * danger


# ═══════════════════════════════════════════════════════════════════════════════
# 统一的投影缓存构建
# ═══════════════════════════════════════════════════════════════════════════════

def _build_pretrain_kfac_cache(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: SAFELoRAHyperParams,
    force_recompute: bool = False,
) -> Dict[str, Dict]:
    """
    步骤1：计算预训练数据的 K-FAC 协方差缓存。

    Returns:
        {layer_name: {"A": Tensor, "B": Tensor, "num_samples": int}}
    """
    print("[SAFE-LoRA] 步骤1：计算预训练 K-FAC 协方差...")
    layer_names = [hparams.rewrite_module_tmp.format(l) for l in hparams.layers]

    stats_dict = layer_stats_kfac_one_pass(
        model=model, tokenizer=tok,
        layer_names=layer_names, stats_dir=STATS_DIR,
        ds_name=hparams.mom2_dataset, to_collect=["mom2"],
        sample_size=hparams.mom2_n_samples,
        precision=hparams.mom2_dtype,
        force_recompute=force_recompute,
    )

    cov_cache = {}
    for layer_name in layer_names:
        A, B, num_samples = stats_dict.pop(layer_name)
        cov_cache[layer_name] = {
            "A": A.to("cpu", dtype=torch.float32),
            "B": B.to("cpu", dtype=torch.float32),
            "num_samples": num_samples,
        }
        del A, B
    return cov_cache


def _build_safety_kfac_cache(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: SAFELoRAHyperParams,
    requests: List[Dict],
    pretrain_cov_cache: Dict[str, Dict],
) -> Optional[Dict[str, Dict]]:
    """
    步骤2（可选）：用安全编辑数据计算 K-FAC，用于对比分析。

    使用 harmful_prompt → safe_response 对来捕获"安全偏移"的激活模式。
    layer_stats_kfac_with_txt_tgt 会自动用编辑请求数据计算各层激活协方差。

    Returns:
        {layer_name: {"A_safety": Tensor, "B_safety": Tensor}} 或 None（如果禁用）
    """
    if not hparams.use_contrastive_analysis:
        print("[SAFE-LoRA] 步骤2：跳过（use_contrastive_analysis=False）")
        return None

    print("[SAFE-LoRA] 步骤2：计算安全数据 K-FAC 协方差（对比分析）...")

    # 从编辑请求中提取安全数据
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]

    if len(texts) == 0:
        print("[SAFE-LoRA] 警告：无编辑请求，跳过安全 K-FAC")
        return None

    layer_names = [hparams.rewrite_module_tmp.format(l) for l in hparams.layers]

    try:
        safety_stats = layer_stats_kfac_with_txt_tgt(
            model, tok,
            layer_names=layer_names,
            txt=texts, tgt=targets,
            precision=hparams.mom2_dtype,
            sample_size=min(hparams.safety_kfac_samples, len(texts)),
            to_collect=["mom2"],
            add_pretrain_data=False,
            pretrain_sample_size=0,
        )

        safety_cache = {}
        for layer_name in layer_names:
            A_s, B_s, num_s = safety_stats.pop(layer_name)
            safety_cache[layer_name] = {
                "A_safety": A_s.to("cpu", dtype=torch.float32),
                "B_safety": B_s.to("cpu", dtype=torch.float32),
            }
            del A_s, B_s
        return safety_cache
    except Exception as e:
        print(f"[SAFE-LoRA] 警告：安全 K-FAC 计算失败 ({e})，回退到纯谱方法")
        return None


def build_safe_lora_projection_cache(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: SAFELoRAHyperParams,
    requests: List[Dict],
    force_recompute: bool = False,
) -> Dict[str, Dict]:
    """
    构建 SAFE-LoRA 投影缓存（步骤1-3的完整流程）。

    对每个目标层：
      1. 计算预训练 K-FAC → eigh → 特征值 + 特征向量
      2. 连续谱映射 → sensitivity 向量
      3. (可选) 计算安全 K-FAC → 对比分析 → safety_score 向量
      4. 计算 danger 向量（预缓存，训练中不再变化）
      5. 将 Ua, Ub, danger_a, danger_b 打包

    Returns:
        layer_to_proj_cache: {
            layer_name: {
                "Ua":       Tensor(d_in,  d_in),   # 输入特征向量（升序排列）
                "Ub":       Tensor(d_out, d_out),   # 输出特征向量
                "danger_a": Tensor(d_in,),           # 输入方向危险度 ∈ [0,1]
                "danger_b": Tensor(d_out,),          # 输出方向危险度 ∈ [0,1]
            }
        }
    """
    # 步骤1：预训练 K-FAC
    pretrain_cache = _build_pretrain_kfac_cache(model, tok, hparams, force_recompute)

    # 步骤2：安全数据 K-FAC
    safety_cache = _build_safety_kfac_cache(model, tok, hparams, requests, pretrain_cache)

    # 步骤3：逐层特征分解 + 连续谱分析 + 对比分析
    print("[SAFE-LoRA] 步骤3：特征分解 + 连续谱分析 + 对比分析...")
    layer_to_proj_cache = {}

    layer_names = [hparams.rewrite_module_tmp.format(l) for l in hparams.layers]
    for layer_name in layer_names:
        A = pretrain_cache[layer_name]["A"].to(dtype=torch.float32)
        B = pretrain_cache[layer_name]["B"].to(dtype=torch.float32)

        # 非 Llama/Phi 架构交换 A/B
        if not _is_llama_or_phi(hparams.model_name):
            A, B = B, A

        # 特征分解（eigh 返回升序特征值和对应特征向量）
        Sa, Ua = torch.linalg.eigh(A)  # Sa: (d_in,) 升序, Ua: (d_in, d_in)
        Sb, Ub = torch.linalg.eigh(B)  # Sb: (d_out,) 升序, Ub: (d_out, d_out)

        # ─── 层次一：连续谱敏感度 ──────────────────────────────────────────
        sensitivity_a = compute_spectrum_sensitivity(Sa, hparams.spectrum_temperature)
        sensitivity_b = compute_spectrum_sensitivity(Sb, hparams.spectrum_temperature)

        # ─── 层次二：对比安全方向发现 ──────────────────────────────────────
        if safety_cache is not None and layer_name in safety_cache:
            A_safety = safety_cache[layer_name]["A_safety"].to(dtype=torch.float32)
            B_safety = safety_cache[layer_name]["B_safety"].to(dtype=torch.float32)

            if not _is_llama_or_phi(hparams.model_name):
                A_safety, B_safety = B_safety, A_safety

            # 在预训练特征基下计算安全数据激活方差
            safety_score_a = compute_safety_scores(
                Ua, A_safety, smoothing=hparams.safety_score_smoothing
            )
            safety_score_b = compute_safety_scores(
                Ub, B_safety, smoothing=hparams.safety_score_smoothing
            )
            print(
                f"  [{layer_name}] 对比分析完成: "
                f"safety_score_a mean={safety_score_a.mean().item():.3f}, "
                f"safety_score_b mean={safety_score_b.mean().item():.3f}"
            )
        else:
            # 无安全数据 K-FAC：回退为纯谱方法
            safety_score_a = torch.zeros_like(sensitivity_a)
            safety_score_b = torch.zeros_like(sensitivity_b)
            print(
                f"  [{layer_name}] 纯谱模式（无对比分析）: "
                f"sensitivity_a mean={sensitivity_a.mean().item():.3f}, "
                f"sensitivity_b mean={sensitivity_b.mean().item():.3f}"
            )

        # ─── 计算 danger 向量（预缓存） ────────────────────────────────────
        danger_a = compute_danger_vector(sensitivity_a, safety_score_a)
        danger_b = compute_danger_vector(sensitivity_b, safety_score_b)

        # 统计信息
        n_high_danger_a = (danger_a > 0.5).sum().item()
        n_high_danger_b = (danger_b > 0.5).sum().item()
        print(
            f"  [{layer_name}] danger_a: {n_high_danger_a}/{len(danger_a)} "
            f"high-danger dirs (>0.5), "
            f"danger_b: {n_high_danger_b}/{len(danger_b)} high-danger dirs"
        )

        # 保存缓存
        layer_to_proj_cache[layer_name] = {
            "Ua": Ua.cpu(),           # (d_in,  d_in)
            "Ub": Ub.cpu(),           # (d_out, d_out)
            "danger_a": danger_a.cpu(),  # (d_in,)
            "danger_b": danger_b.cpu(),  # (d_out,)
        }

        del A, B, Sa, Sb, Ua, Ub, sensitivity_a, sensitivity_b
        del safety_score_a, safety_score_b, danger_a, danger_b
        if safety_cache is not None and layer_name in safety_cache:
            del A_safety, B_safety
        torch.cuda.empty_cache()

    # 清理
    del pretrain_cache
    if safety_cache is not None:
        del safety_cache
    torch.cuda.empty_cache()

    return layer_to_proj_cache


# ═══════════════════════════════════════════════════════════════════════════════
# 层次三：ΔW 级联合投影 + SVD 回写
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def project_delta_w_and_rewrite(
    module,
    proj_cache: Dict,
    leak: float,
    adapter_name: str = "default",
) -> float:
    """
    对单个 LoRA 层的 ΔW = B @ A 做连续谱衰减投影，并通过 SVD 回写 lora_A / lora_B。

    投影公式：
      ΔW = B @ A                          (d_out, d_in)  低秩权重增量
      C = Ub.T @ ΔW @ Ua                  (d_out, d_in)  转到预训练特征基
      M_atten = outer(atten_b, atten_a)   (d_out, d_in)  联合衰减矩阵
      C_proj = C * M_atten.T              (d_out, d_in)  逐方向衰减
      ΔW_proj = Ub @ C_proj @ Ua.T        (d_out, d_in)  转回权重空间
      SVD(ΔW_proj * scaling) → new_A (r, d_in), new_B (d_out, r)

    联合衰减矩阵的含义：
      M_atten[i,j] = 输出方向 i 和输入方向 j 的联合衰减因子
      只有输入和输出都是安全方向时，该分量才完全保留

    Args:
        module: PEFT LoRA Linear 层
        proj_cache: 该层的投影缓存 {"Ua", "Ub", "danger_a", "danger_b"}
        leak: 当前泄漏率 ∈ [leak_min, leak_max]
        adapter_name: LoRA adapter 名称

    Returns:
        projected_norm: 投影后 ΔW 的 Frobenius 范数（用于监控）
    """
    # 检查必要条件
    if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
        return 0.0

    weight_A = module.lora_A[adapter_name].weight  # (r, d_in)
    weight_B = module.lora_B[adapter_name].weight  # (d_out, r)
    scaling = module.scaling[adapter_name]
    r = weight_A.shape[0]

    # 获取投影缓存并迁移到正确的设备/数据类型
    base_layer = module.base_layer if hasattr(module, "base_layer") else module
    dev = base_layer.weight.device
    dtype = base_layer.weight.dtype

    Ua = proj_cache["Ua"].to(device=dev, dtype=torch.float32)
    Ub = proj_cache["Ub"].to(device=dev, dtype=torch.float32)
    danger_a = proj_cache["danger_a"].to(device=dev, dtype=torch.float32)
    danger_b = proj_cache["danger_b"].to(device=dev, dtype=torch.float32)

    # 从 danger 和当前 leak 计算衰减因子
    atten_a = compute_attenuation_from_danger(danger_a, leak)  # (d_in,)
    atten_b = compute_attenuation_from_danger(danger_b, leak)  # (d_out,)

    # ΔW = B @ A
    BA = weight_B.to(torch.float32) @ weight_A.to(torch.float32)  # (d_out, d_in)

    # 转到预训练特征基
    C = Ub.T @ BA @ Ua  # (d_out, d_in)，每一列对应输入方向，每一行对应输出方向

    # 联合衰减矩阵（外积 → 逐元素衰减）
    M_atten = torch.outer(atten_b, atten_a)  # (d_out, d_in)
    C_proj = C * M_atten

    # 转回权重空间
    BA_proj = Ub @ C_proj @ Ua.T  # (d_out, d_in)
    BA_scaled = BA_proj * scaling

    # SVD 分解，取前 r 个奇异值重建低秩表示
    U, S, Vh = torch.linalg.svd(BA_scaled, full_matrices=False)
    S_sqrt = torch.sqrt(S[:r].clamp(min=0.0))

    # 新 lora_A (r, d_in) 和 lora_B (d_out, r)
    new_A = (S_sqrt.unsqueeze(1) * Vh[:r]).to(dtype=dtype)
    new_B = (U[:, :r] * S_sqrt.unsqueeze(0)).to(dtype=dtype)

    weight_A.data.copy_(new_A)
    weight_B.data.copy_(new_B)

    # scaling 已编码进权重，将 lora_alpha 设为 r 使 scaling=1.0
    module.lora_alpha[adapter_name] = r
    module.scaling[adapter_name] = 1.0

    projected_norm = BA_scaled.norm().item()
    return projected_norm


def project_all_lora_layers(
    peft_model,
    layer_to_proj_cache: Dict[str, Dict],
    leak: float,
    adapter_name: str = "default",
) -> Dict[str, float]:
    """
    遍历 peft_model 所有 LoRA 层，执行 ΔW 投影 + SVD 回写。

    Args:
        peft_model: PEFT 包装的模型
        layer_to_proj_cache: 各层投影缓存 {layer_name: {Ua, Ub, danger_a, danger_b}}
        leak: 当前泄漏率
        adapter_name: adapter 名称

    Returns:
        layer_norms: {layer_name: projected_norm} 各层投影后的 ΔW 范数
    """
    layer_norms = {}
    for name, module in peft_model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            continue

        # 匹配投影缓存
        matched_cache = None
        for layer_name, cache in layer_to_proj_cache.items():
            clean = layer_name[:-len(".weight")] if layer_name.endswith(".weight") else layer_name
            if clean in name:
                matched_cache = cache
                break

        if matched_cache is None:
            continue

        proj_norm = project_delta_w_and_rewrite(module, matched_cache, leak, adapter_name)
        layer_norms[name] = proj_norm

    return layer_norms


# ═══════════════════════════════════════════════════════════════════════════════
# 渐进约束收紧
# ═══════════════════════════════════════════════════════════════════════════════

def compute_leak_upper_bound(
    step: int,
    total_steps: int,
    leak_max: float = 0.5,
    leak_min: float = 0.01,
) -> float:
    """
    余弦退火：从 leak_max 平滑过渡到 leak_min。

    训练初期（leak → leak_max）：允许较多探索，优化器有更大自由度找到安全任务的可行解
    训练末期（leak → leak_min）：收紧约束，强制 ΔW 收敛到安全子空间

    退火曲线：
      leak
      ↑
      leak_max ┤╲
               │ ╲___
               │     ╲___
      leak_min ┤         ╲___
               └────────────────→ step

    Args:
        step: 当前步数（0-indexed）
        total_steps: 总步数
        leak_max: 初始泄漏率上界
        leak_min: 最终泄漏率上界

    Returns:
        leak: 当前步的泄漏率上界
    """
    if total_steps <= 1:
        return leak_min
    progress = step / (total_steps - 1)
    progress = min(progress, 1.0)
    return leak_min + (leak_max - leak_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════════════
# 层匹配工具
# ═══════════════════════════════════════════════════════════════════════════════

def _match_layer_cache(param_name: str, layer_to_proj_cache: Dict) -> Optional[Dict]:
    """将参数名匹配到对应的投影缓存。"""
    for layer_name, cache in layer_to_proj_cache.items():
        clean = layer_name[:-len(".weight")] if layer_name.endswith(".weight") else layer_name
        if clean in param_name:
            return cache
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 主训练入口
# ═══════════════════════════════════════════════════════════════════════════════

def apply_safe_lora_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: SAFELoRAHyperParams,
    return_orig_weights: bool = False,
    keep_original_weight: bool = False,
    **kwargs,
) -> AutoModelForCausalLM:
    """
    SAFE-LoRA 主训练入口。

    完整流程：
      Phase 0 — 准备数据，规范化 tokenizer
      Phase 1 — 构建 SAFE-LoRA 投影缓存（预训练 K-FAC + 安全对比分析）
      Phase 2 — 挂载标准 LoRA
      Phase 3 — 训练循环：
                  forward → backward → optimizer.step
                  → ΔW 谱投影 → SVD 回写（层次三）
                  → 渐进收紧 leak（层次一+二通过 danger 预缓存参与）
      Phase 4 — merge_and_unload()

    Args:
        model: 原始预训练模型
        tok: tokenizer
        requests: 编辑请求列表 [{"prompt": ..., "target_new": ..., ...}, ...]
        hparams: SAFELoRA 超参数
        kwargs: 可选 {'tracker': ExperimentTracker}

    Returns:
        编辑后的模型（LoRA 已合并）
    """
    print("=" * 60)
    print("[SAFE-LoRA] 开始训练")
    print("=" * 60)
    tracker = kwargs.get("tracker", None)
    device = model.device

    # ── Phase 0: 准备 ────────────────────────────────────────────────────────
    if tok.padding_side != "right":
        tok.padding_side = "right"

    for i, request in enumerate(requests):
        if request["target_new"] and request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]

    # ── Phase 1: 构建 SAFE-LoRA 投影缓存 ─────────────────────────────────────
    print("[SAFE-LoRA] Phase 1: 构建投影缓存...")
    layer_to_proj_cache = build_safe_lora_projection_cache(
        model, tok, hparams, requests
    )

    # ── Phase 2: 挂载 LoRA ───────────────────────────────────────────────────
    print("[SAFE-LoRA] Phase 2: 挂载标准 LoRA...")
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

    # 标准 Adam 优化器（无需自定义投影优化器，投影在 step 后通过 ΔW 级别完成）
    optimizer = torch.optim.Adam(
        peft_model.parameters(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )

    # ── Phase 3: 训练循环 ────────────────────────────────────────────────────
    print("[SAFE-LoRA] Phase 3: 训练循环（ΔW 谱投影 + 渐进收紧）...")
    texts = [r["prompt"] for r in requests]
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

                # ─── 层次三：ΔW 级联合投影 ──────────────────────────────────
                if (step + 1) % hparams.project_every_n_steps == 0:
                    current_leak = compute_leak_upper_bound(
                        step, hparams.num_steps,
                        hparams.leak_max, hparams.leak_min,
                    )
                    layer_norms = project_all_lora_layers(
                        peft_model, layer_to_proj_cache,
                        leak=current_leak,
                    )
                else:
                    current_leak = compute_leak_upper_bound(
                        step, hparams.num_steps,
                        hparams.leak_max, hparams.leak_min,
                    )

            total_loss += loss.item()

        # 日志
        num_batches = max(1, math.ceil(len(texts) / hparams.batch_size))
        avg_loss = total_loss / num_batches
        current_leak = compute_leak_upper_bound(
            step, hparams.num_steps,
            hparams.leak_max, hparams.leak_min,
        )

        if tracker is not None:
            tracker.log({
                "SAFELoRA/loss": avg_loss,
                "SAFELoRA/leak": current_leak,
                "SAFELoRA/step": step + 1,
            })

        print(
            f"[SAFE-LoRA] Step {step+1}/{hparams.num_steps}  "
            f"loss={avg_loss:.4f}  leak={current_leak:.4f}"
        )

        if avg_loss < 1e-3:
            print("[SAFE-LoRA] 损失收敛，提前结束训练")
            break

    # ── Phase 4: 最终投影 + 合并 ──────────────────────────────────────────────
    print("[SAFE-LoRA] Phase 4: 最终 ΔW 投影 + merge_and_unload...")
    # 用 leak_min 做最终投影，确保 ΔW 完全在安全子空间内
    final_norms = project_all_lora_layers(
        peft_model, layer_to_proj_cache,
        leak=hparams.leak_min,
    )
    if final_norms:
        total_delta = sum(final_norms.values())
        print(f"[SAFE-LoRA] 最终投影完成，总 ΔW 范数 = {total_delta:.4f}")

    peft_model = peft_model.merge_and_unload()
    print("=" * 60)
    print("[SAFE-LoRA] 训练完成")
    print("=" * 60)
    return peft_model


# ═══════════════════════════════════════════════════════════════════════════════
# 训练辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_loss(
    model,
    tok: AutoTokenizer,
    texts: List[str],
    targets: List[str],
    device: torch.device,
    hparams: SAFELoRAHyperParams,
) -> torch.Tensor:
    """
    计算编辑损失：拼接 prompt + target_new，仅对 target 部分计算 loss。

    参照现有代码的通用模式：
      - prompt 部分的 label 设为 -100（忽略）
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
    labels[labels == tok.pad_token_id] = -100

    for i, prompt in enumerate(texts):
        prompt_len = len(
            tok(prompt, add_special_tokens=True, truncation=True,
                max_length=hparams.max_length)["input_ids"]
        )
        labels[i, :prompt_len] = -100

    return model(**encodings, labels=labels).loss


def _chunks(lst: List, n: int):
    """将列表按大小 n 切分。"""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]
