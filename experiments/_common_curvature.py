#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
共享曲率原语模块（RQ2 / RQ3 / RQ5 复用）。

本模块是 `experiments/run_experiment1_rho_vs_hc.py` 中可复用原语的 **冻结副本**
（逐字复制，不改动实验一），并新增三个过滤函数：
    - absolute_basis(cap_factor, damping)   : H_e = I 退化的广义基（RQ3 Baseline A）
    - soft_filter(g, basis_a, basis_b, lam) : h_soft(ρ)=1/(1+λρ)（RQ2/RQ3 共用）
    - hard_filter(g, basis_a, basis_b, tau) : h_hard(ρ)=1[ρ≤τ]（RQ2 hard 分支）

为什么不重构实验一去 import 本模块：
    实验一已在服务器跑通（Phase 1 完成）；本地因 `higher` 缺失 + 无模型/数据，
    无法本地验证重构不破坏实验一。复制是零回归风险：实验一逐字不动，
    本模块是冻结副本。两份 generalized_basis 不会漂移——它们是自检过的冻结代码，
    且本模块自带 self_check_kronecker / self_check_filter_identity，任何分歧
    会在服务器自检阶段暴露。`_test_edit_loss_equiv.py` 已是第二份逐字副本且正常工作。

下游脚本（RQ2/RQ3/RQ5）用法：
    sys.path.insert(0, repo_root)  # 在各自脚本头部
    from experiments._common_curvature import (
        load_model_and_hparams, build_factors, generalized_basis, absolute_basis,
        soft_filter, hard_filter, measure_pair, _compute_edit_gradient,
        build_edit_batches, edit_loss, build_capability_batches, capability_kl,
        _safe_corr, topk_precision, set_seed, COMPUTE_DEVICE, EPS_RHO, ...
    )

注意符号约定：
    _compute_edit_gradient 返回 **uphill** 梯度 dL/dW（同实验一）；
    下游取下降方向 g_descent = -g 后再 soft_filter / hard_filter，
    最后 θ' = θ_0 + α·U（U 已是下降方向，故用 +）。
"""

import os
import sys
import random
from typing import Dict, List, Optional, Tuple

# 以 `python experiments/<script>.py` 运行时，sys.path[0] 是 experiments/ 而非仓库根目录，
# 导致 easyeditor、utils 等根级模块不可导入；这里把仓库根目录加入搜索路径。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dotenv import load_dotenv

load_dotenv()

HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_DIR")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor.models.crispedit.utils import (
    update_model_and_tokenizer_with_appropriate_padding_token,
)
from easyeditor.models.curpedit import AdamHyperParams
from easyeditor.models.curpedit.utils import (
    _build_cov_cache_from_hparams,
    _is_llama_or_phi,
)
from easyeditor.models.rome.layer_stats import (
    load_stats_ds,
)
from easyeditor.models.rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    length_collation,
)
from easyeditor.util.runningstats import make_loader

import utils as root_utils  # prepare_requests_from_data_type

# --------------------------------------------------------------------------- #
# 全局常量
# --------------------------------------------------------------------------- #
SEED = 69
ALPHAS_REL_DEFAULT: Tuple[float, ...] = (1e-4, 5e-4, 1e-3, 5e-3)
EPS_RHO = 1e-12
DELTA_E_FLOOR = 1e-4
KL_PER_TOKEN_CLAMP = 100.0  # 单 token KL 上限（nats），防极端离群点
RESTORE_ATOL = 1e-7
SOURCE_NAMES = ("top_cap", "generalized", "random", "edit_gradient")

# 因子矩阵与线性代数（eigh / Cholesky / 矩阵乘）的计算设备。
# 优先 CUDA 加速；CUDA 不可用时回退 CPU（设 CUDA_VISIBLE_DEVICES="" 可强制 CPU）。
# 注意：cap_A/edit_A 可达 [14336,14336]≈3.3GB(fp32)，eigh 需额外同等量级工作区，
# 显存吃紧时可改回 CPU 或减小规模。
COMPUTE_DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.T)


def _check_square(matrix: torch.Tensor, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got {tuple(matrix.shape)}.")


def _chunks(arr, n):
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# --------------------------------------------------------------------------- #
# 1. 模型 / hparams 加载（复制 run_crispedit.py:228-236 + :30-35 模式）
# --------------------------------------------------------------------------- #
def load_model_and_hparams(model_key: str):
    """返回 (model, tok, hparams)。"""
    hparams = AdamHyperParams.from_hparams(f"./hparams/CurpEdit/{model_key}")

    MODEL_NAME = hparams.model_name
    print(f"[load] model_name = {MODEL_NAME}")
    if HF_CACHE_DIR and os.path.exists(HF_CACHE_DIR + MODEL_NAME):
        MODEL_NAME = HF_CACHE_DIR + MODEL_NAME
    print(f"[load] resolved path = {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", local_files_only=True
    )
    model, tokenizer = update_model_and_tokenizer_with_appropriate_padding_token(
        model, tokenizer, hparams
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer, hparams


# --------------------------------------------------------------------------- #
# 2. 因子构建（绕过 weight-keyed projection_cache 间接层，直接取单层 cov_cache）
# --------------------------------------------------------------------------- #
def build_factors(model, tok, hparams, layer_idx: int) -> Dict:
    """
    调 _build_cov_cache_from_hparams 计算全部目标层 K-FAC 协方差，再取出指定层。
    原始 cov_cache 键为 "A"/"B"/"task_A"/"task_B"（非 cap_A/edit_A）；
    这里按 Llama no-swap 约定映射：
        cap_A  = cache["A"]      ([in, in])
        cap_B  = cache["B"]      ([out, out])
        edit_A = cache["task_A"] ([in, in])
        edit_B = cache["task_B"] ([out, out])
    """
    layer_to_cov = _build_cov_cache_from_hparams(model, tok, hparams, force_recompute=False)

    layer_name = hparams.rewrite_module_tmp.format(layer_idx)
    if layer_name not in layer_to_cov:
        # 兜底：按层索引扫描（兼容命名差异）
        candidates = [k for k in layer_to_cov if f"layers.{layer_idx}." in k]
        if not candidates:
            raise KeyError(
                f"layer {layer_name} not in cov_cache; available={list(layer_to_cov.keys())}"
            )
        layer_name = candidates[0]
    cache = layer_to_cov[layer_name]

    if "task_A" not in cache or "task_B" not in cache:
        raise RuntimeError(
            f"cov_cache for {layer_name} missing task_A/task_B (edit factors); "
            "check task_mom2_dataset / task_mom2_n_samples in YAML."
        )

    # 取该层权重参数（就地编辑对象，保 device_map 分片）
    weight_param: Optional[torch.nn.Parameter] = None
    for n, p in model.named_parameters():
        if n == layer_name:
            weight_param = p
            break
    if weight_param is None:
        raise RuntimeError(f"cannot locate parameter '{layer_name}' in model")

    # Llama 走 no-swap 分支：A=[in,in] 与 weight[out,in] 的 in 维一致
    is_llama = _is_llama_or_phi(hparams.model_name)
    A_in, B_out = cache["A"], cache["B"]
    tA_in, tB_out = cache["task_A"], cache["task_B"]
    if not is_llama:
        # 非 llama/phi/qwen 走 swap 分支：A/B 互换角色
        A_in, B_out = B_out, A_in
        tA_in, tB_out = tB_out, tA_in

    out_dim, in_dim = weight_param.shape
    assert A_in.shape[0] == in_dim, f"cap_A in-dim {A_in.shape[0]} != weight in {in_dim}"
    assert B_out.shape[0] == out_dim, f"cap_B out-dim {B_out.shape[0]} != weight out {out_dim}"

    factors = {
        "layer_name": layer_name,
        "cap_A": A_in.to(COMPUTE_DEVICE, dtype=torch.float32).contiguous(),
        "cap_B": B_out.to(COMPUTE_DEVICE, dtype=torch.float32).contiguous(),
        "edit_A": tA_in.to(COMPUTE_DEVICE, dtype=torch.float32).contiguous(),
        "edit_B": tB_out.to(COMPUTE_DEVICE, dtype=torch.float32).contiguous(),
        "weight_param": weight_param,
        "theta0": weight_param.data.detach().clone(),
        "out_dim": out_dim,
        "in_dim": in_dim,
    }
    print(
        f"[factors] layer={layer_name} out/in=({out_dim},{in_dim}) "
        f"cap_A={tuple(factors['cap_A'].shape)} cap_B={tuple(factors['cap_B'].shape)} "
        f"is_llama_no_swap={is_llama}"
    )
    return factors


# --------------------------------------------------------------------------- #
# 3. 曲率数学
# --------------------------------------------------------------------------- #
def generalized_basis(
    edit_factor: torch.Tensor,
    cap_factor: torch.Tensor,
    damping: float,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    复制自 projected_adam.py:301-360 的 Cholesky 白化核心（不实例化 ProjectedAdam）。
    返回 (q, dual_q, cap_eigs)：
        Q^T edit_factor Q = I,   Q^T cap_factor Q = diag(cap_eigs)。
    cap_eigs 即广义特征值（相对曲率谱）。

    解的是阻尼广义特征问题  H_c q = ρ (H_e + eps·I) q，
    eps = damping * mean(|diag(H_e)|)（与 ProjectedAdam._generalized_basis 完全一致）。
    """
    _check_square(edit_factor, "edit_factor")
    _check_square(cap_factor, "cap_factor")
    edit_factor = _symmetrize(edit_factor)
    cap_factor = _symmetrize(cap_factor)

    n = edit_factor.shape[0]
    trace_scale = edit_factor.diagonal().abs().mean().clamp(min=1e-12)
    eps = damping * trace_scale
    eye = torch.eye(n, device=edit_factor.device, dtype=edit_factor.dtype)
    edit_factor_reg = edit_factor + eps * eye

    if verbose:
        eigvals = torch.linalg.eigvalsh(edit_factor)
        cond = eigvals.max() / eigvals.clamp(min=1e-20).min()
        eigvals_reg = torch.linalg.eigvalsh(edit_factor_reg)
        cond_reg = eigvals_reg.max() / eigvals_reg.clamp(min=1e-20).min()
        print(
            f"  [gen_basis] n={n} cond(before)={cond.item():.3e} "
            f"cond(after)={cond_reg.item():.3e} eps={eps.item():.3e}"
        )

    L = torch.linalg.cholesky(edit_factor_reg)
    tmp = torch.linalg.solve_triangular(L, cap_factor, upper=False)
    whitened_cap = torch.linalg.solve_triangular(L, tmp.T, upper=False).T
    cap_eigs, cap_vecs = torch.linalg.eigh(whitened_cap)
    q = torch.linalg.solve_triangular(L.transpose(-1, -2), cap_vecs, upper=True)
    dual_q = L @ cap_vecs
    cap_eigs = torch.clamp(cap_eigs, min=0.0)

    if not (torch.isfinite(q).all() and torch.isfinite(dual_q).all()
            and torch.isfinite(cap_eigs).all()):
        raise RuntimeError("generalized basis produced non-finite values, check factor conditioning")

    return q.contiguous(), dual_q.contiguous(), cap_eigs.contiguous()


def absolute_basis(
    cap_factor: torch.Tensor,
    damping: float,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    H_e = I 退化的广义基（RQ3 Baseline A：仅用 H_c 绝对曲率过滤）。

    广义特征问题 H_c q = ρ H_e q 在 H_e = I 时退化为标准特征问题 H_c q = ρ q，
    ρ 即 H_c 的特征值（绝对曲率谱）。这里直接以 edit_factor = I 调用
    generalized_basis，保证：
      - 与 generalized_basis 同签名、同下游过滤公式（soft_filter / hard_filter）；
      - 同 eps = damping * mean(|diag(I)|) = damping * 1 的阻尼规则（damping parity），
        避免把阻尼差异与 H_e 结构差异混淆。

    返回 (q, dual_q, cap_eigs)。注意阻尼下 q 与 dual_q 相差 sqrt(1+damping) 标量
    （damping→0 时 q=dual_q=eigh(cap).vectors），但 soft/hard filter 中该标量
    在 (dual_q_b^T g dual_q_a) 与 (q_b · q_a^T) 之间精确抵消，故过滤结果与
    damping→0 极限一致；cap_eigs = eigval(H_c)/(1+damping)（整体缩放被 RQ3 的
    per-method λ 尺度对齐吸收）。
    """
    _check_square(cap_factor, "cap_factor")
    n = cap_factor.shape[0]
    eye = torch.eye(n, device=cap_factor.device, dtype=cap_factor.dtype)
    if verbose:
        print(f"  [abs_basis] H_e=I reduction, n={n}, damping={damping}")
    return generalized_basis(eye, cap_factor, damping, verbose=verbose)


def kfac_quad_form(V: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> float:
    """
    Kronecker 二次型迹公式：c(V) = trace(B @ V @ A @ V.T)。
    V:[out,in], A:[in,in], B:[out,out]。返回 Python float。
    V 常驻 CPU 以省显存，A/B 在 COMPUTE_DEVICE；
    这里把 V 临时搬到 A 的设备上做乘法，取回 Python float 后即释放。
    """
    V = V.to(device=A.device, dtype=A.dtype)
    # 用两步乘法控制显存：M = V @ A  ([out,in])，tmp = B @ V ([out,in])
    VA = V @ A                       # [out, in]
    return float(torch.trace(B @ (VA @ V.T)).clamp(min=0.0).item())


def normalize_frobenius(V: torch.Tensor) -> torch.Tensor:
    # V 常驻 CPU：K 条方向累积可达数十 GB，放 GPU 会 OOM；
    # kfac_quad_form 会按需把 V 临时搬到因子所在设备做乘法。
    V = V.to("cpu", dtype=torch.float32)
    norm = V.norm().item()
    if norm < 1e-30:
        raise RuntimeError("direction V has near-zero Frobenius norm; cannot normalize")
    return V / norm


# --------------------------------------------------------------------------- #
# 3b. 谱过滤函数（新增：RQ2 / RQ3 共用）
# --------------------------------------------------------------------------- #
# basis = (q, dual_q, cap_eigs) 元组，由 generalized_basis / absolute_basis 返回。
# A 侧 basis 维度 [in,in]，B 侧 basis 维度 [out,out]；g 形状 [out,in]（与权重同形）。


def _joint_eigs(basis_a: Tuple, basis_b: Tuple) -> torch.Tensor:
    """ρ_ij = outer(clamp(eig_b, min=0), clamp(eig_a, min=0))，形状 [out, in]。"""
    _, _, eig_a = basis_a
    _, _, eig_b = basis_b
    return torch.outer(
        torch.clamp(eig_b.flatten(), min=0.0),
        torch.clamp(eig_a.flatten(), min=0.0),
    )


def soft_filter(
    g: torch.Tensor,
    basis_a: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    basis_b: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    lam: float,
) -> torch.Tensor:
    """
    h_soft(ρ) = 1/(1+λρ) 软谱过滤。逐字复刻 projected_adam.py:704-714：
        coeffs = dual_q_b^T @ g @ dual_q_a
        ρ_ij   = outer(clamp(eig_b,min=0), clamp(eig_a,min=0))
        denom  = 1 + λ · ρ_ij
        U      = q_b @ (coeffs / denom.clamp(min=1e-12)) @ q_a^T

    g 须为下降方向（g = -∇_W L_edit）；返回过滤后更新 U（与 g 同形 [out,in]，
    同设备/类型）。lam=0 时 U ≡ g（无保护），lam→∞ 时 U→0（全保护）。
    """
    q_a, dual_q_a, _ = basis_a
    q_b, dual_q_b, _ = basis_b
    coeffs = dual_q_b.T @ g @ dual_q_a                      # [out, in]
    rho_ij = _joint_eigs(basis_a, basis_b)                  # [out, in]
    denom = 1.0 + float(lam) * rho_ij
    filtered = q_b @ (coeffs / denom.clamp(min=1e-12)) @ q_a.T
    return filtered


def hard_filter(
    g: torch.Tensor,
    basis_a: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    basis_b: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tau: float,
) -> torch.Tensor:
    """
    h_hard(ρ) = 1[ρ ≤ τ] 硬低曲率投影：
        coeffs = dual_q_b^T @ g @ dual_q_a
        mask   = (ρ_ij <= τ).float()
        U      = q_b @ (coeffs * mask) @ q_a^T

    τ 在 clamp 后的 ρ_ij 分布上取（调用方按分位/绝对值给）：
      τ < 0        → 全抑制（U≈0，全保护，与 soft λ→∞ 角点对齐）
      τ ≥ max(ρ)   → 全保留（U≡g，与 soft λ=0 角点对齐）
    """
    q_a, dual_q_a, _ = basis_a
    q_b, dual_q_b, _ = basis_b
    coeffs = dual_q_b.T @ g @ dual_q_a                      # [out, in]
    rho_ij = _joint_eigs(basis_a, basis_b)                  # [out, in]
    mask = (rho_ij <= float(tau)).to(coeffs.dtype)
    filtered = q_b @ (coeffs * mask) @ q_a.T
    return filtered


# --------------------------------------------------------------------------- #
# 4. 编辑梯度 / 编辑 loss
# --------------------------------------------------------------------------- #
def _compute_edit_gradient(
    model, tok, edit_txt: List[str], edit_tgt: List[str], weight_param: torch.nn.Parameter
) -> torch.Tensor:
    """
    对 L_edit（target_new 的因果 LM CE，prompt mask -100）做一次反向，返回与 weight 同形状的梯度。
    参考 calculate_request_loss 的前向/mask 逻辑。

    返回 **uphill** 梯度 dL/dW（CPU fp32）。下游过滤/步进须取下降方向 g_descent = -g。
    """
    weight_param.requires_grad_(True)
    # 只让目标层可导，其它冻结
    model.eval()
    txt_edit, tgt_eval = edit_txt, edit_tgt
    batch_size = 32
    grad_acc = None
    n_tokens = 0

    model.zero_grad(set_to_none=True)
    for txt_chunk, tgt_chunk in zip(_chunks(txt_edit, batch_size), _chunks(tgt_eval, batch_size)):
        inputs_targets = [t + g for t, g in zip(txt_chunk, tgt_chunk)]
        encodings = tok(inputs_targets, return_tensors="pt", padding=True).to(model.device)
        labels = encodings["input_ids"].clone()
        for i, prompt in enumerate(txt_chunk):
            prompt_len = len(tok.encode(prompt, add_special_tokens=True))
            labels[i, :prompt_len] = -100
        labels[labels == tok.pad_token_id] = -100

        outputs = model(**encodings)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        valid = (shift_labels != -100).sum().item()
        if valid == 0:
            continue
        (loss / valid).backward()
        if weight_param.grad is not None:
            g = weight_param.grad.detach().clone()
            grad_acc = g if grad_acc is None else (grad_acc + g)
            n_tokens += valid
        weight_param.grad = None
        model.zero_grad(set_to_none=True)

    weight_param.requires_grad_(False)
    if grad_acc is None:
        raise RuntimeError("no valid edit tokens; cannot compute edit gradient")
    grad_acc = grad_acc / max(n_tokens, 1)
    return grad_acc.to("cpu", dtype=torch.float32).contiguous()


def build_edit_batches(
    tok, edit_txt: List[str], edit_tgt: List[str], batch_size: int = 32
) -> List[Dict]:
    """
    预分词编辑样本并按 batch_size 打包 padding，构造 labels（prompt + pad 位置置 -100）。
    全程复用，省掉 measure_pair 内 308× 的重复分词。
    数值与 calculate_request_loss 一致：sample_size == len(edit_txt) 时无随机抽样，
    prompt_len = len(tok.encode(prompt, add_special_tokens=True))，pad 位置由
    labels[labels == pad_id] = -100 统一 mask。
    返回 List[{input_ids, attention_mask, labels}]（CPU long）。
    """
    pad_id = tok.pad_token_id
    fill_id = pad_id if pad_id is not None else 0

    encoded: List[Tuple[List[int], int]] = []
    for t, g in zip(edit_txt, edit_tgt):
        ids = tok(t + g, return_tensors=None, add_special_tokens=True)["input_ids"]
        prompt_len = len(tok.encode(t, add_special_tokens=True))
        encoded.append((ids, prompt_len))

    batches: List[Dict] = []
    for chunk in _chunks(encoded, batch_size):
        max_len = max(len(ids) for ids, _ in chunk)
        input_ids = torch.full((len(chunk), max_len), fill_id, dtype=torch.long)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long)
        labels = torch.full((len(chunk), max_len), -100, dtype=torch.long)
        for i, (ids, prompt_len) in enumerate(chunk):
            L = len(ids)
            ids_t = torch.tensor(ids, dtype=torch.long)
            input_ids[i, :L] = ids_t
            attention_mask[i, :L] = 1
            labels[i, :L] = ids_t
        # 与 calculate_request_loss 一致：先 mask 所有 pad_id 位置（含 padding），再 mask prompt 前缀
        if pad_id is not None:
            labels[labels == pad_id] = -100
        for i, (_, prompt_len) in enumerate(chunk):
            labels[i, :prompt_len] = -100
        batches.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })
    return batches


def edit_loss(model, edit_batches: List[Dict]) -> float:
    """
    批量计算 L_edit（target_new token 的平均 CE），与 calculate_request_loss 数值一致。
    累加在 GPU 标量张量上，循环结束一次性 .item()。
    """
    model.eval()
    total_loss = torch.zeros((), device=model.device, dtype=torch.float32)
    total_tokens = torch.zeros((), device=model.device, dtype=torch.float32)
    with torch.no_grad():
        for batch in edit_batches:
            batch = dict_to_(batch, model.device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += loss
            total_tokens += (shift_labels != -100).sum()

    tt = total_tokens.item()
    if tt == 0:
        return 0.0
    return total_loss.item() / tt


# --------------------------------------------------------------------------- #
# 5. 能力 KL（参照 calculate_cache_loss 数据管线，CE→KL 改造）
# --------------------------------------------------------------------------- #
def build_capability_batches(
    tok, ds_name: str, sample_size: int, batch_tokens: int
) -> List[Dict]:
    """
    预分词 + 预采样，把能力评测数据一次性物化成固定 batch 列表（CPU 张量）。
    采样种子/数量固定（random_sample=SEED, sample_size=N_cap），故每次测量结果一致——
    物化一次后供全部 measure_pair 复用，省掉 308× 的数据加载/分词/成桶开销。
    返回 List[batch_dict]，每个 batch_dict 含 input_ids/position_ids/attention_mask（CPU）。
    """
    raw_ds = load_stats_ds(ds_name)
    maxlen = 512  # 与 calculate_cache_loss 一致
    ds = TokenizedDataset(raw_ds["val"], tok, maxlen=maxlen)
    loader = make_loader(
        ds,
        sample_size=sample_size,
        batch_size=1,
        collate_fn=length_collation(batch_tokens),
        pin_memory=False,
        random_sample=SEED,
        num_workers=0,
    )
    batches: List[Dict] = []
    for batch_group in loader:
        for batch in batch_group:
            batches.append(batch)
    return batches


def _forward_logits(model, batch) -> torch.Tensor:
    batch = dict_to_(batch, model.device)
    out = model(**batch, use_cache=False)
    return out.logits if hasattr(out, "logits") else out


def capability_kl(
    model,
    tok,
    batches: List[Dict],
    ref_cache: Optional[List[torch.Tensor]] = None,
) -> Tuple[float, Optional[List[torch.Tensor]]]:
    """
    计算 D_cap(θ',θ_0) = (1/N) Σ_t KL(p_θ0(·|x_t) ‖ p_θ'(·|x_t))，按 token 平均。
    batches：由 build_capability_batches 预构建的固定 batch 列表（CPU 张量，全程复用）。
    ref_cache：若提供，则跳过 θ_0 的前向（已缓存 ref logits）；否则现场前向 θ_0 并缓存。
    返回 (kl_value, ref_cache_out)。ref_cache_out 为 None 时不缓存。
    """
    pad_id = tok.pad_token_id
    model_dtype = next(model.parameters()).dtype

    total_kl = torch.zeros((), device=model.device, dtype=torch.float32)
    total_tokens = torch.zeros((), device=model.device, dtype=torch.float32)
    new_cache: Optional[List[torch.Tensor]] = [] if ref_cache is None else None
    cache_idx = 0

    model.eval()
    with torch.no_grad():
        for batch in batches:
            labels = batch["input_ids"].clone()
            labels[labels == 0] = -100
            if pad_id is not None:
                labels[labels == pad_id] = -100
            # 参考分布（θ_0）：现场前向或读缓存
            if ref_cache is None:
                ref_logits = _forward_logits(model, batch).float()
                if new_cache is not None:
                    # fp16 落盘省一半 CPU 内存（vocab 128256 × tokens 较大）
                    new_cache.append(ref_logits.to(torch.float16).cpu())
            else:
                ref_logits = ref_cache[cache_idx].to(
                    model.device, dtype=model_dtype
                ).float()
                cache_idx += 1

            # 当前分布（θ'）：要求调用方已把权重改到 θ' 再调用
            cur_logits = _forward_logits(model, batch).float()

            # shift：tokens < n predict n
            shift_ref = ref_logits[..., :-1, :].contiguous()
            shift_cur = cur_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            mask = (shift_labels != -100)
            log_ref = torch.log_softmax(shift_ref, dim=-1)
            log_cur = torch.log_softmax(shift_cur, dim=-1)
            p_ref = log_ref.exp()
            kl_tok = (p_ref * (log_ref - log_cur)).sum(dim=-1)  # [B, T-1]
            kl_tok = torch.clamp(kl_tok, max=KL_PER_TOKEN_CLAMP)

            total_kl += (kl_tok * mask.float()).sum()
            total_tokens += mask.sum()

            del ref_logits, cur_logits, shift_ref, shift_cur, log_ref, log_cur, p_ref

    total_kl_v = total_kl.item()
    total_tokens_v = total_tokens.item()
    if total_tokens_v == 0:
        return 0.0, (new_cache if ref_cache is None else None)
    return total_kl_v / total_tokens_v, (new_cache if ref_cache is None else None)


# --------------------------------------------------------------------------- #
# 6. 单方向测量
# --------------------------------------------------------------------------- #
def measure_pair(
    model,
    tok,
    factors: Dict,
    V_k: torch.Tensor,
    alpha: float,
    edit_batches: List[Dict],
    cap_batches: List[Dict],
    ref_cache: Optional[List[torch.Tensor]],
) -> Tuple[float, float, float]:
    """
    在 θ_0 + α·V_k 下测量 (ΔE, ΔC, R)。
    save-modify-restore：用 .copy_() 原地改单层权重，try/finally 必恢复。
    edit_batches：预构建的编辑 batch 列表（批量前向，替代 calculate_request_loss 的逐条 batch=1）。
    cap_batches：预构建的能力评测 batch 列表（全程复用）。

    **V_k 须为下降方向**（内部 θ' = θ_0 + α·V_k，故 V_k = -∇L 才是下降）。
    返回 (edit_loss_prime, delta_C, 0.0)；ΔE 由调用方用 baseline_edit_loss - edit_loss_prime 算。
    """
    weight_param = factors["weight_param"]
    theta0 = factors["theta0"]
    device = weight_param.device
    V_dev = V_k.to(device=device, dtype=weight_param.dtype)

    try:
        # θ_0 + α·V_k（就地修改，保 device_map 分片）
        weight_param.data.copy_(theta0 + alpha * V_dev)

        # ΔE 的 θ' 侧：L_edit(θ_0+αV_k)；baseline L_edit(θ_0) 由调用方预先算好
        edit_loss_prime = edit_loss(model, edit_batches)
        # ΔC：此时权重在 θ'，KL(p_θ0 ‖ p_θ')；ref_cache 提供 θ_0 的 logits 避免重算
        delta_C, _ = capability_kl(
            model, tok, cap_batches, ref_cache=ref_cache,
        )
    finally:
        weight_param.data.copy_(theta0)
        assert torch.allclose(weight_param.data, theta0, atol=RESTORE_ATOL), \
            "weight restore failed after measure_pair"

    return edit_loss_prime, delta_C, 0.0  # R 由调用方用 baseline 算


# --------------------------------------------------------------------------- #
# 7. 汇总 / 相关性
# --------------------------------------------------------------------------- #
def _safe_corr(x: List[float], y: List[float]) -> Dict:
    """Pearson + Spearman + p 值；样本不足返回 None。"""
    if len(x) < 3:
        return {"n": len(x), "pearson_r": None, "pearson_p": None,
                "spearman_r": None, "spearman_p": None}
    try:
        from scipy.stats import pearsonr, spearmanr
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        return {"n": len(x), "pearson_r": float(pr), "pearson_p": float(pp),
                "spearman_r": float(sr), "spearman_p": float(sp)}
    except Exception as e:
        return {"n": len(x), "error": str(e)}


def topk_precision(predictor: List[float], target: List[float], ks: List[int]) -> Dict:
    """
    top-k 安全方向精度：按 predictor 升序（低=安全）取前 k 个，
    看其落入 target 升序前 k 个的比例。多个 k。
    """
    n = len(predictor)
    p = np.argsort(np.asarray(predictor))  # 升序索引
    out = {}
    for k in ks:
        if k > n:
            continue
        pred_top = set(p[:k].tolist())
        tgt_top = set(np.argsort(np.asarray(target))[:k].tolist())
        out[f"top{k}"] = len(pred_top & tgt_top) / k
    return out


# --------------------------------------------------------------------------- #
# 7b. 多源单位方向构造（复制自实验一 build_directions，供 RQ3 预测器相关性比较复用）
# --------------------------------------------------------------------------- #
def build_directions(
    factors: Dict,
    K_per_source: int,
    edit_txt: List[str],
    edit_tgt: List[str],
    model,
    tok,
    hparams,
    eig_device=COMPUTE_DEVICE,
) -> List[Dict]:
    """
    返回 [{V, source, c, e, rho, meta}]，V 已归一化到 ||V||_F = 1。
    来源（与实验一一致，保证 RQ3 预测器比较与 RQ1 可对照）：
      (a) top_cap : cap_B 顶部特征向量 ⊗ cap_A 顶部特征向量（H_c 主方向）
      (b) generalized : 取 top-ρ / bottom-ρ / 随机广义特征方向
      (c) random : 高斯随机方向
      (d) edit_gradient : L_edit 一次反向得 g，归一化 ±g
    """
    cap_A, cap_B = factors["cap_A"], factors["cap_B"]
    edit_A, edit_B = factors["edit_A"], factors["edit_B"]
    out_dim, in_dim = factors["out_dim"], factors["in_dim"]
    damping = float(getattr(hparams, "newton_damping", 1e-5))

    directions: List[Dict] = []

    # ---- (a) top_cap：分别 eigh cap_A / cap_B，取顶部 K 个张量积 ----
    print(f"[dirs] (a) top_cap: eigh(cap_A)/eigh(cap_B) on {eig_device} ...")
    eig_a_vals, eig_a_vecs = torch.linalg.eigh(cap_A.to(eig_device))   # 升序
    eig_b_vals, eig_b_vecs = torch.linalg.eigh(cap_B.to(eig_device))
    top_a = eig_a_vecs[:, -K_per_source:]   # [in, K]
    top_b = eig_b_vecs[:, -K_per_source:]   # [out, K]
    cnt = 0
    for i in range(K_per_source):
        for j in range(K_per_source):
            if cnt >= K_per_source:
                break
            V = torch.outer(top_b[:, j], top_a[:, i])  # [out, in]
            V = normalize_frobenius(V)
            c = kfac_quad_form(V, cap_A, cap_B)
            e = kfac_quad_form(V, edit_A, edit_B)
            directions.append({"V": V, "source": "top_cap",
                               "c": c, "e": e, "rho": c / (e + EPS_RHO),
                               "meta": {"a_eig": float(eig_a_vals[-(i + 1)]),
                                        "b_eig": float(eig_b_vals[-(j + 1)])}})
            cnt += 1
        if cnt >= K_per_source:
            break

    # ---- (b) generalized：A 侧与 B 侧广义特征分解，ρ_ij = outer(eig_b, eig_a) ----
    print("[dirs] (b) generalized: Cholesky whitening (A-side & B-side) ...")
    q_a, _, eig_a_gen = generalized_basis(edit_A, cap_A, damping)  # [in, in]
    q_b, _, eig_b_gen = generalized_basis(edit_B, cap_B, damping)  # [out, out]
    rho_outer = torch.outer(eig_b_gen, eig_a_gen)  # [out, in] = ρ_ij
    flat = rho_outer.flatten()
    top_rho_idx = torch.argsort(flat, descending=True)[:K_per_source].tolist()
    for idx in top_rho_idx:
        j, i = divmod(idx, eig_a_gen.shape[0])
        V = torch.outer(q_b[:, j], q_a[:, i])
        V = normalize_frobenius(V)
        c = kfac_quad_form(V, cap_A, cap_B)
        e = kfac_quad_form(V, edit_A, edit_B)
        directions.append({"V": V, "source": "generalized",
                           "c": c, "e": e, "rho": c / (e + EPS_RHO),
                           "meta": {"kind": "top_rho", "rho_outer": float(flat[idx]),
                                    "a_eig": float(eig_a_gen[i]), "b_eig": float(eig_b_gen[j])}})
    bot_rho_idx = torch.argsort(flat, descending=False)[:K_per_source].tolist()
    for idx in bot_rho_idx:
        j, i = divmod(idx, eig_a_gen.shape[0])
        V = torch.outer(q_b[:, j], q_a[:, i])
        V = normalize_frobenius(V)
        c = kfac_quad_form(V, cap_A, cap_B)
        e = kfac_quad_form(V, edit_A, edit_B)
        directions.append({"V": V, "source": "generalized",
                           "c": c, "e": e, "rho": c / (e + EPS_RHO),
                           "meta": {"kind": "bottom_rho", "rho_outer": float(flat[idx]),
                                    "a_eig": float(eig_a_gen[i]), "b_eig": float(eig_b_gen[j])}})
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    n_a, n_b = eig_a_gen.shape[0], eig_b_gen.shape[0]
    for _ in range(K_per_source):
        i = int(torch.randint(0, n_a, (1,), generator=rng).item())
        j = int(torch.randint(0, n_b, (1,), generator=rng).item())
        V = torch.outer(q_b[:, j], q_a[:, i])
        V = normalize_frobenius(V)
        c = kfac_quad_form(V, cap_A, cap_B)
        e = kfac_quad_form(V, edit_A, edit_B)
        directions.append({"V": V, "source": "generalized",
                           "c": c, "e": e, "rho": c / (e + EPS_RHO),
                           "meta": {"kind": "random_gen", "a_eig": float(eig_a_gen[i]),
                                    "b_eig": float(eig_b_gen[j])}})

    # ---- (c) random：高斯随机方向 ----
    print("[dirs] (c) random: gaussian directions ...")
    for _ in range(K_per_source):
        V = torch.randn(out_dim, in_dim, generator=torch.Generator(device="cpu").manual_seed(SEED + len(directions)))
        V = normalize_frobenius(V)
        c = kfac_quad_form(V, cap_A, cap_B)
        e = kfac_quad_form(V, edit_A, edit_B)
        directions.append({"V": V, "source": "random",
                           "c": c, "e": e, "rho": c / (e + EPS_RHO), "meta": {}})

    # ---- (d) edit_gradient：L_edit 一次反向 ----
    print("[dirs] (d) edit_gradient: one backward on L_edit ...")
    grad = _compute_edit_gradient(model, tok, edit_txt, edit_tgt, factors["weight_param"])
    g_norm = grad.norm().item()
    if g_norm < 1e-30:
        raise RuntimeError("edit gradient has near-zero norm; cannot form edit_gradient direction")
    for sign, tag in ((-1.0, "descent"), (1.0, "ascent")):
        V = (sign * grad / g_norm)
        c = kfac_quad_form(V, cap_A, cap_B)
        e = kfac_quad_form(V, edit_A, edit_B)
        directions.append({"V": V, "source": "edit_gradient",
                           "c": c, "e": e, "rho": c / (e + EPS_RHO),
                           "meta": {"sign": tag, "grad_norm": g_norm}})

    print(f"[dirs] built {len(directions)} directions "
          f"(sources: " + ", ".join(f"{s}={sum(1 for d in directions if d['source']==s)}" for s in SOURCE_NAMES) + ")")
    return directions


def self_check_generalized_rho(directions: List[Dict]) -> None:
    """source=generalized 方向：|ρ − c/e| < 1e-3（edit-正交基下 e≈1 ⇒ |ρ−c|<1e-3）。"""
    worst = 0.0
    n_checked = 0
    for d in directions:
        if d["source"] != "generalized":
            continue
        e = d["e"]
        if e < 1e-20:
            continue
        diff = abs(d["rho"] - d["c"] / d["e"])
        worst = max(worst, diff)
        n_checked += 1
    print(f"[self_check] generalized |ρ−c/e|: n={n_checked} worst={worst:.3e}")
    if worst > 1e-3 and n_checked > 0:
        worst2 = max(
            (abs(d["rho"] - d["c"]) for d in directions if d["source"] == "generalized" and d["e"] < 1.0 + 1e-2 and d["e"] > 1.0 - 1e-2),
            default=0.0,
        )
        print(f"[self_check] (alt) |ρ−c| on e≈1 directions: worst={worst2:.3e}")
        if worst2 > 1e-3:
            raise RuntimeError(f"generalized ρ=c/e self-check FAILED (worst={worst:.3e})")
    print("[self_check] generalized ρ=c/e OK")


# --------------------------------------------------------------------------- #
# 8. 编辑请求加载
# --------------------------------------------------------------------------- #
def get_edit_requests(data_type: str, N_edit: int) -> Tuple[List[str], List[str]]:
    """取 wiki 编辑请求的前 N_edit 条，应用 target_new 前导空格归一化。"""
    requests = root_utils.prepare_requests_from_data_type(data_type)
    requests = requests[:N_edit]
    txt, tgt = [], []
    for r in requests:
        t = r["prompt"]
        g = r["target_new"]
        if g and g[0] != " ":
            g = " " + g
        txt.append(t)
        tgt.append(g)
    return txt, tgt


# --------------------------------------------------------------------------- #
# 9. 数学自检
# --------------------------------------------------------------------------- #
def self_check_kronecker(factors: Dict, block: int = 256) -> None:
    """
    小子块上验证 trace(B V A V^T) = vec(V)^T (B^T ⊗ A) vec(V)。
    用 cap_A/cap_B（同源 K-FAC Fisher 结构）显式构造 Kronecker 积对照。

    注意 vec 顺序：torch 的 V.flatten() 是行优先 = 列优先 vec(V^T)，
    故 Kronecker 因子必须是 (B^T ⊗ A)，而不是 (A ⊗ B)——
    两者仅经换位置换才相等，直接用 (A⊗B) 会引入 ~1e-2 量级的虚假误差。
    """
    A = factors["cap_A"][:block, :block].clone().to("cpu")  # 自检固定 CPU，保证确定性
    B = factors["cap_B"][:block, :block].clone().to("cpu")
    rng = torch.Generator(device="cpu").manual_seed(123)
    V = torch.randn(block, block, generator=rng)
    lhs = kfac_quad_form(V, A, B)
    # trace(B V A V^T) = vec(V^T)^T (B^T ⊗ A) vec(V^T)；
    # 行优先 flatten(V) = 列优先 vec(V^T)，故用 kron(B^T, A)。
    K = torch.kron(B.t().contiguous(), A)  # 仅小块可行：[block^2, block^2]
    rhs = float((V.flatten() @ K @ V.flatten()).item())
    rel = abs(lhs - rhs) / (abs(rhs) + 1e-30)
    print(f"[self_check] Kronecker identity: trace={lhs:.6e} kron={rhs:.6e} rel_err={rel:.3e}")
    if rel > 1e-4:
        raise RuntimeError(f"Kronecker identity self-check FAILED (rel_err={rel:.3e})")
    print("[self_check] Kronecker identity OK")


def self_check_filter_identity(
    basis_a: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    basis_b: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    g: torch.Tensor,
) -> None:
    """
    过滤函数角点一致性自检（RQ2/RQ3 共用）：
      1. 双正交性：q_a^T @ dual_q_a ≈ I, q_b^T @ dual_q_b ≈ I
      2. soft_filter(lam=0) ≈ g            （无保护角点）
      3. hard_filter(tau=+inf) ≈ g         （全保留角点，与 soft lam=0 对齐）
      4. hard_filter(tau=-1) ≈ 0           （全抑制角点，与 soft lam→∞ 对齐）

    g 为任意测试矩阵（形状 [out,in]，与 basis 维度匹配）。
    """
    q_a, dual_q_a, _ = basis_a
    q_b, dual_q_b, _ = basis_b

    # 1. 双正交性
    bi_a = q_a.T @ dual_q_a
    bi_b = q_b.T @ dual_q_b
    err_bi = max(
        float((bi_a - torch.eye(bi_a.shape[0], device=bi_a.device, dtype=bi_a.dtype)).abs().max()),
        float((bi_b - torch.eye(bi_b.shape[0], device=bi_b.device, dtype=bi_b.dtype)).abs().max()),
    )
    print(f"[self_check] biorthogonality max|Q^T DualQ - I| = {err_bi:.3e}")
    if err_bi > 1e-4:
        raise RuntimeError(f"biorthogonality self-check FAILED (err={err_bi:.3e})")

    g_dev = g.to(device=q_a.device, dtype=q_a.dtype)
    g_norm = g_dev.norm().item()

    # 2. soft lam=0 ≈ g
    u_soft0 = soft_filter(g_dev, basis_a, basis_b, 0.0)
    err_soft0 = float((u_soft0 - g_dev).abs().max())
    print(f"[self_check] soft(lam=0) max|U - g| = {err_soft0:.3e}  (||g||={g_norm:.3e})")
    if err_soft0 / (g_norm + 1e-30) > 1e-4:
        raise RuntimeError(f"soft lam=0 identity self-check FAILED (rel_err={err_soft0/(g_norm+1e-30):.3e})")

    # 3. hard tau=+inf ≈ g
    u_hard_inf = hard_filter(g_dev, basis_a, basis_b, float("inf"))
    err_hard_inf = float((u_hard_inf - g_dev).abs().max())
    print(f"[self_check] hard(tau=+inf) max|U - g| = {err_hard_inf:.3e}")
    if err_hard_inf / (g_norm + 1e-30) > 1e-4:
        raise RuntimeError(f"hard tau=+inf identity self-check FAILED (rel_err={err_hard_inf/(g_norm+1e-30):.3e})")

    # 4. hard tau=-1 ≈ 0
    u_hard_zero = hard_filter(g_dev, basis_a, basis_b, -1.0)
    err_hard_zero = float(u_hard_zero.abs().max())
    print(f"[self_check] hard(tau=-1) max|U| = {err_hard_zero:.3e}")
    if err_hard_zero / (g_norm + 1e-30) > 1e-6:
        raise RuntimeError(f"hard tau=-1 zero self-check FAILED (rel={err_hard_zero/(g_norm+1e-30):.3e})")

    print("[self_check] filter identity OK (biorthogonal + lam=0/tau=inf/tau=-1 corners)")


if __name__ == "__main__":
    # 模块自检入口：用小随机因子验证 absolute_basis / soft_filter / hard_filter 的角点一致性。
    # 不依赖 easyeditor/服务器模型，可在任意环境跑（含本地）。
    torch.manual_seed(0)
    n_out, n_in = 24, 16
    cap_A = _symmetrize(torch.randn(n_in, n_in))
    cap_B = _symmetrize(torch.randn(n_out, n_out))
    edit_A = _symmetrize(torch.randn(n_in, n_in))
    edit_B = _symmetrize(torch.randn(n_out, n_out))
    # 保证 PSD：A += alpha I
    cap_A += 2.0 * torch.eye(n_in)
    cap_B += 2.0 * torch.eye(n_out)
    edit_A += 2.0 * torch.eye(n_in)
    edit_B += 2.0 * torch.eye(n_out)

    g = torch.randn(n_out, n_in)

    print("=== absolute_basis (H_e=I) filter self-check ===")
    ba_abs = absolute_basis(cap_A, damping=1e-5, verbose=False)
    bb_abs = absolute_basis(cap_B, damping=1e-5, verbose=False)
    self_check_filter_identity(ba_abs, bb_abs, g)

    print("\n=== generalized_basis (real H_e) filter self-check ===")
    ba_rel = generalized_basis(edit_A, cap_A, damping=1e-5, verbose=False)
    bb_rel = generalized_basis(edit_B, cap_B, damping=1e-5, verbose=False)
    self_check_filter_identity(ba_rel, bb_rel, g)

    print("\n_common_curvature.py 模块自检全部通过。")
