#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ruff: noqa: E402
"""
实验一：核心假设验证 —— ρ(v) 是否优于 H_c？

验证相对曲率
    ρ(v) = v^T H_c v / v^T H_e v
是否比绝对曲率
    c(v) = v^T H_c v
更好地预测安全编辑的风险/收益比 R = ΔC / ΔE。

判定标准（docs/相对曲率安全模型编辑_实验规划.md §4.5）：
    若 Figure B (x=ρ(v), y=ΔC/ΔE) 的相关性显著强于
       Figure A (x=v^T H_c v, y=ΔC/ΔE)，
    则核心假设成立。

设计要点（详见 C:\\Users\\29308\\.claude\\plans\\sorted-hatching-wave.md）：
  - 载体：LLM CurpEdit 路径（Llama-3-8B-Instruct），单层 down_proj 编辑；
  - 独立可运行脚本，复用现有原语，不改动编辑器主流程；
  - ΔC = 能力 KL 散度 D_cap(θ',θ_0) = (1/N) Σ KL(p_θ0(·|x) ‖ p_θ(·|x))；
  - ΔE = L_edit(θ_0) − L_edit(θ_0 + α V_k)；
  - 所有候选方向按编辑梯度统一到下降半空间；
  - 多个固定小步长仅用于 edit-only line search，每个方向只保留 ΔE 最大的一条主记录；
  - ΔE 未超过阈值的方向标记无效，R=null，不使用人为分母 floor；
  - save-modify-restore：用 .copy_() 原地改单层权重，try/finally 必恢复（保 device_map 分片）。

K-FAC 曲率用迹公式（避免构造 58M×58M 完整 H）：
    c = trace(cap_B @ V @ cap_A @ V.T)
    e = trace(edit_B @ V @ edit_A @ V.T)
    ρ = c / (e + ε)

主流程：模型/缓存 -> 候选方向 -> 曲率预测 -> 固定步长编辑探针 ->
选择最大编辑收益的方向记录 -> 有效方向相关性与 bootstrap 汇总。

运行（从仓库根目录）：
    CUDA_VISIBLE_DEVICES=0 python experiments/run_experiment1_rho_vs_hc.py \\
        --model llama3-8b --layer 19 --K 60 --N_alpha 4 --N_cap 64 --N_edit 20 \\
        --out_dir logs/experiment1/

Multi-layer run (one independent result directory per layer):
    python experiments/run_experiment1_rho_vs_hc.py \\
        --model llama3-8b --layers 19 20 21 --out_dir logs/experiment1/

冒烟测试：
    python experiments/run_experiment1_rho_vs_hc.py --K 8 --N_alpha 2 --N_cap 16 --N_edit 4
"""

import argparse
import json
import os
import sys
import random
from typing import Dict, List, Optional, Tuple

# 以 `python experiments/run_experiment1_rho_vs_hc.py` 运行时，sys.path[0] 是
# experiments/ 而非仓库根目录，导致 easyeditor、utils 等根级模块不可导入；
# 这里把仓库根目录加入搜索路径。
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


# --------------------------------------------------------------------------- #
# 1. 模型 / hparams 加载（复制 run_crispedit.py:228-236 + :30-35 模式）
# --------------------------------------------------------------------------- #
def load_model_and_hparams(model_key: str):
    """Load the frozen base model and the CurpEdit configuration used by the probe."""
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
def build_factors(
    model,
    tok,
    hparams,
    layer_idx: int,
    layer_to_cov: Optional[Dict] = None,
) -> Dict:
    """
    # A multi-layer run builds one shared cov-cache, then selects one layer's
    # four factors here. Keeping this lookup separate lets every layer reuse
    # the expensive K-FAC statistics without coupling their measurements.
    调 _build_cov_cache_from_hparams 计算全部目标层 K-FAC 协方差，再取出指定层。
    原始 cov_cache 键为 "A"/"B"/"task_A"/"task_B"（非 cap_A/edit_A）；
    这里按 Llama no-swap 约定映射：
        cap_A  = cache["A"]      ([in, in])
        cap_B  = cache["B"]      ([out, out])
        edit_A = cache["task_A"] ([in, in])
        edit_B = cache["task_B"] ([out, out])
    """
    if layer_to_cov is None:
        layer_to_cov = _build_cov_cache_from_hparams(
            model, tok, hparams, force_recompute=False
        )

    # Resolve the requested layer from either a per-layer cache or a shared
    # cache containing every layer requested by --layers.
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
    # Whitening by H_e turns the generalized problem H_c q = rho H_e q into
    # an ordinary symmetric eigendecomposition. The returned q/dual_q pair
    # is also used by the generalized direction builder below.
    复制自 projected_adam.py:301-360 的 Cholesky 白化核心（不实例化 ProjectedAdam）。
    返回 (q, dual_q, cap_eigs)：
        Q^T edit_factor Q = I,   Q^T cap_factor Q = diag(cap_eigs)。
    cap_eigs 即广义特征值（相对曲率谱）。
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


def kfac_quad_form(V: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> float:
    """
    # The full Hessian would be (out*in)^2 entries. The Kronecker trace form
    # computes the same quadratic form without materializing that matrix.
    Kronecker 二次型迹公式：c(V) = trace(B @ V @ A @ V.T)。
    V:[out,in], A:[in,in], B:[out,out]。返回 Python float。
    V 常驻 CPU 以省显存（K 条方向累积可达数十 GB），A/B 在 COMPUTE_DEVICE；
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
# 4. 方向构造（K is the exact total budget across five direction families
#    plus one deterministic edit-gradient direction）
# --------------------------------------------------------------------------- #
def build_directions(
    factors: Dict,
    K_total: int,
    edit_txt: List[str],
    edit_tgt: List[str],
    model,
    tok,
    hparams,
    eig_device=COMPUTE_DEVICE,
) -> List[Dict]:
    """
    返回 [{V, source, c, e, rho, meta}]，V 已归一化到 ||V||_F = 1。
    来源：
      (a) top_cap : cap_B 顶部特征向量 ⊗ cap_A 顶部特征向量（H_c 主方向）
      (b) generalized : 取 top-ρ / bottom-ρ / 随机广义特征方向
      (c) random : 高斯随机方向
      (d) edit_gradient : L_edit 一次反向得到的下降方向
    """
    if K_total < 1:
        raise ValueError("K must be at least 1")

    # K is an exact unique-direction budget. One slot is reserved for the
    # deterministic edit-gradient direction; the remaining slots are split
    # across five candidate families so the final count is exactly K.
    family_names = (
        "top_cap", "gen_top", "gen_bottom", "gen_random", "random"
    )
    remaining = K_total - 1  # reserve one deterministic edit-gradient direction
    base_count, remainder = divmod(remaining, len(family_names))
    family_counts = {
        name: base_count + (index < remainder)
        for index, name in enumerate(family_names)
    }

    cap_A, cap_B = factors["cap_A"], factors["cap_B"]
    edit_A, edit_B = factors["edit_A"], factors["edit_B"]
    out_dim, in_dim = factors["out_dim"], factors["in_dim"]
    damping = float(getattr(hparams, "newton_damping", 1e-5))

    directions: List[Dict] = []
    print("[dirs] computing edit gradient for direction orientation ...")
    # The gradient is used only to orient candidates, not to select their
    # curvature score. This keeps the candidate comparison interpretable.
    edit_grad = _compute_edit_gradient(
        model, tok, edit_txt, edit_tgt, factors["weight_param"]
    )

    # ---- (a) top_cap：分别 eigh cap_A / cap_B，取顶部 K 个张量积 ----
    print(f"[dirs] (a) top_cap: eigh(cap_A)/eigh(cap_B) on {eig_device} ...")
    eig_a_vals, eig_a_vecs = torch.linalg.eigh(cap_A.to(eig_device))   # 升序
    eig_b_vals, eig_b_vecs = torch.linalg.eigh(cap_B.to(eig_device))
    # 顶部（最大特征值）= 末尾 K 个
    top_count = family_counts["top_cap"]
    top_a = eig_a_vecs[:, -top_count:] if top_count else eig_a_vecs[:, :0]
    top_b = eig_b_vecs[:, -top_count:] if top_count else eig_b_vecs[:, :0]
    cnt = 0
    for i in range(top_count):
        for j in range(top_count):
            if cnt >= top_count:
                break
            V = torch.outer(top_b[:, j], top_a[:, i])  # [out, in]
            directions.append(_finalize_direction(
                V,
                "top_cap",
                {"a_eig": float(eig_a_vals[-(i + 1)]),
                 "b_eig": float(eig_b_vals[-(j + 1)])},
                edit_grad,
                cap_A,
                cap_B,
                edit_A,
                edit_B,
            ))
            cnt += 1
        if cnt >= top_count:
            break

    # ---- (b) generalized：A 侧与 B 侧广义特征分解，ρ_ij = outer(eig_b, eig_a) ----
    print("[dirs] (b) generalized: Cholesky whitening (A-side & B-side) ...")
    q_a, _, eig_a_gen = generalized_basis(edit_A, cap_A, damping)  # [in, in]
    q_b, _, eig_b_gen = generalized_basis(edit_B, cap_B, damping)  # [out, out]
    # 1) top-ρ：取外积最大的若干 (i,j)
    rho_outer = torch.outer(eig_b_gen, eig_a_gen)  # [out, in] = ρ_ij
    flat = rho_outer.flatten()
    top_rho_idx = torch.argsort(flat, descending=True)[:family_counts["gen_top"]].tolist()
    for idx in top_rho_idx:
        j, i = divmod(idx, eig_a_gen.shape[0])  # i 沿 in, j 沿 out
        V = torch.outer(q_b[:, j], q_a[:, i])
        directions.append(_finalize_direction(
            V,
            "generalized",
            {"kind": "top_rho", "rho_outer": float(flat[idx]),
             "a_eig": float(eig_a_gen[i]), "b_eig": float(eig_b_gen[j])},
            edit_grad,
            cap_A,
            cap_B,
            edit_A,
            edit_B,
        ))
    # 2) bottom-ρ（最安全方向）
    bot_rho_idx = torch.argsort(flat, descending=False)[:family_counts["gen_bottom"]].tolist()
    for idx in bot_rho_idx:
        j, i = divmod(idx, eig_a_gen.shape[0])
        V = torch.outer(q_b[:, j], q_a[:, i])
        directions.append(_finalize_direction(
            V,
            "generalized",
            {"kind": "bottom_rho", "rho_outer": float(flat[idx]),
             "a_eig": float(eig_a_gen[i]), "b_eig": float(eig_b_gen[j])},
            edit_grad,
            cap_A,
            cap_B,
            edit_A,
            edit_B,
        ))
    # 3) 随机广义方向
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    n_a, n_b = eig_a_gen.shape[0], eig_b_gen.shape[0]
    for _ in range(family_counts["gen_random"]):
        i = int(torch.randint(0, n_a, (1,), generator=rng).item())
        j = int(torch.randint(0, n_b, (1,), generator=rng).item())
        V = torch.outer(q_b[:, j], q_a[:, i])
        directions.append(_finalize_direction(
            V,
            "generalized",
            {"kind": "random_gen", "a_eig": float(eig_a_gen[i]),
             "b_eig": float(eig_b_gen[j])},
            edit_grad,
            cap_A,
            cap_B,
            edit_A,
            edit_B,
        ))

    # ---- (c) random：高斯随机方向 ----
    print("[dirs] (c) random: gaussian directions ...")
    for _ in range(family_counts["random"]):
        V = torch.randn(out_dim, in_dim, generator=torch.Generator(device="cpu").manual_seed(SEED + len(directions)))
        directions.append(_finalize_direction(
            V, "random", {}, edit_grad, cap_A, cap_B, edit_A, edit_B
        ))

    # ---- (d) edit_gradient：L_edit 一次反向 ----
    print("[dirs] (d) edit_gradient: oriented descent direction ...")
    g_norm = edit_grad.norm().item()
    if g_norm < 1e-30:
        raise RuntimeError("edit gradient has near-zero norm; cannot form edit_gradient direction")
    directions.append(_finalize_direction(
        -edit_grad / g_norm,
        "edit_gradient",
        {"sign": "descent", "grad_norm": g_norm},
        edit_grad,
        cap_A,
        cap_B,
        edit_A,
        edit_B,
    ))

    print(f"[dirs] built {len(directions)} directions "
          f"(sources: " + ", ".join(f"{s}={sum(1 for d in directions if d['source']==s)}" for s in SOURCE_NAMES) + ")")
    if len(directions) != K_total:
        raise RuntimeError(
            f"direction budget mismatch: requested K={K_total}, built {len(directions)}"
        )
    return directions


def _compute_edit_gradient(
    model, tok, edit_txt: List[str], edit_tgt: List[str], weight_param: torch.nn.Parameter
) -> torch.Tensor:
    """
    对 L_edit（target_new 的因果 LM CE，prompt mask -100）做一次反向，返回与 weight 同形状的梯度。
    参考 calculate_request_loss 的前向/mask 逻辑。
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


def _finalize_direction(
    V: torch.Tensor,
    source: str,
    meta: Dict,
    edit_grad: torch.Tensor,
    cap_A: torch.Tensor,
    cap_B: torch.Tensor,
    edit_A: torch.Tensor,
    edit_B: torch.Tensor,
) -> Dict:
    """Normalize, orient, and score one candidate direction.

    Curvature scores are unchanged by V -> -V, but the edit-loss slope is not.
    We therefore flip every candidate into the half-space expected to improve
    the edit objective before measuring finite-step behavior.
    """
    V = normalize_frobenius(V)
    grad = edit_grad.to(device=V.device, dtype=V.dtype)
    g_v = float((grad * V).sum().item())
    flipped = g_v > 0.0
    if flipped:
        V = -V
        g_v = -g_v

    c = kfac_quad_form(V, cap_A, cap_B)
    e = kfac_quad_form(V, edit_A, edit_B)
    direction_meta = dict(meta)
    direction_meta["flipped_to_descent"] = flipped
    return {
        "V": V,
        "source": source,
        "c": c,
        "e": e,
        "rho": c / (e + EPS_RHO),
        "edit_slope": -g_v,
        "meta": direction_meta,
    }


def build_edit_batches(
    tok, edit_txt: List[str], edit_tgt: List[str], batch_size: int = 32
) -> List[Dict]:
    """
    # Tokenization and padding are deterministic, so materialize them once
    # and reuse the CPU batches for every direction/step trial.
    预分词编辑样本并按 batch_size 打包 padding，构造 labels（prompt + pad 位置置 -100）。
    全程复用，省掉 measure_pair 内 K×N_alpha 次的重复分词。
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
    # Only target tokens contribute: prompt tokens are context, not the edit
    # objective being improved.
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
    # The same capability examples must be reused for every perturbation;
    # otherwise changes in the sampled data would be mixed into Delta C.
    预分词 + 预采样，把能力评测数据一次性物化成固定 batch 列表（CPU 张量）。
    采样种子/数量固定（random_sample=SEED, sample_size=N_cap），故每次测量结果一致——
    物化一次后供全部 measure_pair 复用，省掉 K×N_alpha 次的数据加载/分词/成桶开销。
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
    # ref_cache must represent theta_0. measure_pair creates it before changing
    # the weight when caching is disabled, then this function evaluates theta'.
    计算 D_cap(θ',θ_0) = (1/N) Σ_t KL(p_θ0(·|x_t) ‖ p_θ'(·|x_t))，按 token 平均。
    batches：由 build_capability_batches 预构建的固定 batch 列表（CPU 张量，全程复用）。
    ref_cache：若提供，则跳过 θ_0 的前向（已缓存 ref logits）；否则现场前向 θ_0 并缓存。
    返回 (kl_value, ref_cache_out)。ref_cache_out 为 None 时不缓存。

    优化点（相比原版）：
      - 不再每次重建 loader / 重载数据集 / 重新分词；
      - 累加在 GPU 标量张量上，循环结束一次性 .item()，去掉每 batch 同步；
      - 删除每 batch 的 torch.cuda.empty_cache()（~20k 次同步的反模式）。
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

            mask = mask.to(kl_tok.device)
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
    # This is a transactional probe: modify exactly one layer, measure both
    # effects, and restore theta_0 in finally even if a forward pass fails.
    在 θ_0 + α·V_k 下测量 (ΔE, ΔC, R)。
    save-modify-restore：用 .copy_() 原地改单层权重，try/finally 必恢复。
    edit_batches：预构建的编辑 batch 列表（批量前向，替代 calculate_request_loss 的逐条 batch=1）。
    cap_batches：预构建的能力评测 batch 列表（全程复用）。
    """
    weight_param = factors["weight_param"]
    theta0 = factors["theta0"]
    device = weight_param.device
    V_dev = V_k.to(device=device, dtype=weight_param.dtype)
    effective_ref_cache = ref_cache
    if effective_ref_cache is None:
        # Build the reference while the model is still exactly at theta_0.
        # Doing this after the perturbation would compare theta' with itself.
        _, effective_ref_cache = capability_kl(
            model, tok, cap_batches, ref_cache=None,
        )

    try:
        # θ_0 + α·V_k（就地修改，保 device_map 分片）
        weight_param.data.copy_(theta0 + alpha * V_dev)

        # ΔE 的 θ' 侧：L_edit(θ_0+αV_k)；baseline L_edit(θ_0) 由调用方预先算好
        edit_loss_prime = edit_loss(model, edit_batches)
        # ΔC：此时权重在 θ'，KL(p_θ0 ‖ p_θ')；ref_cache 提供 θ_0 的 logits 避免重算
        delta_C, _ = capability_kl(
            model, tok, cap_batches, ref_cache=effective_ref_cache,
        )
    finally:
        weight_param.data.copy_(theta0)
        assert torch.allclose(weight_param.data, theta0, atol=RESTORE_ATOL), \
            "weight restore failed after measure_pair"

    return edit_loss_prime, delta_C, 0.0  # R 由调用方用 baseline 算


# --------------------------------------------------------------------------- #
# 7. 汇总 / 相关性 / 绘图
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


def _bootstrap_spearman_difference(
    records: List[Dict], n_bootstrap: int = 2000
) -> Dict:
    """Stratified direction-level CI for Spearman(rho,R)-Spearman(c,R).

    Resampling within source keeps the source mixture fixed and avoids treating
    the four line-search trials of one direction as independent observations.
    """
    if len(records) < 3:
        return {"n": len(records), "n_bootstrap": 0, "ci95": None}

    try:
        from scipy.stats import spearmanr
    except Exception as exc:
        return {"n": len(records), "n_bootstrap": 0, "error": str(exc)}

    source_indices = {}
    for index, record in enumerate(records):
        source_indices.setdefault(record["source"], []).append(index)

    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(n_bootstrap):
        sampled = []
        for indices in source_indices.values():
            sampled.extend(rng.choice(indices, size=len(indices), replace=True))
        c = [records[index]["c"] for index in sampled]
        rho = [records[index]["rho"] for index in sampled]
        target = [records[index]["R"] for index in sampled]
        # Tuple unpacking works with both SciPy 1.9's SpearmanrResult
        # (correlation, pvalue) and newer SignificanceResult versions
        # (statistic, pvalue).
        c_corr, _ = spearmanr(c, target)
        rho_corr, _ = spearmanr(rho, target)
        c_corr = float(c_corr)
        rho_corr = float(rho_corr)
        difference = rho_corr - c_corr
        if np.isfinite(difference):
            differences.append(difference)

    if not differences:
        return {"n": len(records), "n_bootstrap": 0, "ci95": None}
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "n": len(records),
        "n_bootstrap": len(differences),
        "mean": float(np.mean(differences)),
        "ci95": [float(low), float(high)],
    }


def correlate_and_plot(records: List[Dict], out_dir: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    # Persist the expensive direction measurements before optional statistical
    # analysis or plotting, so a dependency/version error cannot discard them.
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Only the selected primary record per direction enters the endpoint. The
    # line-search trials remain attached to the record for diagnostics, while
    # invalid directions cannot manufacture a positive denominator for R.
    valid_records = [
        r for r in records
        if r.get("valid", False)
        and r.get("R") is not None
        and np.isfinite(r["R"])
    ]
    c_all = [r["c"] for r in valid_records]
    rho_all = [r["rho"] for r in valid_records]
    R_all = [r["R"] for r in valid_records]

    corr_summary = {
        "n_total_directions": len(records),
        "n_valid_directions": len(valid_records),
        "n_invalid_directions": len(records) - len(valid_records),
        "overall": {
            "c_vs_R": _safe_corr(c_all, R_all),
            "rho_vs_R": _safe_corr(rho_all, R_all),
        },
        "by_source": {},
        "by_selected_alpha": {},
    }

    for source in sorted({r["source"] for r in valid_records}):
        sub = [r for r in valid_records if r["source"] == source]
        target = [r["R"] for r in sub]
        corr_summary["by_source"][source] = {
            "c_vs_R": _safe_corr([r["c"] for r in sub], target),
            "rho_vs_R": _safe_corr([r["rho"] for r in sub], target),
        }

    bootstrap = _bootstrap_spearman_difference(valid_records)
    corr_summary["spearman_difference_bootstrap"] = bootstrap

    # Selected-alpha groups are diagnostics only; the primary endpoint uses
    # exactly one independently selected record per direction.
    alphas = sorted(set(r["alpha"] for r in valid_records))
    for a in alphas:
        sub = [r for r in valid_records if abs(r["alpha"] - a) < 1e-15]
        c_a = [r["c"] for r in sub]
        rho_a = [r["rho"] for r in sub]
        R_a = [r["R"] for r in sub]
        corr_summary["by_selected_alpha"][f"{a:.1e}"] = {
            "c_vs_R": _safe_corr(c_a, R_a),
            "rho_vs_R": _safe_corr(rho_a, R_a),
            "topk_c": topk_precision(c_a, R_a, [5, 10, 20]),
            "topk_rho": topk_precision(rho_a, R_a, [5, 10, 20]),
        }

    # 判定
    ov_c = corr_summary["overall"]["c_vs_R"]
    ov_rho = corr_summary["overall"]["rho_vs_R"]
    if ov_c.get("spearman_r") is not None and ov_rho.get("spearman_r") is not None:
        ci95 = bootstrap.get("ci95")
        better = (
            ov_rho["spearman_r"] > 0.0
            and ov_rho["spearman_r"] > ov_c["spearman_r"]
            and ci95 is not None
            and ci95[0] > 0.0
        )
        corr_summary["verdict"] = (
            "rho_stronger_than_c" if better else "rho_advantage_not_established"
        )
        corr_summary["verdict_detail"] = (
            f"Spearman(rho,R)={ov_rho['spearman_r']:.4f} vs "
            f"Spearman(c,R)={ov_c['spearman_r']:.4f}; "
            f"stratified bootstrap CI95(delta)={ci95}"
        )
    else:
        corr_summary["verdict"] = "insufficient_samples"

    # 写 JSON
    with open(os.path.join(out_dir, "correlations.json"), "w", encoding="utf-8") as f:
        json.dump(corr_summary, f, indent=2, ensure_ascii=False)

    if not valid_records:
        print("[plot] no valid directions; skipping figures")
        return corr_summary

    # 绘图（可选，matplotlib 缺失则跳过）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for fname, xv, title in (
            ("fig_a_c_vs_R.png", c_all, "Figure A: c(v)=v^T H_c v  vs  R=ΔC/ΔE"),
            ("fig_b_rho_vs_R.png", rho_all, "Figure B: ρ(v)=v^T H_c v / v^T H_e v  vs  R=ΔC/ΔE"),
        ):
            fig, ax = plt.subplots(figsize=(6, 5))
            colors = {"top_cap": "tab:blue", "generalized": "tab:orange",
                      "random": "tab:green", "edit_gradient": "tab:red"}
            for r in valid_records:
                ax.scatter(r["x"] if False else (r["c"] if fname.startswith("fig_a") else r["rho"]),
                           r["R"], s=12, alpha=0.6,
                           color=colors.get(r["source"], "gray"))
            ax.set_xlabel("c(v)" if fname.startswith("fig_a") else "ρ(v)")
            ax.set_ylabel("R = ΔC / ΔE")
            ax.set_title(title)
            ax.set_xscale("log")
            ax.set_yscale("log")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, fname), dpi=150)
            plt.close(fig)
        print(f"[plot] figures written to {out_dir}")
    except Exception as e:
        print(f"[plot] matplotlib unavailable or failed: {e}")

    print("\n========== 实验一结果 ==========")
    print(json.dumps(corr_summary.get("verdict_detail", corr_summary), ensure_ascii=False, indent=2))
    return corr_summary


# --------------------------------------------------------------------------- #
# 8. 数学自检（有效性检查 1 & 2）
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
        # edit-正交基下 e≈1，故也检查 |ρ−c|
        worst2 = max(
            (abs(d["rho"] - d["c"]) for d in directions if d["source"] == "generalized" and d["e"] < 1.0 + 1e-2 and d["e"] > 1.0 - 1e-2),
            default=0.0,
        )
        print(f"[self_check] (alt) |ρ−c| on e≈1 directions: worst={worst2:.3e}")
        if worst2 > 1e-3:
            raise RuntimeError(f"generalized ρ=c/e self-check FAILED (worst={worst:.3e})")
    print("[self_check] generalized ρ=c/e OK")


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _chunks(arr, n):
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


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
# main
# --------------------------------------------------------------------------- #
def _parse_layer_indices(
    layer: Optional[int], layers: Optional[List[str]]
) -> List[int]:
    """Parse the legacy single-layer flag or the multi-layer flag."""
    if layers is None:
        return [19 if layer is None else layer]

    parsed: List[int] = []
    for value in layers:
        for token in value.split(","):
            token = token.strip()
            if token:
                parsed.append(int(token))
    if not parsed:
        raise ValueError("--layers must contain at least one layer index")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"duplicate layer index in --layers: {parsed}")
    return parsed


def main(
    argv=None,
    _layer_override: Optional[int] = None,
    _shared_model_state: Optional[Dict] = None,
):
    parser = argparse.ArgumentParser(description="实验一：ρ vs H_c 核心假设验证")
    parser.add_argument("--model", default="llama3-8b")
    parser.add_argument("--layer", type=int, default=None,
                        help="single layer index (legacy option; default: 19)")
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        metavar="LAYER",
        help="multiple layer indices, e.g. --layers 19 20 21 or --layers 19,20,21",
    )
    parser.add_argument("--K", type=int, default=60, help="最终唯一候选方向总数")
    parser.add_argument("--N_alpha", type=int, default=4)
    parser.add_argument("--N_cap", type=int, default=64, help="能力 KL 序列长度")
    parser.add_argument("--N_edit", type=int, default=20, help="编辑样本数")
    parser.add_argument(
        "--min_edit_gain",
        type=float,
        default=1e-8,
        help="方向进入相关性分析所需的最小正编辑收益",
    )
    parser.add_argument("--out_dir", default="logs/experiment1/")
    parser.add_argument("--data_type", default="wiki")
    parser.add_argument("--cap_ds", default=None, help="能力 KL 数据集（默认 mom2_dataset）")
    parser.add_argument("--batch_tokens", type=int, default=2048)
    parser.add_argument("--edit_batch_size", type=int, default=32,
                        help="编辑 loss 前向批大小（原 calculate_request_loss 硬编码 batch=1）")
    parser.add_argument("--no_cache_ref_logits", action="store_true",
                        help="关闭 θ_0 能力 logits 缓存（默认开启；关闭后每次 measure 重算 θ_0 前向，慢约 2×）")
    parser.add_argument("--skip_self_check", action="store_true")
    args = parser.parse_args(argv)

    try:
        layer_indices = _parse_layer_indices(args.layer, args.layers)
    except ValueError as exc:
        parser.error(str(exc))

    if args.layer is not None and args.layers is not None:
        parser.error("--layer and --layers are mutually exclusive")
    if args.min_edit_gain < 0:
        parser.error("--min_edit_gain must be non-negative")

    # Multi-layer mode is a sequence of independent probes, not a joint update
    # across layers. The model and cov-cache are shared; each layer still gets
    # its own factors, directions, measurements, and output directory.
    if _layer_override is None and len(layer_indices) > 1:
        set_seed(SEED)
        model, tok, hparams = load_model_and_hparams(args.model)
        original_layers = list(hparams.layers)
        hparams.layers = layer_indices
        print(
            f"[factors] requested layers={layer_indices}; "
            f"hparams.layers {original_layers} -> {hparams.layers}"
        )
        layer_to_cov = _build_cov_cache_from_hparams(
            model, tok, hparams, force_recompute=False
        )
        shared_model_state = {
            "model": model,
            "tok": tok,
            "hparams": hparams,
            "layer_to_cov": layer_to_cov,
        }
        summary = []
        for layer_idx in layer_indices:
            print(f"\n######## multi-layer run: layer {layer_idx} ########")
            main(
                argv,
                _layer_override=layer_idx,
                _shared_model_state=shared_model_state,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            summary.append({
                "layer_index": layer_idx,
                "output_dir": os.path.join(args.out_dir, f"layer_{layer_idx}"),
            })
        os.makedirs(args.out_dir, exist_ok=True)
        with open(
            os.path.join(args.out_dir, "layers_summary.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"[done] multi-layer summary written to {args.out_dir}")
        return summary

    selected_layer = (
        _layer_override
        if _layer_override is not None
        else layer_indices[0]
    )
    args.layer = selected_layer
    args.layers = None
    if _layer_override is not None:
        args.out_dir = os.path.join(args.out_dir, f"layer_{selected_layer}")

    set_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.K < 1:
        parser.error("--K must be at least 1")
    alphas_rel = ALPHAS_REL_DEFAULT[: args.N_alpha]
    if not alphas_rel:
        parser.error("--N_alpha must be at least 1")

    # 1) 模型 / 因子
    if _shared_model_state is None:
        model, tok, hparams = load_model_and_hparams(args.model)
        original_layers = list(hparams.layers)
        hparams.layers = [args.layer]
        print(
            f"[factors] restricted hparams.layers "
            f"{original_layers} -> {hparams.layers}"
        )
        factors = build_factors(model, tok, hparams, args.layer)
    else:
        model = _shared_model_state["model"]
        tok = _shared_model_state["tok"]
        hparams = _shared_model_state["hparams"]
        factors = build_factors(
            model,
            tok,
            hparams,
            args.layer,
            layer_to_cov=_shared_model_state["layer_to_cov"],
        )

    cap_ds = args.cap_ds or getattr(hparams, "mom2_dataset", "wikipedia")

    # 2) 数学自检
    if not args.skip_self_check:
        print("\n=== 数学自检 ===")
        self_check_kronecker(factors)
        # 方向构造后还会再检 generalized ρ

    # 3) 编辑请求
    edit_txt, edit_tgt = get_edit_requests(args.data_type, args.N_edit)
    print(f"[edit] {len(edit_txt)} edit samples from data_type={args.data_type}")

    # 4) 方向构造：c/rho are predictors; edit_slope only orients candidates.
    print("\n=== 构造方向 ===")
    directions = build_directions(
        factors, args.K, edit_txt, edit_tgt, model, tok, hparams, eig_device=COMPUTE_DEVICE
    )
    if not args.skip_self_check:
        self_check_generalized_rho(directions)

    # 5) Baseline: freeze the reference objective and materialize all reusable
    # data before any weight perturbation is attempted.
    print("\n=== baseline ===")
    # Edit batches are reused across every K * N_alpha trial.
    # 数值与 calculate_request_loss 一致（sample_size == len(edit_txt) 无随机抽样）。
    edit_batches = build_edit_batches(tok, edit_txt, edit_tgt, batch_size=args.edit_batch_size)
    baseline_edit_loss = edit_loss(model, edit_batches)
    print(f"[baseline] L_edit(θ_0) = {baseline_edit_loss:.6f} "
          f"({len(edit_batches)} edit batches, batch_size≤{args.edit_batch_size})")

    # Capability batches are also fixed so Delta C compares models on the same
    # token set rather than on a new random sample each time.
    print(f"[baseline] building capability batches (cap_ds={cap_ds}, N_cap={args.N_cap}) ...")
    cap_batches = build_capability_batches(
        tok, cap_ds, sample_size=args.N_cap, batch_tokens=args.batch_tokens
    )
    print(f"[baseline] {len(cap_batches)} capability batches ready")

    # θ_0 的能力 logits 缓存：KL 中 θ_0 全程不变，算一次即可。
    # 默认开启（--no_cache_ref_logits 可关）；不缓存则每次 measure 重做 N_cap 次前向。
    ref_cache = None
    if not args.no_cache_ref_logits:
        print("[baseline] caching θ_0 capability logits (fp16) ...")
        # 权重当前即 θ_0（未被改过）；ref_cache=None 时现场前向并缓存
        _, ref_cache = capability_kl(
            model, tok, cap_batches, ref_cache=None,
        )
        # ref_cache 现含 θ_0 的 logits；后续 measure 时传它避免重算 θ_0 前向

    # 6) Measure each candidate on a small common alpha grid. The grid is used
    # only to choose the edit-improving trial; it is not treated as K*N_alpha
    # independent correlation samples.
    print("\n=== 测量 ===")
    theta0 = factors["theta0"]
    theta0_norm = theta0.norm().item()
    weight_param = factors["weight_param"]

    records: List[Dict] = []
    n_deltaE_le0 = 0
    n_failed_measurements = 0
    total = len(directions) * len(alphas_rel)
    done = 0

    for di, d in enumerate(directions):
        V_k = d["V"]
        trials = []
        for alpha_rel in alphas_rel:
            alpha = alpha_rel * theta0_norm
            done += 1
            print(f"[{done}/{total}] dir#{di} ({d['source']}) α_rel={alpha_rel:.0e} ...", flush=True)
            try:
                edit_loss_prime, delta_C, _ = measure_pair(
                    model, tok, factors, V_k, alpha,
                    edit_batches, cap_batches, ref_cache,
                )
            except Exception as e:
                print(f"  [warn] measure failed: {e}; skipping")
                n_failed_measurements += 1
                continue

            delta_E = baseline_edit_loss - edit_loss_prime
            if delta_E <= 0:
                n_deltaE_le0 += 1
            trial_valid = delta_E > args.min_edit_gain
            trial_R = (
                delta_C / (delta_E + EPS_RHO)
                if trial_valid
                else None
            )
            trials.append({
                "alpha": alpha,
                "alpha_rel": alpha_rel,
                "L_edit_theta_prime": edit_loss_prime,
                "delta_E": delta_E,
                "delta_C": delta_C,
                "R": trial_R,
                "valid": trial_valid,
            })
            print(f"  c={d['c']:.4e} e={d['e']:.4e} ρ={d['rho']:.4e} "
                  f"ΔE={delta_E:.4e} ΔC={delta_C:.4e} "
                  f"R={trial_R if trial_R is not None else 'invalid'}")

        # Select using edit gain alone, independently of c/rho. This prevents
        # the predictor under test from deciding which trial is retained.
        best = max(trials, key=lambda trial: trial["delta_E"], default=None)
        valid = best is not None and best["valid"]
        record = {
            "V_id": di,
            "source": d["source"],
            "c": d["c"],
            "e": d["e"],
            "rho": d["rho"],
            "edit_slope": d["edit_slope"],
            "alpha_quad": d["edit_slope"] / (d["e"] + EPS_RHO),
            "L_edit_theta0": baseline_edit_loss,
            "valid": valid,
            "invalid_reason": None if valid else (
                "no_positive_edit_gain" if best is not None
                else "all_measurements_failed"
            ),
            "alpha": best["alpha"] if best is not None else None,
            "alpha_rel": best["alpha_rel"] if best is not None else None,
            "L_edit_theta_prime": (
                best["L_edit_theta_prime"] if best is not None else None
            ),
            "delta_E": best["delta_E"] if best is not None else None,
            "delta_C": best["delta_C"] if best is not None else None,
            "R": best["R"] if valid else None,
            "line_search": trials,
            "meta": d.get("meta", {}),
        }
        records.append(record)
        print(
            f"[select] dir#{di} valid={valid} "
            f"alpha_rel={record['alpha_rel']} delta_E={record['delta_E']} "
            f"R={record['R']}"
        )

    # 7) 全局恢复断言
    assert torch.allclose(weight_param.data, theta0, atol=RESTORE_ATOL), \
        "GLOBAL weight drift detected after all measurements"

    # 8) Diagnostics and JSON. Alpha sensitivity is retained as a diagnostic,
    # while the endpoint remains one selected record per direction.
    diagnostics = {
        "n_directions": len(directions),
        "n_records": len(records),
        "n_measurements": done,
        "n_failed_measurements": n_failed_measurements,
        "n_valid_directions": sum(r["valid"] for r in records),
        "n_invalid_directions": sum(not r["valid"] for r in records),
        "n_alphas": len(alphas_rel),
        "alphas_rel": list(alphas_rel),
        "pct_deltaE_le0": (n_deltaE_le0 / max(done, 1)) * 100.0,
        "pct_invalid_directions": (
            sum(not r["valid"] for r in records) / max(len(records), 1)
        ) * 100.0,
        "selection_rule": "max_delta_E_over_fixed_alpha_grid",
        "min_edit_gain": args.min_edit_gain,
        "theta0_norm": theta0_norm,
        "layer": factors["layer_name"],
        "cap_A_cond": float(
            (torch.linalg.eigvalsh(factors["cap_A"]).max()
             / torch.linalg.eigvalsh(factors["cap_A"]).clamp(min=1e-20).min()).item()
        ),
        "edit_A_cond": float(
            (torch.linalg.eigvalsh(factors["edit_A"]).max()
             / torch.linalg.eigvalsh(factors["edit_A"]).clamp(min=1e-20).min()).item()
        ),
        "baseline_edit_loss": baseline_edit_loss,
    }
    # Step-size sensitivity is diagnostic only and does not add correlation samples.
    alpha_indep = {}
    for record in records:
        rs = [
            trial["R"] for trial in record["line_search"]
            if trial["R"] is not None
        ]
        if len(rs) >= 2:
            alpha_indep[record["V_id"]] = {
                "R_min": min(rs),
                "R_max": max(rs),
                "ratio": max(rs) / (min(rs) + EPS_RHO),
            }
    diagnostics["alpha_independence"] = alpha_indep

    with open(os.path.join(args.out_dir, "diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    # 9) 相关性 + 绘图
    print("\n=== 汇总 ===")
    if records:
        correlate_and_plot(records, args.out_dir)
    else:
        print("[warn] no records produced; nothing to correlate")

    print(f"\n[done] outputs in {args.out_dir}")
    print(
        f"[diag] invalid directions = "
        f"{diagnostics['pct_invalid_directions']:.1f}%  "
        f"trial %ΔE≤0 = {diagnostics['pct_deltaE_le0']:.1f}%"
    )
    print(f"[diag] cap_A cond = {diagnostics['cap_A_cond']:.3e}  "
          f"edit_A cond = {diagnostics['edit_A_cond']:.3e}")


if __name__ == "__main__":
    main()
