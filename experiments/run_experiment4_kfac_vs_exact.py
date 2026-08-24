#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验二（RQ4，doc §5）：K-FAC 是否能可靠逼近完整 generalized curvature？

独立自包含（纯 torch + scipy，不依赖 easyeditor/服务器 LLM）。在小模型上比较：
  - exact : 完整经验 Fisher F_exact = (1/N) Σ vec(∇_W L_n) vec(∇_W L_n)^T
            = (1/N) Σ kron(g_n g_n^T, a_n a_n^T)   （[out·in, out·in] 全矩阵）
            广义特征问题 F_c q = ρ F_e q 用 Cholesky 白化在【全矩阵】上精确求解。
  - K-FAC : F_kfac = kron(B̄, Ā)，Ā=(1/N)Σa_n a_n^T，B̄=(1/N)Σg_n g_n^T；
            A 侧解 A_c q_a = α A_e q_a、B 侧解 B_c q_b = β B_e q_b；
            ρ_ij^KFAC = outer(β, α) = b_i·a_j，特征向量 = q_b[:,j] ⊗ q_a[:,i]。

两 regime（分离误差源，doc §10）：
  (1) 单线性层 z = W a：K-FAC 误差只剩 cross-sample 项 Σkron(B_n,A_n)/N ≠ kron(B̄,Ā)，
      是干净基准（a 与 g 仍通过 g_n = W a_n − y_n 相关 → 可测近似误差）。
  (2) 2–3 层 MLP：a 与 g 都依赖上下游权重 → 同时混淆 within-sample a-g 独立性假设
      与 cross-sample 项，是真实场景。

逐样本梯度（关键，deep MLP 需逐样本而非 summed loss.backward()）：
  torch.func.functional_call + torch.func.vmap(torch.func.grad(...)) 取逐样本 ∇_W L_n；
  a_n 由前向 hook 取层输入；g_n = (∇_W L_n @ a_n)/||a_n||² 精确还原（rank-1 外积）。

vec/kron 约定（与实验一 self_check_kronecker 一致）：
  torch V.flatten() 行优先；vec_rowmajor(g a^T) = kron(g, a)；
  trace(B V A V^T) = vec(V)^T kron(B, A) vec(V)（B 对称）。

指标（doc §5.2）：谱相关、top-k 方向对齐、子空间角、projector 误差、
  filter 误差/cosine、真实能力保持、True-Hessian vs Fisher gap（次要诊断）。

运行（从仓库根目录，服务器端）：
    python experiments/run_experiment4_kfac_vs_exact.py --out_dir logs/exp4/
冒烟（仅小宽度）：
    python experiments/run_experiment4_kfac_vs_exact.py --widths 16 --edit_batches 32 --out_dir logs/exp4/
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from experiments._common_curvature import generalized_basis, set_seed, EPS_RHO, RESTORE_ATOL


# --------------------------------------------------------------------------- #
# 数据生成
# --------------------------------------------------------------------------- #
def make_linear_data(W_star: torch.Tensor, N: int, noise_std: float, gen: torch.Generator):
    """z = W_star a + noise；a ~ N(0, I)。返回 a [N,in], y [N,out]。"""
    out_dim, in_dim = W_star.shape
    a = torch.randn(N, in_dim, generator=gen)
    y = a @ W_star.t()
    if noise_std > 0:
        y = y + noise_std * torch.randn(N, out_dim, generator=gen)
    return a, y


class TinyMLP(nn.Module):
    """2–3 层 MLP，编辑中间一层 W（in_dim→out_dim）。"""
    def __init__(self, in_dim, hid, out_dim, n_layers, seed):
        super().__init__()
        torch.manual_seed(seed)
        layers = []
        dims = [in_dim] + [hid] * (n_layers - 1) + [out_dim]
        for i in range(n_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        # 编辑层：中间那个 Linear（n_layers=3 → index 2 即第 2 个 Linear）
        linear_idxs = [i for i, m in enumerate(self.net) if isinstance(m, nn.Linear)]
        self.edit_linear_idx = linear_idxs[len(linear_idxs) // 2]
        self.edit_layer = self.net[self.edit_linear_idx]

    def forward(self, x):
        return self.net(x)


def make_mlp_data(teacher: TinyMLP, N: int, in_dim: int, noise_std: float, gen: torch.Generator):
    """a ~ N(0, I_in)，y = teacher(a) + noise。"""
    a = torch.randn(N, in_dim, generator=gen)
    with torch.no_grad():
        y = teacher(a)
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    return a, y


# --------------------------------------------------------------------------- #
# 逐样本梯度（vmap + functional_call）
# --------------------------------------------------------------------------- #
def per_sample_grad_and_activations(
    model: nn.Module, edit_layer: nn.Linear, inputs: torch.Tensor, targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回 (a_n [N,in], g_n [N,out], gW_n [N,out,in])。
      a_n   : edit_layer 的输入激活（前向 hook）
      gW_n  : 逐样本 ∇_W L_n（vmap(grad)）
      g_n   : 输出梯度 = gW_n @ a_n / ||a_n||²（rank-1 精确还原）
    L_n = 0.5 ||y_n − model(a_n)||²（per-sample MSE，对 out 维求和）。
    """
    N = inputs.shape[0]
    params = {k: v for k, v in model.named_parameters()}
    # 找 edit_layer 的参数名
    layer_param_name = None
    for n, p in model.named_parameters():
        if p is edit_layer.weight:
            layer_param_name = n
            break
    assert layer_param_name is not None, "edit_layer.weight not found in model parameters"

    # 前向 hook 抓 edit_layer 输入
    captured = {}

    def hook_fn(module, inp, out):
        # inp 是 tuple，第一项是输入 [N, in]
        captured["a"] = inp[0].detach().clone()

    handle = edit_layer.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        _ = model(inputs)  # 触发 hook
    a_n = captured["a"]  # [N, in]
    handle.remove()

    # 逐样本 grad
    def loss_one(params, x, y):
        out = torch.func.functional_call(model, params, (x.unsqueeze(0),))[0]  # [out]
        return 0.5 * ((out - y) ** 2).sum()

    grad_fn = torch.func.grad(loss_one)
    grads = torch.func.vmap(grad_fn, in_dims=(None, 0, 0))(params, inputs, targets)
    gW_n = grads[layer_param_name]  # [N, out, in]

    # rank-1 还原 g_n = gW_n @ a_n / ||a_n||²
    a_sq = (a_n ** 2).sum(dim=1, keepdim=True).clamp(min=1e-12)  # [N,1]
    g_n = torch.bmm(gW_n, a_n.unsqueeze(2)).squeeze(2) / a_sq    # [N, out]

    return a_n, g_n, gW_n


# --------------------------------------------------------------------------- #
# Fisher 构造
# --------------------------------------------------------------------------- #
def kfac_factors(a_n: torch.Tensor, g_n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ā = (1/N)Σ a_n a_n^T [in,in]，B̄ = (1/N)Σ g_n g_n^T [out,out]。"""
    A = (a_n.t() @ a_n) / a_n.shape[0]
    B = (g_n.t() @ g_n) / g_n.shape[0]
    return A.contiguous(), B.contiguous()


def exact_fisher(a_n: torch.Tensor, g_n: torch.Tensor, gW_n: torch.Tensor) -> torch.Tensor:
    """
    F_exact = (1/N) Σ vec(gW_n) vec(gW_n)^T  [d,d], d=out·in，行优先 vec。
    等价于 (1/N)Σ kron(g_n g_n^T, a_n a_n^T)（rank-1 时两者一致，自检验证）。
    直接从 gW_n 构造最稳健（不依赖 rank-1 还原）。
    """
    N, out_dim, in_dim = gW_n.shape
    d = out_dim * in_dim
    V = gW_n.reshape(N, d)  # 行优先 flatten，[N, d]
    F = (V.t() @ V) / N     # [d, d]
    return 0.5 * (F + F.t()).contiguous()


# --------------------------------------------------------------------------- #
# 广义特征分解
# --------------------------------------------------------------------------- #
def exact_generalized_eig(F_c: torch.Tensor, F_e: torch.Tensor, damping: float):
    """
    全矩阵广义特征 F_c q = ρ F_e q（Cholesky 白化）。
    eps = damping * mean(|diag(F_e)|)（与 K-FAC/ProjectedAdam 同 damping 规则 → damping parity）。
    返回 (rho [d], q [d,d], dual_q [d,d])：q^T F_e q = I, q^T F_c q = diag(rho)。
    """
    d = F_e.shape[0]
    trace_scale = F_e.diagonal().abs().mean().clamp(min=1e-12)
    eps = damping * trace_scale
    eye = torch.eye(d, device=F_e.device, dtype=F_e.dtype)
    F_e_reg = 0.5 * (F_e + F_e.t()) + eps * eye
    L = torch.linalg.cholesky(F_e_reg)
    tmp = torch.linalg.solve_triangular(L, F_c, upper=False)
    whitened = torch.linalg.solve_triangular(L, tmp.t(), upper=False).t()
    rho, vecs = torch.linalg.eigh(whitened)  # 升序
    q = torch.linalg.solve_triangular(L.t(), vecs, upper=True)
    dual_q = L @ vecs
    rho = torch.clamp(rho, min=0.0)
    return rho.contiguous(), q.contiguous(), dual_q.contiguous()


def kfac_generalized(A_c, A_e, B_c, B_e, damping):
    """
    K-FAC：A 侧 generalized_basis(A_e, A_c)、B 侧 generalized_basis(B_e, B_c)。
    返回 (basis_a, basis_b)，ρ_ij = outer(β, α)。
    """
    basis_a = generalized_basis(A_e, A_c, damping, verbose=False)  # A_c q_a = α A_e q_a
    basis_b = generalized_basis(B_e, B_c, damping, verbose=False)  # B_c q_b = β B_e q_b
    return basis_a, basis_b


# --------------------------------------------------------------------------- #
# 过滤（exact 全矩阵 vs K-FAC Kronecker）
# --------------------------------------------------------------------------- #
def soft_filter_full(g_vec: torch.Tensor, q: torch.Tensor, dual_q: torch.Tensor, rho: torch.Tensor, lam: float) -> torch.Tensor:
    """exact 全矩阵 soft filter：U = q @ (dual_q^T g / (1+λρ))。g_vec [d]，返回 [d]。"""
    coeffs = dual_q.t() @ g_vec       # [d]
    denom = 1.0 + float(lam) * rho    # [d]
    return q @ (coeffs / denom.clamp(min=1e-12))


def soft_filter_kfac(g_mat: torch.Tensor, basis_a, basis_b, lam: float) -> torch.Tensor:
    """K-FAC Kronecker soft filter：复用 _common_curvature 软过滤逻辑。g_mat [out,in]，返回 [out,in]。"""
    q_a, dual_q_a, eig_a = basis_a
    q_b, dual_q_b, eig_b = basis_b
    coeffs = dual_q_b.t() @ g_mat @ dual_q_a          # [out, in]
    rho_ij = torch.outer(torch.clamp(eig_b.flatten(), min=0.0),
                         torch.clamp(eig_a.flatten(), min=0.0))
    denom = 1.0 + float(lam) * rho_ij
    return q_b @ (coeffs / denom.clamp(min=1e-12)) @ q_a.t()


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def self_check_vec_convention(a_n, g_n, gW_n, F_exact):
    """
    1. v^T F_exact v == (1/N) Σ (∇_W L_n · V)²          （Fisher 定义）
    2. v^T F_exact v == (1/N) Σ (g_n^T V a_n)²           （rank-1 外积形式）
       ⇒ 确认行优先 vec 约定与实验一 self_check_kronecker 一致。
    """
    N, out_dim, in_dim = gW_n.shape
    d = out_dim * in_dim
    gen = torch.Generator(device="cpu").manual_seed(123)
    v = torch.randn(d, generator=gen)
    V = v.reshape(out_dim, in_dim)  # 行优先 unflatten

    lhs = float((v @ F_exact @ v).item())
    # (1/N) Σ (∇_W L_n · V)²，∇_W L_n · V = trace(gW_n^T V) = Σ gW_n V
    dot = torch.bmm(gW_n, V.unsqueeze(0).expand(N, out_dim, in_dim)).sum(dim=(1, 2))  # [N]
    rhs1 = float((dot ** 2).mean().item())
    # (1/N) Σ (g_n^T V a_n)²
    gva = torch.bmm(g_n.unsqueeze(1), torch.bmm(V.unsqueeze(0).expand(N, out_dim, in_dim), a_n.unsqueeze(2))).squeeze()  # [N]
    rhs2 = float((gva ** 2).mean().item())

    rel1 = abs(lhs - rhs1) / (abs(rhs1) + 1e-30)
    rel2 = abs(lhs - rhs2) / (abs(rhs2) + 1e-30)
    print(f"[self_check] vec convention: v^TFv={lhs:.6e} (Σ∇L·V²/N)={rhs1:.6e} rel={rel1:.3e}")
    print(f"[self_check]                (Σ(g^TVa)²/N)={rhs2:.6e} rel={rel2:.3e}")
    if rel1 > 1e-4 or rel2 > 1e-4:
        raise RuntimeError(f"vec convention self-check FAILED (rel1={rel1:.3e} rel2={rel2:.3e})")
    print("[self_check] vec convention OK (row-major, consistent with experiment1)")


def self_check_kfac_quadform(A_c, B_c, F_kfac):
    """v^T F_kfac v == trace(B̄ V Ā V^T)（K-FAC 二次型）。"""
    out_dim = B_c.shape[0]
    in_dim = A_c.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(456)
    v = torch.randn(out_dim * in_dim, generator=gen)
    V = v.reshape(out_dim, in_dim)
    lhs = float((v @ F_kfac @ v).item())
    rhs = float(torch.trace(B_c @ V @ A_c @ V.t()).clamp(min=0.0).item())
    rel = abs(lhs - rhs) / (abs(rhs) + 1e-30)
    print(f"[self_check] K-FAC quadform: v^TFv={lhs:.6e} trace(BVAV^T)={rhs:.6e} rel={rel:.3e}")
    if rel > 1e-4:
        raise RuntimeError(f"K-FAC quadform self-check FAILED (rel={rel:.3e})")
    print("[self_check] K-FAC quadform OK")


def self_check_independence_demo(out_dim, in_dim, N, damping):
    """
    独立性演示：a_n ⊥ g_n（独立采样）→ F_exact = (1/N)Σkron(B_n,A_n) → kron(B̄,Ā)
    （大数定律，误差 ~ 1/√N）。K-FAC 在独立情形下应近似无误差。
    相关情形（g_n = W a_n − y 派生）→ 显著误差。clean 演示 K-FAC 的独立性假设。
    """
    print(f"[self_check] independence demo (out={out_dim}, in={in_dim}, N={N}) ...")
    gen = torch.Generator(device="cpu").manual_seed(789)

    # 独立情形
    a_ind = torch.randn(N, in_dim, generator=gen)
    g_ind = torch.randn(N, out_dim, generator=gen)
    gW_ind = torch.bmm(g_ind.unsqueeze(2), a_ind.unsqueeze(1))  # [N,out,in] = g a^T
    F_ex_ind = exact_fisher(a_ind, g_ind, gW_ind)
    A_ind, B_ind = kfac_factors(a_ind, g_ind)
    F_kf_ind = torch.kron(B_ind, A_ind)
    err_ind = float((F_ex_ind - F_kf_ind).norm().item() / (F_ex_ind.norm().item() + 1e-30))

    # 相关情形：g_n = W a_n − y（线性 MSE 的输出梯度）
    W = torch.randn(out_dim, in_dim, generator=gen) * 0.3
    y = torch.randn(N, out_dim, generator=gen) * 0.3
    a_dep = torch.randn(N, in_dim, generator=gen)
    g_dep = a_dep @ W.t() - y                          # [N,out]，依赖 a_dep
    gW_dep = torch.bmm(g_dep.unsqueeze(2), a_dep.unsqueeze(1))
    F_ex_dep = exact_fisher(a_dep, g_dep, gW_dep)
    A_dep, B_dep = kfac_factors(a_dep, g_dep)
    F_kf_dep = torch.kron(B_dep, A_dep)
    err_dep = float((F_ex_dep - F_kf_dep).norm().item() / (F_ex_dep.norm().item() + 1e-30))

    print(f"  independent: ||F_exact − F_kfac||/||F_exact|| = {err_ind:.4e}  (期望 ~1/√N={1/np.sqrt(N):.4e})")
    print(f"  dependent  : ||F_exact − F_kfac||/||F_exact|| = {err_dep:.4e}  (期望显著 > independent)")
    if not (err_dep > err_ind):
        print(f"  [warn] dependent error ({err_dep:.3e}) not > independent ({err_ind:.3e}); "
              "independence assumption demo inconclusive at this N/seed")
    else:
        print("[self_check] independence demo OK (K-FAC error grows with a-g dependence)")
    return {"independent_err": err_ind, "dependent_err": err_dep,
            "expected_indep_scale": float(1.0 / np.sqrt(N))}


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def spectrum_correlation(rho_exact: torch.Tensor, rho_kfac_flat: torch.Tensor) -> Dict:
    """排序后两谱的 Pearson/Spearman + 分布统计。"""
    from scipy.stats import pearsonr, spearmanr
    x = np.sort(rho_exact.detach().cpu().numpy())
    y = np.sort(rho_kfac_flat.detach().cpu().numpy())
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {
        "n": len(x),
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_r": float(sr), "spearman_p": float(sp),
        "exact_min": float(x.min()), "exact_max": float(x.max()),
        "kfac_min": float(y.min()), "kfac_max": float(y.max()),
    }


def topk_direction_overlap(rho_exact, q_exact, basis_a, basis_b, ks: List[int]) -> Dict:
    """
    top-K-FAC 方向 = q_b[:,j] ⊗ q_a[:,i]（行优先 vec），在 exact 特征向量中找 max |overlap|。
    报告 top-k K-FAC 方向与 exact top-k 子空间的最大重叠 + ρ-rank 错配。
    """
    q_a, _, eig_a = basis_a
    q_b, _, eig_b = basis_b
    rho_kfac = torch.outer(torch.clamp(eig_b.flatten(), min=0.0),
                           torch.clamp(eig_a.flatten(), min=0.0)).flatten()
    # K-FAC 方向按 ρ 降序
    kfac_order = torch.argsort(rho_kfac, descending=True)
    # exact 方向按 ρ 降序
    exact_order = torch.argsort(rho_exact, descending=True)

    out = {}
    for k in ks:
        if k > len(rho_kfac):
            continue
        kf_idx = kfac_order[:k]
        ex_idx = exact_order[:k]
        # 构造 K-FAC top-k 方向矩阵 [d, k]
        kf_dirs = []
        for idx in kf_idx.tolist():
            j, i = divmod(idx, eig_a.shape[0])
            d = torch.outer(q_b[:, j], q_a[:, i]).flatten()  # 行优先 vec
            kf_dirs.append(d)
        kf_mat = torch.stack(kf_dirs, dim=1)  # [d, k]
        ex_mat = q_exact[:, ex_idx]           # [d, k]
        # 每个 K-FAC 方向在 exact 全空间的最大 |cos|
        kf_n = kf_mat / (kf_mat.norm(dim=0, keepdim=True) + 1e-30)
        cos = (kf_n.t() @ ex_mat).abs().max(dim=1).values  # [k]
        # 子空间 principal angles（KF subspace vs exact subspace）
        try:
            from scipy.linalg import subspace_angles
            ang = subspace_angles(kf_mat.detach().cpu().numpy(), ex_mat.detach().cpu().numpy())
            mean_angle = float(np.degrees(np.mean(ang)))
            max_angle = float(np.degrees(np.max(ang)))
        except Exception:
            mean_angle, max_angle = None, None
        # projector 误差 ||P_exact − P_kfac||_F
        P_ex = ex_mat @ ex_mat.t()
        P_kf = kf_mat @ kf_mat.t()
        proj_err = float((P_ex - P_kf).norm().item() / (P_ex.norm().item() + 1e-30))
        # ρ-rank 错配：top-k K-FAC 索引中有多少不在 top-k exact 索引
        rank_mismatch = 1.0 - len(set(kf_idx.tolist()) & set(ex_idx.tolist())) / k
        out[f"top{k}"] = {
            "max_cos_mean": float(cos.mean().item()),
            "max_cos_min": float(cos.min().item()),
            "subspace_mean_angle_deg": mean_angle,
            "subspace_max_angle_deg": max_angle,
            "projector_rel_err": proj_err,
            "rho_rank_mismatch": rank_mismatch,
        }
    return out


def filter_error_and_cosine(g_test, basis_a, basis_b, q_exact, dual_q_exact, rho_exact, lam: float) -> Dict:
    """exact 全矩阵 vs K-FAC kron soft filter 的 ||Δ||/||exact|| + cosine。"""
    out_dim, in_dim = g_test.shape
    g_vec = g_test.flatten()  # 行优先
    U_ex_vec = soft_filter_full(g_vec, q_exact, dual_q_exact, rho_exact, lam)
    U_ex = U_ex_vec.reshape(out_dim, in_dim)
    U_kf = soft_filter_kfac(g_test, basis_a, basis_b, lam)
    diff = (U_ex - U_kf)
    rel = float(diff.norm().item() / (U_ex.norm().item() + 1e-30))
    cos = float((U_ex.flatten() @ U_kf.flatten()).item()
                / (U_ex.norm().item() * U_kf.norm().item() + 1e-30))
    return {"lam": lam, "rel_err": rel, "cosine": cos,
            "U_exact_norm": float(U_ex.norm().item()),
            "U_kfac_norm": float(U_kf.norm().item())}


def real_capability_preservation(model, edit_layer, W0, g_test, basis_a, basis_b,
                                 q_exact, dual_q_exact, rho_exact, cap_inputs, cap_targets,
                                 alpha: float, lams: List[float]) -> Dict:
    """
    θ_0 + α·U 作用于 W，测 held-out capability MSE（越低越好）。
    比较 exact-filtered vs KFAC-filtered 更新的下游能力保持差异。
    """
    baseline_mse = Fnn.mse_loss(model(cap_inputs), cap_targets).item()
    out = {"baseline_cap_mse": baseline_mse, "alpha": alpha, "by_lam": {}}
    out_dim, in_dim = W0.shape
    g_vec = g_test.flatten()
    for lam in lams:
        U_ex = soft_filter_full(g_vec, q_exact, dual_q_exact, rho_exact, lam).reshape(out_dim, in_dim)
        U_kf = soft_filter_kfac(g_test, basis_a, basis_b, lam)
        rec = {}
        for tag, U in (("exact", U_ex), ("kfac", U_kf)):
            try:
                edit_layer.data.copy_(W0 + alpha * U)
                with torch.no_grad():
                    mse = Fnn.mse_loss(model(cap_inputs), cap_targets).item()
                rec[tag] = {"cap_mse": mse, "delta_mse": mse - baseline_mse}
            finally:
                edit_layer.data.copy_(W0)
                assert torch.allclose(edit_layer.data, W0, atol=RESTORE_ATOL), "weight restore failed"
        rec["exact_minus_kfac_mse"] = rec["exact"]["cap_mse"] - rec["kfac"]["cap_mse"]
        out["by_lam"][f"{lam:.3e}"] = rec
    return out


def true_hessian_diagnostic(model, edit_layer, edit_inputs, edit_targets, F_e_exact) -> Dict:
    """
    True Hessian of mean edit loss w.r.t. vec(edit_layer.weight)，行优先。
    报告 ||H_true − F_e_exact||/||F_e_exact|| 作为 Fisher-vs-Hessian gap 诊断
    （次要诊断：经验 Fisher 才是 K-FAC 的 apples-to-apples 参照）。
    gap 在此处计算（H 仅在此局部可得，不入 JSON）。
    """
    layer_param_name = None
    for n, p in model.named_parameters():
        if p is edit_layer.weight:
            layer_param_name = n
            break

    def mean_loss(flat_w):
        params = {k: v for k, v in model.named_parameters()}
        params[layer_param_name] = flat_w.reshape(edit_layer.weight.shape)
        out = torch.func.functional_call(model, params, (edit_inputs,))
        return Fnn.mse_loss(out, edit_targets)

    try:
        H = torch.autograd.functional.hessian(mean_loss, edit_layer.weight.detach().clone().flatten())
        H = 0.5 * (H + H.t())
        # MSE loss 的 Hessian = (1/N) Σ a_n a_n^T ⊗ ... 经验 Fisher 的 2× 缩放；
        # 这里仅报 gap 量级，不修正常数因子（诊断用途）。
        gap = float((H - F_e_exact).norm().item() / (F_e_exact.norm().item() + 1e-30))
        return {"true_hessian_norm": float(H.norm().item()),
                "true_hessian_shape": list(H.shape),
                "fisher_vs_hessian_gap": gap}
    except Exception as e:
        return {"error": str(e)}


# --------------------------------------------------------------------------- #
# 单配置运行
# --------------------------------------------------------------------------- #
def run_one_config(regime: str, width: int, N_edit: int, N_cap: int, damping: float,
                   noise_std: float, out_dir: str, args) -> Dict:
    print(f"\n{'='*70}\n[{regime}] width={width} N_edit={N_edit} N_cap={N_cap} damping={damping}\n{'='*70}")
    gen = torch.Generator(device="cpu").manual_seed(SEED + width + N_edit)
    torch.manual_seed(SEED + width)

    if regime == "linear":
        # 编辑层就是一个单 Linear；cap teacher 与 edit teacher 不同（不同 W*）
        W0 = torch.randn(width, width, generator=gen) * (1.0 / np.sqrt(width))
        edit_layer = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            edit_layer.weight.copy_(W0)
        model = edit_layer  # 单层即模型
        W_cap_star = torch.randn(width, width, generator=gen) * 0.5
        W_edit_star = torch.randn(width, width, generator=gen) * 0.5
        cap_a, cap_y = make_linear_data(W_cap_star, N_cap, noise_std, gen)
        edit_a, edit_y = make_linear_data(W_edit_star, N_edit, noise_std, gen)
        edit_inputs, edit_targets = edit_a, edit_y
        cap_inputs, cap_targets = cap_a, cap_y
    else:  # mlp
        in_dim, out_dim = width, width
        hid = width
        model = TinyMLP(in_dim, hid, out_dim, n_layers=3, seed=SEED + width)
        edit_layer = model.edit_layer
        W0 = edit_layer.weight.detach().clone()
        teacher = TinyMLP(in_dim, hid, out_dim, n_layers=3, seed=SEED + width + 1)
        cap_a, cap_y = make_mlp_data(teacher, N_cap, in_dim, noise_std, gen)
        edit_a, edit_y = make_mlp_data(teacher, N_edit, in_dim, noise_std, gen)
        edit_inputs, edit_targets = edit_a, edit_y
        cap_inputs, cap_targets = cap_a, cap_y

    W0 = W0.detach().clone()
    out_dim, in_dim = W0.shape

    # 逐样本梯度（capability 侧与 edit 侧）
    print("[grad] per-sample grads (cap + edit) ...")
    a_cap, g_cap, gW_cap = per_sample_grad_and_activations(model, edit_layer, cap_inputs, cap_targets)
    a_edit, g_edit, gW_edit = per_sample_grad_and_activations(model, edit_layer, edit_inputs, edit_targets)

    # Fisher 因子
    A_c, B_c = kfac_factors(a_cap, g_cap)
    A_e, B_e = kfac_factors(a_edit, g_edit)
    F_c_exact = exact_fisher(a_cap, g_cap, gW_cap)
    F_e_exact = exact_fisher(a_edit, g_edit, gW_edit)
    F_c_kfac = torch.kron(B_c, A_c)
    F_e_kfac = torch.kron(B_e, A_e)

    # 自检
    self_check_vec_convention(a_edit, g_edit, gW_edit, F_e_exact)
    self_check_kfac_quadform(A_e, B_e, F_e_kfac)

    # 广义特征
    print("[eig] exact generalized eigh (full matrix) ...")
    rho_exact, q_exact, dual_q_exact = exact_generalized_eig(F_c_exact, F_e_exact, damping)
    print("[eig] K-FAC generalized eigh (per-side) ...")
    basis_a, basis_b = kfac_generalized(A_c, A_e, B_c, B_e, damping)
    rho_kfac = torch.outer(torch.clamp(basis_b[2].flatten(), min=0.0),
                           torch.clamp(basis_a[2].flatten(), min=0.0)).flatten()

    # 指标
    spec = spectrum_correlation(rho_exact, rho_kfac)
    print(f"[metric] spectrum: spearman={spec['spearman_r']:.4f} pearson={spec['pearson_r']:.4f} "
          f"(exact range [{spec['exact_min']:.2e},{spec['exact_max']:.2e}], "
          f"kfac range [{spec['kfac_min']:.2e},{spec['kfac_max']:.2e}])")

    d = out_dim * in_dim
    ks = [k for k in (1, 5, 10, 20) if k <= d]
    topk = topk_direction_overlap(rho_exact, q_exact, basis_a, basis_b, ks)
    for k in ks:
        t = topk[f"top{k}"]
        print(f"[metric] top{k}: max_cos_mean={t['max_cos_mean']:.4f} "
              f"subspace_angle={t['subspace_mean_angle_deg']:.2f}° "
              f"proj_err={t['projector_rel_err']:.4f} rank_mismatch={t['rho_rank_mismatch']:.3f}")

    # filter 误差（测试梯度 = edit 数据在 θ_0 的平均梯度方向）
    g_test_mat = gW_edit.mean(dim=0)  # [out,in]，平均 edit 梯度作为测试方向
    filter_lams = [0.0, 1e-2, 1e0, 1e2]
    filter_metrics = [filter_error_and_cosine(g_test_mat, basis_a, basis_b,
                                              q_exact, dual_q_exact, rho_exact, lam)
                      for lam in filter_lams]
    for fm in filter_metrics:
        print(f"[metric] filter λ={fm['lam']:.0e}: rel_err={fm['rel_err']:.4e} cosine={fm['cosine']:.4f}")

    # 真实能力保持
    alpha = 0.5 * (1.0 / np.sqrt(width))  # 小步长
    real_cap = real_capability_preservation(
        model, edit_layer, W0, g_test_mat, basis_a, basis_b,
        q_exact, dual_q_exact, rho_exact, cap_inputs, cap_targets, alpha, filter_lams,
    )

    # True Hessian 诊断（仅小宽度，避免大 Hessian）
    hess = None
    if width <= 32 and N_edit <= 32:
        print("[diag] true Hessian (Fisher-vs-Hessian gap) ...")
        hess = true_hessian_diagnostic(model, edit_layer, edit_inputs, edit_targets, F_e_exact)
        if "error" not in hess:
            print(f"[diag] fisher_vs_hessian_gap={hess['fisher_vs_hessian_gap']:.4e}")

    return {
        "regime": regime, "width": width, "N_edit": N_edit, "N_cap": N_cap,
        "damping": damping, "out_in_dim": d,
        "spectrum": spec,
        "topk_directions": topk,
        "filter_metrics": filter_metrics,
        "real_capability": real_cap,
        "true_hessian": hess,
        "rho_exact_summary": {
            "min": float(rho_exact.min().item()), "max": float(rho_exact.max().item()),
            "median": float(torch.median(rho_exact).item()),
        },
        "rho_kfac_summary": {
            "min": float(rho_kfac.min().item()), "max": float(rho_kfac.max().item()),
            "median": float(torch.median(rho_kfac).item()),
        },
    }


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def summarize(results: List[Dict], out_dir: str, args) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    # 跨配置聚合：K-FAC 近似质量随 regime/width/N/damping 变化
    by_regime = {"linear": [], "mlp": []}
    for r in results:
        by_regime[r["regime"]].append(r)

    agg = {}
    for regime, rs in by_regime.items():
        if not rs:
            continue
        spearmans = [r["spectrum"]["spearman_r"] for r in rs]
        # top10 子空间角均值
        top10_angles = [r["topk_directions"]["top10"]["subspace_mean_angle_deg"]
                        for r in rs if "top10" in r["topk_directions"]
                        and r["topk_directions"]["top10"]["subspace_mean_angle_deg"] is not None]
        # filter cosine 均值（λ=1）
        filter_cos = []
        for r in rs:
            for fm in r["filter_metrics"]:
                if abs(fm["lam"] - 1.0) < 1e-6:
                    filter_cos.append(fm["cosine"])
        agg[regime] = {
            "n_configs": len(rs),
            "spectrum_spearman_mean": float(np.mean(spearmans)),
            "spectrum_spearman_min": float(np.min(spearmans)),
            "top10_subspace_angle_mean": float(np.mean(top10_angles)) if top10_angles else None,
            "filter_cosine_lam1_mean": float(np.mean(filter_cos)) if filter_cos else None,
        }

    # 判定（doc §5）：K-FAC 在独立/弱相关情形近似好（谱相关高、子空间角小、filter cosine→1），
    # 在强相关/deep MLP 情形显著退化。若两 regime 都近似好 → "kfac_reliable"；
    # 若 mlp 退化但 linear 好 → "kfac_reliable_only_weak_coupling"；
    # 若 linear 也差 → "kfac_unreliable"。
    lin_spear = agg.get("linear", {}).get("spectrum_spearman_mean", 0.0)
    mlp_spear = agg.get("mlp", {}).get("spectrum_spearman_mean", 0.0)
    lin_angle = agg.get("linear", {}).get("top10_subspace_angle_mean", 90.0)
    mlp_angle = agg.get("mlp", {}).get("top10_subspace_angle_mean", 90.0)
    lin_good = (lin_spear > 0.9 and lin_angle is not None and lin_angle < 15)
    mlp_good = (mlp_spear > 0.9 and mlp_angle is not None and mlp_angle < 15)
    if lin_good and mlp_good:
        verdict = "kfac_reliable"
    elif lin_good and not mlp_good:
        verdict = "kfac_reliable_only_weak_coupling"
    elif not lin_good:
        verdict = "kfac_unreliable_even_weak_coupling"
    else:
        verdict = "kfac_partial"

    summary = {
        "rq": "RQ4: does K-FAC reliably approximate full generalized curvature?",
        "n_configs": len(results),
        "by_regime": agg,
        "verdict": verdict,
        "verdict_detail": (
            f"linear: spectrum_spearman={lin_spear:.3f}, top10_angle={lin_angle}°; "
            f"mlp: spectrum_spearman={mlp_spear:.3f}, top10_angle={mlp_angle}°"
        ),
    }

    with open(os.path.join(out_dir, "kfac_vs_exact.json"), "w", encoding="utf-8") as f:
        json.dump({"configs": results, "summary": summary}, f, indent=2, ensure_ascii=False)

    # 绘图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Fig 1: 谱散点（exact vs kfac），每个配置一个子图
        n = len(results)
        if n > 0:
            cols = min(n, 4)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)
            for i, r in enumerate(results):
                ax = axes[i // cols][i % cols]
                # 取该配置的 exact/kfac 谱（从 configs 重新读不便，用 summary 字段画分布对比）
                ax.bar(["exact_min", "kfac_min", "exact_max", "kfac_max"],
                       [r["rho_exact_summary"]["min"], r["rho_kfac_summary"]["min"],
                        r["rho_exact_summary"]["max"], r["rho_kfac_summary"]["max"]],
                       color=["tab:blue", "tab:red", "tab:blue", "tab:red"], alpha=0.6)
                ax.set_title(f"{r['regime']} w={r['width']} N={r['N_edit']} d={r['damping']}\n"
                             f"spearman={r['spectrum']['spearman_r']:.3f}", fontsize=8)
                ax.tick_params(labelsize=7)
            fig.suptitle("RQ4: ρ spectrum range (exact vs K-FAC) per config")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "fig_spectrum.png"), dpi=150)
            plt.close(fig)

        # Fig 2: 近似质量 vs width（spearman + top10 angle）
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for regime, color in (("linear", "tab:blue"), ("mlp", "tab:red")):
            rs = by_regime[regime]
            if not rs:
                continue
            ws = [r["width"] for r in rs]
            sps = [r["spectrum"]["spearman_r"] for r in rs]
            angs = [r["topk_directions"].get("top10", {}).get("subspace_mean_angle_deg")
                    for r in rs]
            angs = [a for a in angs if a is not None]
            ws_ang = [w for w, a in zip(ws, [r["topk_directions"].get("top10", {}).get("subspace_mean_angle_deg") for r in rs]) if a is not None]
            axes[0].scatter(ws, sps, label=regime, color=color, s=60)
            if ws_ang:
                axes[1].scatter(ws_ang, angs, label=regime, color=color, s=60)
        axes[0].set_xlabel("width"); axes[0].set_ylabel("spectrum Spearman(exact, K-FAC)")
        axes[0].set_ylim(-0.05, 1.05); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("width"); axes[1].set_ylabel("top-10 subspace angle (deg)")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        fig.suptitle("RQ4: K-FAC approximation quality vs width & regime")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_quality.png"), dpi=150)
        plt.close(fig)
        print(f"[plot] figures written to {out_dir}")
    except Exception as e:
        print(f"[plot] matplotlib unavailable or failed: {e}")

    print("\n========== RQ4 结果 ==========")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_regime"},
                     ensure_ascii=False, indent=2))
    print(f"\nverdict: {verdict}")
    print(f"  {summary['verdict_detail']}")
    return summary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
SEED = 69

def main():
    parser = argparse.ArgumentParser(description="RQ4: K-FAC vs exact generalized curvature")
    parser.add_argument("--out_dir", default="logs/exp4/")
    parser.add_argument("--widths", type=str, default="16,32,64",
                        help="逗号分隔宽度列表；64 仅单点 spot-check（d=4096 eigh 较慢）")
    parser.add_argument("--edit_batches", type=str, default="8,32,128",
                        help="逗号分隔 N_edit 列表")
    parser.add_argument("--dampings", type=str, default="1e-5,1e-3,1e-1")
    parser.add_argument("--N_cap", type=int, default=256, help="capability 样本数（固定较大，减少 cap Fisher 噪声）")
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--regimes", type=str, default="linear,mlp")
    parser.add_argument("--spotcheck_width64_only", action="store_true",
                        help="64 仅跑默认 N_edit=32、damping=1e-3 单点，不全扫")
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    widths = [int(x) for x in args.widths.split(",")]
    edit_batches = [int(x) for x in args.edit_batches.split(",")]
    dampings = [float(x) for x in args.dampings.split(",")]
    regimes = args.regimes.split(",")

    # 自检：独立性演示（一次性）
    print("=== 独立性自检演示 ===")
    ind_demo = self_check_independence_demo(out_dim=16, in_dim=16, N=512, damping=1e-5)

    results: List[Dict] = []
    for regime in regimes:
        for width in widths:
            for N_edit in edit_batches:
                for damping in dampings:
                    # 64 spot-check：只跑一个 (N_edit, damping) 组合
                    if width == 64 and args.spotcheck_width64_only:
                        if not (N_edit == 32 and abs(damping - 1e-3) < 1e-12):
                            continue
                    # 大配置跳过：width 64 + 大 N_edit 太慢（F_exact 4096² + eigh）
                    if width == 64 and N_edit == 128:
                        print(f"[skip] width=64 N_edit=128 too expensive; skipping")
                        continue
                    try:
                        r = run_one_config(regime, width, N_edit, args.N_cap,
                                           damping, args.noise_std, args.out_dir, args)
                        results.append(r)
                    except Exception as e:
                        print(f"[error] config {regime}/w{width}/N{N_edit}/d{damping} failed: {e}")
                        import traceback; traceback.print_exc()

    summarize(results, args.out_dir, args)
    print(f"\n[done] outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
