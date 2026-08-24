#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验四（RQ3，doc §7）：编辑侧曲率 H_e 是否真的提供额外信息？

对照设计（控制变量：三种/四种基都用同一 soft_filter，隔离 H_e 结构效应；
       soft-vs-hard 形状效应已在 RQ2 单独比较，此处不重复）：
  - Method B  (relative,  真实 H_e) : generalized_basis(edit_A, cap_A) /
                                       generalized_basis(edit_B, cap_B) → ρ_rel = c/e
  - Baseline A (absolute,  H_e = I) : absolute_basis(cap_A) /
                                       absolute_basis(cap_B) → ρ_abs = c（绝对曲率谱）
  - sham_hc   (H_e = H_c,  ρ ≡ 1)   : generalized_basis(cap_A, cap_A) /
                                       generalized_basis(cap_B, cap_B) → 过滤退化为标量 1/(1+λ)
  - sham_random (H_e = 随机 PSD, 可选): 对真实 edit 因子做保谱随机旋转 → 同尺度但方向随机，
                                       测"H_e 的方向内容"是否关键

λ 尺度对齐（修正尺度混淆，doc §7）：
    相对 ρ=c/e 无量纲，绝对 ρ=c 带 H_c 尺度，同一 λ 网格不公平。
    每方法 λ_grid = logspace(-2,2) / median(ρ_method)，使两曲线在
    "几乎不过滤 → 几乎全过滤"的可比强度区间内对照。
    （sham_hc 的 ρ≡1，median=1，λ_grid 即原 logspace。）

两条独立证据线：
  (1) Filter Pareto（主证据）：各方法各自扫 λ，measure_pair 测 (ΔE, ΔC)，
      matched-ΔC 下比较 Method B 的 ΔE 是否 ≥ Baseline A / sham。
  (2) Predictor 相关性（辅助证据）：多源单位方向上 R=ΔC/(ΔE+ε)，
      比较 |Spearman(ρ_rel, R)| vs |Spearman(ρ_abs, R)|（= |Spearman(c, R)|）。
      注：该比较与实验一的 ρ-vs-c 同构，此处作为 H_e 必要性的预测器侧佐证。

判定（doc §7）：
  - matched-ΔC 下 Method B 的 ΔE ≥ Baseline A 且 ≥ sham_hc（两变体都看）；
  - |Spearman(ρ_rel, R)| > |Spearman(ρ_abs, R)|。
  若 Method B 无明显优势 → H_e 必要性存疑（doc §7 末尾），如实报告。

运行（从仓库根目录，服务器端）：
    CUDA_VISIBLE_DEVICES=0 python experiments/run_experiment3_he_necessity.py \\
        --model llama3-8b --layer 19 --N_edit 20 --N_cap 64 --out_dir logs/exp3/
冒烟：
    python experiments/run_experiment3_he_necessity.py --N_edit 8 --N_cap 16 --out_dir logs/exp3/
含随机 PSD 对照：
    python experiments/run_experiment3_he_necessity.py --include_sham_random --out_dir logs/exp3/
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from experiments._common_curvature import (
    SEED, EPS_RHO, DELTA_E_FLOOR, RESTORE_ATOL, COMPUTE_DEVICE,
    set_seed,
    load_model_and_hparams, build_factors,
    generalized_basis, absolute_basis, soft_filter,
    _compute_edit_gradient, build_edit_batches, edit_loss,
    build_capability_batches, capability_kl,
    measure_pair, build_directions, self_check_generalized_rho,
    self_check_kronecker, self_check_filter_identity,
    _safe_corr, topk_precision, kfac_quad_form, normalize_frobenius,
    get_edit_requests,
)


# --------------------------------------------------------------------------- #
# 自检（RQ3 专属）
# --------------------------------------------------------------------------- #
def self_check_sham_he_hc(cap_A: torch.Tensor, cap_B: torch.Tensor, damping: float) -> None:
    """
    sham 对照 H_e=H_c：generalized_basis(cap, cap) 的广义特征值应 ≈ 1（全部）。
    理论：whitened = (H_c+εI)^{-1/2} H_c (H_c+εI)^{-1/2}，特征值 = eig(H_c)/(eig(H_c)+ε)
          ≈ 1 − ε/eig ≈ 1（ε = damping·mean|diag| ~ 1e-5）。
    若不 ≈ 1，sham_hc 的"ρ≡1 标量退化"前提不成立，下游对照失效。
    """
    print("[self_check] sham H_e=H_c: ρ should be ≈1 ...")
    for tag, cap in (("A", cap_A), ("B", cap_B)):
        _, _, eigs = generalized_basis(cap, cap, damping, verbose=False)
        eigs_np = eigs.detach().cpu().numpy()
        dev = float(np.max(np.abs(eigs_np - 1.0)))
        print(f"  [sham_{tag}] n={len(eigs_np)} max|ρ−1|={dev:.3e} (min={eigs_np.min():.4e} max={eigs_np.max():.4e})")
        if dev > 1e-2:
            raise RuntimeError(f"sham H_e=H_c self-check FAILED: ρ not ≈1 (max|ρ−1|={dev:.3e})")
    print("[self_check] sham H_e=H_c OK (ρ ≈ 1, filter degenerates to scalar 1/(1+λ))")


def self_check_absolute_rho_is_curvature(cap_A: torch.Tensor, cap_B: torch.Tensor, damping: float) -> None:
    """
    Baseline A 的 ρ 应正比于绝对曲率谱 outer(eig(H_cB), eig(H_cA))（H_e=I 退化）。
    damping 引入 (1+ε) 缩放（ε~1e-5），故检验"元素比值近常数"而非逐元素相等。
    """
    print("[self_check] absolute_basis ρ ∝ absolute curvature spectrum ...")
    _, _, eig_a_abs = absolute_basis(cap_A, damping, verbose=False)
    _, _, eig_b_abs = absolute_basis(cap_B, damping, verbose=False)
    rho_abs = torch.outer(
        torch.clamp(eig_b_abs.flatten(), min=0.0),
        torch.clamp(eig_a_abs.flatten(), min=0.0),
    ).detach().cpu().numpy()
    raw_a = torch.clamp(torch.linalg.eigvalsh(cap_A.cpu()), min=0.0).numpy()
    raw_b = torch.clamp(torch.linalg.eigvalsh(cap_B.cpu()), min=0.0).numpy()
    raw = np.outer(raw_b, raw_a)
    # 比值（屏蔽 0）
    mask = raw > 1e-12
    ratios = rho_abs[mask] / raw[mask]
    cv = float(ratios.std() / (abs(ratios.mean()) + 1e-30))  # 变异系数
    print(f"  [abs] ratio mean={ratios.mean():.6e} std={ratios.std():.3e} CV={cv:.3e} "
          f"(应≈1/(1+ε_a)(1+ε_b)≈1, CV→0 表示 ρ 严格正比于绝对曲率谱)")
    if cv > 1e-3:
        raise RuntimeError(f"absolute_basis ρ not proportional to curvature spectrum (CV={cv:.3e})")
    print("[self_check] absolute_basis OK (ρ = absolute curvature spectrum up to damping scalar)")


def self_check_sign(
    model, tok, factors, g_descent, edit_batches, cap_batches, ref_cache, alpha,
) -> float:
    """g_descent 须为下降方向：θ_0 + α·g_descent 后 ΔE > 0。"""
    baseline = edit_loss(model, edit_batches)
    edit_prime, delta_C, _ = measure_pair(
        model, tok, factors, g_descent, alpha, edit_batches, cap_batches, ref_cache,
    )
    dE = baseline - edit_prime
    print(f"[self_check] sign: g_descent single step ΔE={dE:.4e} (must be > 0 for descent)")
    if dE <= 0:
        raise RuntimeError(
            f"SIGN BUG: ΔE={dE:.4e} ≤ 0 — g_descent is uphill; "
            "check _compute_edit_gradient returns uphill dL/dW and downstream uses g_descent=-g."
        )
    print("[self_check] sign OK (g_descent is a descent direction)")
    return baseline


# --------------------------------------------------------------------------- #
# 辅助：保谱随机旋转（sham_random：同尺度、随机方向的 H_e）
# --------------------------------------------------------------------------- #
def _random_rotated(factor: torch.Tensor, seed: int) -> torch.Tensor:
    """
    对 factor 做 Q diag(eigvals) Q^T，Q 为随机正交矩阵 → 保谱、随机方向。
    用于 sham_random：H_e 的尺度（特征值分布）与真实一致，但方向内容被打乱。
    若 real-H_e 仍胜 sham_random，则 H_e 的"方向对齐"内容起作用，而非仅尺度。
    """
    n = factor.shape[0]
    f = factor.detach().cpu().to(torch.float32)
    f = 0.5 * (f + f.T)  # 对称化
    eigvals = torch.linalg.eigvalsh(f).clamp(min=0.0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(n, n, generator=g)
    Q, _ = torch.linalg.qr(X)  # Q 正交
    rand = (Q * eigvals.unsqueeze(0)) @ Q.T
    rand = 0.5 * (rand + rand.T)
    return rand.to(device=factor.device, dtype=factor.dtype).contiguous()


def _rho_flat(basis_a: Tuple, basis_b: Tuple) -> torch.Tensor:
    """各方法 clamp 后的 ρ_ij.flatten()，用于 λ 尺度对齐与统计。"""
    _, _, eig_a = basis_a
    _, _, eig_b = basis_b
    return torch.outer(
        torch.clamp(eig_b.flatten(), min=0.0),
        torch.clamp(eig_a.flatten(), min=0.0),
    ).flatten()


# --------------------------------------------------------------------------- #
# λ 网格对齐
# --------------------------------------------------------------------------- #
def build_lambda_grid(rho_flat: torch.Tensor, n_points: int = 8) -> List[float]:
    """
    λ_grid = [0] + logspace(-2, 2, n_points-1) / median(ρ)。
    λ=0 为无保护角点（所有方法 U≡g，共享起点）；其余按 median(ρ) 归一，
    使 λ·median(ρ) 跨 [1e-2, 1e2] → 各方法在可比强度区间对照。
    """
    med = float(torch.median(rho_flat).item())
    if med < 1e-30:
        med = 1.0  # 全零 ρ（理论上不会发生）退化为不归一
    base = np.logspace(-2, 2, n_points - 1).tolist()
    grid = [0.0] + [float(b) / med for b in base]
    return grid


# --------------------------------------------------------------------------- #
# 单方法扫描
# --------------------------------------------------------------------------- #
def run_method_sweep(
    model, tok, factors, basis_a, basis_b, g_descent, alpha,
    edit_batches, cap_batches, ref_cache, baseline_edit_loss,
    label: str, lam_grid: List[float], variant: str,
) -> List[Dict]:
    """
    对每个 λ 计算 U = soft_filter(g_descent, basis_a, basis_b, λ)，按 variant 调整后 measure_pair。
    variant:
      'fixed'  : θ' = θ_0 + α·U          （自然 Pareto）
      'renorm' : U_n = U/||U||_F, θ'=θ_0+α·U_n （matched-norm 比方向质量）
    """
    records: List[Dict] = []
    g_norm = g_descent.norm().item()
    trivial_thresh = 1e-6 * g_norm

    for lam in lam_grid:
        U = soft_filter(g_descent, basis_a, basis_b, lam)
        U = U.to("cpu", dtype=torch.float32)
        u_norm = U.norm().item()
        is_trivial = u_norm < trivial_thresh

        if variant == "renorm":
            if is_trivial:
                records.append({
                    "method": label, "variant": variant, "lam": lam,
                    "alpha": alpha, "U_norm": u_norm, "trivial": True,
                    "delta_E": None, "delta_C": None,
                })
                continue
            V = U / u_norm
        else:
            V = U

        try:
            edit_prime, delta_C, _ = measure_pair(
                model, tok, factors, V, alpha,
                edit_batches, cap_batches, ref_cache,
            )
        except Exception as e:
            print(f"  [warn] measure failed ({label}/{variant} λ={lam}): {e}; skipping")
            continue
        dE = baseline_edit_loss - edit_prime
        records.append({
            "method": label, "variant": variant, "lam": lam,
            "alpha": alpha, "U_norm": u_norm, "trivial": is_trivial,
            "delta_E": dE, "delta_C": delta_C,
            "L_edit_theta_prime": edit_prime,
        })
        print(f"  [{label}/{variant}] λ={lam:.4e} ||U||={u_norm:.4e} "
              f"ΔE={dE:.4e} ΔC={delta_C:.4e}")
    return records


# --------------------------------------------------------------------------- #
# matched-ΔC 跨方法比较
# --------------------------------------------------------------------------- #
def matched_deltaC_comparison(records: List[Dict], methods: List[str], n_bins: int = 8) -> Dict:
    """
    按 ΔC 分 bin（仅非平凡点），取各方法在该 bin 内最大 ΔE。
    Method B 应 ≥ Baseline A / sham。返回 per-variant 的 bin 表与 win-rate。
    """
    out = {}
    for variant in ("fixed", "renorm"):
        sub = [r for r in records if r["variant"] == variant and not r.get("trivial", False)
               and r["delta_E"] is not None and r["delta_C"] is not None]
        if len(sub) < 2:
            out[variant] = {"bins": [], "note": "insufficient non-trivial points"}
            continue
        dcs = [r["delta_C"] for r in sub]
        dc_min, dc_max = min(dcs), max(dcs)
        if dc_max - dc_min < 1e-15:
            out[variant] = {"bins": [], "note": "ΔC range degenerate"}
            continue
        edges = np.linspace(dc_min, dc_max, n_bins + 1)
        bins = []
        method_max_dE = {m: [] for m in methods}
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            entry = {"dc_center": float(0.5 * (lo + hi))}
            for m in methods:
                dEs = [r["delta_E"] for r in sub
                       if r["method"] == m and lo <= r["delta_C"] <= hi]
                entry[f"{m}_max_dE"] = max(dEs) if dEs else None
                if dEs:
                    method_max_dE[m].append(max(dEs))
            bins.append(entry)
        # Method B 胜率：每个 bin 内 relative >= 各对照
        n_beat_abs = sum(1 for b in bins
                         if b.get("relative_max_dE") is not None and b.get("absolute_max_dE") is not None
                         and b["relative_max_dE"] >= b["absolute_max_dE"] - 1e-12)
        n_beat_sham = sum(1 for b in bins
                          if b.get("relative_max_dE") is not None and b.get("sham_hc_max_dE") is not None
                          and b["relative_max_dE"] >= b["sham_hc_max_dE"] - 1e-12)
        n_valid_abs = sum(1 for b in bins
                          if b.get("relative_max_dE") is not None and b.get("absolute_max_dE") is not None)
        n_valid_sham = sum(1 for b in bins
                           if b.get("relative_max_dE") is not None and b.get("sham_hc_max_dE") is not None)
        out[variant] = {
            "bins": bins, "n_bins": len(bins),
            "relative_beats_absolute": n_beat_abs / max(n_valid_abs, 1),
            "relative_beats_sham_hc": n_beat_sham / max(n_valid_sham, 1),
            "n_bins_valid_absolute": n_valid_abs,
            "n_bins_valid_sham_hc": n_valid_sham,
        }
    return out


# --------------------------------------------------------------------------- #
# 预测器相关性（辅助证据线）
# --------------------------------------------------------------------------- #
def predictor_correlation(
    model, tok, factors, hparams, edit_txt, edit_tgt,
    edit_batches, cap_batches, ref_cache, baseline_edit_loss,
    K: int, alpha_rel: float, out_dir: str,
) -> Dict:
    """
    多源单位方向上测 (ΔE, ΔC, R=ΔC/(ΔE+ε))，比较
      |Spearman(ρ_rel, R)|  vs  |Spearman(ρ_abs, R)|  （ρ_abs = c，H_e=I 退化）。
    复用 build_directions（与实验一同源，保证可比）。RQ3 据此判定 H_e 是否提升方向预测。
    """
    print("\n=== 预测器相关性（辅助证据）===")
    g_norm = _compute_edit_gradient(model, tok, edit_txt, edit_tgt, factors["weight_param"]).norm().item()
    alpha = alpha_rel * g_norm
    K_per_source = max(K // 4, 1)

    directions = build_directions(
        factors, K_per_source, edit_txt, edit_tgt, model, tok, hparams, eig_device=COMPUTE_DEVICE,
    )
    self_check_generalized_rho(directions)

    recs: List[Dict] = []
    for di, d in enumerate(directions):
        V = d["V"]
        try:
            edit_prime, delta_C, _ = measure_pair(
                model, tok, factors, V, alpha, edit_batches, cap_batches, ref_cache,
            )
        except Exception as e:
            print(f"  [warn] measure dir#{di} failed: {e}; skipping")
            continue
        dE = baseline_edit_loss - edit_prime
        dE_eff = dE if dE > 0 else DELTA_E_FLOOR
        R = delta_C / (dE_eff + EPS_RHO)
        recs.append({
            "V_id": di, "source": d["source"],
            "c": d["c"], "e": d["e"], "rho_rel": d["rho"],
            "rho_abs": d["c"],  # H_e=I 退化：ρ_abs = c（绝对曲率）
            "delta_E": dE, "delta_C": delta_C, "R": R,
        })

    if len(recs) < 3:
        print("[warn] insufficient predictor records; skipping correlation")
        return {"n": len(recs), "verdict": "insufficient_samples"}

    rho_rel = [r["rho_rel"] for r in recs]
    rho_abs = [r["rho_abs"] for r in recs]
    R = [r["R"] for r in recs]

    corr = {
        "n": len(recs),
        "rho_rel_vs_R": _safe_corr(rho_rel, R),
        "rho_abs_vs_R": _safe_corr(rho_abs, R),
        "topk_rho_rel": topk_precision(rho_rel, R, [5, 10, 20]),
        "topk_rho_abs": topk_precision(rho_abs, R, [5, 10, 20]),
    }
    rrel = corr["rho_rel_vs_R"].get("spearman_r")
    rabs = corr["rho_abs_vs_R"].get("spearman_r")
    if rrel is not None and rabs is not None:
        better = abs(rrel) > abs(rabs)
        corr["verdict"] = "rho_rel_stronger" if better else "rho_abs_not_worse_or_stronger"
        corr["verdict_detail"] = (
            f"|Spearman(ρ_rel,R)|={abs(rrel):.4f} vs |Spearman(ρ_abs,R)|={abs(rabs):.4f} "
            f"(ρ_rel=真实H_e相对曲率, ρ_abs=绝对曲率c)"
        )
    else:
        corr["verdict"] = "insufficient_samples"

    with open(os.path.join(out_dir, "predictor_correlation.json"), "w", encoding="utf-8") as f:
        json.dump({"records": recs, "summary": corr}, f, indent=2, ensure_ascii=False)

    print(f"[predictor] {corr.get('verdict_detail', corr.get('verdict'))}")
    return corr


# --------------------------------------------------------------------------- #
# 汇总与绘图
# --------------------------------------------------------------------------- #
def summarize(
    records: List[Dict], matched: Dict, predictor: Dict,
    methods: List[str], out_dir: str, args,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    fixed = matched.get("fixed", {})
    renorm = matched.get("renorm", {})
    fixed_beat_abs = fixed.get("relative_beats_absolute", 0.0)
    fixed_beat_sham = fixed.get("relative_beats_sham_hc", 0.0)
    renorm_beat_abs = renorm.get("relative_beats_absolute", 0.0)
    renorm_beat_sham = renorm.get("relative_beats_sham_hc", 0.0)

    # 预测器侧
    pred_rrel = predictor.get("rho_rel_vs_R", {}).get("spearman_r")
    pred_rabs = predictor.get("rho_abs_vs_R", {}).get("spearman_r")
    pred_better = (pred_rrel is not None and pred_rabs is not None and abs(pred_rrel) > abs(pred_rabs))

    # 判定：filter Pareto（两变体 Method B 胜 Baseline A 与 sham）+ 预测器侧
    pareto_supports = (fixed_beat_abs > 0.5 and fixed_beat_sham > 0.5
                       and renorm_beat_abs >= 0.4)
    if pareto_supports and pred_better:
        verdict = "he_provides_extra_info"
    elif pareto_supports or pred_better:
        verdict = "he_partial_support"
    else:
        verdict = "he_necessity_doubtful"

    summary = {
        "rq": "RQ3: does edit-side curvature H_e provide extra info?",
        "n_records": len(records),
        "methods": methods,
        "alpha_rel": args.alpha_rel,
        "matched_deltaC": matched,
        "predictor_correlation": predictor,
        "filter_pareto_evidence": {
            "fixed_beats_absolute": fixed_beat_abs,
            "fixed_beats_sham_hc": fixed_beat_sham,
            "renorm_beats_absolute": renorm_beat_abs,
            "renorm_beats_sham_hc": renorm_beat_sham,
        },
        "verdict": verdict,
        "verdict_detail": (
            f"matched-ΔC Method B win-rate: fixed vs abs={fixed_beat_abs:.2f}, "
            f"fixed vs sham={fixed_beat_sham:.2f}, renorm vs abs={renorm_beat_abs:.2f}; "
            f"predictor |ρ_rel|>{abs(pred_rrel) if pred_rrel is not None else 'NA'} "
            f"vs |ρ_abs|={abs(pred_rabs) if pred_rabs is not None else 'NA'} "
            f"(predictor_better={pred_better})"
        ),
    }

    with open(os.path.join(out_dir, "he_necessity.json"), "w", encoding="utf-8") as f:
        json.dump({"records": records, "summary": summary}, f, indent=2, ensure_ascii=False)

    # 绘图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"relative": "tab:blue", "absolute": "tab:red",
                  "sham_hc": "tab:gray", "sham_random": "tab:purple"}
        markers = {"relative": "o", "absolute": "s", "sham_hc": "^", "sham_random": "D"}

        # Fig 1: Filter Pareto（两变体分面）
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, variant in zip(axes, ("fixed", "renorm")):
            for m in methods:
                pts = [(r["delta_C"], r["delta_E"]) for r in records
                       if r["variant"] == variant and r["method"] == m
                       and r["delta_E"] is not None and r["delta_C"] is not None
                       and not r.get("trivial", False)]
                if pts:
                    pts.sort()
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, color=colors.get(m, "k"), marker=markers.get(m, "."),
                            label=m, linewidth=1.5)
            ax.set_xlabel("ΔC = capability KL  (higher = more capability loss)")
            ax.set_ylabel("ΔE = L_edit(θ₀) − L_edit(θ')  (higher = better edit)")
            mv = matched.get(variant, {})
            ax.set_title(f"Variant: {variant}\n(B beats abs={mv.get('relative_beats_absolute',0):.2f}, "
                         f"beats sham={mv.get('relative_beats_sham_hc',0):.2f})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle("RQ3: H_e necessity — relative (real H_e) vs absolute (H_e=I) vs sham")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_pareto.png"), dpi=150)
        plt.close(fig)

        # Fig 2: ρ 谱分布对比（各方法的 ρ_ij 分布）
        fig, ax = plt.subplots(figsize=(7, 5))
        for m, rho_flat in args.rho_dists.items():
            arr = rho_flat.detach().cpu().numpy()
            arr = arr[arr > 0]
            if len(arr):
                ax.hist(np.clip(arr, 1e-6, np.quantile(arr, 0.99)), bins=60, alpha=0.5,
                        label=f"{m} (med={np.median(arr):.2e})", color=colors.get(m, "k"))
        ax.set_xscale("log")
        ax.set_xlabel("ρ_ij (clamp, log scale)")
        ax.set_ylabel("count")
        ax.set_title("RQ3: ρ spectrum per method (relative dimensionless vs absolute curvature-scaled)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_rho_spectrum.png"), dpi=150)
        plt.close(fig)

        print(f"[plot] figures written to {out_dir}")
    except Exception as e:
        print(f"[plot] matplotlib unavailable or failed: {e}")

    print("\n========== RQ3 结果 ==========")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("matched_deltaC", "predictor_correlation")},
                     ensure_ascii=False, indent=2))
    print(f"\nverdict: {verdict}")
    print(f"  {summary['verdict_detail']}")
    return summary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="RQ3: H_e 是否提供额外信息")
    parser.add_argument("--model", default="llama3-8b")
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--N_edit", type=int, default=20)
    parser.add_argument("--N_cap", type=int, default=64)
    parser.add_argument("--out_dir", default="logs/exp3/")
    parser.add_argument("--data_type", default="wiki")
    parser.add_argument("--cap_ds", default=None)
    parser.add_argument("--batch_tokens", type=int, default=2048)
    parser.add_argument("--edit_batch_size", type=int, default=32)
    parser.add_argument("--alpha_rel", type=float, default=1e-3,
                        help="步长 α = alpha_rel · ||g_descent||_F（梯度尺度，非权重尺度）")
    parser.add_argument("--K", type=int, default=24, help="预测器相关性方向数（≈4来源×K/4）")
    parser.add_argument("--n_lambda", type=int, default=8, help="每方法 λ 网格点数（含 0）")
    parser.add_argument("--include_sham_random", action="store_true",
                        help="加入 sham_random（保谱随机旋转 H_e）对照")
    parser.add_argument("--skip_predictor", action="store_true",
                        help="跳过预测器相关性证据线（省时）")
    parser.add_argument("--no_cache_ref_logits", action="store_true")
    parser.add_argument("--skip_self_check", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 模型 / 因子（单层 down_proj）
    model, tok, hparams = load_model_and_hparams(args.model)
    original_layers = list(hparams.layers)
    hparams.layers = [args.layer]
    print(f"[factors] restricted hparams.layers {original_layers} -> {hparams.layers}")
    factors = build_factors(model, tok, hparams, args.layer)

    cap_ds = args.cap_ds or getattr(hparams, "mom2_dataset", "wikipedia")
    damping = float(getattr(hparams, "factor_damping",
                           getattr(hparams, "newton_damping", 1e-5)))

    cap_A, cap_B = factors["cap_A"], factors["cap_B"]
    edit_A, edit_B = factors["edit_A"], factors["edit_B"]

    # 2) 编辑梯度（uphill dL/dW）→ 下降方向 g_descent = -g
    edit_txt, edit_tgt = get_edit_requests(args.data_type, args.N_edit)
    print(f"[edit] {len(edit_txt)} edit samples from data_type={args.data_type}")
    g = _compute_edit_gradient(model, tok, edit_txt, edit_tgt, factors["weight_param"])
    g_descent = (-g).contiguous()
    g_norm = g_descent.norm().item()
    args.g_norm = g_norm
    print(f"[grad] ||g_descent||_F = {g_norm:.6e}  (damping={damping})")

    # 3) 各方法基
    print("\n=== 构造各方法基 ===")
    bases: Dict[str, Tuple] = {}
    bases["relative"] = (
        generalized_basis(edit_A, cap_A, damping),
        generalized_basis(edit_B, cap_B, damping),
    )
    bases["absolute"] = (
        absolute_basis(cap_A, damping),
        absolute_basis(cap_B, damping),
    )
    bases["sham_hc"] = (
        generalized_basis(cap_A, cap_A, damping),
        generalized_basis(cap_B, cap_B, damping),
    )
    methods = ["relative", "absolute", "sham_hc"]
    if args.include_sham_random:
        rand_edit_A = _random_rotated(edit_A, seed=SEED + 1)
        rand_edit_B = _random_rotated(edit_B, seed=SEED + 2)
        bases["sham_random"] = (
            generalized_basis(rand_edit_A, cap_A, damping),
            generalized_basis(rand_edit_B, cap_B, damping),
        )
        methods.append("sham_random")

    # 4) 自检
    if not args.skip_self_check:
        print("\n=== 数学自检 ===")
        self_check_kronecker(factors)
        # relative 基的过滤角点一致性
        self_check_filter_identity(
            bases["relative"][0], bases["relative"][1],
            g_descent.to(bases["relative"][0][0].device, bases["relative"][0][0].dtype),
        )
        # absolute 基的过滤角点一致性
        self_check_filter_identity(
            bases["absolute"][0], bases["absolute"][1],
            g_descent.to(bases["absolute"][0][0].device, bases["absolute"][0][0].dtype),
        )
        # RQ3 专属：sham_hc 的 ρ≡1、absolute 的 ρ=绝对曲率谱
        self_check_sham_he_hc(cap_A, cap_B, damping)
        self_check_absolute_rho_is_curvature(cap_A, cap_B, damping)

    # 物化预算
    edit_batches = build_edit_batches(tok, edit_txt, edit_tgt, batch_size=args.edit_batch_size)
    cap_batches = build_capability_batches(tok, cap_ds, args.N_cap, args.batch_tokens)
    ref_cache = None
    if not args.no_cache_ref_logits:
        print("[baseline] caching θ_0 capability logits ...")
        _, ref_cache = capability_kl(model, tok, cap_batches, ref_cache=None)
    alpha = args.alpha_rel * g_norm

    if not args.skip_self_check:
        baseline_edit_loss = self_check_sign(
            model, tok, factors, g_descent, edit_batches, cap_batches, ref_cache, alpha,
        )
    else:
        baseline_edit_loss = edit_loss(model, edit_batches)
    print(f"[baseline] L_edit(θ_0) = {baseline_edit_loss:.6f}")

    # 5) 各方法 λ 网格（按 median(ρ) 对齐）+ 扫描
    print("\n=== λ 网格对齐（按 median(ρ) 归一）===")
    lam_grids: Dict[str, List[float]] = {}
    rho_dists: Dict[str, torch.Tensor] = {}
    for m in methods:
        rf = _rho_flat(bases[m][0], bases[m][1])
        rho_dists[m] = rf
        lam_grids[m] = build_lambda_grid(rf, n_points=args.n_lambda)
        print(f"  [{m}] median(ρ)={float(torch.median(rf)):.4e} max(ρ)={float(rf.max()):.4e} "
              f"λ_grid={[f'{x:.3e}' for x in lam_grids[m]]}")
    args.rho_dists = rho_dists

    records: List[Dict] = []
    for variant in ("fixed", "renorm"):
        print(f"\n=== sweep variant={variant} α={alpha:.4e} ===")
        for m in methods:
            print(f"--- method={m} ---")
            records += run_method_sweep(
                model, tok, factors, bases[m][0], bases[m][1], g_descent, alpha,
                edit_batches, cap_batches, ref_cache, baseline_edit_loss,
                m, lam_grids[m], variant,
            )

    # 6) 全局恢复断言
    weight_param = factors["weight_param"]
    theta0 = factors["theta0"]
    assert torch.allclose(weight_param.data, theta0, atol=RESTORE_ATOL), \
        "GLOBAL weight drift detected after all measurements"

    # 7) matched-ΔC 比较
    matched = matched_deltaC_comparison(records, methods)

    # 8) 预测器相关性（辅助证据）
    if args.skip_predictor:
        predictor = {"n": 0, "verdict": "skipped"}
    else:
        predictor = predictor_correlation(
            model, tok, factors, hparams, edit_txt, edit_tgt,
            edit_batches, cap_batches, ref_cache, baseline_edit_loss,
            args.K, args.alpha_rel, args.out_dir,
        )

    # 9) 汇总 + 绘图
    summarize(records, matched, predictor, methods, args.out_dir, args)
    print(f"\n[done] outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
