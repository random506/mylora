#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验三（RQ2，doc §6）：soft 谱过滤 h_soft(ρ)=1/(1+λρ) 是否优于
absolute-curvature hard 低曲率投影 h_hard(ρ)=1[ρ≤τ]？

设计（单步 proximal 评估，复用 experiments/_common_curvature.py）：
  1. 单层 down_proj LLM 因子 + wiki 编辑梯度 g_descent = -∇_W L_edit（下降方向）；
  2. 相对曲率广义基（A 侧 generalized_basis(edit_A, cap_A)、B 侧 generalized_basis(edit_B, cap_B)），
     ρ_ij = outer(clamp(eig_b,min=0), clamp(eig_a,min=0))；
  3. soft 扫 λ 网格、hard 扫 τ 分位网格，各自 U = filter(g_descent)；
  4. θ' = θ_0 + α·U，measure_pair 测 (ΔE, ΔC)；α 取梯度尺度 α=α_rel·||g_descent||_F；
  5. 两变体：(i) fixed-α 自然 Pareto；(ii) Frobenius 重归一 U/||U|| 后 matched-norm 比方向质量。

公平性控制（doc §6）：相同基础编辑器、相同层、相同 cap 统计、相同 α；
soft/hard 共用同一 ρ 谱（仅过滤形状不同），故比较纯为 soft-vs-hard 形状效应。
角点对齐：λ=0 ≡ τ≥max(ρ)（无保护，U=g_descent）；λ→∞ ≡ τ=0（保留 ρ==0 子空间）。

判定（doc §6）：
  - soft 曲线平滑无断崖；hard 在特征值间隙处 ΔE 断崖（保守）；
  - matched-ΔC 下 soft 的 ΔE ≥ hard（两变体都看）；
  - soft 保留 hard 丢弃的中-ρ 方向 → 同 ΔC 下 ΔE 更高。

运行（从仓库根目录，服务器端）：
    CUDA_VISIBLE_DEVICES=0 python experiments/run_experiment2_soft_vs_hard.py \\
        --model llama3-8b --layer 19 --N_edit 20 --N_cap 64 --out_dir logs/exp2/
冒烟：
    python experiments/run_experiment2_soft_vs_hard.py --N_edit 8 --N_cap 16 --out_dir logs/exp2/
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
    SEED, EPS_RHO, DELTA_E_FLOOR, RESTORE_ATOL, ALPHAS_REL_DEFAULT, COMPUTE_DEVICE,
    set_seed, _chunks,
    load_model_and_hparams, build_factors,
    generalized_basis, soft_filter, hard_filter,
    _compute_edit_gradient, build_edit_batches, edit_loss,
    build_capability_batches, capability_kl,
    measure_pair, self_check_kronecker, self_check_filter_identity,
    get_edit_requests,
)


# --------------------------------------------------------------------------- #
# 自检（RQ2 专属）
# --------------------------------------------------------------------------- #
def self_check_corner_alignment(
    basis_a: Tuple, basis_b: Tuple, g_descent: torch.Tensor,
) -> None:
    """
    角点对齐自检：
      - soft(λ=0) ≡ hard(τ=+inf) ≡ g_descent   （无保护角点，已在 self_check_filter_identity 覆盖，此处复验）
      - soft(λ=1e8) ≡ hard(τ=0)                 （全保护角点：两者都保留 ρ==0 子空间）
    这是 soft/hard 在两端对齐的数学保证——中间差异纯来自过滤形状。
    """
    g = g_descent.to(device=basis_a[0].device, dtype=basis_a[0].dtype)
    u_soft_inf = soft_filter(g, basis_a, basis_b, 1e8)
    u_hard_0 = hard_filter(g, basis_a, basis_b, 0.0)
    err = float((u_soft_inf - u_hard_0).abs().max())
    g_norm = g.norm().item()
    rel = err / (g_norm + 1e-30)
    print(f"[self_check] corner soft(λ=1e8) vs hard(τ=0): max|Δ|={err:.3e} rel={rel:.3e}")
    if rel > 1e-4:
        raise RuntimeError(f"corner alignment self-check FAILED (rel={rel:.3e})")
    print("[self_check] corner alignment OK (soft λ→∞ == hard τ=0 == ρ==0 subspace)")


def self_check_sign(
    model, tok, factors, g_descent, edit_batches, cap_batches, ref_cache, alpha,
) -> float:
    """
    符号自检：g_descent 须为下降方向——θ_0 + α·g_descent 后 ΔE = L(θ_0)−L(θ') > 0。
    若 ΔE ≤ 0 说明 g 被当成下降（符号反了），下游所有 soft/hard 步都会 uphill。
    """
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
# 扫描
# --------------------------------------------------------------------------- #
def retention_stats(g: torch.Tensor, basis_a, basis_b, method: str, strength: float) -> Dict:
    """记录 ||U||_F 与系数保留比例，用于解释 Pareto。"""
    q_a, dual_q_a, _ = basis_a
    q_b, dual_q_b, _ = basis_b
    g_dev = g.to(device=q_a.device, dtype=q_a.dtype)
    coeffs = dual_q_b.T @ g_dev @ dual_q_a
    rho_ij = torch.outer(
        torch.clamp(basis_b[2].flatten(), min=0.0),
        torch.clamp(basis_a[2].flatten(), min=0.0),
    )
    coeffs_abs = coeffs.abs()
    total_mass = float(coeffs_abs.sum().item()) + 1e-30
    if method == "soft":
        denom = 1.0 + float(strength) * rho_ij
        kept = coeffs_abs / denom.clamp(min=1e-12)
        kept_mass = float(kept.sum().item()) / total_mass
        kept_frac = float(((rho_ij > 0) & (1.0 / denom.clamp(min=1e-12) > 1e-6)).float().mean().item())
    else:  # hard
        mask = (rho_ij <= float(strength)).to(coeffs.dtype)
        kept = coeffs_abs * mask
        kept_mass = float(kept.sum().item()) / total_mass
        kept_frac = float(mask.mean().item())
    return {"kept_mass": kept_mass, "kept_frac": kept_frac}


def run_sweep(
    model, tok, factors, basis_a, basis_b, g_descent, alpha,
    edit_batches, cap_batches, ref_cache, baseline_edit_loss,
    method: str, strengths: List[float], variant: str,
) -> List[Dict]:
    """
    对每个 strength 计算 U = filter(g_descent)，按 variant 调整后 measure_pair。
    variant:
      'fixed'      : θ' = θ_0 + α·U            （自然 Pareto，||U|| 随 strength 变）
      'renorm'     : U_n = U/||U||_F, θ' = θ_0 + α·U_n  （matched-norm 比方向质量）
    """
    records: List[Dict] = []
    g_norm = g_descent.norm().item()
    trivial_thresh = 1e-6 * g_norm

    for s in strengths:
        if method == "soft":
            U = soft_filter(g_descent, basis_a, basis_b, s)
        else:
            U = hard_filter(g_descent, basis_a, basis_b, s)
        U = U.to("cpu", dtype=torch.float32)
        u_norm = U.norm().item()
        ret = retention_stats(g_descent, basis_a, basis_b, method, s)
        is_trivial = u_norm < trivial_thresh

        if variant == "renorm":
            if is_trivial:
                # 重归一在近零更新上退化（放大数值噪声），跳过该点
                records.append({
                    "method": method, "variant": variant, "strength": s,
                    "alpha": alpha, "U_norm": u_norm, "trivial": True,
                    "delta_E": None, "delta_C": None,
                    "kept_mass": ret["kept_mass"], "kept_frac": ret["kept_frac"],
                })
                continue
            V = U / u_norm
            measure_alpha = alpha
        else:  # fixed
            V = U
            measure_alpha = alpha

        try:
            edit_prime, delta_C, _ = measure_pair(
                model, tok, factors, V, measure_alpha,
                edit_batches, cap_batches, ref_cache,
            )
        except Exception as e:
            print(f"  [warn] measure failed ({method}/{variant} s={s}): {e}; skipping")
            continue
        dE = baseline_edit_loss - edit_prime
        records.append({
            "method": method, "variant": variant, "strength": s,
            "alpha": measure_alpha, "U_norm": u_norm, "trivial": is_trivial,
            "delta_E": dE, "delta_C": delta_C,
            "L_edit_theta_prime": edit_prime,
            "kept_mass": ret["kept_mass"], "kept_frac": ret["kept_frac"],
        })
        print(f"  [{method}/{variant}] s={s:.4e} ||U||={u_norm:.4e} "
              f"ΔE={dE:.4e} ΔC={delta_C:.4e} kept_mass={ret['kept_mass']:.3f}")
    return records


# --------------------------------------------------------------------------- #
# Pareto 汇总与绘图
# --------------------------------------------------------------------------- #
def matched_deltaC_comparison(records: List[Dict], n_bins: int = 8) -> Dict:
    """
    按 ΔC 分 bin（仅非平凡点），取各方法在该 bin 内最大 ΔE。
    soft 应 ≥ hard（两变体分别看）。返回每变体每方法的 (ΔC_center, max_ΔE) 列表。
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
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            soft_dE = max((r["delta_E"] for r in sub
                           if r["method"] == "soft" and lo <= r["delta_C"] <= hi), default=None)
            hard_dE = max((r["delta_E"] for r in sub
                           if r["method"] == "hard" and lo <= r["delta_C"] <= hi), default=None)
            if soft_dE is not None or hard_dE is not None:
                bins.append({
                    "dc_center": float(0.5 * (lo + hi)),
                    "soft_max_dE": soft_dE, "hard_max_dE": hard_dE,
                    "soft_ge_hard": (soft_dE is not None and hard_dE is not None
                                     and soft_dE >= hard_dE - 1e-12),
                })
        n_soft_wins = sum(1 for b in bins if b["soft_ge_hard"])
        out[variant] = {
            "bins": bins,
            "n_bins": len(bins),
            "n_soft_wins": n_soft_wins,
            "soft_win_rate": n_soft_wins / max(len(bins), 1),
        }
    return out


def summarize(records: List[Dict], out_dir: str, args) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    matched = matched_deltaC_comparison(records)

    # 定性：soft 平滑 vs hard 断崖
    def cliffiness(sub: List[Dict]) -> float:
        """相邻 strength 的 ΔE 跳跃最大值 / ΔE 中位数（越大越断崖式）。"""
        ds = sorted(sub, key=lambda r: r["strength"])
        dEs = [r["delta_E"] for r in ds if r["delta_E"] is not None]
        if len(dEs) < 3:
            return 0.0
        jumps = [abs(dEs[i + 1] - dEs[i]) for i in range(len(dEs) - 1)]
        med = max(np.median(dEs), 1e-15)
        return max(jumps) / med

    qual = {}
    for variant in ("fixed", "renorm"):
        vsub = [r for r in records if r["variant"] == variant and not r.get("trivial", False)]
        soft_sub = [r for r in vsub if r["method"] == "soft"]
        hard_sub = [r for r in vsub if r["method"] == "hard"]
        qual[variant] = {
            "soft_cliffiness": cliffiness(soft_sub),
            "hard_cliffiness": cliffiness(hard_sub),
            "hard_more_cliffy": cliffiness(hard_sub) > cliffiness(soft_sub),
        }

    # 判定
    fixed_wr = matched.get("fixed", {}).get("soft_win_rate", 0.0)
    renorm_wr = matched.get("renorm", {}).get("soft_win_rate", 0.0)
    fixed_cliff = qual.get("fixed", {}).get("hard_more_cliffy", False)
    soft_better = (fixed_wr > 0.5) and (renorm_wr >= 0.4) and fixed_cliff
    verdict = "soft_dominates_hard" if soft_better else (
        "soft_comparable" if (fixed_wr >= 0.4 or renorm_wr >= 0.4) else "hard_not_worse_or_inconclusive"
    )

    summary = {
        "rq": "RQ2: soft vs hard spectral filtering",
        "n_records": len(records),
        "alpha_rel": args.alpha_rel,
        "alpha_abs": args.alpha_rel * args.g_norm if args.g_norm else None,
        "matched_deltaC": matched,
        "qualitative_cliffiness": qual,
        "soft_win_rate_fixed": fixed_wr,
        "soft_win_rate_renorm": renorm_wr,
        "verdict": verdict,
        "verdict_detail": (
            f"matched-ΔC soft win-rate: fixed={fixed_wr:.2f}, renorm={renorm_wr:.2f}; "
            f"hard more cliffy (fixed)={fixed_cliff}"
        ),
    }

    with open(os.path.join(out_dir, "pareto.json"), "w", encoding="utf-8") as f:
        json.dump({"records": records, "summary": summary}, f, indent=2, ensure_ascii=False)

    # 绘图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Fig 1: Pareto（两变体分面）
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, variant in zip(axes, ("fixed", "renorm")):
            for method, color, marker in (("soft", "tab:blue", "o"), ("hard", "tab:red", "s")):
                pts = [(r["delta_C"], r["delta_E"]) for r in records
                       if r["variant"] == variant and r["method"] == method
                       and r["delta_E"] is not None and r["delta_C"] is not None
                       and not r.get("trivial", False)]
                if pts:
                    xs, ys = zip(*sorted(pts))
                    ax.plot(xs, ys, color=color, marker=marker, label=method, linewidth=1.5)
            ax.set_xlabel("ΔC = capability KL  (higher = more capability loss)")
            ax.set_ylabel("ΔE = L_edit(θ₀) − L_edit(θ')  (higher = better edit)")
            ax.set_title(f"Variant: {variant}\n(matched-ΔC soft win-rate="
                         f"{matched.get(variant,{}).get('soft_win_rate',0):.2f})")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.suptitle("RQ2: Soft (1/(1+λρ)) vs Hard (1[ρ≤τ]) spectral filtering Pareto")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_pareto.png"), dpi=150)
        plt.close(fig)

        # Fig 2: ||U|| vs strength + kept_mass
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, metric, ylabel in zip(axes, ("U_norm", "kept_mass"),
                                      ("||U||_F", "kept coefficient mass")):
            for variant, ls in (("fixed", "-"), ("renorm", "--")):
                for method, color in (("soft", "tab:blue"), ("hard", "tab:red")):
                    pts = [(r["strength"], r[metric]) for r in records
                           if r["variant"] == variant and r["method"] == method
                           and not r.get("trivial", False)]
                    if pts:
                        pts.sort()
                        xs, ys = zip(*pts)
                        ax.plot(xs, ys, color=color, linestyle=ls, marker=".",
                                label=f"{method}/{variant}")
            ax.set_xlabel("strength (λ for soft, τ for hard)")
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle("RQ2: update norm & retained mass vs filter strength")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_norm_mass.png"), dpi=150)
        plt.close(fig)

        # Fig 3: coeffs histogram（soft 收缩 vs hard 截断，取中等 strength 点）
        fig, ax = plt.subplots(figsize=(7, 5))
        g_dev = args.g_descent_dev
        coeffs = args.basis_b[1].T @ g_dev @ args.basis_a[1]
        rho_ij = torch.outer(
            torch.clamp(args.basis_b[2].flatten(), min=0.0),
            torch.clamp(args.basis_a[2].flatten(), min=0.0),
        )
        soft_mid = 1.0
        hard_tau = float(torch.quantile(rho_ij.flatten(), 0.5).item())
        soft_coeffs = (coeffs / (1.0 + soft_mid * rho_ij).clamp(min=1e-12)).flatten().abs().cpu().numpy()
        hard_coeffs = (coeffs * (rho_ij <= hard_tau).to(coeffs.dtype)).flatten().abs().cpu().numpy()
        raw_coeffs = coeffs.flatten().abs().cpu().numpy()
        ax.hist(np.clip(raw_coeffs, 0, np.quantile(raw_coeffs, 0.99)), bins=60, alpha=0.4, label="raw", color="gray")
        ax.hist(np.clip(soft_coeffs, 0, np.quantile(raw_coeffs, 0.99)), bins=60, alpha=0.5, label=f"soft λ={soft_mid}", color="tab:blue")
        ax.hist(np.clip(hard_coeffs, 0, np.quantile(raw_coeffs, 0.99)), bins=60, alpha=0.5, label=f"hard τ={hard_tau:.2e}(50%)", color="tab:red")
        ax.set_xlabel("|coefficient| in (dual_q) basis")
        ax.set_ylabel("count")
        ax.set_title("RQ2: soft shrinks vs hard truncates coefficient spectrum")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_coeffs_hist.png"), dpi=150)
        plt.close(fig)
        print(f"[plot] figures written to {out_dir}")
    except Exception as e:
        print(f"[plot] matplotlib unavailable or failed: {e}")

    print("\n========== RQ2 结果 ==========")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("matched_deltaC",)},
                     ensure_ascii=False, indent=2))
    print(f"\nverdict: {verdict}")
    print(f"  {summary['verdict_detail']}")
    return summary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="RQ2: soft vs hard 谱过滤 Pareto")
    parser.add_argument("--model", default="llama3-8b")
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--N_edit", type=int, default=20)
    parser.add_argument("--N_cap", type=int, default=64)
    parser.add_argument("--out_dir", default="logs/exp2/")
    parser.add_argument("--data_type", default="wiki")
    parser.add_argument("--cap_ds", default=None)
    parser.add_argument("--batch_tokens", type=int, default=2048)
    parser.add_argument("--edit_batch_size", type=int, default=32)
    parser.add_argument("--alpha_rel", type=float, default=1e-3,
                        help="步长 α = alpha_rel · ||g_descent||_F（梯度尺度，非权重尺度）")
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

    # 2) 编辑梯度（uphill dL/dW）→ 下降方向 g_descent = -g
    edit_txt, edit_tgt = get_edit_requests(args.data_type, args.N_edit)
    print(f"[edit] {len(edit_txt)} edit samples from data_type={args.data_type}")
    g = _compute_edit_gradient(model, tok, edit_txt, edit_tgt, factors["weight_param"])
    g_descent = (-g).contiguous()
    g_norm = g_descent.norm().item()
    args.g_norm = g_norm
    print(f"[grad] ||g_descent||_F = {g_norm:.6e}  (damping={damping})")

    # 3) 广义基（A 侧 [in,in]、B 侧 [out,out]），ρ_ij = outer(eig_b, eig_a)
    print("\n=== 广义基（相对曲率）===")
    basis_a = generalized_basis(factors["edit_A"], factors["cap_A"], damping)
    basis_b = generalized_basis(factors["edit_B"], factors["cap_B"], damping)

    # 4) 自检
    if not args.skip_self_check:
        print("\n=== 数学自检 ===")
        self_check_kronecker(factors)
        self_check_filter_identity(basis_a, basis_b, g_descent.to(basis_a[0].device, basis_a[0].dtype))
        self_check_corner_alignment(basis_a, basis_b, g_descent)
        # 编辑 loss 物化
        edit_batches = build_edit_batches(tok, edit_txt, edit_tgt, batch_size=args.edit_batch_size)
        cap_batches = build_capability_batches(tok, cap_ds, args.N_cap, args.batch_tokens)
        ref_cache = None
        if not args.no_cache_ref_logits:
            print("[baseline] caching θ_0 capability logits ...")
            _, ref_cache = capability_kl(model, tok, cap_batches, ref_cache=None)
        alpha = args.alpha_rel * g_norm
        baseline_edit_loss = self_check_sign(
            model, tok, factors, g_descent, edit_batches, cap_batches, ref_cache, alpha,
        )
        print(f"[baseline] L_edit(θ_0) = {baseline_edit_loss:.6f}")
    else:
        edit_batches = build_edit_batches(tok, edit_txt, edit_tgt, batch_size=args.edit_batch_size)
        cap_batches = build_capability_batches(tok, cap_ds, args.N_cap, args.batch_tokens)
        ref_cache = None
        if not args.no_cache_ref_logits:
            print("[baseline] caching θ_0 capability logits ...")
            _, ref_cache = capability_kl(model, tok, cap_batches, ref_cache=None)
        alpha = args.alpha_rel * g_norm
        baseline_edit_loss = edit_loss(model, edit_batches)

    # 5) strength 网格
    soft_lams = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e6]
    rho_flat = torch.outer(
        torch.clamp(basis_b[2].flatten(), min=0.0),
        torch.clamp(basis_a[2].flatten(), min=0.0),
    ).flatten()
    rho_max = float(rho_flat.max().item())
    hard_taus = [-1.0] + [float(torch.quantile(rho_flat, q).item())
                          for q in (0.10, 0.25, 0.50, 0.75, 0.90)] + [rho_max]
    print(f"[sweep] soft λ={soft_lams}")
    print(f"[sweep] hard τ={hard_taus}  (ρ_max={rho_max:.4e})")

    # 6) 扫描（两变体 × 两方法）
    records: List[Dict] = []
    alpha = args.alpha_rel * g_norm
    for variant in ("fixed", "renorm"):
        print(f"\n=== sweep variant={variant} α={alpha:.4e} ===")
        records += run_sweep(model, tok, factors, basis_a, basis_b, g_descent, alpha,
                             edit_batches, cap_batches, ref_cache, baseline_edit_loss,
                             "soft", soft_lams, variant)
        records += run_sweep(model, tok, factors, basis_a, basis_b, g_descent, alpha,
                             edit_batches, cap_batches, ref_cache, baseline_edit_loss,
                             "hard", hard_taus, variant)

    # 7) 全局恢复断言
    weight_param = factors["weight_param"]
    theta0 = factors["theta0"]
    assert torch.allclose(weight_param.data, theta0, atol=RESTORE_ATOL), \
        "GLOBAL weight drift detected after all measurements"

    # 8) 汇总 + 绘图（histogram 用到的对象挂 args）
    args.g_descent_dev = g_descent.to(basis_a[0].device, basis_a[0].dtype)
    args.basis_a = basis_a
    args.basis_b = basis_b
    summarize(records, args.out_dir, args)
    print(f"\n[done] outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
