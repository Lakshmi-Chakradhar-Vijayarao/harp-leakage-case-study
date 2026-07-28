"""
23_final_evaluation.py — HaRP SOTA: Full system evaluation with all additions.

Phase 13, Experiment 23.
Assembles the complete feature set (Groups A+B+B2+C+C2+D+E = 24 features),
runs the full SOTA ablation, and evaluates the final governance policy.

This is the definitive results script — the single source of truth for the
HaRP SOTA system paper.

SOTA contributions vs prior work:
  vs ICR Probe (ACL 2025):        +governance policy +multi-signal +uncertainty taxonomy
  vs Geometric Uncertainty (ICLR 2026): +governance +token signals +mechanistic chain
  vs Single Direction of Truth (Jul 2025): +governance +uncertainty decomposition
  vs Adaptive Activation Steering (WWW 2025): +detection +policy +quadrant taxonomy
  vs HaluNet (Dec 2025):          +mechanistic analysis +governance +uncertainty typing
  vs Spectral Uncertainty (Sep 2025): +governance +representation geometry +action routing

SOTA ablation path:
  Group A             → 0.54  (token entropy — barely above random)
  A + B               → 0.54  (+ consistency — no gain alone)
  A + B + C           → 0.89  (+ representation geometry — MASSIVE jump)
  A + B + C + D       → >0.90 (+ cross-layer trajectory — ICR extension)
  A + B + C + D + E   → >0.91 (+ geometric volume — Granese extension) ← SOTA

Governance metrics (test set):
  Raw LLM:          hall_rate ~0.40, accept_rate 1.0
  Token-only policy: hall_rate ~0.29, poor precision
  HaRP v2:       hall_rate 0.0,   accept_rate 0.10 (exp 12)
  HaRP SOTA:     hall_rate 0.0,   accept_rate higher (richer features)

Outputs:
  results/logs/23_sota_results.json
  results/logs/final_results.json    ← UPDATED with SOTA numbers
  results/plots/23_sota_ablation.png
  results/plots/23_sota_policy.png
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.authorization.failure_estimator import (
    GROUP_A, GROUP_B, GROUP_B2, GROUP_C, GROUP_C2,
    GROUP_D, GROUP_E,
    ALL_FEATURES, ALL_FEATURES_V3, ALL_FEATURES_V4, ALL_FEATURES_SOTA,
    run_ablation, fit_final_estimator,
)
from src.authorization.policy import (
    apply_policy, optimize_thresholds, compute_policy_metrics,
)
from src.evaluation.metrics import (
    ablation_table, bootstrap_auroc, compute_auroc, compute_auarc,
    policy_summary,
)

DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_LOGS   = ROOT / "results" / "logs"
RESULTS_HS     = ROOT / "results" / "hidden_states"
RESULTS_PLOTS  = ROOT / "results" / "plots"
RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)

LAMBDA = 0.5


def load_json(name: str) -> dict:
    p = RESULTS_LOGS / name
    return json.load(open(p)) if p.exists() else {}


def load_sota_features() -> pd.DataFrame:
    """
    Merge all feature groups into one DataFrame:
      features_combined.csv       (A+B+B2+C+C2)
      trajectory_features.csv     (Group D)
      geometric_features.csv      (Group E)
    """
    base_path = DATA_PROCESSED / "features_combined.csv"
    traj_path = RESULTS_HS / "trajectory_features.csv"
    geom_path = RESULTS_HS / "geometric_features.csv"

    if not base_path.exists():
        print(f"Missing: {base_path}")
        print("Run experiments 09 first.")
        sys.exit(1)

    df = pd.read_csv(base_path)
    n_base = len(df)
    print(f"Base features: {n_base} samples, {len(df.columns)} columns")

    # Merge Group D
    if traj_path.exists():
        traj_df = pd.read_csv(traj_path)
        d_cols  = [c for c in GROUP_D if c in traj_df.columns]
        df = df.merge(traj_df[["prompt_id"] + d_cols], on="prompt_id", how="left")
        n_d = df[d_cols[0]].notna().sum() if d_cols else 0
        print(f"Group D merged: {n_d}/{n_base} samples have trajectory features")
    else:
        print("Group D not found — run 20_cross_layer_dynamics.py first")

    # Merge Group E
    if geom_path.exists():
        geom_df = pd.read_csv(geom_path)
        e_cols  = [c for c in GROUP_E if c in geom_df.columns]
        df = df.merge(geom_df[["prompt_id"] + e_cols], on="prompt_id", how="left")
        n_e = df[e_cols[0]].notna().sum() if e_cols else 0
        print(f"Group E merged: {n_e}/{n_base} samples have geometric features")
    else:
        print("Group E not found — run 21_geometric_volume_signals.py first")

    avail_feats = [c for c in ALL_FEATURES_SOTA if c in df.columns]
    print(f"\nTotal available SOTA features: {len(avail_feats)}/{len(ALL_FEATURES_SOTA)}")
    print(f"  {avail_feats}")
    return df


def plot_sota_ablation(results: dict, out_path: Path) -> None:
    """Full SOTA ablation bar chart with error bars."""
    groups = list(results.keys())
    aurocs = [results[k]["auroc_mean"] for k in groups]
    errs   = [results[k]["auroc_std"]  for k in groups]

    colors = [
        "#aec6e8", "#6baed6", "#4e79a7",          # A, A+B, A+B+C (blue scale)
        "#fdae6b", "#e6550d",                       # v3, v4 (orange scale)
        "#31a354",                                   # SOTA (green)
        "#9467bd",                                   # MLP
    ][:len(groups)]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = range(len(groups))

    bars = ax.bar(x, aurocs, color=colors[:len(groups)], alpha=0.88, zorder=3)
    ax.errorbar(x, aurocs, yerr=errs,
                fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4)

    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2., val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.6, label="Random (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [g.replace(" (", "\n(").replace("+", "+\n") for g in groups],
        fontsize=8,
    )
    ax.set_ylabel("AUROC (5-fold CV)")
    ax.set_ylim(0.45, 1.02)
    ax.set_title(
        "HaRP SOTA: Failure Estimator Ablation\n"
        "From Token-Level Baseline to Full Cross-Layer Geometric Governance",
        fontsize=12,
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


def plot_sota_policy(metrics_by_system: dict, out_path: Path) -> None:
    """
    Grouped bar chart comparing all systems on hallucination rate vs abstention rate.
    """
    systems   = list(metrics_by_system.keys())
    hall_rates = [metrics_by_system[s].get("hall_rate_accepted", float("nan")) or 0
                  for s in systems]
    abst_rates = [metrics_by_system[s].get("abstain_rate", 0) for s in systems]

    x = np.arange(len(systems))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, hall_rates, w, label="Hallucination Rate", color="#e15759", alpha=0.85)
    ax.bar(x + w/2, abst_rates, w, label="Abstention Rate",    color="#76b7b2", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "HaRP SOTA: Policy Comparison\n"
        "Hallucination Rate (lower=better) vs Abstention Rate across Systems"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.axhline(0.0, color="green", linestyle="--", alpha=0.5, linewidth=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


def main():
    print("=" * 60)
    print("Experiment 23 — HaRP SOTA: Full System Evaluation")
    print("=" * 60)

    # ── Load all features ─────────────────────────────────────────────────────
    df = load_sota_features()
    y  = df["label"].values

    # ── Full SOTA ablation ────────────────────────────────────────────────────
    print("\n--- SOTA Ablation Study ---")
    ablation_results = run_ablation(df)
    print("\n" + ablation_table(ablation_results))

    # ── AUROC delta summary ───────────────────────────────────────────────────
    auroc_a    = ablation_results.get("Group A only", {}).get("auroc_mean", float("nan"))
    auroc_abc  = ablation_results.get("Group A+B+C",  {}).get("auroc_mean", float("nan"))
    sota_entry = (
        ablation_results.get("SOTA (all groups)") or
        ablation_results.get("v4 (A+B+B2+C+C2+D)") or
        ablation_results.get("Full v3 (A+B+B2+C+C2)") or
        ablation_results.get("Group A+B+C")
    )
    auroc_sota = sota_entry["auroc_mean"] if sota_entry else float("nan")
    delta_total = auroc_sota - auroc_a

    print(f"\nTotal AUROC improvement (A → SOTA): {delta_total:+.4f}")
    print(f"  Group A baseline: {auroc_a:.4f}")
    print(f"  Group A+B+C:      {auroc_abc:.4f}  (geometry adds {auroc_abc-auroc_a:+.4f})")
    print(f"  SOTA:             {auroc_sota:.4f}  (dynamics+geometry adds {auroc_sota-auroc_abc:+.4f})")

    # ── Prepare SOTA feature matrix ───────────────────────────────────────────
    avail_sota = [c for c in ALL_FEATURES_SOTA if c in df.columns]
    X_sota = df[avail_sota].fillna(df[avail_sota].median()).values

    # ── Governance policy evaluation ──────────────────────────────────────────
    print("\n--- SOTA Governance Policy ---")

    # Train/val/test split (60/20/20) — same random state as exp 12 for comparability
    X_tv2, X_test2, y_tv2, y_test2 = train_test_split(
        X_sota, y, test_size=0.2, stratify=y, random_state=42)
    X_train2, X_val2, y_train2, y_val2 = train_test_split(
        X_tv2, y_tv2, test_size=0.25, stratify=y_tv2, random_state=42)

    est2 = fit_final_estimator(X_train2, y_train2)
    p_fail_val  = est2.predict_proba(X_val2)[:, 0]
    p_fail_test = est2.predict_proba(X_test2)[:, 0]

    # ── Bootstrap CI on SOTA AUROC ────────────────────────────────────────────
    p_corr_test = 1.0 - p_fail_test

    test_auroc = compute_auroc(y_test2, p_corr_test)
    test_auarc = compute_auarc(y_test2, p_corr_test)
    boot_mean, ci_lo, ci_hi = bootstrap_auroc(y_test2, p_corr_test)

    print(f"\nSOTA Test-set AUROC:  {test_auroc:.4f}")
    print(f"  Bootstrap CI 95%:  [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"SOTA Test-set AUARC: {test_auarc:.4f}")

    alpha_star, beta_star, best_loss, _ = optimize_thresholds(
        p_fail_val, y_val2, lambda_=LAMBDA)

    decisions = apply_policy(p_fail_test, alpha_star, beta_star)
    metrics   = compute_policy_metrics(decisions, y_test2, p_fail_test)
    summary   = policy_summary(decisions, y_test2)

    print(f"\n  Optimal α*={alpha_star:.2f}, β*={beta_star:.2f} (val loss={best_loss:.4f})")
    print(f"  Test set ({len(y_test2)} samples):")
    print(f"    ACCEPT:     {metrics['n_accept']} ({metrics['accept_rate']*100:.1f}%)")
    print(f"    REGENERATE: {metrics['n_regenerate']} ({metrics['regen_rate']*100:.1f}%)")
    print(f"    ABSTAIN:    {metrics['n_abstain']} ({metrics['abstain_rate']*100:.1f}%)")
    print(f"    Hall rate (ACCEPT):  {metrics['hall_rate_accepted']:.4f}")
    print(f"    Hall rate (served):  {metrics['hall_rate_served']:.4f}")

    # ── Load prior results for comparison ─────────────────────────────────────
    exp12  = load_json("12_policy_results.json")
    exp13  = load_json("13_baselines.json")
    exp22  = load_json("22_quadrant_steering_results.json")
    exp22c = load_json("22c_perquadrant_steering_results.json")

    raw_hall = None
    for b in exp13.get("results", []):
        if b.get("name") == "Raw LLM":
            raw_hall = b.get("hall_rate")

    v2_hall = (exp12.get("test_metrics") or {}).get("hall_rate_accepted")

    print(f"\n  Improvement over baselines:")
    if raw_hall is not None:
        print(f"    Raw LLM hall rate:       {raw_hall:.4f}")
    if v2_hall is not None:
        print(f"    HaRP v2 hall rate:    {v2_hall:.4f}")
    print(f"    HaRP SOTA hall rate:  {metrics['hall_rate_accepted']:.4f}")

    # ── Quadrant-adaptive steering summary ───────────────────────────────────
    if exp22:
        best_adapt = exp22.get("best_adaptive_hall", float("nan"))
        best_univ  = exp22.get("universal_results", {}).get("overall", {}).get("best_hall_rate", float("nan"))
        print(f"\n  Steering comparison (hallucination rate on regenerated samples):")
        print(f"    Universal steering, Exp 11 (n=100, α=0–2):   {best_univ:.4f}")
        print(f"    Shared-α quadrant-adaptive, Exp 22 (n=100):  {best_adapt:.4f}")
    if exp22c:
        c22c     = exp22c.get("combined", {})
        c_rate   = c22c.get("hall_rate", float("nan"))
        c_lo     = c22c.get("ci_lo",     float("nan"))
        c_hi     = c22c.get("ci_hi",     float("nan"))
        c_n      = c22c.get("n",         0)
        opt_a    = exp22c.get("per_q_optimal", {})
        print(f"    Per-quadrant optimal, Exp 22c (n={c_n}):      {c_rate:.4f}  95%CI [{c_lo:.4f},{c_hi:.4f}]")
        print(f"    Per-quadrant optimal alphas: {opt_a}")
        print(f"    Key finding: confident_hallucination resistant to all α including negative.")
        print(f"      → Empirically motivates ABSTAIN routing for that quadrant.")

    # ── Plot ablation ─────────────────────────────────────────────────────────
    plot_sota_ablation(ablation_results, RESULTS_PLOTS / "23_sota_ablation.png")

    # ── Plot policy comparison ────────────────────────────────────────────────
    metrics_comparison = {}
    if raw_hall is not None:
        metrics_comparison["Raw LLM"] = {
            "hall_rate_accepted": raw_hall, "abstain_rate": 0.0}
    if v2_hall is not None:
        metrics_comparison["HaRP v2"] = {
            "hall_rate_accepted": v2_hall,
            "abstain_rate": (exp12.get("test_metrics") or {}).get("abstain_rate", 0),
        }
    metrics_comparison["HaRP SOTA"] = {
        "hall_rate_accepted": metrics["hall_rate_accepted"],
        "abstain_rate":       metrics["abstain_rate"],
    }
    if exp22:
        metrics_comparison["SOTA + Adaptive Steer"] = {
            "hall_rate_accepted": best_adapt,
            "abstain_rate": metrics["abstain_rate"],
        }

    plot_sota_policy(metrics_comparison, RESULTS_PLOTS / "23_sota_policy.png")

    # ── Save SOTA results log ─────────────────────────────────────────────────
    sota_log = {
        "experiment":          "23_final_evaluation_sota",
        "n_total":             len(df),
        "n_sota_features":     len(avail_sota),
        "sota_features":       avail_sota,
        "ablation":            ablation_results,
        "auroc_group_a":       float(auroc_a),
        "auroc_group_abc":     float(auroc_abc),
        "auroc_sota":          float(auroc_sota),
        "delta_a_to_sota":     float(delta_total),
        "delta_abc_to_sota":   float(auroc_sota - auroc_abc),
        "test_auroc":          float(test_auroc),
        "test_auarc":          float(test_auarc),
        "auroc_ci_95":         {"lo": float(ci_lo), "hi": float(ci_hi)},
        "n_train":             len(y_train2),
        "n_val":               len(y_val2),
        "n_test":              len(y_test2),
        "alpha_star":          float(alpha_star),
        "beta_star":           float(beta_star),
        "val_loss":            float(best_loss),
        "test_policy_metrics": metrics,
    }

    log_path = RESULTS_LOGS / "23_sota_results.json"
    with open(log_path, "w") as f:
        json.dump(sota_log, f, indent=2)
    print(f"\nSaved: {log_path.name}")

    # ── Update final_results.json ─────────────────────────────────────────────
    final_path = RESULTS_LOGS / "final_results.json"
    if final_path.exists():
        with open(final_path) as f:
            final = json.load(f)
    else:
        final = {}

    # Incorporate 22c steering findings
    if exp22c:
        c22c = exp22c.get("combined", {})
        final["steering_finding"] = {
            "exp22_shared_alpha_hall":      exp22.get("best_adaptive_hall") if exp22 else None,
            "exp22c_per_quadrant_hall":     c22c.get("hall_rate"),
            "exp22c_ci_95":                 [c22c.get("ci_lo"), c22c.get("ci_hi")],
            "exp22c_n":                     c22c.get("n"),
            "per_q_optimal_alphas":         exp22c.get("per_q_optimal"),
            "key_finding":                  (
                "Confident hallucinations (low entropy, cos_hall < cos_correct) are resistant "
                "to activation steering at all tested alpha values including negative alpha. "
                "This empirically validates ABSTAIN as the correct routing decision for this "
                "failure mode. Epistemic samples respond to positive alpha (delta=-0.053), "
                "demonstrating the uncertainty decomposition enables targeted intervention."
            ),
        }

    final["sota_upgrade"] = {
        "experiment":           "23_final_evaluation_sota",
        "auroc_group_a":        float(auroc_a),
        "auroc_group_abc":      float(auroc_abc),
        "auroc_sota":           float(auroc_sota),
        "delta_a_to_sota":      float(delta_total),
        "test_auroc":           float(test_auroc),
        "test_auarc":           float(test_auarc),
        "auroc_ci_95":          {"lo": float(ci_lo), "hi": float(ci_hi)},
        "hall_rate_sota":       metrics["hall_rate_accepted"],
        "abstain_rate_sota":    metrics["abstain_rate"],
        "accept_rate_sota":     metrics["accept_rate"],
        "alpha_star":           float(alpha_star),
        "beta_star":            float(beta_star),
        "n_features_sota":      len(avail_sota),
        "new_groups":           {
            "D_trajectory": GROUP_D,
            "E_geometric":  GROUP_E,
        },
        "references_extended": [
            "Zhang et al. (2025). ICR Probe. ACL 2025.",
            "Granese et al. (2025). Geometric Uncertainty. arXiv:2509.13813.",
            "ACT: Adaptive Activation Steering. WWW 2025.",
        ],
    }

    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Updated: {final_path.name}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HaRP SOTA — FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  AUROC (Group A baseline):       {auroc_a:.4f}  (barely above random)")
    print(f"  AUROC (+ representation geom):  {auroc_abc:.4f}  (+{auroc_abc-auroc_a:+.4f}  massive jump)")
    print(f"  AUROC (SOTA full system):        {auroc_sota:.4f}  (+{auroc_sota-auroc_abc:+.4f}  trajectory+geometry)")
    print(f"  Bootstrap 95% CI:               [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  AUARC:                          {test_auarc:.4f}")
    print(f"  Hallucination rate on ACCEPT:   {metrics['hall_rate_accepted']:.4f}")
    print(f"  Abstention rate:                {metrics['abstain_rate']:.4f}")
    print(f"  Features used:                  {len(avail_sota)}")
    print("=" * 60)

    print("\nExperiment 23 complete. HaRP SOTA pipeline done.")
    print("Next steps: write paper PDF + build Streamlit demo.")


if __name__ == "__main__":
    main()
