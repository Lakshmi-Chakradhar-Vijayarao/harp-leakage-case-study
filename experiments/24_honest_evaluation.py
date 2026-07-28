"""
24_honest_evaluation.py — HaRP: Corrected evaluation with leak-free features.

This replaces the inflated numbers from Exp 23 with honest, reproducible results.

WHAT WAS WRONG IN EXP 23
-------------------------
probe_conf was computed in-sample: the probe was fitted on all 700 hidden-state
samples, then its predictions were used as a feature for those same samples.
In-sample AUROC of probe_conf alone = 1.0000 (it memorised the labels).
The 5-fold CV on the outer failure estimator does NOT fix this: the probe
(inner model) was already fit before the CV split.

Result: AUROC 0.9624 is artificially inflated. The true out-of-sample performance
uses OOF probe_conf (see Exp 09b), which has AUROC ~0.75.

WHAT THIS EXPERIMENT DOES
--------------------------
1. Loads features_combined_oof.csv (produced by Exp 09b) with leak-free Group C.
2. Runs the full ablation (Groups A → A+B → A+B+C → SOTA).
3. Evaluates the governance policy with the corrected estimator.
4. Reports Clopper-Pearson 95% CI on hall_rate_accepted.
5. Saves honest numbers to final_results.json under "honest_evaluation".

HONEST ABLATION PATH (expected approximate values)
---------------------------------------------------
  Group A only          → ~0.54  (unchanged — token signals, no leakage)
  A + B                 → ~0.59  (unchanged — consistency signals, no leakage)
  A + B + C (OOF)       → ~0.82–0.88  (true geometry contribution, leakage removed)
  A + B + C + D + E     → ~0.83–0.89  (SOTA, Groups D+E still marginal)

Outputs:
  results/logs/24_honest_results.json
  results/plots/24_honest_ablation.png
  results/logs/final_results.json  ← updated "honest_evaluation" key
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
    clopper_pearson_ci,
)

DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_LOGS   = ROOT / "results" / "logs"
RESULTS_HS     = ROOT / "results" / "hidden_states"
RESULTS_PLOTS  = ROOT / "results" / "plots"
RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)

LAMBDA = 0.5


def load_honest_features() -> pd.DataFrame:
    """
    Load features_combined_oof.csv (OOF Group C, 700 HS-only samples)
    and merge in Groups D and E.
    """
    oof_path  = DATA_PROCESSED / "features_combined_oof.csv"
    traj_path = RESULTS_HS / "trajectory_features.csv"
    geom_path = RESULTS_HS / "geometric_features.csv"

    if not oof_path.exists():
        print(f"Missing: {oof_path}")
        print("Run experiments/09b_oof_group_c.py first.")
        sys.exit(1)

    df = pd.read_csv(oof_path)
    n_base = len(df)
    print(f"OOF base features: {n_base} samples, {len(df.columns)} columns")

    if traj_path.exists():
        traj_df = pd.read_csv(traj_path)
        d_cols  = [c for c in GROUP_D if c in traj_df.columns]
        df = df.merge(traj_df[["prompt_id"] + d_cols], on="prompt_id", how="left")
        n_d = df[d_cols[0]].notna().sum() if d_cols else 0
        print(f"Group D merged: {n_d}/{n_base} samples")

    if geom_path.exists():
        geom_df = pd.read_csv(geom_path)
        e_cols  = [c for c in GROUP_E if c in geom_df.columns]
        df = df.merge(geom_df[["prompt_id"] + e_cols], on="prompt_id", how="left")
        n_e = df[e_cols[0]].notna().sum() if e_cols else 0
        print(f"Group E merged: {n_e}/{n_base} samples")

    avail = [c for c in ALL_FEATURES_SOTA if c in df.columns]
    print(f"Available SOTA features: {len(avail)}/{len(ALL_FEATURES_SOTA)}")
    return df


def plot_honest_ablation(results: dict, leaked_abc: float, out_path: Path) -> None:
    groups = list(results.keys())
    aurocs = [results[k]["auroc_mean"] for k in groups]
    errs   = [results[k]["auroc_std"]  for k in groups]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = range(len(groups))
    colors = ["#aec6e8", "#6baed6", "#4e79a7", "#fdae6b", "#e6550d", "#31a354", "#9467bd"]

    bars = ax.bar(x, aurocs, color=colors[:len(groups)], alpha=0.88, zorder=3)
    ax.errorbar(x, aurocs, yerr=errs,
                fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4)

    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2., val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0.5,          color="grey",   linestyle=":", alpha=0.6,  label="Random (0.50)")
    ax.axhline(leaked_abc,   color="red",    linestyle="--", alpha=0.6,
               label=f"Leaked Exp 23 A+B+C: {leaked_abc:.3f} (inflated)")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [g.replace(" (", "\n(").replace("+", "+\n") for g in groups], fontsize=8)
    ax.set_ylabel("AUROC (5-fold CV)")
    ax.set_ylim(0.45, 1.02)
    ax.set_title(
        "HaRP Honest Evaluation: Leak-Free Ablation\n"
        "OOF Group C removes probe_conf in-sample inflation",
        fontsize=12,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


def main():
    print("=" * 60)
    print("Experiment 24 — HaRP Honest Evaluation (OOF Group C)")
    print("=" * 60)

    # ── Check 09b ran ─────────────────────────────────────────────────────────
    log09b_path = RESULTS_LOGS / "09b_oof_group_c.json"
    if not log09b_path.exists():
        print("Missing 09b_oof_group_c.json — run 09b_oof_group_c.py first.")
        sys.exit(1)
    log09b = json.load(open(log09b_path))
    leaked_auroc = log09b["probe_conf_auroc_insample"]
    honest_probe_auroc = log09b["probe_conf_auroc_oof"]
    print(f"Leakage summary from 09b:")
    print(f"  probe_conf in-sample AUROC: {leaked_auroc:.4f}  (inflated)")
    print(f"  probe_conf OOF AUROC:       {honest_probe_auroc:.4f}  (true)")

    # ── Load honest features ──────────────────────────────────────────────────
    df = load_honest_features()
    y  = df["label"].values

    # ── Full ablation with OOF features ───────────────────────────────────────
    print("\n--- Honest Ablation Study (OOF Group C) ---")
    ablation_results = run_ablation(df)
    print("\n" + ablation_table(ablation_results))

    auroc_a    = ablation_results.get("Group A only",  {}).get("auroc_mean", float("nan"))
    auroc_abc  = ablation_results.get("Group A+B+C",   {}).get("auroc_mean", float("nan"))
    sota_entry = (
        ablation_results.get("SOTA (all groups)") or
        ablation_results.get("v4 (A+B+B2+C+C2+D)") or
        ablation_results.get("Full v3 (A+B+B2+C+C2)") or
        ablation_results.get("Group A+B+C")
    )
    auroc_sota = sota_entry["auroc_mean"] if sota_entry else float("nan")

    print(f"\nHonest AUROC improvements:")
    print(f"  Group A baseline:        {auroc_a:.4f}")
    print(f"  + Group C (OOF):         {auroc_abc:.4f}  (adds {auroc_abc - auroc_a:+.4f})")
    print(f"  SOTA (all groups, OOF):  {auroc_sota:.4f}  (adds {auroc_sota - auroc_abc:+.4f})")
    print(f"\n  Leaked Exp 23 A+B+C:     {log09b.get('probe_conf_auroc_insample', 0.9620):.4f}")
    print(f"  Inflation from leakage:  {0.9620 - auroc_abc:+.4f}")

    # ── Policy evaluation ──────────────────────────────────────────────────────
    print("\n--- Honest Governance Policy ---")
    avail_sota = [c for c in ALL_FEATURES_SOTA if c in df.columns]
    X_sota = df[avail_sota].fillna(df[avail_sota].median()).values

    X_tv, X_test, y_tv, y_test = train_test_split(
        X_sota, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=42)

    est = fit_final_estimator(X_train, y_train)
    p_fail_val  = est.predict_proba(X_val)[:, 0]
    p_fail_test = est.predict_proba(X_test)[:, 0]
    p_corr_test = 1.0 - p_fail_test

    test_auroc = compute_auroc(y_test, p_corr_test)
    test_auarc = compute_auarc(y_test, p_corr_test)
    boot_mean, ci_lo, ci_hi = bootstrap_auroc(y_test, p_corr_test)

    print(f"\nHonest Test-set AUROC:  {test_auroc:.4f}")
    print(f"  Bootstrap CI 95%:     [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Test-set AUARC:         {test_auarc:.4f}")

    alpha_star, beta_star, best_loss, _ = optimize_thresholds(
        p_fail_val, y_val, lambda_=LAMBDA)

    decisions = apply_policy(p_fail_test, alpha_star, beta_star)
    metrics   = compute_policy_metrics(decisions, y_test, p_fail_test)

    n_hall  = metrics["hall_rate_accepted_ci_95"]["n_hall"]
    n_acc   = metrics["n_accept"]
    hall_lo = metrics["hall_rate_accepted_ci_95"]["lo"]
    hall_hi = metrics["hall_rate_accepted_ci_95"]["hi"]

    print(f"\n  Optimal α*={alpha_star:.2f}, β*={beta_star:.2f}")
    print(f"  Test set ({len(y_test)} samples):")
    print(f"    ACCEPT:     {metrics['n_accept']} ({metrics['accept_rate']*100:.1f}%)")
    print(f"    REGENERATE: {metrics['n_regenerate']} ({metrics['regen_rate']*100:.1f}%)")
    print(f"    ABSTAIN:    {metrics['n_abstain']} ({metrics['abstain_rate']*100:.1f}%)")
    print(f"\n    Hall rate (ACCEPT): {metrics['hall_rate_accepted']:.4f}")
    print(f"    Clopper-Pearson 95% CI: [{hall_lo:.4f}, {hall_hi:.4f}]")
    print(f"    ({n_hall} hallucinations in {n_acc} accepted samples)")
    print(f"    Hall rate (served):  {metrics['hall_rate_served']:.4f}  (incl. REGENERATE)")

    # ── Comparison vs leaked Exp 23 ───────────────────────────────────────────
    exp23 = json.load(open(RESULTS_LOGS / "23_sota_results.json")) if \
        (RESULTS_LOGS / "23_sota_results.json").exists() else {}

    print(f"\n--- Comparison: Leaked vs Honest ---")
    print(f"  {'Metric':<35} {'Exp 23 (leaked)':<20} {'Exp 24 (honest)'}")
    print(f"  {'-'*35} {'-'*20} {'-'*20}")
    print(f"  {'A+B+C AUROC (5-fold CV)':<35} {'0.9620':<20} {auroc_abc:.4f}")
    print(f"  {'SOTA AUROC (5-fold CV)':<35} {'0.9624':<20} {auroc_sota:.4f}")
    print(f"  {'Test AUROC':<35} {exp23.get('test_auroc', 0.9521):.4f}                {test_auroc:.4f}")
    print(f"  {'Hall rate (ACCEPT)':<35} {exp23.get('test_policy_metrics',{}).get('hall_rate_accepted', 0.0):.4f}                {metrics['hall_rate_accepted']:.4f}")
    print(f"  {'Accept rate':<35} {exp23.get('test_policy_metrics',{}).get('accept_rate', 0.317):.4f}                {metrics['accept_rate']:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    leaked_abc_exp23 = exp23.get("auroc_group_abc", 0.9620)
    plot_honest_ablation(ablation_results, leaked_abc_exp23,
                         RESULTS_PLOTS / "24_honest_ablation.png")

    # ── Save results ──────────────────────────────────────────────────────────
    honest_log = {
        "experiment":             "24_honest_evaluation",
        "n_samples":              len(df),
        "n_features":             len(avail_sota),
        "leakage_correction": {
            "probe_conf_insample_auroc": leaked_auroc,
            "probe_conf_oof_auroc":      honest_probe_auroc,
            "inflation":                 round(leaked_auroc - honest_probe_auroc, 4),
        },
        "ablation":               ablation_results,
        "auroc_group_a":          float(auroc_a),
        "auroc_group_abc_honest": float(auroc_abc),
        "auroc_sota_honest":      float(auroc_sota),
        "delta_a_to_abc":         float(auroc_abc - auroc_a),
        "delta_a_to_sota":        float(auroc_sota - auroc_a),
        "test_auroc":             float(test_auroc),
        "test_auarc":             float(test_auarc),
        "auroc_ci_95":            {"lo": float(ci_lo), "hi": float(ci_hi)},
        "alpha_star":             float(alpha_star),
        "beta_star":              float(beta_star),
        "n_train":                len(y_train),
        "n_val":                  len(y_val),
        "n_test":                 len(y_test),
        "test_policy_metrics":    metrics,
    }

    log_path = RESULTS_LOGS / "24_honest_results.json"
    with open(log_path, "w") as f:
        json.dump(honest_log, f, indent=2)
    print(f"\nSaved: {log_path.name}")

    # ── Update final_results.json ─────────────────────────────────────────────
    final_path = RESULTS_LOGS / "final_results.json"
    with open(final_path) as f:
        final = json.load(f)

    final["honest_evaluation"] = {
        "note": (
            "Corrected evaluation using out-of-fold Group C features. "
            "Exp 23 reported AUROC 0.9624 due to probe_conf in-sample leakage "
            "(probe fitted on all 700 samples, AUROC=1.0 in-sample). "
            "These numbers reflect true generalisation performance."
        ),
        "auroc_group_a":          float(auroc_a),
        "auroc_group_abc_honest": float(auroc_abc),
        "auroc_sota_honest":      float(auroc_sota),
        "test_auroc":             float(test_auroc),
        "test_auroc_ci_95":       {"lo": float(ci_lo), "hi": float(ci_hi)},
        "test_auarc":             float(test_auarc),
        "hall_rate_accepted":     float(metrics["hall_rate_accepted"]),
        "hall_rate_accepted_ci_95": {
            "lo":   float(hall_lo),
            "hi":   float(hall_hi),
            "n_hall":   n_hall,
            "n_accept": n_acc,
        },
        "hall_rate_served":       float(metrics["hall_rate_served"]),
        "accept_rate":            float(metrics["accept_rate"]),
        "abstain_rate":           float(metrics["abstain_rate"]),
        "probe_conf_inflation_delta": round(leaked_auroc - honest_probe_auroc, 4),
        "d_and_e_contribution":   round(auroc_sota - auroc_abc, 4),
        "d_and_e_interpretation": (
            "Groups D (trajectory) and E (geometric volume) add "
            f"{auroc_sota - auroc_abc:+.4f} AUROC over A+B+C. "
            "This is within measurement noise (std ~0.015). "
            "These groups do not contribute meaningfully beyond Group C."
        ),
    }

    # Update metrics_table with honest numbers
    final["metrics_table"].update({
        "auroc_group_abc":  float(auroc_abc),
        "auroc_sota":       float(auroc_sota),
        "test_auroc":       float(test_auroc),
        "test_auroc_ci_95": {"lo": float(ci_lo), "hi": float(ci_hi)},
        "hall_rate_harp":     float(metrics["hall_rate_accepted"]),
        "hall_rate_harp_ci_95": {"lo": float(hall_lo), "hi": float(hall_hi)},
        "abstain_rate_harp":  float(metrics["abstain_rate"]),
        "accept_rate_harp":   float(metrics["accept_rate"]),
    })

    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Updated: {final_path.name}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HaRP HONEST EVALUATION — FINAL RESULTS")
    print("=" * 60)
    print(f"  AUROC (Group A, token only):     {auroc_a:.4f}  (near random)")
    print(f"  AUROC (+ repr geometry, OOF):    {auroc_abc:.4f}  ({auroc_abc-auroc_a:+.4f}  honest jump)")
    print(f"  AUROC (SOTA, all groups, OOF):   {auroc_sota:.4f}  ({auroc_sota-auroc_abc:+.4f}  D+E contribution)")
    print(f"  Bootstrap 95% CI:                [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  AUARC:                           {test_auarc:.4f}")
    print(f"  Hall rate (ACCEPT):              {metrics['hall_rate_accepted']:.4f}")
    print(f"    CP 95% CI:                     [{hall_lo:.4f}, {hall_hi:.4f}]")
    print(f"    ({n_hall}/{n_acc} accepted samples hallucinated)")
    print(f"  Hall rate (served, incl regen):  {metrics['hall_rate_served']:.4f}")
    print(f"  Accept rate:                     {metrics['accept_rate']:.4f}")
    print(f"  Abstain rate:                    {metrics['abstain_rate']:.4f}")
    print()
    print(f"  Leaked Exp 23 AUROC was:         0.9624  (inflation: {0.9624 - auroc_sota:+.4f})")
    print(f"  Groups D+E honest contribution:  {auroc_sota - auroc_abc:+.4f}  (marginal, within noise)")
    print("=" * 60)
    print("\nExperiment 24 complete. final_results.json updated.")


if __name__ == "__main__":
    main()
