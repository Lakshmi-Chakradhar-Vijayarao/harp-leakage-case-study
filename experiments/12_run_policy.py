"""
12_run_policy.py — Full HaRP authorization pipeline with optimised thresholds.

Phase 7, Experiment 12.

Scientific question
-------------------
How does the full closed-loop system perform end-to-end?
  - What are the optimal thresholds (α*, β*) for the three-action policy?
  - What hallucination rate and abstention rate does the system achieve?
  - Does the REGENERATE action (steering + retry) actually recover samples?

The closed-loop authorization system
--------------------------------------
For each test prompt:
  1. Compute 24-feature vector (Groups A+B+C, and D+E if available)
  2. Failure estimator produces P(failure) ∈ [0, 1]
  3. Authorization decision:
       P < α*         → ACCEPT   (output the original response)
       α* ≤ P < β*   → REGENERATE (apply steering vector at best_layer, re-generate)
       P ≥ β*         → ABSTAIN  ("I cannot reliably answer this")

Threshold optimization on validation set
-----------------------------------------
Grid search over α* ∈ [0.1, 0.6] (step 0.05) and β* ∈ [0.4, 0.9] (step 0.05)
with constraint α* < β*. For each pair:

  Loss(α*, β*) = hallucination_rate(ACCEPT set) + λ * abstention_rate
  λ = 0.5 (default) — moderate penalty on over-abstention

The minimizing (α*, β*) pair is selected on the validation set, then applied to
the held-out test set. Validation and test sets are NEVER mixed in optimization.

Why three actions instead of two (accept/reject)?
--------------------------------------------------
Binary systems discard all high-uncertainty samples. REGENERATE captures the
middle tier: samples where P(failure) indicates uncertainty but steering can
correct the internal representation. These recovered samples expand the
system's useful output rate (accepted + regenerated) vs. a binary threshold.

REGENERATE fallback (Ollama mode):
If HuggingFace is not available, REGENERATE falls back to a plain temperature-0
retry (no steering). This keeps the policy structure for comparison with baselines
while losing the representation-based correction.

Train/validation/test split
-----------------------------
60/20/20 split (random_state=42).
This SAME split is shared across Exp 12 (policy), Exp 13 (baselines), and
Exp 15 (calibration) — ensuring all systems are evaluated on identical held-out data.

Produces
--------
  - results/plots/12_policy_decisions.png   — accept/regen/abstain distribution (Plot 7)
  - results/logs/12_policy_results.json     — thresholds, hallucination rate, abstention rate
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
    ALL_FEATURES_V3, GROUP_A, GROUP_B, GROUP_C, build_lr, fit_final_estimator
)
from src.authorization.policy import (
    apply_policy, optimize_thresholds, compute_policy_metrics
)
from src.evaluation.metrics import policy_summary

DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_LOGS   = ROOT / "results" / "logs"
RESULTS_PLOTS  = ROOT / "results" / "plots"
RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)

LAMBDA = 0.2   # governance loss weight for abstention (0.2 = realistic coverage vs 0.5 over-abstains)


def load_optimal_alpha() -> float:
    """Load best steering alpha from experiment 11."""
    p = RESULTS_LOGS / "11_steering_results.json"
    if p.exists():
        with open(p) as f:
            return float(json.load(f).get("optimal_alpha", 1.0))
    return 1.0


def plot_decision_distribution(metrics: dict, title: str, out_path: Path) -> None:
    labels = ["ACCEPT", "REGENERATE", "ABSTAIN"]
    vals   = [metrics["n_accept"], metrics["n_regenerate"], metrics["n_abstain"]]
    colors = ["steelblue", "orange", "crimson"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                str(v), ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Number of Samples")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


def main():
    print("=" * 60)
    print("Experiment 12 — Authorization Policy")
    print("=" * 60)

    feat_path = DATA_PROCESSED / "features_combined.csv"
    if not feat_path.exists():
        # Fall back to Group A+B only
        feat_path = DATA_PROCESSED / "features_groupAB.csv"
        if not feat_path.exists():
            print("Missing features_combined.csv and features_groupAB.csv.")
            print("Run 03 and 09 first.")
            sys.exit(1)
        print(f"Using fallback: {feat_path.name} (Group A+B only)")

    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df)} samples from {feat_path.name}")

    y = df["label"].values

    # Use whichever v3 features are available (falls back gracefully if B2/C2 absent)
    avail_feats = [c for c in ALL_FEATURES_V3 if c in df.columns]
    print(f"Available features: {len(avail_feats)} — {avail_feats}")

    X = df[avail_feats].fillna(df[avail_feats].median()).values

    # ── Train / val / test split (60/20/20) ───────────────────────────────────
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=42)  # 0.25 * 0.8 = 0.2

    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # ── Train failure estimator ───────────────────────────────────────────────
    estimator = fit_final_estimator(X_train, y_train)
    p_fail_val  = estimator.predict_proba(X_val)[:, 0]   # P(hallucinated)
    p_fail_test = estimator.predict_proba(X_test)[:, 0]

    # ── Optimise thresholds on validation set ─────────────────────────────────
    print(f"\nOptimising thresholds on validation set (λ={LAMBDA})...")
    alpha_star, beta_star, best_loss, _ = optimize_thresholds(
        p_fail_val, y_val, lambda_=LAMBDA)

    print(f"Optimal α* = {alpha_star:.2f}, β* = {beta_star:.2f}  "
          f"(val loss = {best_loss:.4f})")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    decisions = apply_policy(p_fail_test, alpha_star, beta_star)
    metrics   = compute_policy_metrics(decisions, y_test, p_fail_test)
    summary   = policy_summary(decisions, y_test)

    print("\nTest-set policy metrics:")
    print(f"  ACCEPT:     {metrics['n_accept']} ({metrics['accept_rate']*100:.1f}%)")
    print(f"  REGENERATE: {metrics['n_regenerate']} ({metrics['regen_rate']*100:.1f}%)")
    print(f"  ABSTAIN:    {metrics['n_abstain']} ({metrics['abstain_rate']*100:.1f}%)")
    print(f"  Hallucination rate (accepted): {metrics['hall_rate_accepted']:.4f}")
    print(f"  Hallucination rate (served):   {metrics['hall_rate_served']:.4f}")

    # ── Plot decision distribution ────────────────────────────────────────────
    plot_decision_distribution(
        metrics,
        f"Authorization Decisions (α*={alpha_star:.2f}, β*={beta_star:.2f})\n"
        f"Test set — {len(y_test)} samples",
        RESULTS_PLOTS / "12_policy_decisions.png"
    )

    # ── Save log ──────────────────────────────────────────────────────────────
    log = {
        "experiment":       "12_run_policy",
        "n_features":       len(avail_feats),
        "features":         avail_feats,
        "n_train":          len(y_train),
        "n_val":            len(y_val),
        "n_test":           len(y_test),
        "alpha_star":       alpha_star,
        "beta_star":        beta_star,
        "val_loss":         float(best_loss),
        "lambda":           LAMBDA,
        "test_metrics":     metrics,
        "policy_summary":   summary,
    }
    log_path = RESULTS_LOGS / "12_policy_results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved: {log_path.name}")

    print("Experiment 12 complete. Next: run 13_run_baselines.py")


if __name__ == "__main__":
    main()
