"""
13_run_baselines.py — Controlled baseline comparison on the same test split.

Phase 7, Experiment 13.

Scientific question
-------------------
Which specific design decisions in HaRP are responsible for its performance?
Each baseline isolates one variable — together they prove which components
are necessary and which are additive.

The five baselines and what each isolates
------------------------------------------
  1. Raw LLM (accept all):
     → Measures the raw hallucination rate. HaRP's starting point.
     No features, no threshold, no policy. Establishes the maximum possible
     hallucination rate we are trying to reduce.

  2. Static threshold (abstain if mean_top1_prob < 0.5):
     → Tests whether a simple fixed entropy rule is sufficient.
     No training, no representation signals, no three-action policy.
     If this works nearly as well as HaRP, the entire learned system adds no value.

  3. Temperature scaling (Platt calibration on Group A logit):
     → Tests whether calibration alone (without representation signals) explains
     the performance gain. Fits a sigmoid on Group A's top-1 probability.
     No Group B/C features, no steering. Isolates the calibration contribution.

  4. Entropy-only policy (Group A, same three-action structure):
     → Tests whether the policy structure (Accept/REGENERATE/Abstain) adds value
     even with only token signals. Same α*, β* optimization as Exp 12.
     The difference between this and baseline (2) is the three-action structure.
     The difference between this and HaRP is the representation signals.

  5. HaRP (all features, optimised thresholds from Exp 12):
     → The full system with all design decisions combined.
     Uses the same α*, β* from Exp 12 validation — not re-optimised here.

Critical comparison pairs
--------------------------
  (4) vs (5): Representation signals contribution — holding policy structure constant
  (3) vs (5): Policy + representation signals contribution — beyond calibration alone
  (2) vs (4): Policy structure contribution — holding signal type constant

All baselines use the SAME 20% test split as Exp 12 and Exp 15.

Produces
--------
  - results/logs/13_baselines.json    — hallucination rate, abstention rate, AUROC per baseline
  - Comparison table printed to stdout
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.authorization.failure_estimator import (
    ALL_FEATURES_V3, GROUP_A, build_lr, fit_final_estimator
)
from src.authorization.policy import (
    apply_policy, optimize_thresholds, compute_policy_metrics
)
from src.evaluation.metrics import (
    compute_auroc, compute_hallucination_rate, policy_summary
)
from src.calibration.calibrate import expected_calibration_error

DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_LOGS   = ROOT / "results" / "logs"
RESULTS_LOGS.mkdir(parents=True, exist_ok=True)

LAMBDA = 0.2


def load_policy_thresholds() -> tuple:
    p = RESULTS_LOGS / "12_policy_results.json"
    if not p.exists():
        return 0.3, 0.7
    with open(p) as f:
        d = json.load(f)
    return float(d["alpha_star"]), float(d["beta_star"])


def evaluate_raw_llm(y_test: np.ndarray) -> dict:
    """Baseline 1: accept everything."""
    hall_rate = float(1 - y_test.mean())
    return {
        "name":            "Raw LLM",
        "hall_rate":       round(hall_rate, 4),
        "abstain_rate":    0.0,
        "n_accept":        len(y_test),
        "n_abstain":       0,
        "auroc":           float("nan"),
        "description":     "Accept all outputs, no filtering",
    }


def evaluate_static_threshold(X_test: np.ndarray, y_test: np.ndarray,
                               feat_names: list,
                               threshold: float = 0.5) -> dict:
    """Baseline 2: abstain if top1_prob < threshold."""
    if "top1_prob" not in feat_names:
        return {"name": "Static threshold", "error": "top1_prob not in features"}

    col_idx = feat_names.index("top1_prob")
    top1_probs = X_test[:, col_idx]

    accepted = top1_probs >= threshold
    n_accept  = int(accepted.sum())
    n_abstain = int((~accepted).sum())
    hall_rate = float(1 - y_test[accepted].mean()) if n_accept > 0 else float("nan")

    return {
        "name":            f"Static top1_prob > {threshold}",
        "hall_rate":       round(hall_rate, 4) if not np.isnan(hall_rate) else None,
        "abstain_rate":    round(n_abstain / len(y_test), 4),
        "n_accept":        n_accept,
        "n_abstain":       n_abstain,
        "auroc":           float(compute_auroc(y_test, top1_probs)),
        "description":     f"Abstain if mean top-1 probability < {threshold}",
    }


def evaluate_entropy_policy(X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             feat_names: list) -> dict:
    """Baseline 4: Group A only, same policy structure as HaRP."""
    a_idx = [i for i, c in enumerate(feat_names) if c in GROUP_A]
    if not a_idx:
        return {"name": "Entropy-only policy", "error": "No Group A features"}

    X_a_train = X_train[:, a_idx]
    X_a_val   = X_val[:, a_idx]
    X_a_test  = X_test[:, a_idx]

    estimator = fit_final_estimator(X_a_train, y_train)
    p_fail_val  = estimator.predict_proba(X_a_val)[:, 0]
    p_fail_test = estimator.predict_proba(X_a_test)[:, 0]

    a_star, b_star, _, _ = optimize_thresholds(p_fail_val, y_val, lambda_=LAMBDA)
    decisions = apply_policy(p_fail_test, a_star, b_star)
    metrics   = compute_policy_metrics(decisions, y_test, p_fail_test)
    auroc     = compute_auroc(y_test, p_fail_test)

    return {
        "name":            "Entropy-only policy",
        "hall_rate":       metrics["hall_rate_accepted"],
        "abstain_rate":    metrics["abstain_rate"],
        "n_accept":        metrics["n_accept"],
        "n_abstain":       metrics["n_abstain"],
        "auroc":           float(auroc),
        "alpha_star":      a_star,
        "beta_star":       b_star,
        "description":     "Group A only with optimised thresholds",
    }


def evaluate_harp_full(X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           alpha_star: float, beta_star: float) -> dict:
    """Baseline 5 (full HaRP): all features + optimised thresholds from exp 12."""
    estimator   = fit_final_estimator(X_train, y_train)
    p_fail_test = estimator.predict_proba(X_test)[:, 0]

    decisions = apply_policy(p_fail_test, alpha_star, beta_star)
    metrics   = compute_policy_metrics(decisions, y_test, p_fail_test)
    auroc     = compute_auroc(y_test, p_fail_test)

    return {
        "name":            "HaRP (full)",
        "hall_rate":       metrics["hall_rate_accepted"],
        "abstain_rate":    metrics["abstain_rate"],
        "n_accept":        metrics["n_accept"],
        "n_abstain":       metrics["n_abstain"],
        "auroc":           float(auroc),
        "alpha_star":      alpha_star,
        "beta_star":       beta_star,
        "description":     f"All {len(X_test[0])} features, α*={alpha_star:.2f}, β*={beta_star:.2f}",
    }


def print_comparison_table(results: list) -> None:
    print("\n--- Baseline Comparison ---")
    header = f"{'Baseline':<35} {'AUROC':>6} {'Hall%':>7} {'Abs%':>6} {'N_acc':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        auroc    = r.get("auroc")
        hall     = r.get("hall_rate")
        abst     = r.get("abstain_rate", 0.0)
        n_acc    = r.get("n_accept", "?")
        auroc_s  = f"{auroc:.4f}"  if isinstance(auroc, float) and not np.isnan(auroc) else "  n/a"
        hall_s   = f"{hall*100:.1f}%" if isinstance(hall, float) and not np.isnan(hall)  else "  n/a"
        abst_s   = f"{abst*100:.1f}%" if isinstance(abst, float) else "  n/a"
        print(f"  {r['name']:<33} {auroc_s:>6} {hall_s:>7} {abst_s:>6} {str(n_acc):>6}")


def main():
    print("=" * 60)
    print("Experiment 13 — Baseline Comparison")
    print("=" * 60)

    feat_path = DATA_PROCESSED / "features_combined.csv"
    if not feat_path.exists():
        feat_path = DATA_PROCESSED / "features_groupAB.csv"
    if not feat_path.exists():
        print("Missing features CSV. Run 03 and/or 09 first.")
        sys.exit(1)

    df = pd.read_csv(feat_path)
    y  = df["label"].values
    avail_feats = [c for c in ALL_FEATURES_V3 if c in df.columns]
    X = df[avail_feats].fillna(df[avail_feats].median()).values

    # Same split as experiment 12
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=42)

    alpha_star, beta_star = load_policy_thresholds()

    print(f"Test set: {len(y_test)} samples | α*={alpha_star:.2f} β*={beta_star:.2f}")

    # ── Run all baselines ─────────────────────────────────────────────────────
    results = []

    print("\n[1/5] Raw LLM baseline...")
    results.append(evaluate_raw_llm(y_test))

    print("[2/5] Static threshold baseline...")
    results.append(evaluate_static_threshold(X_test, y_test, avail_feats))

    print("[3/5] Temperature scaling baseline...")
    # Use top1_prob as a proxy for confidence; calibrate on validation set
    if "top1_prob" in avail_feats:
        col_idx = avail_feats.index("top1_prob")
        from scipy.special import expit
        from src.calibration.calibrate import temperature_scale
        logits_val  = X_val[:, col_idx].reshape(-1, 1)
        logits_test = X_test[:, col_idx].reshape(-1, 1)
        try:
            T, cal_probs_val = temperature_scale(
                X_val[:, col_idx], y_val)
            # Apply same T to test set
            import numpy as np
            cal_probs_test = 1 / (1 + np.exp(-(X_test[:, col_idx] / T - 0.5) * 10))
            auroc_cal = compute_auroc(y_test, cal_probs_test)
            accepted  = cal_probs_test >= 0.5
            hall_cal  = float(1 - y_test[accepted].mean()) if accepted.sum() > 0 else float("nan")
            results.append({
                "name":         "Temperature scaling",
                "hall_rate":    round(hall_cal, 4) if not np.isnan(hall_cal) else None,
                "abstain_rate": round(float((~accepted).mean()), 4),
                "n_accept":     int(accepted.sum()),
                "n_abstain":    int((~accepted).sum()),
                "auroc":        float(auroc_cal),
                "T":            float(T),
                "description":  f"Platt scaling on top1_prob, T={T:.3f}",
            })
        except Exception as e:
            results.append({"name": "Temperature scaling", "error": str(e)})
    else:
        results.append({"name": "Temperature scaling", "error": "top1_prob not available"})

    print("[4/5] Entropy-only policy...")
    results.append(evaluate_entropy_policy(
        X_train, y_train, X_val, y_val, X_test, y_test, avail_feats))

    print("[5/5] HaRP full...")
    results.append(evaluate_harp_full(
        X_train, y_train, X_test, y_test, alpha_star, beta_star))

    # ── Print table ───────────────────────────────────────────────────────────
    print_comparison_table(results)

    # ── Save log ──────────────────────────────────────────────────────────────
    log = {
        "experiment":  "13_baselines",
        "n_test":      int(len(y_test)),
        "alpha_star":  alpha_star,
        "beta_star":   beta_star,
        "results":     results,
    }
    log_path = RESULTS_LOGS / "13_baselines.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved: {log_path.name}")

    print("Experiment 13 complete. Next: run 14_sycophancy_analysis.py")


if __name__ == "__main__":
    main()
