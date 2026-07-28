"""
04_train_estimator_v1.py — Train failure estimator on Group A features only.

Phase 3, Experiment 4.
Trains Logistic Regression on the 6 token-level signals (Group A).
Evaluates with 5-fold stratified cross-validation.
Compares against two simple baselines:
  - Random baseline (AUROC = 0.50)
  - Static threshold: abstain if mean top-1 prob < 0.5

This establishes the AUROC floor before adding representation signals.
Expected AUROC: 0.60 – 0.67.

Output:
  results/logs/04_estimator_v1.json
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.authorization.failure_estimator import (
    GROUP_A, build_lr, evaluate_estimator
)
from src.evaluation.metrics import (
    compute_auroc, bootstrap_auroc, feature_importance_lr, ablation_table
)

DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_LOGS   = ROOT / "results" / "logs"
RESULTS_LOGS.mkdir(parents=True, exist_ok=True)


def static_threshold_auroc(df: pd.DataFrame, col: str = "top1_prob",
                            threshold: float = 0.5) -> float:
    """
    Baseline: predict hallucinated (0) if top1_prob < threshold.
    Returns AUROC of this rule.
    """
    if col not in df.columns:
        return float("nan")
    y_true = df["label"].values
    y_pred_prob = df[col].fillna(0.5).values  # treat NaN as uncertain
    return compute_auroc(y_true, y_pred_prob)


def main():
    print("=" * 60)
    print("Experiment 04 — Failure Estimator v1 (Group A only)")
    print("=" * 60)

    feat_path = DATA_PROCESSED / "features_groupAB.csv"
    if not feat_path.exists():
        print(f"Missing: {feat_path}\nRun 03_label_responses.py first.")
        sys.exit(1)

    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df)} samples from features_groupAB.csv")

    y = df["label"].values
    print(f"Label distribution: {int((y==1).sum())} correct / "
          f"{int((y==0).sum())} hallucinated ({100*(y==0).mean():.1f}% hall)")

    # ── Group A features ──────────────────────────────────────────────────────
    avail_a = [c for c in GROUP_A if c in df.columns]
    if not avail_a:
        print("No Group A features found. Ensure 03 ran correctly.")
        sys.exit(1)

    X_a = df[avail_a].fillna(df[avail_a].median()).values
    print(f"\nGroup A features ({len(avail_a)}): {avail_a}")

    nan_before = df[avail_a].isna().mean().mean()
    print(f"NaN rate before fill: {nan_before*100:.1f}%")

    # ── Train / test split (for final metrics + feature importance) ───────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_a, y, test_size=0.2, stratify=y, random_state=42)

    # ── Cross-validated evaluation ────────────────────────────────────────────
    print("\nCross-validation (5-fold stratified):")
    print("-" * 60)
    cv_result = evaluate_estimator(X_a, y, model_type="lr",
                                   label="Group A only (6 token signals)")
    print("-" * 60)

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\nBaselines:")
    rand_auroc = 0.5
    static_auroc = static_threshold_auroc(df)
    print(f"  Random baseline:       AUROC = 0.5000")
    print(f"  Static top1_prob > 0.5: AUROC = {static_auroc:.4f}")

    # ── Feature importance ────────────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    final_model = build_lr()
    final_model.fit(X_tr, y_tr)

    importances = feature_importance_lr(final_model, avail_a)
    print("\nFeature importances (by |coefficient|):")
    for item in importances:
        print(f"  {item['feature']:<20}  coef = {item['coefficient']:+.4f}")

    # ── Bootstrap CI on test set ──────────────────────────────────────────────
    y_prob_te = final_model.predict_proba(X_te)[:, 1]
    pt, lo, hi = bootstrap_auroc(y_te, y_prob_te)
    print(f"\nTest-set AUROC: {compute_auroc(y_te, y_prob_te):.4f} "
          f"(95% CI: {lo:.4f} – {hi:.4f})")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "experiment":    "04_estimator_v1",
        "n_samples":     len(df),
        "n_features":    len(avail_a),
        "features":      avail_a,
        "cv_auroc_mean": cv_result["auroc_mean"],
        "cv_auroc_std":  cv_result["auroc_std"],
        "cv_f1_mean":    cv_result["f1_mean"],
        "test_auroc":    float(compute_auroc(y_te, y_prob_te)),
        "test_auroc_ci": {"lo_95": lo, "hi_95": hi},
        "baselines": {
            "random":         rand_auroc,
            "static_top1":    float(static_auroc),
        },
        "feature_importances": importances,
    }

    log_path = RESULTS_LOGS / "04_estimator_v1.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {log_path.name}")

    print(f"\nGroup A AUROC baseline: {cv_result['auroc_mean']:.4f} "
          f"± {cv_result['auroc_std']:.4f}")
    print("Experiment 04 complete. Next: run 05_extract_hidden_states.py")


if __name__ == "__main__":
    main()
