"""
15_calibration.py — Probability calibration analysis: ECE and reliability diagrams.

Phase 9, Experiment 15.

Scientific question
-------------------
When HaRP says P(correct) = 0.8, is it actually correct 80% of the time?
A well-calibrated failure estimator is REQUIRED for the authorization policy to work
as designed. Miscalibrated probabilities mean the thresholds (α*, β*) from Exp 12
do not correspond to actual error rates — the policy makes decisions based on
confidence values that do not reflect real-world accuracy.

Expected Calibration Error (ECE)
----------------------------------
  ECE = Σ_b (|B_b| / N) × |accuracy(B_b) - confidence(B_b)|

Samples are grouped into M=10 equally-spaced confidence bins [0,0.1), [0.1,0.2), ...
For each bin b: compute the fraction of correct predictions (accuracy) and the mean
predicted confidence. If accuracy ≈ confidence for all bins: ECE ≈ 0 (perfectly calibrated).

ECE interpretation:
  ECE < 0.05  : well-calibrated — policy thresholds can be trusted
  ECE 0.05–0.10: moderately calibrated — thresholds need slight adjustment
  ECE > 0.10  : poorly calibrated — P(failure) values are misleading

The reliability diagram visualizes this: a calibrated system plots on the diagonal.
Points above the diagonal: model is overconfident (predicts higher confidence than accuracy).
Points below the diagonal: model is underconfident (predicts lower confidence than accuracy).

Three systems compared
-----------------------
  1. Raw LLM (top1_prob as confidence):
     The model's own probability estimate. Expected to be overconfident — Qwen 2.5 3B
     is instruction-tuned to produce confident outputs regardless of correctness.

  2. Temperature scaling (Platt calibration on Group A):
     Fits a sigmoid (T parameter) on the Group A logit to recalibrate top1_prob.
     This is the standard single-parameter calibration method (Guo et al. 2017).
     Expected: better than raw but still limited by the quality of Group A signals.

  3. HaRP failure estimator (full 11+ features):
     The LR-based P(failure) trained on the multi-signal feature set.
     Expected: best ECE, because the richer feature set provides a more accurate
     posterior P(correct | features) than top1_prob alone.

Why HaRP should have lower ECE than temperature scaling
----------------------------------------------------------
Temperature scaling only recalibrates the magnitude of Group A's logit. It cannot
correct the fundamental problem: Group A alone (entropy, top1_prob) is insufficient
to distinguish confident hallucination from confident correct. The multi-signal
estimator can make finer-grained distinctions, leading to probability estimates
that more accurately reflect actual correctness rates.

Produces
--------
  - results/plots/15_reliability_diagram.png   — reliability diagrams, 3 systems (Plot 8)
  - results/logs/15_calibration_results.json   — ECE per system, bin statistics
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

from src.authorization.failure_estimator import ALL_FEATURES_V3, fit_final_estimator
from src.calibration.calibrate import (
    expected_calibration_error, temperature_scale, print_calibration_summary
)

DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_LOGS   = ROOT / "results" / "logs"
RESULTS_PLOTS  = ROOT / "results" / "plots"
RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)


def plot_reliability_diagram(systems: dict, out_path: Path) -> None:
    """
    Plot reliability diagrams for multiple systems.
    systems = {name: (y_true, y_prob, ece)}
    """
    colors = ["#e15759", "#f28e2b", "#4e79a7"]
    markers = ["s", "^", "o"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    for (name, (y_true, y_prob, ece)), color, marker in zip(
            systems.items(), colors, markers):
        _, bins = expected_calibration_error(y_true, y_prob, n_bins=10)
        bin_conf = [b["confidence"] for b in bins if b["confidence"] is not None]
        bin_acc  = [b["accuracy"]   for b in bins if b["accuracy"]   is not None]
        ax.plot(bin_conf, bin_acc, f"-{marker}", color=color,
                label=f"{name} (ECE={ece:.4f})", markersize=7)

    ax.set_xlabel("Mean Predicted Probability (Confidence)")
    ax.set_ylabel("Fraction of Correct Answers (Accuracy)")
    ax.set_title("Reliability Diagram\n"
                 "Diagonal = perfect calibration | Below = overconfident")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


def main():
    print("=" * 60)
    print("Experiment 15 — Calibration (ECE + Reliability Diagram)")
    print("=" * 60)

    feat_path = DATA_PROCESSED / "features_combined.csv"
    if not feat_path.exists():
        feat_path = DATA_PROCESSED / "features_groupAB.csv"
    if not feat_path.exists():
        print("Missing features CSV. Run 03/09 first.")
        sys.exit(1)

    df = pd.read_csv(feat_path)
    y  = df["label"].values
    avail_feats = [c for c in ALL_FEATURES_V3 if c in df.columns]
    X = df[avail_feats].fillna(df[avail_feats].median()).values

    # Same split as experiment 12/13
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_tv, _, y_tv, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=42)

    print(f"Test set: {len(y_test)} samples")

    # ── System 1: Raw LLM ─────────────────────────────────────────────────────
    raw_prob = None
    if "top1_prob" in avail_feats:
        col_idx  = avail_feats.index("top1_prob")
        raw_prob = X_test[:, col_idx]
        ece_raw, _ = expected_calibration_error(y_test, raw_prob)
        print(f"\n[Raw LLM]  ECE = {ece_raw:.5f}")
    else:
        raw_prob = np.full(len(y_test), 0.5)
        ece_raw  = float("nan")
        print("\n[Raw LLM]  top1_prob not available — using 0.5 placeholder")

    # ── System 2: Temperature scaling ────────────────────────────────────────
    try:
        T_star, cal_probs_val = temperature_scale(X_val[:, avail_feats.index("top1_prob")], y_val)
        cal_probs_test = 1 / (1 + np.exp(-(X_test[:, avail_feats.index("top1_prob")] / T_star - 0.5) * 10))
        ece_temp, _ = expected_calibration_error(y_test, cal_probs_test)
        print(f"[Temp. scaling]  T*={T_star:.3f}  ECE = {ece_temp:.5f}")
    except Exception as e:
        cal_probs_test = raw_prob.copy() if raw_prob is not None else np.full(len(y_test), 0.5)
        ece_temp = float("nan")
        print(f"[Temp. scaling]  Failed: {e}")

    # ── System 3: HaRP (full) ──────────────────────────────────────────────
    estimator   = fit_final_estimator(X_train, y_train)
    p_fail_test = estimator.predict_proba(X_test)[:, 0]     # P(hallucinated)
    p_corr_test = 1.0 - p_fail_test                         # P(correct)
    ece_gov, _  = expected_calibration_error(y_test, p_corr_test)
    print(f"[HaRP full]  ECE = {ece_gov:.5f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print_calibration_summary({
        "Raw LLM":          (y_test, raw_prob),
        "Temperature scaling": (y_test, cal_probs_test),
        "HaRP (full)":   (y_test, p_corr_test),
    })

    # ── Plot ──────────────────────────────────────────────────────────────────
    systems_plot = {
        "Raw LLM":             (y_test, raw_prob,        ece_raw),
        "Temperature scaling": (y_test, cal_probs_test,  ece_temp),
        "HaRP (full)":      (y_test, p_corr_test,     ece_gov),
    }
    plot_reliability_diagram(systems_plot, RESULTS_PLOTS / "15_reliability_diagram.png")

    # ── Save log ──────────────────────────────────────────────────────────────
    log = {
        "experiment": "15_calibration",
        "n_test":     int(len(y_test)),
        "systems": {
            "raw_llm":           {"ece": float(ece_raw)  if not np.isnan(ece_raw)  else None},
            "temp_scaling":      {"ece": float(ece_temp) if not np.isnan(ece_temp) else None,
                                  "T_star": float(T_star) if "T_star" in dir() else None},
            "harp_full":      {"ece": float(ece_gov)},
        },
    }
    log_path = RESULTS_LOGS / "15_calibration_results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved: {log_path.name}")

    print("Experiment 15 complete. Next: run 16_full_evaluation.py")


if __name__ == "__main__":
    main()
