"""
calibrate.py — Probability calibration: ECE and reliability diagrams.

Calibration answers: "When HaRP says it's 80% confident, is it right 80%?"

Expected Calibration Error (ECE):
  Divide predictions into M bins (default 10).
  ECE = Σ (|bin| / N) * |accuracy(bin) - confidence(bin)|
  Perfect calibration = 0. Values below 0.05 = well-calibrated.

Reliability diagram:
  X-axis: confidence (mean predicted probability per bin)
  Y-axis: fraction correct in that bin
  Perfect calibration = diagonal line
  Above diagonal = underconfident
  Below diagonal = overconfident

Three systems compared (Plot 8):
  1. Raw LLM (always accepts, confidence = top-1 prob)
  2. Temperature scaling (Platt scaling on logits)
  3. HaRP (full system)
"""

import numpy as np
from typing import Tuple, List, Dict


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, List[Dict]]:
    """
    Compute ECE and return per-bin statistics for reliability diagram.

    Parameters
    ----------
    y_true : np.ndarray [N] — true labels (1=correct, 0=hallucinated)
    y_prob : np.ndarray [N] — predicted P(correct) for each sample

    Returns
    -------
    ece  : float — Expected Calibration Error
    bins : list of dicts — {bin_lower, bin_upper, confidence, accuracy, count}
    """
    bins_stats = []
    ece = 0.0
    n   = len(y_true)

    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)  # include right edge

        count = int(mask.sum())
        if count == 0:
            bins_stats.append({
                "bin_lower":  round(lo, 2),
                "bin_upper":  round(hi, 2),
                "confidence": None,
                "accuracy":   None,
                "count":      0,
            })
            continue

        confidence = float(y_prob[mask].mean())
        accuracy   = float(y_true[mask].mean())

        ece += (count / n) * abs(accuracy - confidence)

        bins_stats.append({
            "bin_lower":  round(lo, 2),
            "bin_upper":  round(hi, 2),
            "confidence": round(confidence, 4),
            "accuracy":   round(accuracy, 4),
            "count":      count,
        })

    return round(ece, 5), bins_stats


def temperature_scale(
    logits: np.ndarray,
    y_true: np.ndarray,
    T_range: Tuple = (0.1, 5.0, 50),
) -> Tuple[float, np.ndarray]:
    """
    Post-hoc temperature scaling (Platt calibration on a single scalar T).

    Finds the temperature T that minimises ECE on the given data.
    Returns (best_T, calibrated_probs).
    """
    from scipy.special import softmax

    temps   = np.linspace(*T_range)
    best_T  = 1.0
    best_ece = float("inf")

    for T in temps:
        scaled_logits = logits / T
        probs = softmax(scaled_logits, axis=-1)
        if probs.ndim == 2:
            probs = probs[:, 1]
        ece, _ = expected_calibration_error(y_true, probs)
        if ece < best_ece:
            best_ece = ece
            best_T   = T

    # Apply best temperature
    calibrated_probs = softmax(logits / best_T, axis=-1)
    if calibrated_probs.ndim == 2:
        calibrated_probs = calibrated_probs[:, 1]

    print(f"Temperature scaling: T* = {best_T:.3f} → ECE = {best_ece:.5f}")
    return best_T, calibrated_probs


def print_calibration_summary(
    systems: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_bins: int = 10,
):
    """
    Print ECE for multiple systems side-by-side.

    Parameters
    ----------
    systems : dict mapping system_name → (y_true, y_prob)
    """
    print("Calibration Summary:")
    print("-" * 45)
    for name, (y_true, y_prob) in systems.items():
        ece, _ = expected_calibration_error(y_true, y_prob, n_bins)
        print(f"  {name:<30}  ECE = {ece:.5f}")
    print("-" * 45)
