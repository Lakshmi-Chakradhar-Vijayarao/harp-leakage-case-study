"""
metrics.py — Central evaluation metrics for HaRP.

All functions return plain Python types (float, int, dict) for JSON serialisability.
Bootstrap confidence intervals use 1000 resamples with seed=42.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
)
from scipy.stats import beta as _beta_dist


def clopper_pearson_ci(
    n_successes: int,
    n_trials: int,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """
    Exact (Clopper-Pearson) binomial confidence interval for a proportion.

    Parameters
    ----------
    n_successes : int — number of events (e.g. hallucinations observed)
    n_trials    : int — total trials (e.g. n_accepted)
    ci          : float — confidence level (default 0.95)

    Returns (lower_bound, upper_bound).
    Both are 0.0/1.0 at boundary cases (0 successes or all successes).
    """
    if n_trials == 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - ci
    lo = (
        float(_beta_dist.ppf(alpha / 2, n_successes, n_trials - n_successes + 1))
        if n_successes > 0 else 0.0
    )
    hi = (
        float(_beta_dist.ppf(1 - alpha / 2, n_successes + 1, n_trials - n_successes))
        if n_successes < n_trials else 1.0
    )
    return lo, hi


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUROC — area under the ROC curve. Returns nan if only one class present."""
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray,
               zero_division: int = 0) -> float:
    return float(f1_score(y_true, y_pred, zero_division=zero_division))


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_hallucination_rate(labels: np.ndarray,
                               accepted_mask: np.ndarray) -> float:
    """
    Fraction of accepted answers that are hallucinated.
    labels=1 → correct, labels=0 → hallucinated.
    """
    if int(accepted_mask.sum()) == 0:
        return float("nan")
    return float(1.0 - labels[accepted_mask].mean())


def bootstrap_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for AUROC.
    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng    = np.random.default_rng(seed)
    n      = len(y_true)
    scores = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(roc_auc_score(yt, yp))
        except ValueError:
            pass

    if not scores:
        return float("nan"), float("nan"), float("nan")

    alpha = (1 - ci) / 2
    return (
        float(np.mean(scores)),
        float(np.percentile(scores, 100 * alpha)),
        float(np.percentile(scores, 100 * (1 - alpha))),
    )


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict:
    """Full classification metrics in a single serialisable dict."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    report = {
        "accuracy":   compute_accuracy(y_true, y_pred),
        "f1":         compute_f1(y_true, y_pred),
        "precision":  float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":     float(recall_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_total":    len(y_true),
        "n_positive": int(y_true.sum()),
        "n_negative": int((y_true == 0).sum()),
    }
    if y_prob is not None:
        point, lo, hi = bootstrap_auroc(y_true, y_prob)
        report["auroc"]    = compute_auroc(y_true, y_prob)
        report["auroc_ci"] = {"mean": point, "lo_95": lo, "hi_95": hi}
    return report


def feature_importance_lr(pipeline, feature_names: List[str]) -> List[Dict]:
    """
    Extract sorted feature importances from a fitted LR pipeline.
    Returns list of {feature, coefficient, abs_coef} sorted by |coef|.
    """
    clf  = pipeline.named_steps["clf"]
    coef = clf.coef_[0]
    return sorted(
        [{"feature": n, "coefficient": float(c), "abs_coef": float(abs(c))}
         for n, c in zip(feature_names, coef)],
        key=lambda x: x["abs_coef"],
        reverse=True,
    )


def policy_summary(decisions: List[str], labels: np.ndarray) -> Dict:
    """Compute policy-level metrics from a list of decision strings and true labels."""
    d = np.array(decisions)
    n = len(d)

    accept_mask  = d == "ACCEPT"
    regen_mask   = d == "REGENERATE"
    abstain_mask = d == "ABSTAIN"
    served_mask  = accept_mask | regen_mask

    result = {
        "n_total":      n,
        "n_accept":     int(accept_mask.sum()),
        "n_regenerate": int(regen_mask.sum()),
        "n_abstain":    int(abstain_mask.sum()),
        "accept_rate":  round(float(accept_mask.mean()), 4),
        "regen_rate":   round(float(regen_mask.mean()), 4),
        "abstain_rate": round(float(abstain_mask.mean()), 4),
    }
    result["hall_rate_accepted"] = (
        round(float(1 - labels[accept_mask].mean()), 4)
        if accept_mask.sum() > 0 else float("nan")
    )
    result["hall_rate_served"] = (
        round(float(1 - labels[served_mask].mean()), 4)
        if served_mask.sum() > 0 else float("nan")
    )
    return result


def compute_auarc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 100,
) -> float:
    """
    AUARC — Area Under the Accuracy-Rejection Curve.

    The 2025 standard metric for selective prediction (SelectLLM, ACL 2025).
    Better than AUROC for evaluating the coverage-accuracy trade-off:
      - Sort samples by confidence (ascending uncertainty = descending confidence)
      - At each rejection threshold, compute accuracy on retained samples
      - AUARC = area under accuracy(retention_fraction) curve

    AUARC = 1.0: model perfectly identifies which answers are correct
    AUARC = base_rate: model uncertainty is no better than random abstention

    Returns
    -------
    float in [0, 1]
    """
    # confidence = 1 - P(failure) = P(correct)
    # sort by descending confidence (reject least confident first)
    order = np.argsort(y_prob)[::-1]   # highest confidence first
    y_sorted = y_true[order]

    n = len(y_sorted)
    accs = []
    for k in range(1, n + 1):
        # retain top-k most confident
        accs.append(float(y_sorted[:k].mean()))

    # AUARC = mean accuracy across all retention levels
    return float(np.mean(accs))


def ablation_table(results: Dict[str, Dict]) -> str:
    """
    Format an ablation results dict into a markdown table string.

    Input: {label: {auroc_mean, auroc_std, f1_mean, ...}}
    """
    lines = [
        "| Model | AUROC | ±std | F1 |",
        "|-------|-------|------|----|",
    ]
    for label, r in results.items():
        lines.append(
            f"| {label} "
            f"| {r.get('auroc_mean', float('nan')):.4f} "
            f"| {r.get('auroc_std',  float('nan')):.4f} "
            f"| {r.get('f1_mean',    float('nan')):.4f} |"
        )
    return "\n".join(lines)
