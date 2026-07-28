"""
policy.py — Authorization policy: ACCEPT / REGENERATE / ABSTAIN.

The policy converts P(failure) from the failure estimator into an action.
Thresholds alpha and beta are optimised on the validation set.

Decision rule:
  P < alpha           → ACCEPT      (high confidence, output the answer)
  alpha <= P < beta   → REGENERATE  (uncertain, retry with steering)
  P >= beta           → ABSTAIN     (high failure risk, say "I don't know")

Cost function (minimised during threshold optimisation):
  Loss = hallucination_rate(accepted) + lambda * abstention_rate
  lambda controls the abstention penalty:
    lambda=0.5  = moderate penalty (default)
    lambda=1.0  = abstention as costly as hallucination
    lambda=0.25 = prefer abstaining (safety-first)

REGENERATE behaviour:
  1. Retry same prompt with temperature=0.9
  2. Apply steering vector during forward pass (if HF mode available)
  3. If retry still >= alpha → escalate to ABSTAIN
  Max 1 retry (no infinite loops).
"""

import numpy as np
from typing import List, Tuple, Optional
from src.evaluation.metrics import clopper_pearson_ci


ACTIONS = ("ACCEPT", "REGENERATE", "ABSTAIN")


def decide(
    p_failure: float,
    alpha: float = 0.3,
    beta: float  = 0.7,
) -> str:
    """
    Apply the authorization policy to a single failure probability.

    Returns one of: "ACCEPT", "REGENERATE", "ABSTAIN"
    """
    if p_failure < alpha:
        return "ACCEPT"
    elif p_failure < beta:
        return "REGENERATE"
    else:
        return "ABSTAIN"


def apply_policy(
    p_failures: np.ndarray,
    alpha: float = 0.3,
    beta: float  = 0.7,
) -> List[str]:
    """Apply policy to an array of failure probabilities."""
    return [decide(float(p), alpha, beta) for p in p_failures]


def compute_policy_metrics(
    decisions: List[str],
    labels: np.ndarray,
    p_failures: np.ndarray,
) -> dict:
    """
    Compute hallucination rate, abstention rate, and accuracy on accepted answers.

    Parameters
    ----------
    decisions  : list of "ACCEPT" / "REGENERATE" / "ABSTAIN" per sample
    labels     : np.ndarray [N] — 1=correct, 0=hallucinated (ground truth)

    Returns dict with policy metrics.
    """
    n = len(decisions)
    accepted_mask  = np.array([d == "ACCEPT"     for d in decisions])
    regen_mask     = np.array([d == "REGENERATE" for d in decisions])
    abstain_mask   = np.array([d == "ABSTAIN"    for d in decisions])

    n_accepted = accepted_mask.sum()
    n_regen    = regen_mask.sum()
    n_abstain  = abstain_mask.sum()

    # Hallucination rate among ACCEPT decisions
    if n_accepted > 0:
        hall_rate_accepted = float(1 - labels[accepted_mask].mean())
    else:
        hall_rate_accepted = float("nan")

    # Overall hallucination rate (what the user sees, excluding abstentions)
    served_mask = accepted_mask | regen_mask
    if served_mask.sum() > 0:
        hall_rate_served = float(1 - labels[served_mask].mean())
    else:
        hall_rate_served = float("nan")

    # Clopper-Pearson 95% CI on hall_rate_accepted
    n_hall_accepted = int((~labels[accepted_mask].astype(bool)).sum()) if n_accepted > 0 else 0
    hall_ci_lo, hall_ci_hi = clopper_pearson_ci(n_hall_accepted, int(n_accepted))

    return {
        "n_total":                  n,
        "n_accept":                 int(n_accepted),
        "n_regenerate":             int(n_regen),
        "n_abstain":                int(n_abstain),
        "accept_rate":              round(n_accepted / n, 4),
        "regen_rate":               round(n_regen    / n, 4),
        "abstain_rate":             round(n_abstain  / n, 4),
        "hall_rate_accepted":       round(hall_rate_accepted, 4),
        "hall_rate_accepted_ci_95": {
            "lo": round(hall_ci_lo, 4) if not isinstance(hall_ci_lo, float) or not (hall_ci_lo != hall_ci_lo) else float("nan"),
            "hi": round(hall_ci_hi, 4) if not isinstance(hall_ci_hi, float) or not (hall_ci_hi != hall_ci_hi) else float("nan"),
            "n_hall": n_hall_accepted,
            "n_accept": int(n_accepted),
            "note": "Clopper-Pearson exact binomial 95% CI",
        },
        "hall_rate_served":         round(hall_rate_served, 4),
    }


def optimize_thresholds(
    p_failures: np.ndarray,
    labels: np.ndarray,
    lambda_: float = 0.5,
    alpha_range: Tuple = (0.1, 0.55, 0.05),
    beta_range:  Tuple = (0.5, 0.95, 0.05),
) -> Tuple[float, float, float, List[dict]]:
    """
    Sweep alpha and beta to minimise the governance loss.

    Loss = hallucination_rate(accepted) + lambda * abstention_rate

    Returns (best_alpha, best_beta, best_loss, sweep_results)
    """
    sweep_results = []
    best = (0.3, 0.7, float("inf"))

    alphas = np.arange(*alpha_range)
    betas  = np.arange(*beta_range)

    for alpha in alphas:
        for beta in betas:
            if alpha >= beta:
                continue
            decisions = apply_policy(p_failures, alpha, beta)
            metrics   = compute_policy_metrics(decisions, labels, p_failures)

            hall_rate   = metrics["hall_rate_accepted"]
            abstain_rate = metrics["abstain_rate"]

            if np.isnan(hall_rate):
                continue  # skip if nothing was accepted

            loss = hall_rate + lambda_ * abstain_rate

            sweep_results.append({
                "alpha":        round(float(alpha), 3),
                "beta":         round(float(beta), 3),
                "loss":         round(loss, 5),
                "hall_rate":    round(hall_rate, 4),
                "abstain_rate": round(abstain_rate, 4),
            })

            if loss < best[2]:
                best = (float(alpha), float(beta), loss)

    print(f"Threshold optimisation (λ={lambda_}):")
    print(f"  Best alpha = {best[0]:.2f}, beta = {best[1]:.2f}, loss = {best[2]:.4f}")

    return best[0], best[1], best[2], sweep_results
