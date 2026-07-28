"""
token_signals.py — Group A: Token-level uncertainty signals from logprobs.

All functions take a list of token log-probabilities (natural log scale)
and return a single float. Input is token_logprobs from query_ollama.py.

If logprobs are unavailable (Ollama fallback), all functions return np.nan.
The experiment scripts check for nan and substitute consistency-based features.

Six signals:
  1. predictive_entropy     — avg entropy across all tokens
  2. max_token_entropy      — worst-case uncertainty point
  3. logit_variance         — spread of top-k logits (needs top_logprobs)
  4. confidence_gap         — avg p(top1) - p(top2) (needs top_logprobs)
  5. mean_top1_prob         — avg confidence level across tokens
  6. first_token_entropy    — early signal: hesitation at position 0
"""

import numpy as np
from typing import Optional


def _log_to_prob(log_p: float) -> float:
    """Convert a single log-probability to a probability, clamped to [0,1]."""
    return float(np.clip(np.exp(log_p), 0.0, 1.0))


def predictive_entropy(token_logprobs: Optional[list]) -> float:
    """
    Average token-level entropy across the sequence.

    For each token, treat the top-1 probability as a Bernoulli distribution:
      H(t) = -p * log(p) - (1-p) * log(1-p)
    Then average across all tokens.

    Higher = model is more uncertain overall.
    """
    if not token_logprobs:
        return float("nan")

    entropies = []
    for lp in token_logprobs:
        p = _log_to_prob(lp)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        h = -p * np.log(p) - (1 - p) * np.log(1 - p)
        entropies.append(h)
    return float(np.mean(entropies))


def max_token_entropy(token_logprobs: Optional[list]) -> float:
    """
    Maximum token-level entropy across the sequence.

    Identifies the single token where the model was most uncertain.
    Useful for detecting "surprise" positions that mean-pooling masks.
    """
    if not token_logprobs:
        return float("nan")

    entropies = []
    for lp in token_logprobs:
        p = _log_to_prob(lp)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        h = -p * np.log(p) - (1 - p) * np.log(1 - p)
        entropies.append(h)
    return float(np.max(entropies))


def logit_variance(top_logprobs: Optional[list]) -> float:
    """
    Average variance of top-k logits across tokens.

    top_logprobs is a list of dicts: [{token: log_prob}, ...] per token.
    High variance = model's top candidates are spread out (uncertain).
    Low variance  = model strongly prefers one token (confident).
    """
    if not top_logprobs:
        return float("nan")

    variances = []
    for token_dict in top_logprobs:
        if not isinstance(token_dict, dict):
            continue
        logit_vals = [v for v in token_dict.values() if isinstance(v, (int, float))]
        if len(logit_vals) < 2:
            continue
        variances.append(float(np.var(logit_vals)))

    return float(np.mean(variances)) if variances else float("nan")


def confidence_gap(top_logprobs: Optional[list]) -> float:
    """
    Average gap between top-1 and top-2 token probabilities.

    confidence_gap(t) = p(top1) - p(top2) at each token, then averaged.
    High gap   = model is decisive (confident, sharp distribution).
    Near-zero  = model is torn between two options (uncertain).
    """
    if not top_logprobs:
        return float("nan")

    gaps = []
    for token_dict in top_logprobs:
        if not isinstance(token_dict, dict) or len(token_dict) < 2:
            continue
        sorted_probs = sorted(
            [np.exp(lp) for lp in token_dict.values() if isinstance(lp, (int, float))],
            reverse=True
        )
        if len(sorted_probs) < 2:
            continue
        gap = sorted_probs[0] - sorted_probs[1]
        gaps.append(float(gap))

    return float(np.mean(gaps)) if gaps else float("nan")


def mean_top1_prob(token_logprobs: Optional[list]) -> float:
    """
    Average of the top-1 (greedy) probability across all tokens.

    Directly interpretable: if this is 0.9, the model was 90% confident
    on average. Below 0.5 across a response = overall low-confidence output.
    """
    if not token_logprobs:
        return float("nan")

    probs = [_log_to_prob(lp) for lp in token_logprobs]
    return float(np.mean(probs))


def first_token_entropy(token_logprobs: Optional[list]) -> float:
    """
    Entropy at the very first generated token.

    If the model hesitates at token 0, it's uncertain about where to start —
    an early warning signal that often precedes hallucination.
    """
    if not token_logprobs or len(token_logprobs) == 0:
        return float("nan")

    lp = token_logprobs[0]
    p  = _log_to_prob(lp)
    p  = np.clip(p, 1e-10, 1 - 1e-10)
    return float(-p * np.log(p) - (1 - p) * np.log(1 - p))


def extract_all(token_logprobs: Optional[list],
                top_logprobs:   Optional[list]) -> dict:
    """
    Compute all 6 Group A features at once.

    Returns a dict with keys matching the feature vector definition.
    Values are float — np.nan if logprobs unavailable.
    """
    return {
        "entropy":         predictive_entropy(token_logprobs),
        "max_entropy":     max_token_entropy(token_logprobs),
        "logit_var":       logit_variance(top_logprobs),
        "conf_gap":        confidence_gap(top_logprobs),
        "top1_prob":       mean_top1_prob(token_logprobs),
        "first_entropy":   first_token_entropy(token_logprobs),
    }
