"""
consistency_signals.py — Group B: Semantic consistency across N model runs.

Distinguishes epistemic uncertainty (model doesn't know) from aleatoric
uncertainty (question is genuinely ambiguous).

Core insight:
  Low entropy + low consistency  = DANGEROUS: confident but unstable
  High entropy + low consistency = Honest uncertainty: safe to abstain
  High entropy + high consistency = Hard question, model knows it's hard
  Low entropy + high consistency  = Confident and stable: safe to accept

Implementation:
  Uses sentence-transformers cosine similarity (NOT keyword overlap).
  Semantic similarity handles paraphrases correctly:
    "Paris is the capital" ≈ "The capital of France is Paris" → high sim
  Keyword overlap would score these as low similarity — a false signal.

Dependency: sentence-transformers (all-MiniLM-L6-v2, ~80MB)
"""

import numpy as np
from typing import Optional


# Lazy-load the encoder to avoid startup cost when not needed
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def semantic_consistency(responses: list[str]) -> float:
    """
    Compute mean pairwise cosine similarity across N response strings.

    For 3 responses [r1, r2, r3]:
      sim(r1, r2), sim(r1, r3), sim(r2, r3) → mean of 3 values

    Returns
    -------
    float in [0, 1] — 1.0 = all responses identical in meaning
                     0.0 = responses completely unrelated
    Returns np.nan if fewer than 2 responses provided.
    """
    if len(responses) < 2:
        return float("nan")

    # Clean empty responses
    responses = [r.strip() for r in responses if r.strip()]
    if len(responses) < 2:
        return float("nan")

    encoder = _get_encoder()
    embeddings = encoder.encode(responses, normalize_embeddings=True)
    # Cosine similarity = dot product when vectors are L2-normalised
    sim_matrix = embeddings @ embeddings.T

    # Extract upper triangle (pairwise, no diagonal)
    n = len(responses)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(float(sim_matrix[i, j]))

    return float(np.mean(pairs))


def consistency_variance(responses: list[str]) -> float:
    """
    Variance of pairwise cosine similarities.

    High variance = responses are inconsistent in different ways.
    Low variance  = either consistently similar or consistently dissimilar.
    """
    if len(responses) < 3:
        return float("nan")

    responses = [r.strip() for r in responses if r.strip()]
    if len(responses) < 3:
        return float("nan")

    encoder = _get_encoder()
    embeddings = encoder.encode(responses, normalize_embeddings=True)
    sim_matrix = embeddings @ embeddings.T

    n = len(responses)
    pairs = [float(sim_matrix[i, j])
             for i in range(n) for j in range(i + 1, n)]

    return float(np.var(pairs))


def extract_all(responses: list[str]) -> dict:
    """
    Compute all Group B consistency features from a list of N responses.

    Returns dict with keys:
        consistency_score    : mean pairwise cosine similarity
        consistency_variance : variance of pairwise similarities
    """
    return {
        "consistency_score":    semantic_consistency(responses),
        "consistency_variance": consistency_variance(responses),
    }
