"""
geometric_signals.py — Group E: Geometric volume signals over response embeddings.

Extends the Geometric Uncertainty framework (Granese et al., ICLR 2026 submission,
arXiv:2509.13813) from a standalone detector into a governance feature set.

Core insight (Granese et al.):
  The convex hull volume of response embeddings measures semantic dispersion.
  They prove: Volume(convex_hull(embeddings)) ↔ entropy of the response distribution.
  High volume = responses explore a large semantic region = uncertain.
  Near-zero volume = all responses collapse to the same point = confident.

HaRP extension:
  We compute this volume per-sample (not per-batch) using the N=3 consistency
  responses already stored in features_groupAB.csv. Three signals are extracted:

  simplex_volume   : Area of the triangle formed by 3 response embeddings
                     via the Gram-matrix determinant formula. This is the
                     HaRP operationalisation of Geometric Volume.
                     High = wide semantic spread = uncertain
                     Low  = tight cluster = confident (or sycophantically stuck)

  embedding_spread : Maximum pairwise L2 distance between response embeddings.
                     Captures the worst-case disagreement between any two runs.
                     Complements simplex_volume (which averages all pairs).

  centroid_drift   : Standard deviation of distances from each response embedding
                     to the centroid. Low std = all responses equidistant from
                     center (symmetric uncertainty). High std = one outlier
                     response (model produced one very different answer once).

Why this is stronger than pairwise cosine similarity (Group B):
  - Geometric volume captures the SHAPE of the response cloud, not just average
    pairwise similarity. Two responses can have high cosine similarity but still
    span a large geometric volume if the third is far from both.
  - The Gram-matrix formula is O(N^2) in embedding dimension, not O(N^2) in
    vocabulary space — computationally efficient.
  - Theoretically grounded: volume is a proper uncertainty measure (see Granese
    et al. proof that it upper-bounds entropy).

Reference:
  Granese et al. (2025). Geometric Uncertainty for Detecting and Correcting
  Hallucinations in LLMs. arXiv:2509.13813 (under review ICLR 2026).
"""

import numpy as np
from typing import List


# Lazy-load sentence-transformer encoder
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def _embed_responses(responses: List[str]) -> np.ndarray:
    """
    Encode N responses into L2-normalised embeddings.

    Returns np.ndarray [N, embedding_dim]
    """
    responses_clean = [r.strip() for r in responses if r.strip()]
    encoder = _get_encoder()
    embeddings = encoder.encode(responses_clean, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def simplex_volume(embeddings: np.ndarray) -> float:
    """
    Compute the volume of the simplex formed by N embeddings
    via the Gram-matrix determinant formula.

    For N=3 this is the area of the triangle (the HaRP default).
    For general N, this is the (N-1)-dimensional simplex volume.

    Parameters
    ----------
    embeddings : np.ndarray [N, d] — L2-normalised embeddings

    Returns
    -------
    float — simplex volume (in embedding space units). 0 = degenerate (all same).
    """
    N = embeddings.shape[0]
    if N < 2:
        return 0.0

    # Use the first embedding as origin, compute difference vectors
    origin = embeddings[0]
    diffs = embeddings[1:] - origin      # [N-1, d]

    # Gram matrix G[i,j] = diffs[i] · diffs[j]
    G = diffs @ diffs.T                  # [N-1, N-1]

    # Volume = sqrt(det(G)) / (N-1)!
    det_val = np.linalg.det(G)
    if det_val <= 0:
        return 0.0

    # Factorial for normalisation
    factorial = 1
    for k in range(1, N):
        factorial *= k

    return float(np.sqrt(det_val) / factorial)


def embedding_spread(embeddings: np.ndarray) -> float:
    """
    Maximum pairwise L2 distance between N embeddings.

    This is the diameter of the response cloud — the worst-case semantic
    disagreement between any two model outputs.
    """
    N = embeddings.shape[0]
    if N < 2:
        return 0.0

    max_dist = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(embeddings[i] - embeddings[j]))
            if d > max_dist:
                max_dist = d

    return max_dist


def centroid_drift(embeddings: np.ndarray) -> float:
    """
    Standard deviation of L2 distances from each embedding to the centroid.

    Low  = symmetric uncertainty (all responses equally far from mean)
    High = one outlier response (model occasionally produced something very
           different — a sign of instability rather than uncertainty)
    """
    if len(embeddings) < 2:
        return 0.0

    centroid = embeddings.mean(axis=0)
    dists = np.array([
        float(np.linalg.norm(e - centroid))
        for e in embeddings
    ])
    return float(dists.std())


def extract_all(
    responses: List[str],
) -> dict:
    """
    Compute all 3 Group E geometric features from N response strings.

    Requires sentence-transformers (all-MiniLM-L6-v2) to be installed.
    Uses the SAME encoder as Group B (consistency_signals.py) — shares the
    lazy-loaded instance when called in sequence.

    Parameters
    ----------
    responses : list[str] — N response texts (typically N=3 from consistency check)

    Returns
    -------
    dict with keys:
        simplex_volume   : float
        embedding_spread : float
        centroid_drift   : float
    All return np.nan if fewer than 2 non-empty responses.
    """
    clean = [r.strip() for r in responses if r.strip()]

    if len(clean) < 2:
        return {
            "simplex_volume":    float("nan"),
            "embedding_spread":  float("nan"),
            "centroid_drift":    float("nan"),
        }

    embeddings = _embed_responses(clean)

    return {
        "simplex_volume":    simplex_volume(embeddings),
        "embedding_spread":  embedding_spread(embeddings),
        "centroid_drift":    centroid_drift(embeddings),
    }


def geometric_class_analysis(
    volumes: np.ndarray,
    spreads: np.ndarray,
    drifts: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Compare geometric feature distributions between correct and hallucinated samples.

    Returns dict with per-feature mean_correct, mean_hallucinated, delta.
    """
    results = {}
    for name, vals in [
        ("simplex_volume",    volumes),
        ("embedding_spread",  spreads),
        ("centroid_drift",    drifts),
    ]:
        mask_c = labels == 1
        mask_h = labels == 0
        mu_c = float(np.nanmean(vals[mask_c])) if mask_c.sum() > 0 else float("nan")
        mu_h = float(np.nanmean(vals[mask_h])) if mask_h.sum() > 0 else float("nan")
        delta = float(mu_c - mu_h) if not (np.isnan(mu_c) or np.isnan(mu_h)) else float("nan")

        results[name] = {
            "mean_correct":      round(mu_c, 6),
            "mean_hallucinated": round(mu_h, 6),
            "delta":             round(delta, 6),
        }
    return results
