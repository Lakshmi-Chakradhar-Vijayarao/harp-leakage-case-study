"""
semantic_entropy.py — Group B+: NLI-based semantic entropy (Kuhn et al., ICLR 2023).

Extends cosine consistency with a semantically principled uncertainty measure:
  - Cluster responses by bidirectional NLI entailment (same meaning = same cluster)
  - Compute entropy over cluster probability distribution
  - Invariant to linguistic rephrasing: "Paris is the capital" ≡ "The capital is Paris"

Key insight from Kuhn et al.:
  Standard sequence-level entropy conflates linguistic variance with semantic uncertainty.
  Semantic entropy aggregates all ways to say the same thing into a single cluster,
  giving a cleaner uncertainty estimate.

Novel contribution vs. SelfCheckGPT:
  SelfCheckGPT uses BERTScore/NLI for consistency detection (binary per-sentence).
  We use semantic entropy as a continuous uncertainty feature in a learned estimator
  combined with representation-level signals — no existing paper does this combination.

Model: cross-encoder/nli-deberta-v3-small (~85MB, fast CPU inference)
  Outputs: contradiction / neutral / entailment
  Bidirectional entailment threshold: both directions must predict entailment.

Dependency: sentence-transformers >= 2.2 (cross-encoder support)
"""

import numpy as np
from typing import List, Optional, Tuple


# Lazy-load to avoid startup cost
_nli_model = None


def _get_nli_model():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder(
            "cross-encoder/nli-deberta-v3-small",
            max_length=512,
        )
    return _nli_model


def _nli_predict_batch(
    pairs: List[Tuple[str, str]],
) -> np.ndarray:
    """
    Run NLI on a batch of (premise, hypothesis) pairs.

    Returns array [N, 3] with softmax scores:
        col 0 = contradiction
        col 1 = entailment
        col 2 = neutral
    Note: deberta-v3-small label order: contradiction(0), entailment(1), neutral(2)
    """
    model = _get_nli_model()
    scores = model.predict(pairs, apply_softmax=True)
    return np.array(scores)  # [N, 3]


def _are_equivalent(
    r1: str,
    r2: str,
    entail_threshold: float = 0.5,
) -> bool:
    """
    Two responses are semantically equivalent iff NLI predicts entailment in BOTH directions.

    This is the bidirectional entailment criterion from Kuhn et al. (2023).
    """
    pairs = [(r1, r2), (r2, r1)]
    scores = _nli_predict_batch(pairs)
    # Column 1 = entailment probability
    return (scores[0, 1] >= entail_threshold and
            scores[1, 1] >= entail_threshold)


def cluster_responses(
    responses: List[str],
    entail_threshold: float = 0.5,
) -> List[List[int]]:
    """
    Cluster responses by semantic equivalence using bidirectional NLI.

    Algorithm (greedy, O(N^2)):
      For each response, assign to the first cluster whose representative
      is bidirectionally entailed. If none, start a new cluster.

    Returns
    -------
    List of clusters, each cluster is a list of response indices.
    E.g. [[0, 2], [1], [3, 4]] means responses 0&2 are semantically equivalent, etc.
    """
    clusters: List[List[int]] = []
    cluster_reps: List[str] = []

    for i, resp in enumerate(responses):
        assigned = False
        for c_idx, rep in enumerate(cluster_reps):
            if _are_equivalent(resp, rep, entail_threshold):
                clusters[c_idx].append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])
            cluster_reps.append(resp)

    return clusters


def semantic_entropy(
    responses: List[str],
    entail_threshold: float = 0.5,
) -> float:
    """
    Compute semantic entropy over N responses.

    Steps (Kuhn et al. 2023):
      1. Cluster responses by bidirectional NLI entailment
      2. p(cluster_k) = |cluster_k| / N
      3. H_semantic = -sum_k p_k * log(p_k)

    High semantic entropy = responses disagree on meaning → uncertain
    Low semantic entropy  = responses converge on the same meaning → confident

    Note: H_semantic <= H_sequence always (clustering reduces apparent diversity).
    A large gap H_sequence - H_semantic indicates linguistic variation, not true uncertainty.

    Returns
    -------
    float: semantic entropy in nats. np.nan if fewer than 2 non-empty responses.
    """
    responses = [r.strip() for r in responses if r.strip()]
    if len(responses) < 2:
        return float("nan")

    N = len(responses)
    clusters = cluster_responses(responses, entail_threshold)

    probs = np.array([len(c) / N for c in clusters])
    # Entropy: -sum p * log(p), avoid log(0)
    ent = -np.sum(probs * np.log(probs + 1e-12))
    return float(ent)


def sequence_entropy_estimate(
    responses: List[str],
) -> float:
    """
    Naive sequence-level entropy estimate: treats each distinct response as its own cluster.
    Counts lexically unique responses only (no semantic grouping).

    Used as a baseline to compute the gap vs. semantic entropy.
    Gap = sequence_entropy - semantic_entropy quantifies linguistic noise.
    """
    responses = [r.strip() for r in responses if r.strip()]
    if len(responses) < 2:
        return float("nan")

    N = len(responses)
    from collections import Counter
    counts = Counter(responses)
    probs = np.array([c / N for c in counts.values()])
    ent = -np.sum(probs * np.log(probs + 1e-12))
    return float(ent)


def extract_all(
    responses: List[str],
    entail_threshold: float = 0.5,
) -> dict:
    """
    Compute all semantic entropy features from N responses.

    Returns dict with keys:
        semantic_entropy      : NLI-based cluster entropy (core novel signal)
        n_semantic_clusters   : number of distinct meaning clusters
        semantic_cluster_ratio: n_clusters / N (1.0 = all different, 1/N = all same)
        entropy_gap           : sequence_entropy - semantic_entropy (linguistic noise)
    """
    responses_clean = [r.strip() for r in responses if r.strip()]
    N = len(responses_clean)

    if N < 2:
        return {
            "semantic_entropy":       float("nan"),
            "n_semantic_clusters":    float("nan"),
            "semantic_cluster_ratio": float("nan"),
            "entropy_gap":            float("nan"),
        }

    clusters = cluster_responses(responses_clean, entail_threshold)
    n_clusters = len(clusters)
    probs = np.array([len(c) / N for c in clusters])
    sem_ent = float(-np.sum(probs * np.log(probs + 1e-12)))

    seq_ent = sequence_entropy_estimate(responses_clean)

    return {
        "semantic_entropy":       sem_ent,
        "n_semantic_clusters":    n_clusters,
        "semantic_cluster_ratio": n_clusters / N,
        "entropy_gap":            float(seq_ent - sem_ent) if not np.isnan(seq_ent) else float("nan"),
    }
