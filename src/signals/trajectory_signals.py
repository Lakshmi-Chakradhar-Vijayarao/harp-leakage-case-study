"""
trajectory_signals.py — Group D: Cross-layer hidden state trajectory signals.

Novel contribution extending ICR Probe (Zhang et al., ACL 2025) into a
governance-oriented feature set.

ICR Probe insight: hidden state DYNAMICS across layers are more discriminative
than any single-layer static snapshot. A hallucinating model commits to a
wrong answer at a specific layer and its representation stabilises differently
from a model that is genuinely uncertain.

HaRP extension: We translate this dynamic insight into per-sample features
that feed a learned failure estimator — the first paper to incorporate
cross-layer trajectory signals into a governance policy.

Five trajectory features (all per-sample, computed from last-token hidden states):

  traj_total_variation  : Total path length of the hidden state trajectory
                          sum_{l=1}^{L-1} (1 - cos_sim(h[l], h[l+1]))
                          High = representation kept evolving (unsettled)
                          Low  = representation committed early (decided quickly)

  traj_stabilization_layer : Normalised layer index of first "quiet point" —
                          first l where cos_dist(h[l], h[l+1]) < threshold.
                          Low  = stabilises early (confident, for better or worse)
                          High = never settles (uncertain until the end)

  traj_max_change_layer : Normalised layer index of the single biggest jump.
                          If this aligns with the best probing layer = model
                          makes its factual decision at the right place.
                          Misalignment = anomalous commitment pattern.

  traj_early_commitment : 1 - cos_sim(h[1], h[best_layer])
                          How much did the representation drift between the
                          first transformer layer and the critical probing layer?
                          Low  = early layers already encode the answer correctly
                          High = late layers had to "work hard" to get there

  icr_score             : Information Contribution Rate at best_layer.
                          cos_dist(h[best_layer], h[best_layer-1]) / total_variation
                          Fraction of total trajectory change that occurs exactly
                          at the critical layer. High ICR = sharp, decisive change
                          at the layer that matters. Low ICR = diffuse change.

Reference:
  Zhang et al. (2025). ICR Probe: Tracking Hidden State Dynamics for Reliable
  Hallucination Detection in LLMs. ACL 2025.
"""

import numpy as np
import pandas as pd
from typing import Optional


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity. Range [0, 2]. 0 = identical direction."""
    # Cast to float32 — float16 norms at deep layers can overflow to inf
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    sim = float(np.dot(a, b) / (na * nb))
    return float(1.0 - np.clip(sim, -1.0, 1.0))


def compute_layer_distances(hidden: np.ndarray) -> np.ndarray:
    """
    Compute cosine distances between consecutive layers for one sample.

    Parameters
    ----------
    hidden : np.ndarray [num_layers, hidden_dim]

    Returns
    -------
    np.ndarray [num_layers - 1] — cosine distance at each layer transition
    """
    num_layers = hidden.shape[0]
    dists = np.zeros(num_layers - 1, dtype=np.float32)
    for l in range(num_layers - 1):
        dists[l] = _cosine_distance(hidden[l], hidden[l + 1])
    return dists


def compute_sample_trajectory(
    hidden: np.ndarray,
    best_layer: int,
    stability_threshold: float = 0.005,
) -> dict:
    """
    Compute all 5 trajectory features for a single sample.

    Parameters
    ----------
    hidden            : np.ndarray [num_layers, hidden_dim]
    best_layer        : int — the critical probing layer (from exp06)
    stability_threshold: float — cosine distance below which rep is "stable"

    Returns
    -------
    dict with trajectory feature keys
    """
    num_layers = hidden.shape[0]
    dists = compute_layer_distances(hidden)   # [num_layers - 1]
    total_var = float(dists.sum())

    # traj_total_variation
    traj_total_var = total_var

    # traj_stabilization_layer — first layer where change drops below threshold
    # Search from layer 1 onwards (skip embedding → first transformer transition)
    stab_layer = None
    for l in range(1, len(dists)):
        if dists[l] < stability_threshold:
            stab_layer = l + 1   # +1 because dists[l] = dist between l and l+1
            break
    if stab_layer is None:
        stab_layer_norm = 1.0   # never stabilises
    else:
        stab_layer_norm = float(stab_layer) / max(1, num_layers - 1)

    # traj_max_change_layer — layer of biggest single-step change
    # dists[l] = distance between layers l and l+1
    max_change_idx = int(np.argmax(dists))
    max_change_layer_norm = float(max_change_idx) / max(1, num_layers - 2)

    # traj_early_commitment — drift from layer 1 to best_layer
    if best_layer > 0 and best_layer < num_layers:
        early_commit = _cosine_distance(hidden[1], hidden[best_layer])
    else:
        early_commit = float("nan")

    # icr_score — fraction of total variation that occurs at best_layer transition
    if total_var > 1e-12 and best_layer > 0 and best_layer <= len(dists):
        icr = float(dists[best_layer - 1]) / total_var
    else:
        icr = float("nan")

    return {
        "traj_total_variation":    float(traj_total_var),
        "traj_stabilization_layer": float(stab_layer_norm),
        "traj_max_change_layer":   float(max_change_layer_norm),
        "traj_early_commitment":   float(early_commit) if not np.isnan(early_commit) else float("nan"),
        "icr_score":               float(icr) if not np.isnan(icr) else float("nan"),
    }


def compute_trajectory_features(
    hidden_states: np.ndarray,
    prompt_ids: list,
    best_layer: int,
    stability_threshold: float = 0.005,
) -> pd.DataFrame:
    """
    Compute Group D trajectory features for all samples.

    Parameters
    ----------
    hidden_states : np.ndarray [N, num_layers, hidden_dim]
                    Last-token hidden states for all samples, all layers.
    prompt_ids    : list[str] — prompt IDs matching hidden_states rows
    best_layer    : int — best probing layer from exp06
    stability_threshold : float — cosine dist threshold for stabilisation

    Returns
    -------
    pd.DataFrame with columns: prompt_id + 5 trajectory features
    """
    N = hidden_states.shape[0]
    rows = []

    print(f"Computing trajectory features for {N} samples "
          f"({hidden_states.shape[1]} layers, best_layer={best_layer})...")

    for i in range(N):
        feats = compute_sample_trajectory(
            hidden_states[i],
            best_layer=best_layer,
            stability_threshold=stability_threshold,
        )
        feats["prompt_id"] = prompt_ids[i]
        rows.append(feats)

    df = pd.DataFrame(rows)
    # Reorder columns
    cols = ["prompt_id", "traj_total_variation", "traj_stabilization_layer",
            "traj_max_change_layer", "traj_early_commitment", "icr_score"]
    return df[cols]


def trajectory_class_analysis(
    traj_df: pd.DataFrame,
    labels: np.ndarray,
) -> dict:
    """
    Compute mean trajectory feature values per class for interpretability.

    Returns dict: {feature: {mean_correct, mean_hallucinated, delta}}
    """
    results = {}
    feat_cols = [c for c in traj_df.columns if c != "prompt_id"]

    for feat in feat_cols:
        vals = traj_df[feat].values.astype(float)
        mask_c = labels == 1
        mask_h = labels == 0

        mu_c = float(np.nanmean(vals[mask_c])) if mask_c.sum() > 0 else float("nan")
        mu_h = float(np.nanmean(vals[mask_h])) if mask_h.sum() > 0 else float("nan")
        delta = float(mu_c - mu_h) if not (np.isnan(mu_c) or np.isnan(mu_h)) else float("nan")

        results[feat] = {
            "mean_correct":      round(mu_c, 6),
            "mean_hallucinated": round(mu_h, 6),
            "delta":             round(delta, 6),
        }

    return results
