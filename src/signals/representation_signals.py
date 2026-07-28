"""
representation_signals.py — Group C: Hidden-state geometry features.

These features are computed AFTER Phase 4 identifies the best probing layer.
They require HuggingFace hidden states — never compute in Ollama mode.

Four signals:
  1. cos_dist_correct      — cosine distance from h to μ_correct
  2. cos_dist_hallucinated — cosine distance from h to μ_hallucinated
  3. hidden_norm           — L2 norm of the hidden state at target layer
  4. probe_confidence      — LR probe P(correct) at target layer

These four are the differentiator between v1 and v2 of the failure estimator.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance = 1 - cosine_similarity.
    Range: [0, 2]. 0 = identical direction, 2 = opposite.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return float("nan")
    sim = np.dot(a, b) / (norm_a * norm_b)
    return float(1.0 - np.clip(sim, -1.0, 1.0))


def cos_dist_to_correct(
    h: np.ndarray,
    mu_correct: np.ndarray,
) -> float:
    """
    Cosine distance from a single hidden state h to the mean correct vector.
    Low distance = h is geometrically close to correct representations.
    """
    return cosine_distance(h, mu_correct)


def cos_dist_to_hallucinated(
    h: np.ndarray,
    mu_hallucinated: np.ndarray,
) -> float:
    """
    Cosine distance from h to the mean hallucinated vector.
    Low distance = h is close to hallucinated representations (bad sign).
    """
    return cosine_distance(h, mu_hallucinated)


def hidden_norm(h: np.ndarray) -> float:
    """
    L2 norm of the hidden state vector.
    Abnormally high or low norms can indicate distributional shift.
    """
    return float(np.linalg.norm(h))


def probe_confidence(
    h: np.ndarray,
    probe,  # fitted sklearn LogisticRegression
) -> float:
    """
    Probability output of the best-layer LR probe: P(correct | h).
    High value = probe thinks this hidden state looks like a correct answer.
    """
    h_2d = h.reshape(1, -1)
    return float(probe.predict_proba(h_2d)[0, 1])


def compute_class_means(
    hidden_states: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute μ_correct and μ_hallucinated from a set of labeled hidden states.

    Parameters
    ----------
    hidden_states : np.ndarray [N, hidden_dim]
    labels        : np.ndarray [N] — 1=correct, 0=hallucinated

    Returns (mu_correct, mu_hallucinated) — both shape [hidden_dim]
    """
    mu_correct      = hidden_states[labels == 1].mean(axis=0)
    mu_hallucinated = hidden_states[labels == 0].mean(axis=0)
    return mu_correct, mu_hallucinated


def compute_oof_group_c(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    prompt_ids: List,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Compute out-of-fold (OOF) Group C features to avoid data leakage.

    The core problem with the standard approach: the probe and class centroids are
    fitted on ALL samples, then evaluated on those same samples. probe_conf has
    AUROC=1.0 in-sample. This inflates the failure estimator's AUROC.

    Fix: for each CV fold, fit the probe and compute centroids on the TRAINING
    fold only, then compute probe_conf / cos_correct / cos_hallucinated on the
    held-out TEST fold. h_norm is label-free and unaffected.

    Parameters
    ----------
    hidden_states : np.ndarray [N, hidden_dim]  — hidden states at best_layer
    labels        : np.ndarray [N]              — 1=correct, 0=hallucinated
    prompt_ids    : list [N]                    — prompt identifiers for merging
    n_splits      : int                         — number of CV folds (default 5)

    Returns
    -------
    pd.DataFrame with columns:
        prompt_id, probe_conf_oof, cos_correct_oof, cos_hallucinated_oof, h_norm
    Rows are in the same order as the input hidden_states.
    """
    N = len(labels)
    probe_conf_oof      = np.full(N, float("nan"))
    cos_correct_oof     = np.full(N, float("nan"))
    cos_hallucinated_oof = np.full(N, float("nan"))
    h_norms             = np.array([float(np.linalg.norm(hidden_states[i]))
                                    for i in range(N)])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(hidden_states, labels)):
        X_train, X_test = hidden_states[train_idx], hidden_states[test_idx]
        y_train = labels[train_idx]

        # Fit probe on training fold only
        probe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, class_weight="balanced",
                max_iter=1000, random_state=42,
            )),
        ])
        probe.fit(X_train, y_train)

        # Compute centroids from training fold only
        mu_correct  = X_train[y_train == 1].mean(axis=0)
        mu_hall     = X_train[y_train == 0].mean(axis=0)

        # Compute OOF predictions for test fold
        probe_conf_oof[test_idx] = probe.predict_proba(X_test)[:, 1]

        for i, global_i in enumerate(test_idx):
            h = X_test[i]
            cos_correct_oof[global_i]      = cosine_distance(h, mu_correct)
            cos_hallucinated_oof[global_i] = cosine_distance(h, mu_hall)

    return pd.DataFrame({
        "prompt_id":            prompt_ids,
        "probe_conf_oof":       probe_conf_oof,
        "cos_correct_oof":      cos_correct_oof,
        "cos_hallucinated_oof": cos_hallucinated_oof,
        "h_norm":               h_norms,
    })


def extract_all(
    h: np.ndarray,
    mu_correct: np.ndarray,
    mu_hallucinated: np.ndarray,
    probe,
) -> dict:
    """
    Compute all 4 Group C features for a single hidden state vector.

    Returns dict with keys:
        cos_correct      : cosine distance to μ_correct
        cos_hallucinated : cosine distance to μ_hallucinated
        h_norm           : L2 norm of h
        probe_conf       : LR probe confidence P(correct)
    """
    return {
        "cos_correct":      cos_dist_to_correct(h, mu_correct),
        "cos_hallucinated": cos_dist_to_hallucinated(h, mu_hallucinated),
        "h_norm":           hidden_norm(h),
        "probe_conf":       probe_confidence(h, probe),
    }
