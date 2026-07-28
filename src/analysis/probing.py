"""
probing.py — Layer-wise linear probing of hidden states.

For each transformer layer, train a logistic regression probe on the
last-token hidden state and evaluate hallucination prediction accuracy.

Produces the "probing accuracy curve" (Plot 2) and identifies best_layer —
the layer whose hidden state is most predictive of hallucination.

best_layer feeds into:
  - Group C feature extraction (representation_signals.py)
  - Activation steering (steering/activation_steering.py)
  - Logit lens comparison (analysis/logit_lens.py)
"""

import numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score


def _build_probe(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, class_weight="balanced",
            max_iter=1000, random_state=42
        )),
    ])


def probe_single_layer(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    layer_idx: int,
    n_splits: int = 5,
) -> Dict:
    """
    Train and cross-validate a probe at one layer.

    Parameters
    ----------
    hidden_states : np.ndarray [N, num_layers, hidden_dim]
                    Last-token hidden states for all samples.
    labels        : np.ndarray [N] — 1=correct, 0=hallucinated
    layer_idx     : int — which layer to probe

    Returns dict: layer, auroc_mean, auroc_std, acc_mean
    """
    X = hidden_states[:, layer_idx, :]  # [N, hidden_dim]
    y = labels

    # Reduce splits if class is small
    min_class = min(int(y.sum()), int((y == 0).sum()))
    actual_splits = max(2, min(n_splits, min_class))

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    scoring = {
        "auroc": make_scorer(roc_auc_score, response_method="predict_proba"),
        "accuracy": "accuracy",
    }
    cv = cross_validate(_build_probe(), X, y, cv=skf, scoring=scoring)

    return {
        "layer":      layer_idx,
        "auroc_mean": float(cv["test_auroc"].mean()),
        "auroc_std":  float(cv["test_auroc"].std()),
        "acc_mean":   float(cv["test_accuracy"].mean()),
    }


def probe_all_layers(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
) -> List[Dict]:
    """
    Run probing across all layers and identify the best layer.

    Parameters
    ----------
    hidden_states : np.ndarray [N, num_layers, hidden_dim]
    labels        : np.ndarray [N]

    Returns list of dicts sorted by layer index.
    Prints a summary table and marks the peak layer.
    """
    num_layers = hidden_states.shape[1]
    print(f"Probing {num_layers} layers, N={len(labels)}, "
          f"class balance: {int(labels.sum())}/{int((labels==0).sum())}")
    print("-" * 55)

    results = []
    for layer_idx in range(num_layers):
        r = probe_single_layer(hidden_states, labels, layer_idx, n_splits)
        results.append(r)
        tag = "embed" if layer_idx == 0 else f"L{layer_idx:2d}"
        print(f"  {tag:<8}  AUROC = {r['auroc_mean']:.4f} ± {r['auroc_std']:.4f}")

    peak = max(results, key=lambda r: r["auroc_mean"])
    print("-" * 55)
    print(f"  Best layer: L{peak['layer']} → AUROC {peak['auroc_mean']:.4f}")

    return results


def fit_probe_at_layer(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    layer_idx: int,
) -> Pipeline:
    """
    Fit a probe on the full dataset at a specific layer.
    Returns the fitted Pipeline (for use in Group C feature extraction).
    """
    X = hidden_states[:, layer_idx, :]
    probe = _build_probe()
    probe.fit(X, labels)
    return probe
