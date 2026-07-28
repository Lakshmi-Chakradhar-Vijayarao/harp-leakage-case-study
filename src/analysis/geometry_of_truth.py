"""
geometry_of_truth.py — Linear geometry of correct vs hallucinated representations.

Based on: Marks & Tegmark 2023 "The Geometry of Truth"
Hypothesis: truth has linear structure in hidden state space — a "truth direction"
that separates correct from hallucinated representations.

If PC1 correlates strongly with truth labels (|r| > 0.4), this is evidence
that hallucination is geometrically encoded in the residual stream, not
just distributed across all directions randomly.

Produces: Plot 1 (PCA scatter, correct=blue / hallucinated=red)
"""

import numpy as np
from typing import Dict
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def geometry_analysis(
    h_correct: np.ndarray,
    h_hallucinated: np.ndarray,
    n_components: int = 2,
    cv_folds: int = 5,
) -> Dict:
    """
    Analyse the geometric separation between correct and hallucinated hidden states.

    Parameters
    ----------
    h_correct      : np.ndarray [N_c, hidden_dim] — hidden states for correct responses
    h_hallucinated : np.ndarray [N_h, hidden_dim] — hidden states for hallucinated responses

    Returns
    -------
    dict with keys:
        pc1_correlation   : float — Pearson r between PC1 projection and truth label
                            |r| > 0.4 = linear truth direction exists
        separability      : float — linear SVM 5-fold CV accuracy (upper bound)
        projections       : np.ndarray [N, 2] — PCA-reduced coordinates for plotting
        labels            : np.ndarray [N] — 1=correct, 0=hallucinated
        explained_variance: list[float] — variance ratio per PC
        pca               : fitted PCA object (for projecting new points)
    """
    all_h  = np.vstack([h_correct, h_hallucinated])
    labels = np.array([1] * len(h_correct) + [0] * len(h_hallucinated))

    # Standardise before PCA
    scaler      = StandardScaler()
    all_h_scaled = scaler.fit_transform(all_h)

    # PCA
    pca        = PCA(n_components=n_components, random_state=42)
    projected  = pca.fit_transform(all_h_scaled)    # [N, 2]
    pc1        = projected[:, 0]

    pc1_corr   = float(np.corrcoef(pc1, labels)[0, 1])
    expl_var   = pca.explained_variance_ratio_.tolist()

    # Linear separability via LinearSVC
    svm        = LinearSVC(max_iter=2000, random_state=42, class_weight="balanced")
    sep_scores = cross_val_score(svm, all_h_scaled, labels, cv=cv_folds, scoring="accuracy")
    separability = float(sep_scores.mean())

    # Summary print
    print(f"Geometry of Truth:")
    print(f"  PC1 explains {expl_var[0]*100:.1f}% variance")
    print(f"  PC1 ↔ truth label correlation: r = {pc1_corr:.4f}")
    print(f"  Linear SVM separability: {separability:.4f}")
    if abs(pc1_corr) > 0.4:
        print("  → Strong linear truth direction found (|r| > 0.4)")
    else:
        print("  → Weak linear structure — representations overlap significantly")

    return {
        "pc1_correlation":    pc1_corr,
        "separability":       separability,
        "projections":        projected,
        "labels":             labels,
        "explained_variance": expl_var,
        "pca":                pca,
        "scaler":             scaler,
    }


def compute_truth_direction(
    h_correct: np.ndarray,
    h_hallucinated: np.ndarray,
) -> np.ndarray:
    """
    Compute the mean-difference truth direction vector.
    Used as a lightweight alternative to PCA for the steering vector.

    Returns unit vector of shape [hidden_dim].
    """
    mu_c = h_correct.mean(axis=0)
    mu_h = h_hallucinated.mean(axis=0)
    direction = mu_c - mu_h
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return direction
    return direction / norm
