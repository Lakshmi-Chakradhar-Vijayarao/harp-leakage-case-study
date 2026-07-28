"""
uncertainty_decomposition.py — Formal epistemic-aleatoric uncertainty decomposition.

Core novel theoretical contribution of HaRP.

Classical Bayesian decomposition (Hora 1996, Kendall & Gal 2017):
  Total uncertainty = Epistemic + Aleatoric
  Epistemic: model doesn't know → reducible with more data/training
  Aleatoric: inherent in the question → irreducible

For LLMs, we operationalize this using observable signals:

  EPISTEMIC (model knowledge gap):
    HIGH token entropy + LOW semantic consistency
    Model is uncertain AND different runs disagree on meaning
    → The model lacks stable knowledge about this topic
    → Correct action: ABSTAIN

  ALEATORIC (question ambiguity):
    HIGH token entropy + HIGH semantic consistency
    Model is uncertain BUT all runs express the same kind of uncertainty
    → The question is genuinely hard/ambiguous, not a model failure
    → Correct action: ACCEPT with uncertainty caveat

  CONFIDENT HALLUCINATION (dangerous failure):
    LOW token entropy + cos_hallucinated < cos_correct
    Model is confident BUT representation geometry says it's wrong
    cos_correct / cos_hallucinated are cosine DISTANCES (1 − similarity):
    lower = closer. cos_hallucinated < cos_correct means h is geometrically
    closer to the hallucinated class mean than to the correct class mean.
    → Only catchable with representation signals, NOT by entropy alone
    → Correct action: REGENERATE with steering

  SYCOPHANTIC FAILURE:
    LOW token entropy + LOW semantic consistency
    Model is confident BUT different runs diverge on meaning
    → Model is confidently inconsistent (agrees with whatever framing is given)
    → Correct action: REGENERATE

This 4-quadrant taxonomy is the core analytical framework.
No existing paper maps all four failure modes to their signal signatures.

Reference:
  Kendall & Gal (2017). What Uncertainties Do We Need in Bayesian Deep Learning?
  Hou et al. (2024). Decomposing Uncertainty for LLMs through Input Clarification Ensembling. ICML.
  Kuhn et al. (2023). Semantic Uncertainty. ICLR.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ── Quadrant definitions ─────────────────────────────────────────────────────

QUADRANT_DEFINITIONS = {
    "epistemic": {
        "label": "Epistemic\n(Model Ignorance)",
        "condition": "high entropy + low consistency",
        "action": "ABSTAIN",
        "color": "#e15759",
        "description": "Model is uncertain and unstable across runs. "
                        "Reducible with more training data.",
    },
    "aleatoric": {
        "label": "Aleatoric\n(Question Ambiguity)",
        "condition": "high entropy + high consistency",
        "action": "ACCEPT (with caveat)",
        "color": "#f28e2b",
        "description": "Model consistently expresses the same uncertainty. "
                        "Irreducible — the question is genuinely ambiguous.",
    },
    "confident_correct": {
        "label": "Confident Correct",
        "condition": "low entropy + high consistency",
        "action": "ACCEPT",
        "color": "#59a14f",
        "description": "Model is confident and stable. Safe to accept.",
    },
    "confident_hallucination": {
        "label": "Confident\nHallucination",
        "condition": "low entropy + low consistency (or cos_hall < cos_correct)",
        "action": "REGENERATE",
        "color": "#76b7b2",
        "description": "Model is confident but unstable or geometrically misaligned. "
                        "Dangerous — only catchable with Group C signals.",
    },
}


# ── Signal-based quadrant assignment ─────────────────────────────────────────

def assign_quadrant(
    entropy: float,
    consistency: float,
    entropy_threshold: float = 0.5,
    consistency_threshold: float = 0.5,
    cos_correct: Optional[float] = None,
    cos_hallucinated: Optional[float] = None,
) -> str:
    """
    Assign a sample to one of 4 uncertainty quadrants.

    Parameters
    ----------
    entropy              : predictive entropy (Group A)
    consistency          : semantic consistency score (Group B) in [0,1]
    entropy_threshold    : cutoff for "high entropy"
    consistency_threshold: cutoff for "high consistency"
    cos_correct          : cosine DISTANCE to μ_correct (1 − sim); lower = closer to correct mean
    cos_hallucinated     : cosine DISTANCE to μ_hallucinated; lower = closer to hallucinated mean

    Returns
    -------
    str: one of "epistemic", "aleatoric", "confident_correct", "confident_hallucination"
    """
    high_entropy = entropy > entropy_threshold
    high_consist = consistency > consistency_threshold

    if high_entropy and not high_consist:
        return "epistemic"
    elif high_entropy and high_consist:
        return "aleatoric"
    elif not high_entropy and high_consist:
        # Check geometry if available
        if cos_correct is not None and cos_hallucinated is not None:
            # cos_* are cosine DISTANCES (lower = closer).
            # cos_hallucinated < cos_correct → h is closer to hallucinated mean.
            if cos_hallucinated < cos_correct:
                return "confident_hallucination"
        return "confident_correct"
    else:  # low entropy + low consistency
        return "confident_hallucination"


def assign_quadrants_df(
    df: pd.DataFrame,
    entropy_col: str = "entropy",
    consistency_col: str = "consistency_score",
    cos_correct_col: Optional[str] = "cos_correct",
    cos_hallucinated_col: Optional[str] = "cos_hallucinated",
    entropy_threshold: Optional[float] = None,
    consistency_threshold: Optional[float] = None,
) -> pd.Series:
    """
    Assign quadrant labels to all rows in a DataFrame.

    Thresholds default to median values if not specified (data-adaptive).
    """
    if entropy_threshold is None:
        entropy_threshold = float(df[entropy_col].median())
    if consistency_threshold is None:
        consistency_threshold = float(df[consistency_col].median())

    has_geometry = (
        cos_correct_col is not None and cos_correct_col in df.columns and
        cos_hallucinated_col is not None and cos_hallucinated_col in df.columns
    )

    def _assign_row(row):
        return assign_quadrant(
            entropy=row[entropy_col],
            consistency=row[consistency_col],
            entropy_threshold=entropy_threshold,
            consistency_threshold=consistency_threshold,
            cos_correct=row.get(cos_correct_col) if has_geometry else None,
            cos_hallucinated=row.get(cos_hallucinated_col) if has_geometry else None,
        )

    return df.apply(_assign_row, axis=1)


# ── Decomposition metrics ─────────────────────────────────────────────────────

def compute_decomposition_metrics(
    df: pd.DataFrame,
    quadrant_col: str = "quadrant",
    label_col: str = "label",
) -> Dict:
    """
    Compute key metrics for each uncertainty quadrant.

    Returns dict with per-quadrant:
        count, fraction, hallucination_rate, mean_entropy, mean_consistency
    """
    results = {}
    total = len(df)

    for q_name in QUADRANT_DEFINITIONS.keys():
        mask = df[quadrant_col] == q_name
        sub = df[mask]
        n = len(sub)

        if n == 0:
            results[q_name] = {
                "count": 0, "fraction": 0.0,
                "hallucination_rate": float("nan"),
            }
            continue

        hall_rate = float(1 - sub[label_col].mean()) if label_col in sub.columns else float("nan")

        results[q_name] = {
            "count":             n,
            "fraction":          round(n / total, 4),
            "hallucination_rate": round(hall_rate, 4) if not np.isnan(hall_rate) else None,
            "meta":              QUADRANT_DEFINITIONS[q_name],
        }

    return results


def compute_epistemic_aleatoric_ratio(decomp_metrics: Dict) -> float:
    """
    Epistemic-to-aleatoric ratio: higher = model knowledge gaps dominate.
    Values >> 1 → model needs better training
    Values << 1 → questions are inherently ambiguous
    """
    e = decomp_metrics.get("epistemic", {}).get("count", 0)
    a = decomp_metrics.get("aleatoric", {}).get("count", 0)
    if a == 0:
        return float("inf")
    return round(e / a, 4)


def find_optimal_thresholds(
    df: pd.DataFrame,
    entropy_col: str = "entropy",
    consistency_col: str = "consistency_score",
    label_col: str = "label",
    n_grid: int = 20,
) -> Tuple[float, float, float]:
    """
    Grid search for entropy and consistency thresholds that maximize the
    separation of confident hallucinations from confident correct responses.

    Returns (best_entropy_threshold, best_consistency_threshold, best_score)
    where score = hallucination recall in confident_hallucination quadrant
                + (1 - hallucination_rate) in confident_correct quadrant.
    """
    ent_vals = np.percentile(
        df[entropy_col].dropna(), np.linspace(20, 80, n_grid))
    con_vals = np.percentile(
        df[consistency_col].dropna(), np.linspace(20, 80, n_grid))

    best_score = -1.0
    best_et = float(df[entropy_col].median())
    best_ct = float(df[consistency_col].median())

    for et in ent_vals:
        for ct in con_vals:
            quadrants = assign_quadrants_df(
                df,
                entropy_col=entropy_col,
                consistency_col=consistency_col,
                entropy_threshold=et,
                consistency_threshold=ct,
            )
            tmp = df.copy()
            tmp["_q"] = quadrants

            # Score: fraction of hallucinations in confident_hallucination quadrant
            # + fraction of corrects in confident_correct quadrant
            ch_mask = tmp["_q"] == "confident_hallucination"
            cc_mask = tmp["_q"] == "confident_correct"

            hall_in_ch = (tmp[ch_mask][label_col] == 0).sum() / max(1, (df[label_col] == 0).sum())
            corr_in_cc = (tmp[cc_mask][label_col] == 1).sum() / max(1, (df[label_col] == 1).sum())
            score = hall_in_ch + corr_in_cc

            if score > best_score:
                best_score = score
                best_et = et
                best_ct = ct

    return float(best_et), float(best_ct), float(best_score)


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_uncertainty_quadrants(
    df: pd.DataFrame,
    quadrant_col: str = "quadrant",
    entropy_col: str = "entropy",
    consistency_col: str = "consistency_score",
    label_col: str = "label",
    entropy_threshold: float = 0.5,
    consistency_threshold: float = 0.5,
    out_path=None,
    title: str = "HaRP: Uncertainty Decomposition",
) -> None:
    """
    Figure 1 of the paper: 4-quadrant uncertainty decomposition scatter plot.

    X-axis: semantic consistency (Group B)
    Y-axis: predictive entropy (Group A)
    Color: quadrant assignment
    Marker: correct (circle) vs hallucinated (cross)
    Dashed lines: thresholds
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    color_map = {q: QUADRANT_DEFINITIONS[q]["color"] for q in QUADRANT_DEFINITIONS}
    label_map = {q: QUADRANT_DEFINITIONS[q]["label"] for q in QUADRANT_DEFINITIONS}

    fig, ax = plt.subplots(figsize=(9, 7))

    for quadrant, color in color_map.items():
        mask = df[quadrant_col] == quadrant
        sub = df[mask]
        if len(sub) == 0:
            continue

        # Correct: circle, Hallucinated: X
        correct_mask   = sub[label_col] == 1
        hallucinated_m = sub[label_col] == 0

        ax.scatter(
            sub[correct_mask][consistency_col],
            sub[correct_mask][entropy_col],
            c=color, marker="o", s=25, alpha=0.55, zorder=3,
        )
        ax.scatter(
            sub[hallucinated_m][consistency_col],
            sub[hallucinated_m][entropy_col],
            c=color, marker="x", s=40, alpha=0.75, linewidths=1.2, zorder=4,
        )

    # Threshold lines
    ax.axhline(entropy_threshold, color="black", linestyle="--",
               linewidth=1.0, alpha=0.6, zorder=5)
    ax.axvline(consistency_threshold, color="black", linestyle="--",
               linewidth=1.0, alpha=0.6, zorder=5)

    # Quadrant annotations (background text)
    x_lo, x_hi = ax.get_xlim() if ax.get_xlim() != (0, 1) else (0, 1)
    y_lo, y_hi = ax.get_ylim() if ax.get_ylim() != (0, 1) else (0, 2)
    for q_name, q_def in QUADRANT_DEFINITIONS.items():
        xc = (0.05 + consistency_threshold * 0.5) if "hallucination" in q_name or "epistemic" in q_name else (consistency_threshold + 0.5 * (1 - consistency_threshold))
        yc = (entropy_threshold + 0.4) if q_name in ("epistemic", "aleatoric") else (entropy_threshold * 0.5)
        ax.text(xc, yc, q_def["label"],
                ha="center", va="center", fontsize=8, alpha=0.35,
                color=q_def["color"], fontweight="bold", zorder=2)

    # Legend
    legend_elements = [
        Patch(facecolor=color_map[q], label=label_map[q].replace("\n", " "))
        for q in QUADRANT_DEFINITIONS
    ] + [
        Line2D([0], [0], marker="o", color="grey", label="Correct", markersize=6, linestyle="None"),
        Line2D([0], [0], marker="x", color="grey", label="Hallucinated", markersize=7,
               linestyle="None", markeredgewidth=1.5),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_xlabel("Semantic Consistency (Group B) →", fontsize=11)
    ax.set_ylabel("Predictive Entropy (Group A) →", fontsize=11)
    ax.set_title(
        f"{title}\n"
        "Dashed lines = data-adaptive thresholds | "
        "○ correct | × hallucinated",
        fontsize=11,
    )
    ax.grid(alpha=0.2, zorder=0)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Plot saved: {str(out_path).split('/')[-1]}")
    plt.close()


def plot_quadrant_hallucination_rates(
    decomp_metrics: Dict,
    out_path=None,
) -> None:
    """
    Bar chart: hallucination rate per quadrant.
    Validates that the taxonomy aligns with ground truth labels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quadrants = list(QUADRANT_DEFINITIONS.keys())
    rates = [
        decomp_metrics.get(q, {}).get("hallucination_rate") or 0.0
        for q in quadrants
    ]
    colors = [QUADRANT_DEFINITIONS[q]["color"] for q in quadrants]
    labels = [QUADRANT_DEFINITIONS[q]["label"].replace("\n", " ") for q in quadrants]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(quadrants)), rates, color=colors, alpha=0.85, zorder=3)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2., val + 0.01,
                f"{val:.2%}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(range(len(quadrants)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Hallucination Rate")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Hallucination Rate by Uncertainty Quadrant\n"
        "Validates that epistemic and confident_hallucination quadrants have highest error rates"
    )
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.6)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Plot saved: {str(out_path).split('/')[-1]}")
    plt.close()
