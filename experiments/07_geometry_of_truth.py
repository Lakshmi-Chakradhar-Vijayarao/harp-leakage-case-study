"""
07_geometry_of_truth.py — Linear geometry of correct vs hallucinated representations.

Phase 4, Experiment 7.

Scientific question
-------------------
Does truth have linear structure in Qwen 2.5 3B's representation space?
Concretely: at best_layer, do correct and hallucinated hidden states form separable
clusters with a dominant linear "truth direction" — a single axis in the 2048-dimensional
space that cleanly separates the two classes?

Connection to Marks & Tegmark 2023
------------------------------------
Marks & Tegmark showed that for several LLMs (GPT-3, LLaMA), hidden states of true vs.
false statements lie in geometrically separable regions. They found a "truth direction"
that is linearly extractable from hidden states.

MECH-INT contrast (null result at small scale):
MECH-INT's SVD subspace analysis on GPT-2 (117M) yielded AUROC ~0.50 at all layers —
no intrinsic geometric structure. GPT-2 was too small for the truth direction to dominate
the principal components. We expect Qwen 2.5 3B (3B parameters) to show the linear
geometry that GPT-2 could not.

Method
------
PCA on the 500-sample × 2048-dim hidden state matrix at best_layer.
PC1 and PC2 are the dominant variance directions.

Key metric: `pc1_correlation = Pearson r(PC1 projections, truth labels)`
  |r| > 0.4 → strong linear truth direction (PC1 aligns with correct vs. hallucinated)
  |r| < 0.2 → no dominant linear structure at this scale

Why this matters for the rest of HaRP
-----------------------------------------
If pc1_correlation > 0.4, it validates:
  1. Group C features (cos_correct, cos_hallucinated): class-mean distances are meaningful
     because the means lie on opposite ends of a real geometric axis.
  2. Steering vector (Exp 10): μ_correct - μ_hallucinated points in the truth direction —
     steering along it is geometrically well-motivated.
  3. The AUROC jump from A+B to A+B+C (Exp 09): the geometry is what makes representation
     signals work where entropy signals fail.

A weak pc1_correlation would suggest messy geometry and would predict smaller Group C
contributions — the experimental design is self-validating.

Produces
--------
  - results/plots/07_pca_scatter.png        — PCA scatter, correct=blue, hallucinated=red
  - results/logs/07_geometry_results.json   — pc1_correlation, cluster separation, direction
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.geometry_of_truth import geometry_analysis, compute_truth_direction

RESULTS_LOGS  = ROOT / "results" / "logs"
RESULTS_HS    = ROOT / "results" / "hidden_states"
RESULTS_PLOTS = ROOT / "results" / "plots"
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)


def load_best_layer() -> int:
    log_path = RESULTS_LOGS / "06_probing_results.json"
    if not log_path.exists():
        raise FileNotFoundError("Missing 06_probing_results.json — run 06 first.")
    with open(log_path) as f:
        return int(json.load(f)["best_layer"])


def plot_pca_scatter(proj: np.ndarray, labels: np.ndarray,
                     pc1_corr: float, sep: float,
                     best_layer: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(proj[labels == 1, 0], proj[labels == 1, 1],
               c="steelblue", alpha=0.5, s=20, label="Correct", rasterized=True)
    ax.scatter(proj[labels == 0, 0], proj[labels == 0, 1],
               c="crimson", alpha=0.5, s=20, label="Hallucinated", rasterized=True)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"Geometry of Truth — Layer {best_layer}\n"
        f"PC1 ↔ label correlation: r = {pc1_corr:.4f}  |  "
        f"Linear SVM: {sep:.4f}"
    )
    ax.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


def main():
    print("=" * 60)
    print("Experiment 07 — Geometry of Truth")
    print("=" * 60)

    hs_path = RESULTS_HS / "hidden_states.npz"
    if not hs_path.exists():
        print(f"Missing: {hs_path}\nRun 05 first.")
        sys.exit(1)

    best_layer = load_best_layer()
    print(f"Using best layer: L{best_layer}")

    data   = np.load(hs_path, allow_pickle=True)
    hs     = data["last_token_hs"].astype(np.float32)   # [N, L, D]
    labels = data["labels"].astype(int)

    h_layer = hs[:, best_layer, :]   # [N, D]
    h_corr  = h_layer[labels == 1]
    h_hall  = h_layer[labels == 0]

    print(f"Correct: {len(h_corr)} | Hallucinated: {len(h_hall)}")

    # ── Geometry analysis ─────────────────────────────────────────────────────
    result = geometry_analysis(h_corr, h_hall)

    pc1_corr = result["pc1_correlation"]
    sep      = result["separability"]
    expl     = result["explained_variance"]

    print(f"\nResults:")
    print(f"  PC1 explains {expl[0]*100:.1f}% variance")
    print(f"  PC1 ↔ truth label:  r = {pc1_corr:.4f}")
    print(f"  Linear SVM (5-fold): {sep:.4f}")
    if abs(pc1_corr) > 0.4:
        print("  → Strong linear truth direction (|r| > 0.4)")
        print("  → Connects to Marks & Tegmark 2023: truth is linearly encoded")
    else:
        print("  → Weak linear structure — classes overlap")

    # ── Truth direction ───────────────────────────────────────────────────────
    tv = compute_truth_direction(h_corr, h_hall)
    tv_path = RESULTS_HS / "truth_direction.npy"
    np.save(tv_path, tv)
    print(f"\nTruth direction vector saved: {tv_path.name} (shape: {tv.shape})")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_pca_scatter(
        result["projections"], labels,
        pc1_corr, sep, best_layer,
        RESULTS_PLOTS / "07_pca_scatter.png"
    )

    # ── Save log ──────────────────────────────────────────────────────────────
    log = {
        "experiment":       "07_geometry_of_truth",
        "best_layer":       best_layer,
        "pc1_correlation":  pc1_corr,
        "separability":     sep,
        "explained_variance": expl,
        "n_correct":        len(h_corr),
        "n_hallucinated":   len(h_hall),
        "interpretation":   (
            "strong linear truth direction"
            if abs(pc1_corr) > 0.4
            else "weak linear structure"
        ),
    }
    log_path = RESULTS_LOGS / "07_geometry_results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved: {log_path.name}")

    print("\nExperiment 07 complete. Next: run 08_logit_lens.py")


if __name__ == "__main__":
    main()
