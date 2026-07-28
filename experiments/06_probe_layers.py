"""
06_probe_layers.py — Layer-wise probing accuracy curve.

Phase 4, Experiment 6.

Scientific question
-------------------
At which transformer layer does Qwen 2.5 3B first encode a strong hallucination-
predictive signal in its hidden states? And does that signal persist or fade as we
go deeper?

This experiment produces the most important single number in HaRP: best_layer.
Every Group C feature, the steering vector, and the logit lens analysis all depend on
this value. A wrong best_layer propagates errors through all downstream analyses.

Method
------
For each of Qwen 2.5 3B's 37 hidden state positions (embedding + 36 transformer layers),
train a logistic regression probe on the last-token hidden state (shape: [N, hidden_dim])
using 5-fold stratified cross-validation. AUROC is the primary metric.

Why last-token hidden states (not mean-pooled)?
  For autoregressive LMs, the last token accumulates full left-context via causal
  attention — it is the "decision point" where the model encodes its best estimate
  of the answer. MECH-INT (GPT-2) found last-token at L8 outperforms mean-pooled
  (AUROC 0.604 vs 0.583). HaRP applies the same principle to Qwen 2.5 3B.

Why logistic regression (not MLP or gradient boosting)?
  Linear probes are interpretable: their coefficients identify which dimensions carry
  the signal (relevant for Group C feature design). They are also less prone to
  overfitting with N=500 and 2048 hidden dimensions.

Expected pattern
----------------
  Early layers (L0–L10): AUROC ~0.55–0.65, signal is weak and rising
  Mid layers (L11–L25):  AUROC rising, ~0.65–0.80
  Late layers (L26–L36): Stable plateau at maximum AUROC (~0.80–0.89)

The plateau (not a sharp peak) indicates that the hallucination fingerprint is stably
encoded from mid-network through the final layers. best_layer is selected from
the plateau onset — the first layer that achieves maximum CV AUROC.

Connection to MECH-INT
----------------------
MECH-INT (GPT-2, 12 layers) found the peak at L8–L9 (~67–75% depth).
For Qwen 2.5 3B (36 layers), the equivalent depth would be ~L24–L27.
Empirical best_layer may differ — the probing curve determines this without assumption.

Produces
--------
  - results/plots/06_probing_curve.png         — AUROC vs. layer (Plot 2)
  - results/logs/06_probing_results.json       — per-layer AUROC + best_layer

best_layer feeds into:
  - Experiment 07 (geometry of truth — PCA at best_layer)
  - Experiment 08 (logit lens reference layer)
  - Experiment 09 (Group C: cos_correct, cos_hallucinated, probe_confidence)
  - Experiment 10 (steering vector at best_layer)
  - Experiments 20, 22, 23 (trajectory signals reference; steering)
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

from src.analysis.probing import probe_all_layers, fit_probe_at_layer

RESULTS_LOGS = ROOT / "results" / "logs"
RESULTS_HS   = ROOT / "results" / "hidden_states"
RESULTS_PLOTS= ROOT / "results" / "plots"
RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)


def plot_probing_curve(results: list, best_layer: int, out_path: Path) -> None:
    layers = [r["layer"] for r in results]
    aurocs = [r["auroc_mean"] for r in results]
    errs   = [r["auroc_std"]  for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(layers,
                    [a - e for a, e in zip(aurocs, errs)],
                    [a + e for a, e in zip(aurocs, errs)],
                    alpha=0.25, color="steelblue")
    ax.plot(layers, aurocs, "-o", markersize=4, color="steelblue", label="AUROC (5-fold CV)")
    ax.axvline(best_layer, color="crimson", linestyle="--",
               label=f"Best layer: L{best_layer} ({max(aurocs):.4f})")
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.6, label="Random baseline")

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Probing AUROC")
    ax.set_title("Layer-wise Probing Accuracy: Hallucination Prediction\n"
                 "Qwen 2.5 3B — last-token hidden state")
    ax.legend()
    ax.set_xlim(0, max(layers))
    ax.set_ylim(0.45, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


def main():
    print("=" * 60)
    print("Experiment 06 — Layer-wise Probing")
    print("=" * 60)

    hs_path = RESULTS_HS / "hidden_states.npz"
    if not hs_path.exists():
        print(f"Missing: {hs_path}\nRun 05_extract_hidden_states.py first.")
        sys.exit(1)

    print("Loading hidden states...")
    data     = np.load(hs_path, allow_pickle=True)
    hs       = data["last_token_hs"].astype(np.float32)  # [N, L, D]
    labels   = data["labels"].astype(int)

    n, num_layers, hidden_dim = hs.shape
    print(f"Shape: {n} samples × {num_layers} layers × {hidden_dim} dim")
    print(f"Labels: {int((labels==1).sum())} correct / {int((labels==0).sum())} hallucinated")

    # ── Probe every layer ─────────────────────────────────────────────────────
    print("\nProbing all layers...")
    results = probe_all_layers(hs, labels, n_splits=5)

    best = max(results, key=lambda r: r["auroc_mean"])
    best_layer = best["layer"]
    print(f"\nBest layer: L{best_layer}  AUROC = {best['auroc_mean']:.4f} ± {best['auroc_std']:.4f}")

    # ── Fit final probe at best layer ─────────────────────────────────────────
    probe = fit_probe_at_layer(hs, labels, best_layer)
    import joblib, pickle
    probe_path = RESULTS_HS / "best_layer_probe.pkl"
    with open(probe_path, "wb") as f:
        pickle.dump(probe, f)
    print(f"Best-layer probe saved: {probe_path.name}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_probing_curve(results, best_layer,
                       RESULTS_PLOTS / "06_probing_curve.png")

    # ── Save log ──────────────────────────────────────────────────────────────
    log = {
        "experiment":     "06_probe_layers",
        "n_samples":      n,
        "num_layers":     num_layers,
        "best_layer":     best_layer,
        "best_auroc":     best["auroc_mean"],
        "best_auroc_std": best["auroc_std"],
        "all_layers":     results,
    }
    log_path = RESULTS_LOGS / "06_probing_results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved: {log_path.name}")

    print(f"\nBEST LAYER = {best_layer}  (store this for experiments 07–11)")
    print("Experiment 06 complete. Next: run 07_geometry_of_truth.py")


if __name__ == "__main__":
    main()
