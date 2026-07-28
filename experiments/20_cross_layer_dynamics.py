"""
20_cross_layer_dynamics.py — Group D: Cross-layer hidden state trajectory signals.

Phase 11, Experiment 20.
Computes per-sample hidden state trajectory features from the existing
hidden_states.npz (all 37 layers × 300 samples × 2048 dims).

Zero additional model inference required — everything derives from data
already collected in Experiment 05.

Theoretical basis:
  ICR Probe (Zhang et al., ACL 2025) shows cross-layer hidden state dynamics
  are more discriminative than single-layer static probes. We extend this
  insight into a governance feature set.

The 5 trajectory features (Group D):
  traj_total_variation    : total path length of hidden state trajectory
  traj_stabilization_layer: normalised layer where representation settles
  traj_max_change_layer   : normalised layer with the biggest single jump
  traj_early_commitment   : drift from layer 1 to the critical probing layer
  icr_score               : fraction of total change occurring at the best layer

Outputs:
  results/hidden_states/trajectory_features.csv  ← per-sample Group D features
  results/logs/20_cross_layer_dynamics.json
  results/plots/20_trajectory_comparison.png
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.signals.trajectory_signals import (
    compute_trajectory_features,
    trajectory_class_analysis,
)
from src.evaluation.metrics import compute_auroc

RESULTS_LOGS  = ROOT / "results" / "logs"
RESULTS_HS    = ROOT / "results" / "hidden_states"
RESULTS_PLOTS = ROOT / "results" / "plots"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)


def load_prerequisites() -> tuple:
    hs_path    = RESULTS_HS / "hidden_states.npz"
    probe_path = RESULTS_LOGS / "06_probing_results.json"

    for p in [hs_path, probe_path]:
        if not p.exists():
            print(f"Missing: {p}")
            print("Run experiments 05 and 06 first.")
            sys.exit(1)

    data = np.load(hs_path, allow_pickle=True)
    with open(probe_path) as f:
        best_layer = int(json.load(f)["best_layer"])

    return data, best_layer


def plot_trajectory_comparison(class_analysis: dict, out_path: Path) -> None:
    """
    Plot mean trajectory feature values for correct vs hallucinated samples.
    Side-by-side bar chart — one pair per feature.
    """
    features   = list(class_analysis.keys())
    mu_correct = [class_analysis[f]["mean_correct"]      for f in features]
    mu_hall    = [class_analysis[f]["mean_hallucinated"] for f in features]

    x   = np.arange(len(features))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))

    bars_c = ax.bar(x - w/2, mu_correct, w, label="Correct",      color="#4e79a7", alpha=0.85)
    bars_h = ax.bar(x + w/2, mu_hall,    w, label="Hallucinated", color="#e15759", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f.replace("traj_", "").replace("_", "\n") for f in features],
        fontsize=9,
    )
    ax.set_ylabel("Mean Feature Value")
    ax.set_title(
        "Group D: Cross-Layer Trajectory Features\n"
        "Correct vs Hallucinated Responses — HaRP Extension of ICR Probe (ACL 2025)"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Delta annotations
    for i, (c, h) in enumerate(zip(mu_correct, mu_hall)):
        delta = c - h
        y_pos = max(c, h) + 0.005 * max(abs(c), abs(h), 0.001)
        ax.annotate(
            f"Δ={delta:+.3f}",
            xy=(x[i], y_pos),
            ha="center", va="bottom", fontsize=7, color="black",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


def plot_icr_curve(hidden_states: np.ndarray, labels: np.ndarray,
                   best_layer: int, out_path: Path) -> None:
    """
    Plot mean layer-wise cosine distance (trajectory change) for correct vs
    hallucinated samples across all layers. Highlights the best_layer.
    """
    from src.signals.trajectory_signals import compute_layer_distances

    num_layers = hidden_states.shape[1]
    N = hidden_states.shape[0]

    # Compute mean distances per layer for each class
    dists_correct = np.zeros(num_layers - 1)
    dists_hall    = np.zeros(num_layers - 1)
    n_c = n_h = 0

    for i in range(N):
        d = compute_layer_distances(hidden_states[i])
        if labels[i] == 1:
            dists_correct += d
            n_c += 1
        else:
            dists_hall += d
            n_h += 1

    if n_c > 0:
        dists_correct /= n_c
    if n_h > 0:
        dists_hall /= n_h

    layers = np.arange(1, num_layers)   # transition l→l+1

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(layers, dists_correct, "-o", color="#4e79a7", markersize=4,
            linewidth=1.5, label="Correct", alpha=0.85)
    ax.plot(layers, dists_hall, "-o", color="#e15759", markersize=4,
            linewidth=1.5, label="Hallucinated", alpha=0.85)

    # Mark best_layer transition
    if 1 <= best_layer <= len(layers):
        ax.axvline(best_layer, color="black", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"Best probing layer (L{best_layer})")

    ax.set_xlabel("Layer transition (L → L+1)")
    ax.set_ylabel("Mean cosine distance")
    ax.set_title(
        "Cross-Layer Trajectory: Mean Cosine Distance Between Consecutive Layers\n"
        "Correct vs Hallucinated — ICR-inspired Analysis (HaRP)"
    )
    ax.legend()
    ax.grid(alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


def main():
    print("=" * 60)
    print("Experiment 20 — Group D: Cross-Layer Trajectory Signals")
    print("No additional inference required (uses hidden_states.npz)")
    print("=" * 60)

    data, best_layer = load_prerequisites()
    print(f"Best probing layer: L{best_layer}")

    hs     = data["last_token_hs"].astype(np.float32)   # [N, L, D]
    labels = data["labels"].astype(int)
    pids   = data["prompt_ids"].tolist()
    N, num_layers, hidden_dim = hs.shape
    print(f"Hidden states: {N} samples × {num_layers} layers × {hidden_dim} dim")
    print(f"Class balance: {int(labels.sum())} correct / {int((labels==0).sum())} hallucinated")

    # ── Compute trajectory features ───────────────────────────────────────────
    traj_df = compute_trajectory_features(hs, pids, best_layer)
    print(f"\nTrajectory features computed: {traj_df.shape}")

    # ── Class analysis ────────────────────────────────────────────────────────
    print("\n--- Class-level trajectory feature analysis ---")
    print(f"  {'Feature':<30} {'Correct':>10} {'Halluc.':>10} {'Delta':>10}")
    print("  " + "-" * 64)

    class_analysis = trajectory_class_analysis(traj_df, labels)
    for feat, stats in class_analysis.items():
        print(f"  {feat:<30} "
              f"{stats['mean_correct']:>10.4f} "
              f"{stats['mean_hallucinated']:>10.4f} "
              f"{stats['delta']:>+10.4f}")

    # ── AUROC for each feature individually ──────────────────────────────────
    print("\n--- Individual feature AUROC ---")
    feat_aurocs = {}
    feat_cols = [c for c in traj_df.columns if c != "prompt_id"]
    for feat in feat_cols:
        vals = traj_df[feat].values.astype(float)
        mask = ~np.isnan(vals)
        if mask.sum() < 10:
            print(f"  {feat:<30}  not enough data")
            continue
        auroc = compute_auroc(labels[mask], vals[mask])
        feat_aurocs[feat] = round(auroc, 4)
        print(f"  {feat:<30}  AUROC = {auroc:.4f}")

    # ── Save trajectory features CSV ─────────────────────────────────────────
    out_csv = RESULTS_HS / "trajectory_features.csv"
    traj_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name} ({len(traj_df)} rows)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_trajectory_comparison(
        class_analysis,
        RESULTS_PLOTS / "20_trajectory_comparison.png",
    )
    plot_icr_curve(
        hs, labels, best_layer,
        RESULTS_PLOTS / "20_icr_curve.png",
    )

    # ── Save log ──────────────────────────────────────────────────────────────
    log = {
        "experiment":    "20_cross_layer_dynamics",
        "n_samples":     N,
        "num_layers":    num_layers,
        "best_layer":    best_layer,
        "n_features":    len(feat_cols),
        "features":      feat_cols,
        "individual_auroc": feat_aurocs,
        "class_analysis":   class_analysis,
    }
    log_path = RESULTS_LOGS / "20_cross_layer_dynamics.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved: {log_path.name}")

    print("\nExperiment 20 complete. Next: run 21_geometric_volume_signals.py")


if __name__ == "__main__":
    main()
