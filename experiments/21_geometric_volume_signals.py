"""
21_geometric_volume_signals.py — Group E: Geometric volume signals over response embeddings.

Phase 11, Experiment 21.
Uses the 3 consistency responses stored in features_groupAB.csv
(response_run_1, response_run_2, response_run_3) to compute geometric
volume features per sample.

Zero new Ollama queries required — responses were stored by Exp 02.

Theoretical basis:
  Granese et al. (ICLR 2026 submission, arXiv:2509.13813) prove that the
  convex hull volume of response embeddings is an upper bound on semantic entropy.
  We operationalise this as a per-sample triangle area (simplex volume for N=3)
  and two companion signals.

The 3 geometric features (Group E):
  simplex_volume   : area of triangle formed by 3 response embeddings
  embedding_spread : max pairwise L2 distance (worst-case disagreement)
  centroid_drift   : std of distances from centroid (response cloud shape)

Why this is stronger than Group B (pairwise cosine similarity):
  - Captures the SHAPE of the semantic cloud, not just average similarity
  - Simplex volume = 0 iff all responses are collinear (degenerate confidence)
  - Theoretically grounded by Granese et al. as an entropy proxy

Outputs:
  results/hidden_states/geometric_features.csv  ← per-sample Group E features
  results/logs/21_geometric_volume_results.json
  results/plots/21_geometric_comparison.png
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
from tqdm import tqdm

from src.signals.geometric_signals import (
    extract_all as geom_extract,
    geometric_class_analysis,
)
from src.evaluation.metrics import compute_auroc

RESULTS_LOGS   = ROOT / "results" / "logs"
RESULTS_HS     = ROOT / "results" / "hidden_states"
RESULTS_PLOTS  = ROOT / "results" / "plots"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)
RESULTS_HS.mkdir(parents=True, exist_ok=True)


def load_labeled_features() -> pd.DataFrame:
    """
    Load features_groupAB.csv (contains response_run_1/2/3 and labels).
    Falls back to individual labeled CSVs if combined file is missing.
    """
    p = DATA_PROCESSED / "features_groupAB.csv"
    if not p.exists():
        print(f"Missing: {p}")
        print("Run experiments 01, 02, 03 first.")
        sys.exit(1)
    df = pd.read_csv(p)
    print(f"Loaded: {p.name} ({len(df)} rows)")
    return df


def has_response_runs(df: pd.DataFrame) -> bool:
    return all(f"response_run_{j}" in df.columns for j in range(1, 4))


def compute_geometric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Group E features for all rows that have response_run_* columns.

    Returns a new DataFrame with columns:
        prompt_id, simplex_volume, embedding_spread, centroid_drift
    """
    rows = []
    n_valid = 0
    n_skip  = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Geometric features"):
        pid = str(row["prompt_id"])
        responses = []
        for j in range(1, 4):
            r = str(row.get(f"response_run_{j}", "")).strip()
            if r and r != "nan":
                responses.append(r)

        if len(responses) < 2:
            rows.append({
                "prompt_id":       pid,
                "simplex_volume":  float("nan"),
                "embedding_spread":float("nan"),
                "centroid_drift":  float("nan"),
            })
            n_skip += 1
            continue

        feats = geom_extract(responses)
        feats["prompt_id"] = pid
        rows.append(feats)
        n_valid += 1

    print(f"  Computed: {n_valid} valid | {n_skip} skipped (missing responses)")
    return pd.DataFrame(rows)[
        ["prompt_id", "simplex_volume", "embedding_spread", "centroid_drift"]
    ]


def plot_geometric_comparison(class_analysis: dict, out_path: Path) -> None:
    features   = list(class_analysis.keys())
    mu_correct = [class_analysis[f]["mean_correct"]      for f in features]
    mu_hall    = [class_analysis[f]["mean_hallucinated"] for f in features]

    x = np.arange(len(features))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - w/2, mu_correct, w, label="Correct",      color="#4e79a7", alpha=0.85)
    ax.bar(x + w/2, mu_hall,    w, label="Hallucinated", color="#e15759", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f.replace("_", "\n") for f in features],
        fontsize=10,
    )
    ax.set_ylabel("Mean Feature Value")
    ax.set_title(
        "Group E: Geometric Volume Signals\n"
        "Correct vs Hallucinated — HaRP Extension of Granese et al. (ICLR 2026)"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for i, (c, h) in enumerate(zip(mu_correct, mu_hall)):
        delta = c - h
        y_pos = max(c, h) * 1.02 + 0.0001
        ax.annotate(
            f"Δ={delta:+.4f}",
            xy=(x[i], y_pos),
            ha="center", va="bottom", fontsize=9, color="black",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


def main():
    print("=" * 60)
    print("Experiment 21 — Group E: Geometric Volume Signals")
    print("Uses stored response_run_* columns — no Ollama queries needed")
    print("=" * 60)

    df = load_labeled_features()

    if not has_response_runs(df):
        print("\nMissing response_run_1/2/3 columns in features_groupAB.csv.")
        print("These are saved by Exp 02 when run with Ollama active.")
        print("Re-run 02_consistency_check.py to collect them.")
        sys.exit(1)

    n_with_runs = df[[f"response_run_{j}" for j in range(1, 4)]].notna().all(axis=1).sum()
    print(f"Rows with all 3 response runs: {n_with_runs}/{len(df)}")

    # ── Compute geometric features ────────────────────────────────────────────
    print("\nLoading sentence-transformer encoder (all-MiniLM-L6-v2)...")
    geom_df = compute_geometric_features(df)

    # ── Merge with labels for analysis ────────────────────────────────────────
    merged = geom_df.merge(df[["prompt_id", "label"]], on="prompt_id", how="left")
    labels = merged["label"].values.astype(int)

    # ── Class analysis ────────────────────────────────────────────────────────
    print("\n--- Class-level geometric feature analysis ---")
    print(f"  {'Feature':<22} {'Correct':>10} {'Halluc.':>10} {'Delta':>10}")
    print("  " + "-" * 55)

    volumes = merged["simplex_volume"].values.astype(float)
    spreads = merged["embedding_spread"].values.astype(float)
    drifts  = merged["centroid_drift"].values.astype(float)

    class_analysis = geometric_class_analysis(volumes, spreads, drifts, labels)
    for feat, stats in class_analysis.items():
        print(f"  {feat:<22} "
              f"{stats['mean_correct']:>10.5f} "
              f"{stats['mean_hallucinated']:>10.5f} "
              f"{stats['delta']:>+10.5f}")

    # ── Individual AUROC ──────────────────────────────────────────────────────
    print("\n--- Individual feature AUROC ---")
    feat_aurocs = {}
    for feat, vals in [
        ("simplex_volume",    volumes),
        ("embedding_spread",  spreads),
        ("centroid_drift",    drifts),
    ]:
        mask = ~np.isnan(vals)
        if mask.sum() < 10:
            print(f"  {feat:<22}  not enough data")
            continue
        auroc = compute_auroc(labels[mask], vals[mask])
        feat_aurocs[feat] = round(auroc, 4)
        print(f"  {feat:<22}  AUROC = {auroc:.4f}")

    # ── Save geometric features CSV ───────────────────────────────────────────
    out_csv = RESULTS_HS / "geometric_features.csv"
    geom_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name} ({len(geom_df)} rows)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_geometric_comparison(
        class_analysis,
        RESULTS_PLOTS / "21_geometric_comparison.png",
    )

    # ── Save log ──────────────────────────────────────────────────────────────
    n_valid = int((~np.isnan(volumes)).sum())
    log = {
        "experiment":       "21_geometric_volume_signals",
        "n_total":          len(merged),
        "n_valid":          n_valid,
        "individual_auroc": feat_aurocs,
        "class_analysis":   class_analysis,
    }
    log_path = RESULTS_LOGS / "21_geometric_volume_results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved: {log_path.name}")

    print("\nExperiment 21 complete. Next: run 22_quadrant_adaptive_steering.py (GPU/Colab)")


if __name__ == "__main__":
    main()
