"""
22_quadrant_adaptive_steering.py — Quadrant-adaptive activation steering.

Phase 12, Experiment 22.
REQUIRES GPU — run on Google Colab T4.

The Exp 11 finding: universal steering vector yields α=0 as optimal, because
it averages across fundamentally different failure modes. This experiment tests
the HaRP hypothesis that quadrant-specific steering vectors can recover
the correction signal lost in the aggregate.

Theoretical basis:
  Adaptive Activation Steering (ACT, WWW 2025) shows that category-specific
  vectors dramatically outperform universal vectors (↑142% on LLaMA).
  HaRP extends ACT by using the mechanistically-motivated 4-quadrant
  taxonomy (epistemic / aleatoric / confident_hallucination / confident_correct)
  rather than unsupervised clustering.

Method:
  1. Assign each hallucinated example a quadrant from its signal features
     (using features_combined.csv and assign_quadrants_df)
  2. Compute one steering vector per quadrant from the hidden states subset
  3. For each test example: get quadrant → apply quadrant-specific vector
  4. Compare: universal (Exp 11) vs quadrant-adaptive (this experiment)

Alpha grid per quadrant: [0.0, 0.5, 1.0, 1.5, 2.0]
N per quadrant: depends on data — reported in results.

IMPORTANT: Run this in isolation (no Ollama).
  python experiments/22_quadrant_adaptive_steering.py

Colab setup (same as Exp 11):
  Upload harp repo zip, install requirements, run this script.

Outputs:
  results/logs/22_quadrant_steering_results.json
  results/plots/22_quadrant_steering_comparison.png
"""

import sys
import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.labeling.label_responses import label_response
from src.analysis.uncertainty_decomposition import assign_quadrants_df, assign_quadrant
from src.steering.activation_steering import (
    compute_quadrant_steering_vectors,
    generate_steered_adaptive,
    generate_steered,
    QUADRANT_NAMES,
)

RESULTS_LOGS   = ROOT / "results" / "logs"
RESULTS_HS     = ROOT / "results" / "hidden_states"
RESULTS_PLOTS  = ROOT / "results" / "plots"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)

import os as _os
_alphas_env = _os.environ.get("EXP22_ALPHAS", "")
ALPHAS  = [float(a) for a in _alphas_env.split(",")] if _alphas_env else [0.0, 0.5, 1.0, 1.5, 2.0]
N_STEER = 100
SEED    = 42
MIN_SAMPLES_PER_QUADRANT = 5
# Entropy threshold: set via ENTROPY_THRESHOLD env var (default 0.5).
# Clean run uses 0.3704 (median of 700 HS-covered samples).
ENTROPY_THRESHOLD = float(_os.environ.get("ENTROPY_THRESHOLD", "0.5"))
# Consistency threshold: set via CONSISTENCY_THRESHOLD env var.
# If not set, computed dynamically as the median of HS-covered samples (~0.8575).
# All 700 HS samples have consistency > 0.5, so the hardcoded default of 0.5
# collapses the epistemic quadrant. Data-driven median gives 4 balanced quadrants.
_cons_thresh_env = _os.environ.get("CONSISTENCY_THRESHOLD", "")
CONSISTENCY_THRESHOLD_OVERRIDE = float(_cons_thresh_env) if _cons_thresh_env else None


def load_prerequisites() -> tuple:
    sv_path    = RESULTS_HS / "steering_vector.npy"
    hs_path    = RESULTS_HS / "hidden_states.npz"
    probe_path = RESULTS_LOGS / "06_probing_results.json"
    feat_path  = DATA_PROCESSED / "features_combined.csv"

    for p in [sv_path, hs_path, probe_path]:
        if not p.exists():
            print(f"Missing: {p}")
            print("Run experiments 05, 06, 10 first.")
            sys.exit(1)

    sv         = np.load(sv_path)
    data       = np.load(hs_path, allow_pickle=True)
    with open(probe_path) as f:
        best_layer = int(json.load(f)["best_layer"])

    features_df = None
    if feat_path.exists():
        features_df = pd.read_csv(feat_path)

    return sv, data, best_layer, features_df


def assign_sample_quadrants(
    data: dict,
    features_df: Optional[pd.DataFrame],
) -> np.ndarray:
    """
    Assign each hidden-state sample to a quadrant using features_combined.csv.

    Falls back to a simple entropy/consistency rule if features are missing.
    Returns np.ndarray [N] of quadrant strings.
    """
    pids = data["prompt_ids"].tolist()
    N = len(pids)

    if features_df is not None:
        feat_sub = features_df.set_index("prompt_id")

        # Data-driven consistency threshold: median of HS-covered samples.
        # All 700 HS samples have consistency > 0.5, so the default 0.5 collapses
        # the epistemic quadrant. Using the data median (~0.8575) gives 4 balanced quadrants.
        hs_pids = set(pids)
        hs_features = features_df[features_df["prompt_id"].isin(hs_pids)]
        cons_threshold = (
            CONSISTENCY_THRESHOLD_OVERRIDE
            if CONSISTENCY_THRESHOLD_OVERRIDE is not None
            else float(hs_features["consistency_score"].median())
        )
        print(f"  consistency_threshold: {cons_threshold:.4f}  "
              f"({'env override' if CONSISTENCY_THRESHOLD_OVERRIDE is not None else 'data-driven median'})")

        quadrants = []
        for pid in pids:
            if pid in feat_sub.index:
                row = feat_sub.loc[pid]
                ent  = float(row.get("entropy", 0.5)) if not pd.isna(row.get("entropy", np.nan)) else 0.5
                cons = float(row.get("consistency_score", 0.5)) if not pd.isna(row.get("consistency_score", np.nan)) else 0.5
                cos_c = row.get("cos_correct",      None)
                cos_h = row.get("cos_hallucinated", None)
                cos_c = float(cos_c) if cos_c is not None and not pd.isna(cos_c) else None
                cos_h = float(cos_h) if cos_h is not None and not pd.isna(cos_h) else None

                q = assign_quadrant(ent, cons,
                                    cos_correct=cos_c, cos_hallucinated=cos_h,
                                    entropy_threshold=ENTROPY_THRESHOLD,
                                    consistency_threshold=cons_threshold)
            else:
                q = "confident_hallucination"
            quadrants.append(q)
        return np.array(quadrants)
    else:
        # Fallback: assign all to confident_hallucination (selected from hall_idx anyway)
        return np.array(["confident_hallucination"] * N)


def plot_quadrant_comparison(
    adaptive_results: dict,
    universal_results: dict,
    out_path: Path,
) -> None:
    """
    Side-by-side comparison: quadrant-adaptive vs universal steering.
    Left: hallucination rate per quadrant at optimal alpha.
    Right: overall hallucination rate vs alpha for both methods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel 1: optimal hallucination rate per quadrant ─────────────────────
    ax = axes[0]
    q_names  = [q for q in QUADRANT_NAMES if q in adaptive_results]
    adapt_rates  = [adaptive_results[q]["best_hall_rate"]     for q in q_names]
    universal_r  = universal_results.get("overall", {}).get("best_hall_rate", float("nan"))

    x = np.arange(len(q_names))
    w = 0.35
    ax.bar(x - w/2, adapt_rates, w, label="Quadrant-Adaptive", color="#4e79a7", alpha=0.85)
    ax.axhline(universal_r, color="#e15759", linestyle="--", linewidth=2,
               label=f"Universal (best={universal_r:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace("_", "\n") for q in q_names], fontsize=8)
    ax.set_ylabel("Hallucination Rate (optimal α)")
    ax.set_title("Quadrant-Adaptive vs Universal Steering\nHallucination Rate per Quadrant")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    # ── Panel 2: overall rate vs alpha ────────────────────────────────────────
    ax2 = axes[1]
    alphas = sorted([float(a) for a in adaptive_results.get("overall_by_alpha", {}).keys()])
    adapt_overall  = [adaptive_results["overall_by_alpha"][str(a)] for a in alphas]
    universal_line = [universal_results.get("by_alpha", {}).get(str(a), float("nan"))
                      for a in alphas]

    ax2.plot(alphas, adapt_overall,  "-o", color="#4e79a7", markersize=8,
             linewidth=2, label="Quadrant-Adaptive")
    ax2.plot(alphas, universal_line, "-s", color="#e15759", markersize=8,
             linewidth=2, label="Universal (Exp 11)")
    ax2.set_xlabel("Steering Coefficient α")
    ax2.set_ylabel("Hallucination Rate")
    ax2.set_title("Hallucination Rate vs α\nQuadrant-Adaptive vs Universal")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


def main():
    print("=" * 60)
    print("Experiment 22 — Quadrant-Adaptive Activation Steering")
    print("REQUIRES GPU. Run on Google Colab T4.")
    print(f"Alpha grid: {ALPHAS}")
    print("=" * 60)

    universal_sv, data, best_layer, features_df = load_prerequisites()
    print(f"Best layer: L{best_layer}")

    labels    = data["labels"].astype(int)
    prompts   = data["prompts"].tolist()
    gts       = data["ground_truths"].tolist()
    pids      = data["prompt_ids"].tolist()

    # ── Assign quadrants ──────────────────────────────────────────────────────
    print("\nAssigning quadrants to all samples...")
    all_quadrants = assign_sample_quadrants(data, features_df)
    q_counts = {q: int((all_quadrants == q).sum()) for q in QUADRANT_NAMES}
    print("  Quadrant distribution (all samples):")
    for q, n in q_counts.items():
        print(f"    {q:<30} {n:3d}")

    # ── Load HF model ─────────────────────────────────────────────────────────
    from src.model.query_hf import load_model, cleanup
    model, tokenizer = load_model()

    # ── Compute quadrant-specific steering vectors ────────────────────────────
    print("\nComputing quadrant-specific steering vectors...")
    q_vectors = compute_quadrant_steering_vectors(
        model, tokenizer, prompts, labels,
        all_quadrants.tolist(), best_layer,
        min_samples=MIN_SAMPLES_PER_QUADRANT,
    )

    # Select hallucinated test examples
    hall_idx = np.where(labels == 0)[0]
    rng = np.random.default_rng(SEED)
    n   = min(N_STEER, len(hall_idx))
    selected = rng.choice(hall_idx, size=n, replace=False)
    selected_quadrants = all_quadrants[selected]

    print(f"\nSelected {n} hallucinated examples for steering experiment")
    sel_q_counts = {q: int((selected_quadrants == q).sum()) for q in QUADRANT_NAMES}
    print("  Selected quadrant distribution:")
    for q, cnt in sel_q_counts.items():
        print(f"    {q:<30} {cnt:3d}")

    # ── Run quadrant-adaptive steering experiment ─────────────────────────────
    adaptive_results = {"overall_by_alpha": {}}
    for q in QUADRANT_NAMES:
        adaptive_results[q] = {"by_alpha": {}, "best_hall_rate": 1.0, "best_alpha": 0.0}

    for alpha in ALPHAS:
        print(f"\n--- Quadrant-Adaptive α = {alpha} ---")
        per_q = {q: {"corrected": 0, "wrong": 0} for q in QUADRANT_NAMES}
        total_corrected = 0
        total_wrong = 0

        for idx in tqdm(selected, desc=f"α={alpha}"):
            prompt  = str(prompts[idx])
            gt      = str(gts[idx])
            quadrant = all_quadrants[idx]

            try:
                response = generate_steered_adaptive(
                    model, tokenizer, prompt, quadrant,
                    q_vectors, best_layer, alpha,
                    fallback_to_universal=True,
                    universal_sv=universal_sv,
                )
                lbl = label_response(response, gt)
                if lbl == 1:
                    per_q[quadrant]["corrected"] += 1
                    total_corrected += 1
                else:
                    per_q[quadrant]["wrong"] += 1
                    total_wrong += 1
            except Exception as e:
                print(f"\n  Error: {e}")
                per_q[quadrant]["wrong"] += 1
                total_wrong += 1

        total = total_corrected + total_wrong
        overall_hall = total_wrong / total if total > 0 else float("nan")
        adaptive_results["overall_by_alpha"][str(alpha)] = round(overall_hall, 4)
        print(f"  Overall: hall_rate={overall_hall:.4f}  (N={total})")

        for q in QUADRANT_NAMES:
            qc = per_q[q]["corrected"]
            qw = per_q[q]["wrong"]
            qt = qc + qw
            if qt > 0:
                qr = qw / qt
                adaptive_results[q]["by_alpha"][str(alpha)] = round(qr, 4)
                if qr < adaptive_results[q]["best_hall_rate"]:
                    adaptive_results[q]["best_hall_rate"] = round(qr, 4)
                    adaptive_results[q]["best_alpha"]     = alpha
                print(f"    {q:<30} hall_rate={qr:.4f}  N={qt}")
            else:
                adaptive_results[q]["by_alpha"][str(alpha)] = float("nan")

    cleanup()

    # ── Load Exp 11 universal results for comparison ──────────────────────────
    exp11_path = RESULTS_LOGS / "11_steering_results.json"
    universal_results = {}
    if exp11_path.exists():
        with open(exp11_path) as f:
            exp11 = json.load(f)
        by_alpha = {
            str(v["alpha"]): v["hallucination_rate"]
            for v in exp11.get("results", {}).values()
        }
        best_univ_alpha = exp11.get("optimal_alpha", 0.0)
        best_univ_rate  = exp11["results"][str(best_univ_alpha)]["hallucination_rate"]
        universal_results = {
            "by_alpha":      by_alpha,
            "best_alpha":    best_univ_alpha,
            "overall": {"best_hall_rate": best_univ_rate},
        }

    # ── Find best quadrant-adaptive configuration ─────────────────────────────
    best_adapt_alpha = min(
        adaptive_results["overall_by_alpha"].items(),
        key=lambda kv: kv[1] if not (isinstance(kv[1], float) and kv[1] != kv[1]) else 1.0
    )
    print(f"\nBest adaptive α: {best_adapt_alpha[0]} "
          f"(hall_rate={best_adapt_alpha[1]:.4f})")
    if universal_results:
        print(f"Best universal α: {universal_results['best_alpha']} "
              f"(hall_rate={universal_results['overall']['best_hall_rate']:.4f})")
        delta = float(universal_results["overall"]["best_hall_rate"]) - float(best_adapt_alpha[1])
        print(f"Improvement: {delta:+.4f} hallucination rate reduction")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_quadrant_comparison(
        adaptive_results,
        universal_results,
        RESULTS_PLOTS / "22_quadrant_steering_comparison.png",
    )

    # ── Save log ──────────────────────────────────────────────────────────────
    log = {
        "experiment":         "22_quadrant_adaptive_steering",
        "best_layer":         best_layer,
        "n_examples":         n,
        "alphas":             ALPHAS,
        "quadrant_counts":    sel_q_counts,
        "vectors_computed":   {q: (v is not None) for q, v in q_vectors.items()},
        "adaptive_results":   adaptive_results,
        "universal_results":  universal_results,
        "best_adaptive_alpha":  float(best_adapt_alpha[0]),
        "best_adaptive_hall":   float(best_adapt_alpha[1]),
    }
    log_path = RESULTS_LOGS / "22_quadrant_steering_results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved: {log_path.name}")

    print("\nExperiment 22 complete. Next: run 23_final_evaluation.py")


if __name__ == "__main__":
    main()
