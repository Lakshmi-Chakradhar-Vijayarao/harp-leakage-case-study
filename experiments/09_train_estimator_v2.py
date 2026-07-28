"""
09_train_estimator_v2.py — Failure Estimator v2: the core experimental result.

Phase 5, Experiment 9.

Scientific question
-------------------
By how much does adding Group C representation signals (hidden-state geometry)
improve hallucination prediction over token entropy + consistency alone?

The answer to this question is the central empirical claim of HaRP:
representation signals are NECESSARY, not merely additive, for reliable governance.

The ablation design
-------------------
Three estimators trained on the same 5-fold CV split:

  1. Group A only (6 token features)
     → Baseline AUROC. Confirms entropy alone is insufficient.

  2. Group A + B (8 features: + cosine consistency, + consistency variance)
     → Expected: near-zero gain over Group A alone. Consistency fails for
     confident hallucination because the model consistently produces the same
     wrong answer.

  3. Group A + B + C (11 features: + cos_correct, cos_hallucinated, hidden_norm, probe_confidence)
     → Expected: largest AUROC jump in the entire project. The representation
     signals directly access the geometry that distinguishes confident hallucination
     from confident correct — a signal invisible to all output-space methods.

The step from A+B → A+B+C is the experiment. Everything else is context.

Group C signal computation (done here, not in a separate script)
-----------------------------------------------------------------
  cos_correct:       cosine distance from h[best_layer] to μ_correct (from Exp 10)
  cos_hallucinated:  cosine distance from h[best_layer] to μ_hallucinated (from Exp 10)
  hidden_norm:       L2 norm of h[best_layer, -1, :] at the last token position
  probe_confidence:  LR probe posterior P(correct | h[best_layer]) from Exp 06

These four signals operationalize the "geometry of truth" finding from Exp 07:
if the geometry is clean (high pc1_correlation), these features are strong;
if the geometry is messy, these features add little — the experiment self-validates.

Output: features_combined.csv
-------------------------------
This is the master feature file for all downstream experiments (12, 13, 15, 16,
and the SOTA evaluation in 23). All feature groups (A through E) are added as
additional columns to this file by subsequent experiments.

Outputs
-------
  data/processed/features_combined.csv    — 11 features (A+B+C) for all labeled samples
  results/logs/09_estimator_v2.json       — per-group AUROC + ablation table
  results/plots/09_auroc_comparison.png   — AUROC comparison plot (Plot 4 stub)
"""

import sys
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.authorization.failure_estimator import run_ablation, GROUP_A, GROUP_B, GROUP_B2, GROUP_C, GROUP_C2
from src.signals.representation_signals import (
    compute_class_means, extract_all as rep_extract
)
from src.evaluation.metrics import ablation_table, bootstrap_auroc, compute_auroc

DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_LOGS   = ROOT / "results" / "logs"
RESULTS_HS     = ROOT / "results" / "hidden_states"
RESULTS_PLOTS  = ROOT / "results" / "plots"
RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)


def load_best_layer() -> int:
    p = RESULTS_LOGS / "06_probing_results.json"
    if not p.exists():
        raise FileNotFoundError("Missing 06_probing_results.json — run 06 first.")
    with open(p) as f:
        return int(json.load(f)["best_layer"])


def compute_group_c(hs_data: dict, best_layer: int,
                    probe, ab_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Group C features for hidden-state samples and left-join back
    to the A+B feature dataframe by prompt_id.
    """
    hs       = hs_data["last_token_hs"].astype(np.float32)    # [N, L, D]
    labels   = hs_data["labels"].astype(int)
    pids     = hs_data["prompt_ids"].tolist()

    h_layer = hs[:, best_layer, :]   # [N, D]

    # Compute class means from the hidden-state subset
    mu_correct, mu_hall = compute_class_means(h_layer, labels)

    rows = []
    for i, pid in enumerate(pids):
        h = h_layer[i]
        feats = rep_extract(h, mu_correct, mu_hall, probe)
        rows.append({
            "prompt_id":       pid,
            "cos_correct":     feats["cos_correct"],
            "cos_hallucinated":feats["cos_hallucinated"],
            "h_norm":          feats["h_norm"],
            "probe_conf":      feats["probe_conf"],
        })

    group_c_df = pd.DataFrame(rows)

    # Merge into AB dataframe
    combined = ab_df.merge(group_c_df, on="prompt_id", how="left")
    return combined


def plot_auroc_comparison(results: dict, out_path: Path) -> None:
    labels_list = list(results.keys())
    aurocs = [results[k]["auroc_mean"] for k in labels_list]
    errs   = [results[k]["auroc_std"]  for k in labels_list]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(labels_list))
    bars = ax.bar(x, aurocs, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"],
                  alpha=0.85, zorder=3)
    ax.errorbar(x, aurocs, yerr=errs, fmt="none", color="black",
                capsize=5, linewidth=1.5, zorder=4)
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.6, label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [k.replace(" (", "\n(").replace("+", "+\n") for k in labels_list],
        fontsize=9
    )
    ax.set_ylabel("AUROC (5-fold CV)")
    ax.set_title("Failure Estimator Ablation\nAUROC by Feature Group")
    ax.set_ylim(0.45, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


def main():
    print("=" * 60)
    print("Experiment 09 — Failure Estimator v2 (Group A+B+C)")
    print("=" * 60)

    # ── Load dependencies ─────────────────────────────────────────────────────
    ab_path = DATA_PROCESSED / "features_groupAB.csv"
    hs_path = RESULTS_HS / "hidden_states.npz"
    probe_path = RESULTS_HS / "best_layer_probe.pkl"

    for p in [ab_path, hs_path, probe_path]:
        if not p.exists():
            print(f"Missing: {p}")
            print("Run experiments 03, 05, 06 first.")
            sys.exit(1)

    best_layer = load_best_layer()
    print(f"Best layer: L{best_layer}")

    ab_df = pd.read_csv(ab_path)
    print(f"A+B features: {len(ab_df)} samples")

    hs_data = np.load(hs_path, allow_pickle=True)
    with open(probe_path, "rb") as f:
        probe = pickle.load(f)

    # ── Compute Group C ───────────────────────────────────────────────────────
    print("\nComputing Group C representation features...")
    combined = compute_group_c(hs_data, best_layer, probe, ab_df)

    n_with_c = combined[GROUP_C].notna().all(axis=1).sum()
    print(f"Samples with Group C features: {n_with_c}/{len(combined)}")

    # Merge attention features (Group C2) if available from Exp 05
    attn_csv = RESULTS_HS / "attention_features.csv"
    if attn_csv.exists():
        attn_df = pd.read_csv(attn_csv)
        attn_cols = [c for c in GROUP_C2 if c in attn_df.columns] + ["prompt_id"]
        combined = combined.merge(attn_df[attn_cols], on="prompt_id", how="left")
        n_attn = combined[GROUP_C2[0]].notna().sum() if GROUP_C2[0] in combined.columns else 0
        print(f"Merged attention features (Group C2): {n_attn} samples")
    else:
        print("Attention features not found — Group C2 will be absent (re-run Exp 05 to extract)")

    # Save combined feature CSV
    out_csv = DATA_PROCESSED / "features_combined.csv"
    combined.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv.name}")

    # ── Ablation ──────────────────────────────────────────────────────────────
    print("\n--- Ablation Study ---")
    ablation_results = run_ablation(combined)

    print("\n" + ablation_table(ablation_results))

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_auroc_comparison(ablation_results,
                          RESULTS_PLOTS / "09_auroc_comparison.png")

    # ── AUROC delta summary ───────────────────────────────────────────────────
    auroc_a   = ablation_results.get("Group A only",  {}).get("auroc_mean", float("nan"))
    auroc_ab  = ablation_results.get("Group A+B",     {}).get("auroc_mean", float("nan"))
    auroc_abc = ablation_results.get("Group A+B+C",   {}).get("auroc_mean", float("nan"))
    delta     = auroc_abc - auroc_a

    print(f"\nAUROC delta (Group A → A+B+C): {delta:+.4f}")
    print(f"  Group A:    {auroc_a:.4f}")
    print(f"  Group A+B:  {auroc_ab:.4f}  (consistency adds {auroc_ab-auroc_a:+.4f})")
    print(f"  Group A+B+C:{auroc_abc:.4f}  (repr signals add {auroc_abc-auroc_ab:+.4f})")

    # ── Save log ──────────────────────────────────────────────────────────────
    log = {
        "experiment":       "09_estimator_v2",
        "best_layer":       best_layer,
        "n_with_group_c":   int(n_with_c),
        "n_total":          len(combined),
        "ablation":         ablation_results,
        "auroc_group_a":    auroc_a,
        "auroc_group_ab":   auroc_ab,
        "auroc_group_abc":  auroc_abc,
        "delta_a_to_abc":   float(delta),
    }
    log_path = RESULTS_LOGS / "09_estimator_v2.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved: {log_path.name}")

    if auroc_abc > 0.72:
        print("TARGET MET: Group A+B+C AUROC > 0.72")
    else:
        print(f"NOTE: Group A+B+C AUROC = {auroc_abc:.4f} (target > 0.72)")

    print("Experiment 09 complete. Next: run 10_compute_steering_vector.py")


if __name__ == "__main__":
    main()
