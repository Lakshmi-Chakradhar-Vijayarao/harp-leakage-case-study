"""
09b_oof_group_c.py — Leak-free Group C features via out-of-fold computation.

WHY THIS EXISTS
---------------
Experiment 09 computed Group C features (probe_conf, cos_correct, cos_hallucinated)
using a probe and centroids fitted on ALL 700 samples, then evaluated on those same
700 samples. In-sample, probe_conf has AUROC = 1.0000 (perfect label recovery).
When this feature is used in the failure estimator's 5-fold CV, the CV outer loop
does NOT correct for the inner leakage — the probe already saw all labels.

This inflated the reported AUROC from the true generalisation value to ~0.96,
masking the real discriminative power of the system.

FIX
---
For each of the 5 CV folds:
  1. Fit the probe on the TRAINING fold hidden states only.
  2. Compute mu_correct and mu_hallucinated from TRAINING fold only.
  3. Predict probe_conf, cos_correct, cos_hallucinated on the held-out TEST fold.
  4. h_norm is label-free and does not require OOF treatment.

This gives true out-of-fold predictions. The resulting probe_conf_oof has
AUROC ≈ 0.75 (matching Exp 06 cross-validated result), not 1.0.

OUTPUTS
-------
  data/processed/features_combined_oof.csv  — same structure as features_combined.csv
    but Group C columns are replaced with OOF versions:
      probe_conf_oof, cos_correct_oof, cos_hallucinated_oof, h_norm
    Restricted to the 700 samples that have hidden states.

  results/logs/09b_oof_group_c.json  — OOF vs in-sample AUROC comparison
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.signals.representation_signals import compute_oof_group_c

DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_LOGS   = ROOT / "results" / "logs"
RESULTS_HS     = ROOT / "results" / "hidden_states"
RESULTS_LOGS.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("Experiment 09b — Out-of-Fold Group C (Leak-Free)")
    print("=" * 60)

    # ── Load inputs ───────────────────────────────────────────────────────────
    hs_path      = RESULTS_HS / "hidden_states.npz"
    probe_res    = RESULTS_LOGS / "06_probing_results.json"
    combined_csv = DATA_PROCESSED / "features_combined.csv"

    for p in [hs_path, probe_res, combined_csv]:
        if not p.exists():
            print(f"Missing: {p}")
            sys.exit(1)

    best_layer = int(json.load(open(probe_res))["best_layer"])
    print(f"Best layer: L{best_layer}")

    data   = np.load(str(hs_path), allow_pickle=True)
    hs     = data["last_token_hs"].astype(np.float32)    # [N, L, D]
    labels = data["labels"].astype(int)                  # [N]
    pids   = data["prompt_ids"].tolist()

    h_layer = hs[:, best_layer, :]   # [N, D]
    N       = len(labels)
    print(f"Hidden states: {N} samples, layer {best_layer}, dim {h_layer.shape[1]}")
    print(f"Labels: {int((labels==1).sum())} correct / {int((labels==0).sum())} hallucinated")

    # ── Compute in-sample probe_conf for comparison ──────────────────────────
    import pickle
    probe_path = RESULTS_HS / "best_layer_probe.pkl"
    with open(probe_path, "rb") as f:
        probe_insample = pickle.load(f)

    probe_conf_insample = probe_insample.predict_proba(h_layer)[:, 1]
    auroc_insample      = float(roc_auc_score(labels, probe_conf_insample))
    print(f"\nIn-sample probe_conf AUROC (leaked):  {auroc_insample:.4f}  ← biased")

    # ── Compute OOF Group C features ──────────────────────────────────────────
    print("\nComputing OOF Group C features (5-fold, may take ~30s)...")
    oof_df = compute_oof_group_c(h_layer, labels, pids, n_splits=5)

    auroc_oof_probe = float(roc_auc_score(labels, oof_df["probe_conf_oof"].values))
    print(f"OOF probe_conf AUROC (honest):        {auroc_oof_probe:.4f}  ← true generalisation")
    print(f"  Delta (leaked - honest): {auroc_insample - auroc_oof_probe:+.4f}")

    # Also compare cos_correct_oof vs in-sample cos_correct
    fc = pd.read_csv(combined_csv)
    fc_hs = fc[fc["cos_correct"].notna()].copy()
    auroc_cos_insample = float(roc_auc_score(fc_hs["label"], 1 - fc_hs["cos_correct"]))
    fc_hs_merged = fc_hs.merge(oof_df[["prompt_id", "cos_correct_oof"]], on="prompt_id", how="left")
    auroc_cos_oof = float(roc_auc_score(
        fc_hs_merged["label"], 1 - fc_hs_merged["cos_correct_oof"].fillna(0.5)
    ))
    print(f"\nIn-sample cos_correct AUROC:  {auroc_cos_insample:.4f}")
    print(f"OOF      cos_correct AUROC:  {auroc_cos_oof:.4f}")
    print(f"  (cos_correct leakage is minor: each sample contributes 1/N to centroid)")

    # ── Build OOF features_combined ───────────────────────────────────────────
    print("\nBuilding features_combined_oof.csv...")

    # Start from features_combined, restrict to HS samples, replace Group C cols
    fc_oof = fc_hs.copy()

    # Drop in-sample Group C columns
    drop_cols = [c for c in ["cos_correct", "cos_hallucinated", "h_norm", "probe_conf"]
                 if c in fc_oof.columns]
    fc_oof = fc_oof.drop(columns=drop_cols)

    # Merge in OOF Group C
    fc_oof = fc_oof.merge(
        oof_df[["prompt_id", "probe_conf_oof", "cos_correct_oof",
                "cos_hallucinated_oof", "h_norm"]],
        on="prompt_id", how="left",
    )

    # Rename OOF cols to the canonical Group C names so existing code runs unchanged
    fc_oof = fc_oof.rename(columns={
        "probe_conf_oof":       "probe_conf",
        "cos_correct_oof":      "cos_correct",
        "cos_hallucinated_oof": "cos_hallucinated",
    })

    print(f"  Shape: {fc_oof.shape}  (restricted to {len(fc_oof)} HS samples)")
    print(f"  probe_conf AUROC (OOF): {float(roc_auc_score(fc_oof['label'], fc_oof['probe_conf'])):.4f}")

    out_csv = DATA_PROCESSED / "features_combined_oof.csv"
    fc_oof.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv.name}")

    # ── Save log ──────────────────────────────────────────────────────────────
    log = {
        "experiment":              "09b_oof_group_c",
        "best_layer":              best_layer,
        "n_hs_samples":            N,
        "n_splits":                5,
        "probe_conf_auroc_insample": auroc_insample,
        "probe_conf_auroc_oof":    auroc_oof_probe,
        "leakage_delta":           round(auroc_insample - auroc_oof_probe, 4),
        "cos_correct_auroc_insample": auroc_cos_insample,
        "cos_correct_auroc_oof":   auroc_cos_oof,
        "output_csv":              str(out_csv),
        "interpretation": (
            "probe_conf in Exp 09 was computed in-sample (probe fitted on all 700 "
            "samples, then predicted on same 700). In-sample AUROC=1.0 inflated the "
            "failure estimator. OOF AUROC matches Exp 06 CV result (~0.75), "
            "representing the true generalisation performance of the probe signal."
        ),
    }
    log_path = RESULTS_LOGS / "09b_oof_group_c.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved: {log_path.name}")

    print("\n" + "=" * 60)
    print("LEAKAGE SUMMARY")
    print("=" * 60)
    print(f"  probe_conf (in-sample, Exp 09): AUROC = {auroc_insample:.4f}  ← LEAKED")
    print(f"  probe_conf (OOF, this exp):     AUROC = {auroc_oof_probe:.4f}  ← HONEST")
    print(f"  Inflation: {auroc_insample - auroc_oof_probe:+.4f}")
    print(f"\n  cos_correct / cos_hallucinated: minor centroid leakage (1/N),")
    print(f"  corrected here using fold-specific centroids.")
    print("=" * 60)
    print("\nNext: run 24_honest_evaluation.py")


if __name__ == "__main__":
    main()
