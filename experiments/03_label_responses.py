"""
03_label_responses.py — Auto-label responses + extract Group A features.

Phase 2, Experiment 3.
Applies two-tier labeling (exact match then keyword overlap >= 0.5) to all
three datasets. Parses logprobs_json and computes all 6 Group A token signals.
Merges datasets into features_groupAB.csv (11 columns: A+B features + label).

Output:
  data/processed/truthfulqa_labeled.csv
  data/processed/nq_labeled.csv
  data/processed/adversarial_labeled.csv
  data/processed/features_groupAB.csv   ← main input to experiments 04 and 09
  results/logs/03_labeling_summary.json

Spot-check CSVs saved for manual review:
  data/processed/truthfulqa_labeled_spotcheck.csv
  data/processed/nq_labeled_spotcheck.csv
  data/processed/adversarial_labeled_spotcheck.csv
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.labeling.label_responses import (
    label_dataframe, check_class_balance, spot_check
)
from src.signals.token_signals import extract_all as token_extract

DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_LOGS   = ROOT / "results" / "logs"
RESULTS_LOGS.mkdir(parents=True, exist_ok=True)

# Group A feature names (must match failure_estimator.py GROUP_A)
GROUP_A_COLS = ["entropy", "max_entropy", "logit_var", "conf_gap",
                "top1_prob", "first_entropy"]
GROUP_B_COLS = ["consistency_score", "consistency_variance"]


def extract_group_a(df: pd.DataFrame) -> pd.DataFrame:
    """Parse token_logprobs_json and compute all 6 Group A features."""
    rows = {k: [] for k in GROUP_A_COLS}
    for _, row in df.iterrows():
        try:
            token_lp = json.loads(str(row.get("token_logprobs_json", "null")))
            top_lp   = json.loads(str(row.get("top_logprobs_json",   "null")))
        except (json.JSONDecodeError, TypeError):
            token_lp, top_lp = None, None

        # If token_lp contains token strings instead of floats (query_ollama bug
        # with older Ollama dict format), recover logprobs from top_lp.
        if token_lp and top_lp and isinstance(token_lp[0], str):
            token_lp = [d.get('logprob') for d in top_lp if isinstance(d, dict)]
            token_lp = [x for x in token_lp if x is not None] or None

        feats = token_extract(token_lp, top_lp)
        for k in GROUP_A_COLS:
            rows[k].append(feats[k])

    for k, vals in rows.items():
        df[k] = vals
    return df


def process_dataset(name: str, src_csv: Path, out_csv: Path) -> pd.DataFrame:
    if not src_csv.exists():
        print(f"  Skipping {name}: {src_csv.name} not found (run 01/02 first)")
        return None

    print(f"\n[{name}]")
    df = pd.read_csv(src_csv)
    print(f"  Rows loaded: {len(df)}")

    # Label
    df = label_dataframe(df)
    balance = check_class_balance(df["label"])

    # Group A features
    df = extract_group_a(df)

    # NaN count for logprob features
    nan_pct = df[GROUP_A_COLS].isna().mean().mean() * 100
    print(f"  Group A NaN rate: {nan_pct:.1f}% (expected if logprobs unavailable)")

    # Save labeled CSV
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv.name}")

    # Spot-check
    sc = spot_check(df, n=50)
    sc_path = out_csv.parent / out_csv.name.replace(".csv", "_spotcheck.csv")
    sc.to_csv(sc_path, index=False)
    print(f"  Spot-check saved: {sc_path.name} — open and verify 50 labels manually")

    return df


def main():
    print("=" * 60)
    print("Experiment 03 — Labeling + Group A Feature Extraction")
    print("=" * 60)

    datasets = [
        ("TruthfulQA", DATA_PROCESSED / "truthfulqa_responses.csv",
                       DATA_PROCESSED / "truthfulqa_labeled.csv"),
        ("NQ",         DATA_PROCESSED / "nq_responses.csv",
                       DATA_PROCESSED / "nq_labeled.csv"),
        ("Adversarial",DATA_PROCESSED / "adversarial_responses.csv",
                       DATA_PROCESSED / "adversarial_labeled.csv"),
    ]

    all_dfs = []
    for name, src, out in datasets:
        df = process_dataset(name, src, out)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("\nNo datasets found. Run 01_collect_data.py and 02_consistency_check.py first.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\n--- Combined dataset: {len(combined)} samples ---")
    check_class_balance(combined["label"])

    # Build feature CSV (Group A + B + response_run_* + label)
    # response_run_* columns are needed by Exp 17 (NLI semantic entropy).
    feature_cols = (
        ["dataset", "prompt_id", "label"]
        + GROUP_A_COLS
        + [c for c in GROUP_B_COLS if c in combined.columns]
        + [c for c in combined.columns if c.startswith("response_run")]
    )
    if "prompt_type" in combined.columns:
        feature_cols.insert(3, "prompt_type")

    avail = [c for c in feature_cols if c in combined.columns]
    out_features = DATA_PROCESSED / "features_groupAB.csv"
    combined[avail].to_csv(out_features, index=False)
    print(f"\nSaved: features_groupAB.csv ({len(combined)} rows, {len(avail)} cols)")

    # Log summary
    per_ds = {}
    for ds in combined["dataset"].unique():
        sub = combined[combined["dataset"] == ds]
        per_ds[ds] = {
            "n": int(len(sub)),
            "n_correct":      int((sub["label"] == 1).sum()),
            "n_hallucinated": int((sub["label"] == 0).sum()),
            "pct_hall":       round(100 * (sub["label"] == 0).mean(), 1),
        }

    summary = {
        "total":          len(combined),
        "n_correct":      int((combined["label"] == 1).sum()),
        "n_hallucinated": int((combined["label"] == 0).sum()),
        "pct_hall":       round(100 * (combined["label"] == 0).mean(), 1),
        "per_dataset":    per_ds,
        "group_a_nan_pct": round(
            float(combined[GROUP_A_COLS].isna().mean().mean() * 100), 2),
    }
    log_path = RESULTS_LOGS / "03_labeling_summary.json"
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {log_path.name}")

    print("\nIMPORTANT: Open the spotcheck CSVs and manually verify 50 labels.")
    print("Target: >= 30% hallucinated. If below, check labeling threshold.")
    print(f"Current: {summary['pct_hall']}% hallucinated")

    print("\nExperiment 03 complete. Next: run 04_train_estimator_v1.py")


if __name__ == "__main__":
    main()
