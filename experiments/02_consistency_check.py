"""
02_consistency_check.py — Compute Group B semantic consistency signals.

Phase 1, Experiment 2.
For each prompt in all three response CSVs, runs 3 queries at temperature=0.7
and computes semantic consistency (mean pairwise cosine similarity of responses
via sentence-transformers all-MiniLM-L6-v2).

Adds/updates columns:  consistency_score, consistency_variance
Checkpointed: skips prompts that already have a non-NaN consistency_score.

Expected runtime: ~50 mins (3 extra calls/prompt × ~1450 prompts × ~1.5s).
Requires: Ollama running, sentence-transformers installed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.model.query_ollama import query_consistency
from src.signals.consistency_signals import extract_all as consistency_extract

DATA_PROCESSED = ROOT / "data" / "processed"

DATASETS = [
    "truthfulqa_responses.csv",
    "nq_responses.csv",
    "adversarial_responses.csv",
]


def add_consistency(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"  Skipping {csv_path.name} — file not found (run 01 first)")
        return

    df = pd.read_csv(csv_path)

    if "consistency_score" not in df.columns:
        df["consistency_score"]    = np.nan
        df["consistency_variance"] = np.nan
    for j in range(1, 4):
        if f"response_run_{j}" not in df.columns:
            df[f"response_run_{j}"] = pd.array([""] * len(df), dtype=object)

    needs = df["consistency_score"].isna()
    n_todo = int(needs.sum())
    print(f"  {csv_path.name}: {n_todo}/{len(df)} prompts need scoring")

    if n_todo == 0:
        return

    for i, idx in enumerate(tqdm(df.index[needs], desc=csv_path.stem)):
        prompt = str(df.at[idx, "prompt"])
        try:
            responses = query_consistency(prompt, n=3, temperature=0.7)
            sigs = consistency_extract(responses)
            df.at[idx, "consistency_score"]    = sigs["consistency_score"]
            df.at[idx, "consistency_variance"] = sigs["consistency_variance"]
            # Save raw responses for semantic entropy (Exp 17)
            for j, r in enumerate(responses):
                df.at[idx, f"response_run_{j+1}"] = r
        except Exception as e:
            print(f"\n  Error on index {idx}: {e}")

        if (i + 1) % 50 == 0:
            df.to_csv(csv_path, index=False)

    df.to_csv(csv_path, index=False)

    done = df["consistency_score"].notna().sum()
    mean_cs = df["consistency_score"].dropna().mean()
    print(f"  Saved. Scored: {done}/{len(df)} | mean consistency: {mean_cs:.4f}")


def main():
    print("=" * 60)
    print("Experiment 02 — Consistency Signals (Group B)")
    print("Requires: Ollama running, sentence-transformers installed")
    print("=" * 60)

    import requests
    try:
        requests.get("http://localhost:11434", timeout=5)
    except Exception:
        print("\nERROR: Cannot reach Ollama at http://localhost:11434")
        sys.exit(1)

    for fname in DATASETS:
        print(f"\n[Processing] {fname}")
        add_consistency(DATA_PROCESSED / fname)

    print("\n--- Summary ---")
    for fname in DATASETS:
        p = DATA_PROCESSED / fname
        if p.exists():
            df = pd.read_csv(p)
            if "consistency_score" in df.columns:
                cs = df["consistency_score"].dropna()
                print(f"  {fname}: mean={cs.mean():.4f}, std={cs.std():.4f}, "
                      f"n={len(cs)}")

    print("\nExperiment 02 complete. Next: run 03_label_responses.py")


if __name__ == "__main__":
    main()
