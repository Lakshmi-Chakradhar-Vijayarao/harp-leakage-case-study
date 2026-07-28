"""
01_collect_data.py — Query Qwen 2.5 3B via Ollama for all three datasets.

Phase 1, Experiment 1.
Loads TruthfulQA (817), NQ Open (500), and custom adversarial prompts (130).
Queries each via Ollama at temperature=0.0 (deterministic) to get a
stable response and token log-probabilities.

Checkpointed: if output CSV already exists, already-processed prompt_ids
are skipped so you can resume after interruption.

Expected runtime: ~35 mins total (~1.5s per prompt × ~1450 prompts).
Run with Ollama server active: `ollama serve`
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from src.model.query_ollama import query_with_logprobs

DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE = "Question: {question}\n\nAnswer:"

# Quick-run caps: set to None for the full dataset (production run)
# These values give ~500 samples in ~15-20 min for end-to-end validation.
N_TQA_CAP = 500   # full = 817  (increased from 200 for statistical robustness)
N_NQ_CAP  = 300   # full = 500  (increased from 100 for statistical robustness)


# ── Dataset loaders ───────────────────────────────────────────────────────────

def _load_truthfulqa():
    print("  Loading TruthfulQA (generation split)...")
    ds = load_dataset("truthful_qa", "generation", split="validation",
                      trust_remote_code=False)
    records = []
    for i, row in enumerate(ds):
        records.append({
            "dataset":      "truthfulqa",
            "prompt_id":    f"tqa_{i:04d}",
            "prompt":       PROMPT_TEMPLATE.format(question=row["question"]),
            "ground_truth": row["best_answer"],
            "prompt_type":  "factual",
        })
    if N_TQA_CAP is not None:
        records = records[:N_TQA_CAP]
    print(f"  Loaded {len(records)} TruthfulQA prompts.")
    return records


def _load_nq():
    print("  Loading NQ Open (train[:500])...")
    ds = load_dataset("nq_open", split="train[:500]", trust_remote_code=False)
    records = []
    for i, row in enumerate(ds):
        answers = row.get("answer", [])
        gt = answers[0] if answers else ""
        records.append({
            "dataset":      "nq",
            "prompt_id":    f"nq_{i:04d}",
            "prompt":       PROMPT_TEMPLATE.format(question=row["question"]),
            "ground_truth": gt,
            "prompt_type":  "factual",
        })
    if N_NQ_CAP is not None:
        records = records[:N_NQ_CAP]
    print(f"  Loaded {len(records)} NQ prompts.")
    return records


def _load_adversarial():
    adv_path = DATA_RAW / "adversarial_prompts.json"
    if not adv_path.exists():
        raise FileNotFoundError(f"Missing: {adv_path}\nRun: create data/raw/adversarial_prompts.json")
    with open(adv_path) as f:
        adv = json.load(f)
    records = []
    for i, item in enumerate(adv):
        records.append({
            "dataset":      "adversarial",
            "prompt_id":    f"adv_{i:04d}",
            "prompt":       item["prompt"],
            "ground_truth": item["ground_truth"],
            "prompt_type":  item.get("type", "unknown"),
        })
    print(f"  Loaded {len(records)} adversarial prompts.")
    return records


# ── Query + checkpoint ────────────────────────────────────────────────────────

def _process_dataset(records, out_csv):
    """Query Ollama for each record; checkpoint every 50 prompts."""
    done_ids = set()
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        done_ids = set(existing["prompt_id"].tolist())
        n_done = len(done_ids)
        print(f"  Resuming: {n_done}/{len(records)} already done.")

    pending = [r for r in records if r["prompt_id"] not in done_ids]
    if not pending:
        print("  Nothing to do — all prompts already collected.")
        return

    buffer = []
    for rec in tqdm(pending, desc=out_csv.stem):
        try:
            result = query_with_logprobs(rec["prompt"])
            row = {
                **rec,
                "response":            result["response_text"],
                "token_logprobs_json": json.dumps(result["token_logprobs"]),
                "top_logprobs_json":   json.dumps(result["top_logprobs"]),
                "logprobs_available":  result["logprobs_available"],
            }
        except Exception as e:
            print(f"\n  Error on {rec['prompt_id']}: {e}")
            row = {
                **rec,
                "response":            "",
                "token_logprobs_json": "null",
                "top_logprobs_json":   "null",
                "logprobs_available":  False,
            }
        buffer.append(row)

        # Checkpoint every 50
        if len(buffer) % 50 == 0:
            _append_to_csv(buffer, out_csv)
            buffer = []

    if buffer:
        _append_to_csv(buffer, out_csv)

    final = pd.read_csv(out_csv)
    print(f"  Saved {len(final)} rows → {out_csv.name}")


def _append_to_csv(rows, path):
    df_new = pd.DataFrame(rows)
    if path.exists():
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Experiment 01 — Data Collection")
    print("Requires: Ollama running (`ollama serve`)")
    print("=" * 60)

    # Sanity-check Ollama is reachable
    import requests
    try:
        r = requests.get("http://localhost:11434", timeout=5)
    except Exception:
        print("\nERROR: Cannot reach Ollama at http://localhost:11434")
        print("Start it with: ollama serve")
        sys.exit(1)

    print("\n[1/3] TruthfulQA")
    tqa = _load_truthfulqa()
    _process_dataset(tqa, DATA_PROCESSED / "truthfulqa_responses.csv")

    print("\n[2/3] NQ Open")
    nq = _load_nq()
    _process_dataset(nq, DATA_PROCESSED / "nq_responses.csv")

    print("\n[3/3] Adversarial")
    adv = _load_adversarial()
    _process_dataset(adv, DATA_PROCESSED / "adversarial_responses.csv")

    print("\n--- Summary ---")
    for name, fname in [
        ("TruthfulQA", "truthfulqa_responses.csv"),
        ("NQ",         "nq_responses.csv"),
        ("Adversarial","adversarial_responses.csv"),
    ]:
        p = DATA_PROCESSED / fname
        if p.exists():
            df = pd.read_csv(p)
            avail = int(df["logprobs_available"].sum()) if "logprobs_available" in df.columns else "?"
            print(f"  {name}: {len(df)} rows, logprobs available: {avail}")

    print("\nExperiment 01 complete. Next: run 02_consistency_check.py")


if __name__ == "__main__":
    main()
