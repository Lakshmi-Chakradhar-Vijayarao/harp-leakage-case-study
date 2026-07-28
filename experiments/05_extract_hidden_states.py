"""
05_extract_hidden_states.py — Extract Qwen 2.5 3B hidden states via HuggingFace.

Phase 4, Experiment 5.
IMPORTANT — Memory rules:
  1. Close Ollama before running this script (they cannot share RAM).
  2. Close all browser tabs and heavy applications.
  3. Run this file in isolation: python experiments/05_extract_hidden_states.py

Loads 500 labeled samples (balanced: 250 TruthfulQA + 250 NQ, stratified by label).
Extracts the last-token hidden state at every transformer layer.
Saves a single compressed archive for use by experiments 06–11.

Output:
  results/hidden_states/hidden_states.npz
    - last_token_hs: [N, num_layers+1, hidden_dim]  (includes embedding layer)
    - labels:        [N]
    - prompt_ids:    [N]  (string identifiers)
    - prompts:       [N]  (text, for logit lens)
    - ground_truths: [N]

  results/logs/05_hidden_states_meta.json
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_PROCESSED   = ROOT / "data" / "processed"
RESULTS_LOGS     = ROOT / "results" / "logs"
RESULTS_HS       = ROOT / "results" / "hidden_states"
RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
RESULTS_HS.mkdir(parents=True, exist_ok=True)

N_SAMPLES  = 700   # total; split between TQA and NQ (increased for larger dataset)
SEED       = 42


def load_balanced_subset(n: int = N_SAMPLES) -> pd.DataFrame:
    """
    Load n samples balanced across datasets and labels.
    Priority order: TruthfulQA → NQ (adversarial excluded from hidden states).
    """
    dfs = []
    for fname in ["truthfulqa_labeled.csv", "nq_labeled.csv"]:
        p = DATA_PROCESSED / fname
        if p.exists():
            dfs.append(pd.read_csv(p))

    if not dfs:
        raise FileNotFoundError("No labeled CSVs found. Run 03 first.")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["label", "prompt"])
    combined["label"] = combined["label"].astype(int)

    # Stratified sample
    rng = np.random.default_rng(SEED)
    correct   = combined[combined["label"] == 1]
    halluc    = combined[combined["label"] == 0]

    n_hall  = min(len(halluc), n // 2)
    n_corr  = min(len(correct), n - n_hall)

    idx_h = rng.choice(len(halluc), size=n_hall, replace=False)
    idx_c = rng.choice(len(correct), size=n_corr, replace=False)

    subset = pd.concat([
        halluc.iloc[idx_h],
        correct.iloc[idx_c],
    ], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"Subset: {len(subset)} samples — "
          f"{int((subset['label']==1).sum())} correct / "
          f"{int((subset['label']==0).sum())} hallucinated")
    return subset


def extract(subset: pd.DataFrame):
    """Run HF forward passes and collect hidden states + attention signals."""
    from src.model.query_hf import load_model, cleanup, get_hidden_states
    from src.signals.attention_signals import extract_attention_signals

    model, tokenizer = load_model()

    n = len(subset)
    # Determine shape from first pass
    test_result = get_hidden_states(subset["prompt"].iloc[0], return_attentions=True)
    num_layers  = test_result["last_token_hs"].shape[0]
    hidden_dim  = test_result["last_token_hs"].shape[1]
    has_attn    = test_result["attentions"] is not None
    print(f"Model: {num_layers} layers, hidden_dim={hidden_dim}, attention={has_attn}")
    print(f"Collecting {n} hidden states...")

    all_hs        = np.zeros((n, num_layers, hidden_dim), dtype=np.float16)
    attn_features = []   # list of dicts — Group C2 signals per sample
    labels        = np.zeros(n, dtype=np.int8)
    prompt_ids    = []
    prompts_list  = []
    gt_list       = []

    for i, (_, row) in enumerate(tqdm(subset.iterrows(), total=n)):
        result = get_hidden_states(row["prompt"], return_attentions=True)
        all_hs[i] = result["last_token_hs"].astype(np.float16)
        labels[i] = int(row["label"])
        prompt_ids.append(str(row.get("prompt_id", f"sample_{i:04d}")))
        prompts_list.append(str(row["prompt"]))
        gt_list.append(str(row.get("ground_truth", "")))

        if result["attentions"] is not None:
            attn_sigs = extract_attention_signals(result["attentions"])
        else:
            attn_sigs = {
                "attn_entropy_mean": float("nan"),
                "attn_entropy_max":  float("nan"),
                "attn_entropy_std":  float("nan"),
                "attn_head_cv":      float("nan"),
                "attn_top_k_mass":   float("nan"),
            }
        attn_features.append(attn_sigs)

    cleanup()
    return all_hs, labels, prompt_ids, prompts_list, gt_list, num_layers, hidden_dim, attn_features


def main():
    print("=" * 60)
    print("Experiment 05 — Hidden State Extraction (HuggingFace)")
    print("IMPORTANT: Ollama must be stopped before running this.")
    print("=" * 60)

    out_npz = RESULTS_HS / "hidden_states.npz"

    if out_npz.exists():
        data = np.load(out_npz, allow_pickle=True)
        n = data["last_token_hs"].shape[0]
        print(f"Found existing hidden_states.npz with {n} samples. Skipping extraction.")
        return

    print("\nLoading subset...")
    subset = load_balanced_subset(N_SAMPLES)

    print("\nExtracting hidden states + attention signals (~15-30 min)...")
    all_hs, labels, prompt_ids, prompts, gts, num_layers, hidden_dim, attn_feats = extract(subset)

    print(f"\nSaving to {out_npz} ...")
    np.savez_compressed(
        out_npz,
        last_token_hs = all_hs,         # [N, num_layers, hidden_dim], float16
        labels        = labels,          # [N], int8
        prompt_ids    = np.array(prompt_ids, dtype=object),
        prompts       = np.array(prompts,    dtype=object),
        ground_truths = np.array(gts,        dtype=object),
    )

    # Save attention features as CSV (Group C2) for Exp 09
    import pandas as pd
    attn_df = pd.DataFrame(attn_feats)
    attn_df["prompt_id"] = prompt_ids
    attn_df["label"]     = labels
    attn_csv = RESULTS_HS / "attention_features.csv"
    attn_df.to_csv(attn_csv, index=False)
    print(f"Saved attention features: {attn_csv.name}")

    size_mb = out_npz.stat().st_size / (1024 ** 2)
    print(f"Saved {out_npz.name} ({size_mb:.1f} MB)")

    meta = {
        "experiment":  "05_extract_hidden_states",
        "n_samples":   int(len(labels)),
        "n_correct":   int((labels == 1).sum()),
        "n_hallucinated": int((labels == 0).sum()),
        "num_layers":  int(num_layers),
        "hidden_dim":  int(hidden_dim),
        "dtype":       "float16",
        "file_mb":     round(size_mb, 2),
    }
    log_path = RESULTS_LOGS / "05_hidden_states_meta.json"
    with open(log_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {log_path.name}")

    print("\nExperiment 05 complete. Next: run 06_probe_layers.py")
    print("You can restart Ollama now if needed for other experiments.")


if __name__ == "__main__":
    main()
