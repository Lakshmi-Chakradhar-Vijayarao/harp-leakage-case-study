"""
gap_fix_B_halueval.py — Out-of-distribution generalization: HaRP probe on HaluEval.

This answers: does the HaRP probe (trained on TruthfulQA) generalize to a
completely different dataset and answer domain?

Design:
  - Load 500 samples from HaluEval qa_samples (random seed 42: 250 correct + 250 hallucinated)
  - Each HaluEval sample has: question, answer (correct), hallucination (wrong)
  - We construct the input text as: "[question]\n[response]"
    where response is either the correct answer OR the hallucinated answer
  - Extract L32 hidden states using Qwen 2.5 3B (same model as HaRP)
  - Apply the EXISTING HaRP probe (zero-shot OOD transfer — no retraining)
  - Also train a fresh OOF probe on HaluEval hidden states (in-domain upper bound)
  - Report:
      - Zero-shot transfer AUROC (HaRP probe, OOD)
      - In-domain OOF probe AUROC (upper bound for HaluEval)
      - Gap = upper_bound - transfer_auroc  (how much is left on the table)

The question:
  If zero-shot transfer AUROC > 0.65:  probe generalizes across datasets
  If 0.57 < transfer < 0.65:           partial generalization (scale/domain gap)
  If transfer ≤ 0.57:                  probe overfits TruthfulQA; not general

Note on input construction:
  HaRP's probe was trained on hidden states of Qwen's GENERATED responses.
  Here we feed the HaluEval reference text (not Qwen's generation) to Qwen
  as a continuation prompt. This tests whether the probe can read the
  geometry of "given text containing a hallucination" vs "given text
  containing a correct answer" — a stricter test than in-distribution.

Saves:
  data/processed/halueval_hidden_states.npz
  results/logs/gap_fix_B_halueval.json
  results/plots/gap_fix_B_halueval.png

Usage:
  python experiments/gap_fix_B_halueval.py
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PROCESSED    = ROOT / "data" / "processed"
RESULTS_LOGS      = ROOT / "results" / "logs"
RESULTS_PLOTS     = ROOT / "results" / "plots"
HIDDEN_STATES_DIR = ROOT / "results" / "hidden_states"

RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
N_SPLITS     = 5
BEST_LAYER   = 32
N_SAMPLES    = 500   # 250 correct + 250 hallucinated


# ── Data loading ──────────────────────────────────────────────────────────────

def load_halueval(n: int = N_SAMPLES) -> pd.DataFrame:
    """
    Load n/2 correct + n/2 hallucinated samples from HaluEval qa_samples.
    Returns DataFrame with columns: text, label (1=correct, 0=hallucinated)
    """
    from datasets import load_dataset
    print(f"Loading HaluEval qa_samples (n={n})...")
    ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    df = ds.to_pandas()
    df = df.dropna(subset=["question", "answer", "hallucination"])

    # Sample n/2 items (same items used for both correct and hallucinated rows)
    n_each = n // 2
    sample = df.sample(n=min(n_each, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)

    records = []
    for _, row in sample.iterrows():
        q = str(row["question"]).strip()
        a = str(row["answer"]).strip()
        h = str(row["hallucination"]).strip()
        # Correct row
        records.append({"text": f"{q}\n{a}", "label": 1, "question": q, "response": a})
        # Hallucinated row
        records.append({"text": f"{q}\n{h}", "label": 0, "question": q, "response": h})

    out = pd.DataFrame(records)
    print(f"  Loaded {len(out)} rows  |  "
          f"correct={int(out['label'].sum())}  hallucinated={int((out['label']==0).sum())}")
    return out


# ── Hidden state extraction ───────────────────────────────────────────────────

def extract_hidden_states(texts: list, cache_path: Path) -> np.ndarray:
    """
    Extract L32 last-token hidden states from Qwen 2.5 3B for each text.
    Returns array of shape (N, 2048).
    """
    if cache_path.exists():
        data = np.load(cache_path)
        print(f"Loaded cached hidden states from {cache_path.name}  shape={data['hs'].shape}")
        return data["hs"]

    print(f"\nExtracting L{BEST_LAYER} hidden states from Qwen 2.5 3B...")
    from src.model.query_hf import load_model
    import torch

    model, tokenizer = load_model()

    all_hs = []
    for i, text in enumerate(texts):
        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512,
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        if str(model.device) != "cpu":
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        h = outputs.hidden_states[BEST_LAYER][0, -1, :].cpu().float().numpy()
        all_hs.append(h)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(texts)}] extracted")

    result = np.stack(all_hs).astype(np.float16)   # (N, 2048)
    np.savez(cache_path, hs=result)
    print(f"Saved: {cache_path.name}  shape={result.shape}")
    return result


# ── OOF probe ────────────────────────────────────────────────────────────────

def run_oof_probe(X: np.ndarray, y: np.ndarray, label: str) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=1000, random_state=RANDOM_SEED,
        )),
    ])
    skf    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
    mean, std = float(scores.mean()), float(scores.std())
    print(f"  {label:<45}  AUROC = {mean:.4f} ± {std:.4f}")
    return {"label": label, "auroc_mean": round(mean, 4), "auroc_std": round(std, 4)}


# ── Zero-shot transfer (apply HaRP probe directly) ────────────────────────────

def run_zero_shot_transfer(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Apply the HaRP probe (trained on TruthfulQA) directly to HaluEval hidden states.
    No retraining — pure OOD transfer.
    """
    from sklearn.metrics import roc_auc_score

    probe_path = HIDDEN_STATES_DIR / "best_layer_probe.pkl"
    if not probe_path.exists():
        print("  HaRP probe not found — skipping zero-shot transfer.")
        return {"auroc_mean": float("nan"), "note": "probe_not_found"}

    with open(probe_path, "rb") as f:
        probe = pickle.load(f)

    X_f32 = X.astype(np.float32)
    try:
        probs = probe.predict_proba(X_f32)[:, 1]
        auroc = float(roc_auc_score(y, probs))
        print(f"  {'Zero-shot transfer (HaRP→HaluEval)':<45}  AUROC = {auroc:.4f}")
        return {"label": "zero_shot_transfer", "auroc_mean": round(auroc, 4), "auroc_std": None}
    except Exception as e:
        print(f"  Zero-shot transfer failed: {e}")
        return {"auroc_mean": float("nan"), "note": str(e)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Gap Fix B — HaluEval OOD Generalization Benchmark")
    print("=" * 60)

    df = load_halueval(N_SAMPLES)

    cache_path = DATA_PROCESSED / "halueval_hidden_states.npz"
    hs = extract_hidden_states(df["text"].tolist(), cache_path)
    # hs shape: (N, 2048)

    y = df["label"].values.astype(int)
    X = hs.astype(np.float32)

    print("\nRunning probes...")
    in_domain  = run_oof_probe(X, y, "In-domain OOF probe (HaluEval — upper bound)")
    zero_shot  = run_zero_shot_transfer(X, y)

    harp_auroc = 0.775
    baseline   = 0.570

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  HaRP probe on TruthfulQA (in-dist):  {harp_auroc:.4f}")
    print(f"  HaRP probe on HaluEval (zero-shot):   {zero_shot['auroc_mean']:.4f}")
    print(f"  Fresh OOF probe on HaluEval:          {in_domain['auroc_mean']:.4f}")
    print(f"  Token-signal baseline:                {baseline:.4f}")

    zt_auroc = zero_shot["auroc_mean"]
    if not np.isnan(zt_auroc):
        gap = in_domain["auroc_mean"] - zt_auroc
        retention = (zt_auroc - baseline) / max(in_domain["auroc_mean"] - baseline, 1e-6)
        print(f"\n  Transfer gap (upper_bound - zero_shot): {gap:+.4f}")
        print(f"  Signal retention:                       {retention*100:.1f}%")

        if zt_auroc > 0.65:
            verdict = "STRONG_GENERALIZATION"
            finding = (
                f"HaRP probe generalizes strongly to HaluEval (zero-shot AUROC={zt_auroc:.3f}). "
                f"L32 geometry encodes hallucination-relevant structure independent of dataset. "
                f"Signal retention: {retention*100:.0f}% of in-domain upper bound."
            )
        elif zt_auroc > baseline:
            verdict = "PARTIAL_GENERALIZATION"
            finding = (
                f"Partial OOD generalization (zero-shot AUROC={zt_auroc:.3f}). "
                f"Signal persists above baseline ({baseline}) but below 0.65 threshold. "
                f"Domain gap between TruthfulQA (factual) and HaluEval (QA pairs) "
                f"attenuates transfer. In-domain upper bound: {in_domain['auroc_mean']:.3f}."
            )
        else:
            verdict = "NO_GENERALIZATION"
            finding = (
                f"Probe does not generalize to HaluEval (zero-shot AUROC={zt_auroc:.3f} ≤ baseline). "
                f"L32 geometry may be TruthfulQA-specific. "
                f"In-domain HaluEval probe AUROC={in_domain['auroc_mean']:.3f} shows "
                f"{'strong' if in_domain['auroc_mean'] > 0.65 else 'weak'} local geometry signal."
            )
    else:
        verdict = "TRANSFER_SKIPPED"
        finding = "Zero-shot transfer could not be computed."

    print(f"\nVerdict: {verdict}")
    print(f"Finding: {finding}")

    # ── Save ──────────────────────────────────────────────────────────────────
    log = {
        "experiment":           "gap_fix_B_halueval",
        "dataset":              "pminervini/HaluEval (qa_samples)",
        "n_samples":            len(df),
        "model":                "Qwen/Qwen2.5-3B-Instruct",
        "best_layer":           BEST_LAYER,
        "results": {
            "harp_tqa_indist_auroc":   harp_auroc,
            "halueval_zeroshot_auroc": zero_shot,
            "halueval_indomain_auroc": in_domain,
        },
        "verdict":              verdict,
        "key_finding":          finding,
        "methodology": {
            "input_format":   "question + newline + answer/hallucination text",
            "labeling":       "1=correct_answer, 0=hallucinated_answer (from HaluEval ground truth)",
            "zero_shot":      "Apply HaRP probe (TruthfulQA-trained) directly — no retraining",
            "in_domain":      "Fresh 5-fold OOF LR probe trained on HaluEval hidden states",
            "hidden_states":  f"Qwen 2.5 3B L{BEST_LAYER} last-token hidden state (2048 dims)",
        },
    }

    log_path = RESULTS_LOGS / "gap_fix_B_halueval.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved: {log_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        bars_x = ["TruthfulQA\n(HaRP, in-dist)", "HaluEval\n(zero-shot)", "HaluEval\n(in-domain OOF)"]
        bars_y = [harp_auroc,
                  zero_shot["auroc_mean"] if not np.isnan(zero_shot["auroc_mean"]) else 0,
                  in_domain["auroc_mean"]]
        bar_colors = ["#4e79a7", "#e15759", "#59a14f"]

        ax.bar(bars_x, bars_y, color=bar_colors, width=0.5, alpha=0.85)
        ax.axhline(baseline, color="gray", linestyle=":", linewidth=1, label="Token baseline: 0.57")
        ax.axhline(0.65, color="black", linestyle="--", linewidth=1, label="Threshold: 0.65")
        ax.set_ylim(0.4, 0.9)
        ax.set_ylabel("AUROC")
        ax.set_title("HaRP OOD Generalization — TruthfulQA → HaluEval")
        ax.legend(fontsize=8)
        for i, v in enumerate(bars_y):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

        plt.tight_layout()
        plot_path = RESULTS_PLOTS / "gap_fix_B_halueval.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_path.name}")
    except Exception as e:
        print(f"Plot failed (non-critical): {e}")

    print("\nExperiment gap_fix_B complete.")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
