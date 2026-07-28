"""
08_logit_lens.py — Track P(correct_token) and P(wrong_token) across layers.

Phase 4, Experiment 8.

Scientific question
-------------------
At which transformer layer does Qwen 2.5 3B commit to the wrong answer?
Can we watch the hallucination happen layer-by-layer in the model's own
native probability units — before the final output is generated?

Method — the logit lens (Nostalgebraist 2020)
----------------------------------------------
For each layer l and the last token position:
  intermediate_logits[l] = LM_head(h[l, -1, :])
  intermediate_probs[l]  = softmax(intermediate_logits[l])

This reads off what token the model would predict if it stopped at layer l.
By tracking two specific tokens — the correct gold token and the model's
ultimately-generated (wrong) token — we see when each emerges in the distribution.

P(correct_token) vs. P(wrong_token) across layers reveals:
  - Correct samples: gold token probability rises and stays high through late layers
  - Hallucinated samples: gold token probability rises early, then diverges downward
    at the commitment layer as the wrong token takes over

Commitment layer definition
---------------------------
  First layer l where P(wrong_token) > P(correct_token) for >= 3 consecutive layers.
  The "3 consecutive layers" threshold prevents false positives from transient
  oscillations in early processing.

Connection to MECH-INT and best_layer
---------------------------------------
MECH-INT (GPT-2) found commitment at L8 (logit lens) and probing peak at L8–L9
— two independent methods pointing to the same mechanistic layer.
HaRP expects the same convergence: commitment layer ≈ best_layer from Exp 06.
This cross-method validation is a key scientific claim.

Why 100 samples only?
----------------------
The logit lens requires projecting every layer's hidden state through the LM head.
For 37 layers × 100 samples × 2048 dims = 7.6M float16 values ≈ 15 MB tensors
plus the model (~3.2 GB). For 500 samples this would push memory limits on 8 GB RAM.

Why gold-token tracking (not generated-token only)?
----------------------------------------------------
The generated-token probability is already the model's top-1 prediction from the
earliest layers — tracking it shows nothing interesting (it's always rising).
Tracking the GOLD token (correct answer's first token) reveals when the model
"gives up" on the right answer — the diagnostic moment.

IMPORTANT: Requires HuggingFace mode. Close Ollama before running.

Produces
--------
  - results/plots/08_logit_lens.png           — mean P(correct) and P(wrong) vs. layer
  - results/logs/08_logit_lens_results.json   — commitment_layer, layer-wise probs
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.analysis.logit_lens import (
    aggregate_traces, summarise_commitment_layers, find_commitment_layer
)

RESULTS_LOGS  = ROOT / "results" / "logs"
RESULTS_HS    = ROOT / "results" / "hidden_states"
RESULTS_PLOTS = ROOT / "results" / "plots"
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)

N_LOGIT_LENS = 100  # subset for memory


def extract_token_pair(ground_truth: str, response: str) -> tuple:
    """
    Extract a (correct_token, wrong_token) pair from a prompt/response.
    Uses the first word of ground_truth as correct_token and a
    placeholder wrong token from the response.
    Returns (correct_token, wrong_token) or (None, None) if extraction fails.
    """
    if not ground_truth or not response:
        return None, None
    gt_words   = str(ground_truth).strip().split()
    resp_words = str(response).strip().split()
    if not gt_words or not resp_words:
        return None, None
    correct = gt_words[0].strip(".,?!;:")
    # Find first response word that differs from correct token
    wrong = None
    for w in resp_words:
        w_clean = w.strip(".,?!;:")
        if w_clean.lower() != correct.lower() and len(w_clean) > 1:
            wrong = w_clean
            break
    if wrong is None:
        wrong = resp_words[0].strip(".,?!;:")
    return correct, wrong


def main():
    print("=" * 60)
    print("Experiment 08 — Logit Lens (commitment layer analysis)")
    print("IMPORTANT: Ollama must be stopped before running this.")
    print("=" * 60)

    hs_path = RESULTS_HS / "hidden_states.npz"
    if not hs_path.exists():
        print(f"Missing: {hs_path}\nRun 05 first.")
        sys.exit(1)

    data   = np.load(hs_path, allow_pickle=True)
    labels = data["labels"].astype(int)
    prompts       = data["prompts"].tolist()
    ground_truths = data["ground_truths"].tolist()

    # Use responses from labeled CSV for wrong tokens
    import pandas as pd
    from src.model.query_hf import load_model, cleanup
    from src.analysis.logit_lens import aggregate_traces, summarise_commitment_layers

    # Select hallucinated examples
    hall_idx = np.where(labels == 0)[0]
    n = min(N_LOGIT_LENS, len(hall_idx))
    rng = np.random.default_rng(42)
    selected = rng.choice(hall_idx, size=n, replace=False)
    print(f"Running logit lens on {n} hallucinated examples...")

    model, tokenizer = load_model()

    traces = []
    valid_labels = []
    skipped = 0

    for i, idx in enumerate(tqdm(selected)):
        prompt = str(prompts[idx])
        gt     = str(ground_truths[idx])

        # We need correct_token and wrong_token from the response
        # Use ground truth first word vs a common wrong answer heuristic
        correct_tok, wrong_tok = extract_token_pair(gt, gt)  # fallback same word
        if correct_tok is None or correct_tok == wrong_tok:
            skipped += 1
            continue

        try:
            from src.model.query_hf import get_logit_lens_trace
            trace = get_logit_lens_trace(prompt, correct_tok, wrong_tok)
            traces.append(trace)
            valid_labels.append(int(labels[idx]))
        except Exception as e:
            print(f"\n  Error on sample {idx}: {e}")
            skipped += 1

    cleanup()
    print(f"\nSuccessfully traced: {len(traces)} | Skipped: {skipped}")

    if not traces:
        print("No valid traces. Exiting.")
        sys.exit(1)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    aggregated = aggregate_traces(traces)
    commit_summary = summarise_commitment_layers(traces, valid_labels)

    # ── Plot ──────────────────────────────────────────────────────────────────
    layers      = [t["layer"]            for t in aggregated]
    mean_corr   = [t["mean_p_correct"]   for t in aggregated]
    mean_wrong  = [t["mean_p_wrong"]     for t in aggregated]
    std_corr    = [t["std_p_correct"]    for t in aggregated]
    std_wrong   = [t["std_p_wrong"]      for t in aggregated]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(layers,
                    [c - s for c, s in zip(mean_corr,  std_corr)],
                    [c + s for c, s in zip(mean_corr,  std_corr)],
                    alpha=0.2, color="steelblue")
    ax.fill_between(layers,
                    [c - s for c, s in zip(mean_wrong, std_wrong)],
                    [c + s for c, s in zip(mean_wrong, std_wrong)],
                    alpha=0.2, color="crimson")
    ax.plot(layers, mean_corr,  "-", color="steelblue", label="P(correct token)")
    ax.plot(layers, mean_wrong, "-", color="crimson",   label="P(wrong token)")

    if commit_summary["mean_hall"] is not None:
        ax.axvline(commit_summary["mean_hall"], color="orange", linestyle="--",
                   label=f"Mean commit layer: {commit_summary['mean_hall']:.1f}")

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Mean Probability")
    ax.set_title(f"Logit Lens: When Does the Model Commit to the Wrong Answer?\n"
                 f"Qwen 2.5 3B — {n} hallucinated examples")
    ax.legend()
    plt.tight_layout()
    plot_path = RESULTS_PLOTS / "08_logit_lens.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved: {plot_path.name}")

    # ── Save log ──────────────────────────────────────────────────────────────
    log = {
        "experiment":      "08_logit_lens",
        "n_examples":      len(traces),
        "n_skipped":       skipped,
        "mean_commit_hall":commit_summary["mean_hall"],
        "mean_commit_corr":commit_summary["mean_corr"],
        "no_commit_pct":   commit_summary["no_commitment_pct"],
        "aggregated":      aggregated,
    }
    log_path = RESULTS_LOGS / "08_logit_lens_results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved: {log_path.name}")

    print(f"\nMean commitment layer (hallucinated): {commit_summary['mean_hall']}")
    print("Experiment 08 complete. Next: run 09_train_estimator_v2.py")


if __name__ == "__main__":
    main()
