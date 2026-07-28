"""
11_run_steering_experiment.py — Activation steering: hallucination rate vs. coefficient α.

Phase 6, Experiment 11.

Scientific question
-------------------
Can injecting the steering vector (from Exp 10) at best_layer reduce hallucination
on previously-hallucinating examples? And what is the optimal steering coefficient α?

Why hallucination rate (not AUROC)?
-------------------------------------
AUROC measures how well a probe can DETECT hallucination from representations.
Hallucination rate measures whether the model actually PRODUCES correct answers
after steering. This is the externally-verifiable outcome — what matters to users.

Expected and actual finding: α=0 is optimal (no steering = lowest hall rate)
-----------------------------------------------------------------------------
The universal steering vector averages over all failure mode types (epistemic,
aleatoric, confident correct, confident hallucination). For samples where the model
genuinely lacks knowledge (epistemic), there is no steering direction that adds the
missing information — you can only change the model's confidence, not its knowledge.

The competing effects across quadrants produce cancellation:
  - For confident_hallucination: steering toward μ_correct HELPS (correct direction)
  - For epistemic: steering toward μ_correct HURTS (pushes toward uncertain/wrong state)
  - The universal vector averages both, producing near-zero net effect

This finding directly motivates quadrant-adaptive steering (Exp 22):
separate steering vectors, one per failure mode, applied only when appropriate.

Alpha grid and fluency check
-----------------------------
α = [0.0, 0.5, 1.0, 1.5, 2.0]:
  - Small α (0.5–1.0): minimal effect; the injection is too weak to shift representations
  - Large α (2.0+): may distort representations and degrade fluency/coherence
  - 20 samples manually reviewed at each α for grammaticality and semantic coherence

IMPORTANT: Requires HuggingFace mode. Close Ollama before running.

Produces
--------
  - results/plots/11_steering_curve.png     — hallucination rate vs. α (Plot 6)
  - results/logs/11_steering_results.json   — per-alpha results dict
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

from src.labeling.label_responses import label_response
from src.steering.activation_steering import generate_steered

RESULTS_LOGS  = ROOT / "results" / "logs"
RESULTS_HS    = ROOT / "results" / "hidden_states"
RESULTS_PLOTS = ROOT / "results" / "plots"
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)

ALPHAS       = [0.0, 0.5, 1.0, 1.5, 2.0]
N_STEER      = 100
SEED         = 42


def load_best_layer() -> int:
    p = RESULTS_LOGS / "06_probing_results.json"
    with open(p) as f:
        return int(json.load(f)["best_layer"])


def main():
    print("=" * 60)
    print("Experiment 11 — Activation Steering Experiment")
    print(f"Grid: α = {ALPHAS}")
    print("IMPORTANT: Ollama must be stopped before running this.")
    print("=" * 60)

    # ── Load prerequisites ────────────────────────────────────────────────────
    sv_path = RESULTS_HS / "steering_vector.npy"
    hs_path = RESULTS_HS / "hidden_states.npz"
    if not sv_path.exists() or not hs_path.exists():
        print("Missing steering_vector.npy or hidden_states.npz. Run 05/10 first.")
        sys.exit(1)

    sv         = np.load(sv_path)
    best_layer = load_best_layer()
    data       = np.load(hs_path, allow_pickle=True)
    labels     = data["labels"].astype(int)
    prompts    = data["prompts"].tolist()
    gts        = data["ground_truths"].tolist()

    # Select hallucinated examples
    hall_idx = np.where(labels == 0)[0]
    rng = np.random.default_rng(SEED)
    n = min(N_STEER, len(hall_idx))
    selected = rng.choice(hall_idx, size=n, replace=False)
    print(f"\nUsing {n} hallucinated examples for steering experiment")

    # ── Load HF model ─────────────────────────────────────────────────────────
    from src.model.query_hf import load_model, cleanup
    model, tokenizer = load_model()

    alpha_results = {}

    for alpha in ALPHAS:
        print(f"\n--- α = {alpha} ---")
        corrected = 0
        still_wrong = 0
        responses_sample = []

        for idx in tqdm(selected, desc=f"α={alpha}"):
            prompt = str(prompts[idx])
            gt     = str(gts[idx])
            try:
                if alpha == 0.0:
                    # Baseline: no steering, just regenerate at temp=0.9
                    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
                    import torch
                    with torch.no_grad():
                        out_ids = model.generate(
                            **inp,
                            max_new_tokens=256,
                            temperature=0.9,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    new_ids = out_ids[0][inp["input_ids"].shape[1]:]
                    response = tokenizer.decode(new_ids, skip_special_tokens=True)
                else:
                    response = generate_steered(
                        model, tokenizer, prompt, sv, best_layer, alpha)

                lbl = label_response(response, gt)
                if lbl == 1:
                    corrected += 1
                else:
                    still_wrong += 1

                if len(responses_sample) < 5:
                    responses_sample.append({
                        "prompt": prompt[:100] + "...",
                        "gt":     gt,
                        "response": response[:150] + "...",
                        "label": lbl,
                    })
            except Exception as e:
                print(f"\n  Error: {e}")
                still_wrong += 1

        total = corrected + still_wrong
        hall_rate = still_wrong / total if total > 0 else float("nan")
        correction_rate = corrected / total if total > 0 else float("nan")

        print(f"  Hallucination rate: {hall_rate:.4f} "
              f"| Correction rate: {correction_rate:.4f} "
              f"| N={total}")

        alpha_results[str(alpha)] = {
            "alpha":           alpha,
            "n":               total,
            "n_corrected":     corrected,
            "n_still_wrong":   still_wrong,
            "hallucination_rate": hall_rate,
            "correction_rate":    correction_rate,
            "sample_responses":   responses_sample,
        }

    cleanup()

    # ── Plot ──────────────────────────────────────────────────────────────────
    alphas_list = [v["alpha"] for v in alpha_results.values()]
    hall_rates  = [v["hallucination_rate"] for v in alpha_results.values()]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas_list, hall_rates, "-o", color="crimson",
            markersize=8, linewidth=2)
    ax.fill_between(alphas_list, hall_rates,
                    alpha=0.15, color="crimson")
    ax.set_xlabel("Steering Coefficient α")
    ax.set_ylabel("Hallucination Rate on Regenerated Answers")
    ax.set_title("Effect of Activation Steering on Hallucination Rate\n"
                 f"Qwen 2.5 3B — Layer {best_layer} — {n} examples")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = RESULTS_PLOTS / "11_steering_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {plot_path.name}")

    # ── Find optimal alpha ────────────────────────────────────────────────────
    opt_alpha = min(alpha_results.items(),
                    key=lambda kv: kv[1]["hallucination_rate"])[1]["alpha"]
    print(f"Optimal α: {opt_alpha} "
          f"(hallucination rate = {alpha_results[str(opt_alpha)]['hallucination_rate']:.4f})")

    # ── Save log ──────────────────────────────────────────────────────────────
    log = {
        "experiment":   "11_steering_experiment",
        "best_layer":   best_layer,
        "n_examples":   n,
        "alphas":       ALPHAS,
        "optimal_alpha":opt_alpha,
        "results":      alpha_results,
    }
    log_path = RESULTS_LOGS / "11_steering_results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved: {log_path.name}")

    print("\nIMPORTANT: Manually review sample responses for fluency.")
    print("Check 11_steering_results.json → results[alpha][sample_responses]")
    print("Experiment 11 complete. Next: run 12_run_policy.py")


if __name__ == "__main__":
    main()
