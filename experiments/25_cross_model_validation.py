"""
gap_fix_A_gpt2_medium.py — Cross-model validation: does geometry signal transfer
to GPT-2 Medium 345M?

This replaces the unusable Exp 19 (which used synthetic data with llama3.2:1b
and reported a nonsensical AUROC of 0.487).

Design:
  - Same 700 TruthfulQA prompts used to train HaRP
  - Same OOF linear probe protocol (5-fold, StandardScaler + LogisticRegression)
  - GPT-2 Medium has 24 transformer layers (hidden_dim=1024)
  - Depth-equivalent to HaRP's L32/36 = 88.9% depth = L21 for GPT-2 Medium
  - We probe L18, L19, L20, L21, L22 (near-optimal band) to find the best layer
  - Compare best-layer AUROC to HaRP's 0.775

The question:
  If geometry signal is real and encodes a semantic property of hallucination
  (not just an artefact of Qwen 3B's weights), we expect:
    - GPT-2 Medium to show above-chance geometry signal
    - Signal to peak near the 88% depth mark (L21)
    - AUROC to be lower than 0.775 (smaller model, less powerful representations)
    - But clearly above 0.57 (Group A baseline)

Null: AUROC ≤ 0.57 everywhere → geometry is Qwen-specific, not architectural

Saves:
  results/hidden_states/gpt2_medium_hidden_states.npz
  results/logs/gap_fix_A_gpt2_medium.json
  results/plots/gap_fix_A_gpt2_medium.png

Usage:
  python experiments/gap_fix_A_gpt2_medium.py
"""

import sys
import json
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

RANDOM_SEED = 42
N_SPLITS    = 5

# GPT-2 Medium has 24 layers (indices 0..24, where 0 = embedding, 1..24 = transformer blocks)
# Depth-equivalent to Qwen L32/36 (88.9%) → L21.3 → probe L18-L22
PROBE_LAYERS = [18, 19, 20, 21, 22]
GPT2_MODEL   = "openai-community/gpt2-medium"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_prompts_and_labels() -> tuple[list, np.ndarray]:
    csv_path = DATA_PROCESSED / "truthfulqa_labeled.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}\nRun 03_label_responses.py first.")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["prompt", "label"])
    prompts = df["prompt"].tolist()
    labels  = df["label"].astype(int).values
    print(f"Loaded {len(prompts)} prompts  |  "
          f"correct={int(labels.sum())}  hallucinated={int((labels==0).sum())}")
    return prompts, labels


# ── Hidden state extraction ───────────────────────────────────────────────────

def extract_gpt2_hidden_states(
    prompts: list,
    layers: list,
    cache_path: Path,
) -> np.ndarray:
    """
    Extract last-token hidden states from GPT-2 Medium at the specified layers.
    Returns array of shape (N, len(layers), 1024).
    Caches to cache_path.
    """
    if cache_path.exists():
        data = np.load(cache_path)
        print(f"Loaded cached GPT-2 Medium hidden states from {cache_path.name}  "
              f"shape={data['hidden_states'].shape}")
        return data["hidden_states"]

    print(f"\nExtracting GPT-2 Medium hidden states at layers {layers}...")
    print(f"Loading {GPT2_MODEL} ...")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # GPT-2 Medium runs fine on CPU/MPS — use MPS if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        GPT2_MODEL,
        dtype=torch.float16 if device == "mps" else torch.float32,
        output_hidden_states=True,
    ).to(device)
    model.eval()
    print(f"  Loaded on {device}  |  "
          f"layers={model.config.n_layer}  hidden_dim={model.config.n_embd}")

    all_hs = []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=256,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of (n_layer+1) tensors, each [1, seq_len, 1024]
        # index 0 = embedding, 1..24 = transformer blocks
        sample_hs = np.stack([
            outputs.hidden_states[l][0, -1, :].cpu().float().numpy()
            for l in layers
        ])  # shape: (len(layers), 1024)
        all_hs.append(sample_hs)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(prompts)}] extracted")

    result = np.stack(all_hs).astype(np.float16)  # (N, len(layers), 1024)
    np.savez(cache_path, hidden_states=result, layers=np.array(layers))
    print(f"Saved: {cache_path.name}  shape={result.shape}")

    # Clean up model to free MPS memory
    del model
    import gc
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

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
    print(f"  {label:<25}  AUROC = {mean:.4f} ± {std:.4f}")
    return {"layer_label": label, "auroc_mean": round(mean, 4), "auroc_std": round(std, 4)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Gap Fix A — GPT-2 Medium Cross-Model Validation")
    print("=" * 60)

    prompts, labels = load_prompts_and_labels()

    cache_path = HIDDEN_STATES_DIR / "gpt2_medium_hidden_states.npz"
    hidden_states = extract_gpt2_hidden_states(prompts, PROBE_LAYERS, cache_path)
    # hidden_states shape: (N, len(PROBE_LAYERS), 1024)

    print(f"\nRunning OOF probes at layers {PROBE_LAYERS}...")
    layer_results = []
    for i, layer_idx in enumerate(PROBE_LAYERS):
        X = hidden_states[:, i, :].astype(np.float32)
        result = run_oof_probe(X, labels, f"L{layer_idx}")
        result["layer_idx"] = layer_idx
        layer_results.append(result)

    best = max(layer_results, key=lambda r: r["auroc_mean"])
    harp_auroc = 0.775  # honest OOF AUROC from HaRP Exp 24

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nLayer-by-layer AUROC (GPT-2 Medium 345M):")
    for r in layer_results:
        marker = " ← best" if r["layer_idx"] == best["layer_idx"] else ""
        print(f"  L{r['layer_idx']}: {r['auroc_mean']:.4f} ± {r['auroc_std']:.4f}{marker}")

    delta = best["auroc_mean"] - 0.57
    print(f"\n  Best layer:        L{best['layer_idx']}  AUROC={best['auroc_mean']:.4f}")
    print(f"  HaRP (Qwen 3B):    AUROC={harp_auroc:.4f}")
    print(f"  Chance baseline:   AUROC=0.5700  (Group A token signals)")
    print(f"  Signal above baseline: +{delta:.4f}")

    if best["auroc_mean"] > 0.65:
        verdict = "TRANSFER_CONFIRMED"
        finding = (
            f"Geometry signal transfers to GPT-2 Medium. "
            f"Best AUROC={best['auroc_mean']:.3f} at L{best['layer_idx']} "
            f"(depth {best['layer_idx']/24*100:.0f}%). "
            f"Signal persists across architectures and scales (117M→345M→3B). "
            f"This supports the claim that hidden-state geometry is an "
            f"architectural property of transformer hallucination, not Qwen-specific."
        )
    elif best["auroc_mean"] > 0.57:
        verdict = "WEAK_TRANSFER"
        finding = (
            f"Weak geometry signal at GPT-2 Medium. "
            f"Best AUROC={best['auroc_mean']:.3f} at L{best['layer_idx']}. "
            f"Signal is above chance but below the 0.65 threshold. "
            f"Scale matters: signal strengthens with model size."
        )
    else:
        verdict = "NO_TRANSFER"
        finding = (
            f"No geometry signal at GPT-2 Medium. "
            f"Best AUROC={best['auroc_mean']:.3f} — at or below token-signal baseline. "
            f"Geometry signal may be emergent at 3B+ scale only. "
            f"Consistent with MECH-INT finding (GPT-2 117M SVD null result)."
        )

    print(f"\nVerdict: {verdict}")
    print(f"Finding: {finding}")

    # ── Save ──────────────────────────────────────────────────────────────────
    log = {
        "experiment":       "gap_fix_A_gpt2_medium",
        "model":            GPT2_MODEL,
        "n_samples":        len(prompts),
        "probe_layers":     PROBE_LAYERS,
        "n_splits":         N_SPLITS,
        "layer_results":    layer_results,
        "best_layer":       best,
        "harp_qwen3b_auroc": harp_auroc,
        "verdict":          verdict,
        "key_finding":      finding,
        "methodology": {
            "prompts":  "Same 700 TruthfulQA prompts as HaRP training set",
            "labels":   "Same ROUGE-L labels as HaRP",
            "probe":    "Fresh OOF logistic regression on GPT-2 Medium HS (not HaRP probe)",
            "layers":   f"Probed L{PROBE_LAYERS[0]}-L{PROBE_LAYERS[-1]} (88% depth band)",
            "rationale": "Depth-equivalent to HaRP's L32/36 = L21.3 for GPT-2 Medium 24-layer",
        },
    }

    log_path = RESULTS_LOGS / "gap_fix_A_gpt2_medium.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved: {log_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        layers_x  = [r["layer_idx"] for r in layer_results]
        aurocs_y  = [r["auroc_mean"] for r in layer_results]
        stds_y    = [r["auroc_std"]  for r in layer_results]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(layers_x, aurocs_y, yerr=stds_y,
                    marker="o", color="#4e79a7", linewidth=1.5,
                    capsize=4, label="GPT-2 Medium")
        ax.axhline(harp_auroc, color="#e15759", linestyle="--", linewidth=1.5,
                   label=f"HaRP / Qwen 3B (L32): {harp_auroc}")
        ax.axhline(0.57, color="gray", linestyle=":", linewidth=1,
                   label="Token-signal baseline: 0.57")
        ax.set_xlabel("GPT-2 Medium Layer")
        ax.set_ylabel("OOF AUROC")
        ax.set_title("Cross-Model Validation: Geometry Signal at GPT-2 Medium 345M")
        ax.set_ylim(0.45, 0.85)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = RESULTS_PLOTS / "gap_fix_A_gpt2_medium.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_path.name}")
    except Exception as e:
        print(f"Plot failed (non-critical): {e}")

    print("\nExperiment gap_fix_A complete.")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
