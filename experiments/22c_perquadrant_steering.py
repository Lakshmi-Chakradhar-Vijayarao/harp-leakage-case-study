"""
22c_perquadrant_steering.py — Per-quadrant independent alpha optimisation.

Phase 12, Experiment 22c.
REQUIRES GPU — run on Google Colab T4.

Problem with Exp 22:
  A shared alpha swept across all quadrants masks the per-quadrant signal.
  At shared alpha=1.0 the overall result is 0.68 vs universal 0.67 — a null result.
  But the per-quadrant picture shows:
    epistemic:              0.652 → 0.609 at alpha=0.5  (responds to positive alpha)
    aleatoric:              0.714 → 0.571 at alpha=1.0  (needs large push)
    confident_hallucination:0.745 → 0.766 at any positive alpha (resists, worsens)
    confident_correct:      0.444 → 0.556 at any positive alpha (resists)
  The cancellation of improvements and degradations in the aggregate hides the signal.

This experiment:
  1. Sweeps each quadrant's alpha INDEPENDENTLY over a quadrant-specific grid,
     including NEGATIVE alpha for confident_hallucination (destabilise overconfident
     wrong representations rather than forcing toward correct class mean).
  2. Uses ALL 307 hallucinated samples for statistical power (Exp 22 used 100).
  3. Finds the per-quadrant optimal alpha, then runs a COMBINED pass where each
     sample gets its quadrant's optimal alpha applied.
  4. Computes bootstrap 95% CI on the combined result.
  5. Compares: universal (0.67) vs shared-adaptive (0.68) vs per-quadrant-optimal.

Theoretical basis for negative alpha on confident_hallucination:
  These samples have low entropy (model is already committed to the wrong answer)
  and cos_hall < cos_correct (geometrically anchored near the hallucinated class mean).
  Positive alpha h + alpha*v pushes toward the correct class mean — but the model's
  generation momentum at low entropy is strong enough to override a small nudge.
  Negative alpha h - |alpha|*v pushes the representation AWAY from both class means,
  increasing internal uncertainty in the hidden state, which can unlock the generation
  head to consider alternative completions. It is not "steer toward hallucinated" — it
  is "destabilise the overconfident committed state."
  Grounded in: Representation Engineering (Zou et al. 2023) negative-direction results;
  Adaptive Residual-Update Steering (RUDDER, 2025) context-sensitive gating.

Per-quadrant alpha grids:
  epistemic:               [0.0, 0.5, 1.0, 1.5]        (positive — already helps)
  aleatoric:               [0.5, 1.0, 1.5, 2.0]        (large positive needed)
  confident_hallucination: [-1.5, -1.0, -0.5, 0.0]    (negative — destabilise)
  confident_correct:       [-0.5, 0.0]                  (try negative, minimal range)

Expected GPU time on T4: ~90 min (max_new_tokens=128, n=307).

Outputs:
  results/logs/22c_perquadrant_steering_results.json
  results/plots/22c_perquadrant_steering.png
"""

import sys
import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.labeling.label_responses import label_response
from src.analysis.uncertainty_decomposition import assign_quadrant
from src.steering.activation_steering import (
    compute_quadrant_steering_vectors,
    generate_steered_adaptive,
    QUADRANT_NAMES,
)

RESULTS_LOGS   = ROOT / "results" / "logs"
RESULTS_HS     = ROOT / "results" / "hidden_states"
RESULTS_PLOTS  = ROOT / "results" / "plots"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)

import os as _os

# Per-quadrant alpha grids — the core design of this experiment.
# Override any grid via env var e.g. EXP22C_EPISTEMIC_ALPHAS=0.0,0.5,1.0
def _parse_alphas(env_key, default):
    raw = _os.environ.get(env_key, "")
    return [float(a) for a in raw.split(",")] if raw else default

QUADRANT_ALPHAS = {
    "epistemic":               _parse_alphas("EXP22C_EPIST_ALPHAS",  [0.0, 0.5, 1.0, 1.5]),
    "aleatoric":               _parse_alphas("EXP22C_ALEAT_ALPHAS",  [0.5, 1.0, 1.5, 2.0]),
    "confident_hallucination": _parse_alphas("EXP22C_CHALL_ALPHAS",  [-1.5, -1.0, -0.5, 0.0]),
    "confident_correct":       _parse_alphas("EXP22C_CCORR_ALPHAS",  [-0.5, 0.0]),
}

ENTROPY_THRESHOLD    = float(_os.environ.get("ENTROPY_THRESHOLD", "0.3704"))
_cons_env            = _os.environ.get("CONSISTENCY_THRESHOLD", "")
CONSISTENCY_OVERRIDE = float(_cons_env) if _cons_env else None
MIN_SAMPLES          = 5
N_BOOTSTRAP          = 1000
MAX_NEW_TOKENS       = 128   # shorter for speed; TQA/NQ answers are brief
SEED                 = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_prerequisites():
    sv_path    = RESULTS_HS / "steering_vector.npy"
    hs_path    = RESULTS_HS / "hidden_states.npz"
    probe_path = RESULTS_LOGS / "06_probing_results.json"
    feat_path  = DATA_PROCESSED / "features_combined.csv"

    for p in [sv_path, hs_path, probe_path]:
        if not p.exists():
            print(f"Missing: {p}")
            sys.exit(1)

    sv   = np.load(sv_path)
    data = np.load(hs_path, allow_pickle=True)
    with open(probe_path) as f:
        best_layer = int(json.load(f)["best_layer"])

    features_df = pd.read_csv(feat_path) if feat_path.exists() else None
    return sv, data, best_layer, features_df


def assign_sample_quadrants(data, features_df):
    pids = data["prompt_ids"].tolist()

    if features_df is None:
        return np.array(["confident_hallucination"] * len(pids))

    feat_sub = features_df.set_index("prompt_id")
    hs_feat  = features_df[features_df["prompt_id"].isin(set(pids))]
    cons_threshold = (
        CONSISTENCY_OVERRIDE
        if CONSISTENCY_OVERRIDE is not None
        else float(hs_feat["consistency_score"].median())
    )
    print(f"  entropy_threshold:     {ENTROPY_THRESHOLD:.4f}")
    print(f"  consistency_threshold: {cons_threshold:.4f}  (data-driven median)")

    quadrants = []
    for pid in pids:
        if pid in feat_sub.index:
            row   = feat_sub.loc[pid]
            ent   = float(row.get("entropy", 0.5))            if not pd.isna(row.get("entropy",            np.nan)) else 0.5
            cons  = float(row.get("consistency_score", 0.5))  if not pd.isna(row.get("consistency_score",  np.nan)) else 0.5
            cos_c = row.get("cos_correct",      None)
            cos_h = row.get("cos_hallucinated", None)
            cos_c = float(cos_c) if cos_c is not None and not pd.isna(cos_c) else None
            cos_h = float(cos_h) if cos_h is not None and not pd.isna(cos_h) else None
            q = assign_quadrant(ent, cons,
                                entropy_threshold=ENTROPY_THRESHOLD,
                                consistency_threshold=cons_threshold,
                                cos_correct=cos_c, cos_hallucinated=cos_h)
        else:
            q = "confident_hallucination"
        quadrants.append(q)
    return np.array(quadrants)


def bootstrap_ci(outcomes: list, n_bootstrap: int = 1000, seed: int = 0):
    """Bootstrap 95% CI on hallucination rate from binary outcomes (0=hall,1=correct)."""
    rng = np.random.default_rng(seed)
    arr = np.array(outcomes, dtype=float)
    rates = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        rates.append(1.0 - sample.mean())
    rates = np.sort(rates)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def plot_results(per_q_sweep, combined, universal_rate, exp22_rate, out_path):
    """
    Three-panel figure:
      Left:   per-quadrant hall_rate vs alpha (including negative range)
      Middle: combined comparison bar chart with CI
      Right:  per-quadrant optimal rates vs universal baseline
    """
    colors = {
        "epistemic":               "#e15759",
        "aleatoric":               "#f28e2b",
        "confident_hallucination": "#4e79a7",
        "confident_correct":       "#59a14f",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: per-quadrant alpha curves ───────────────────────────────────
    ax = axes[0]
    for q in QUADRANT_NAMES:
        qd = per_q_sweep.get(q, {})
        if not qd:
            continue
        alphas = sorted(qd.keys())
        rates  = [qd[a] for a in alphas]
        ax.plot(alphas, rates, "-o", color=colors[q], linewidth=2,
                markersize=7, label=q.replace("_", " "))

    ax.axhline(universal_rate, color="black", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Universal best ({universal_rate:.3f})")
    ax.axvline(0, color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering coefficient α")
    ax.set_ylabel("Hallucination rate")
    ax.set_title("Per-quadrant α sweep\n(negative α = destabilise confident wrong)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # ── Panel 2: combined comparison with bootstrap CI ───────────────────────
    ax2 = axes[1]
    methods = ["Universal\n(Exp 11)", "Shared-α\n(Exp 22)", "Per-quadrant\nOptimal (22c)"]
    rates2  = [universal_rate, exp22_rate, combined["hall_rate"]]
    ci_lo   = [universal_rate, exp22_rate, combined["ci_lo"]]
    ci_hi   = [universal_rate, exp22_rate, combined["ci_hi"]]
    bcolors = ["#bab0ac", "#bab0ac", "#4e79a7"]

    bars = ax2.bar(methods, rates2, color=bcolors, alpha=0.85, zorder=3)
    for i, (bar, lo, hi) in enumerate(zip(bars, ci_lo, ci_hi)):
        x = bar.get_x() + bar.get_width() / 2
        ax2.plot([x, x], [lo, hi], color="black", linewidth=2, zorder=4)
        ax2.plot([x - 0.1, x + 0.1], [lo, lo], color="black", linewidth=2, zorder=4)
        ax2.plot([x - 0.1, x + 0.1], [hi, hi], color="black", linewidth=2, zorder=4)
        ax2.text(x, rates2[i] + 0.02, f"{rates2[i]:.3f}", ha="center",
                 fontsize=10, fontweight="bold")

    ax2.set_ylabel("Hallucination rate (lower = better)")
    ax2.set_title("Steering method comparison\n(Bootstrap 95% CI on Per-quadrant)")
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", alpha=0.3)
    improvement = universal_rate - combined["hall_rate"]
    ax2.text(0.5, 0.92, f"Improvement over universal: {improvement:+.3f}",
             transform=ax2.transAxes, ha="center", fontsize=9,
             color="#4e79a7" if improvement > 0 else "#e15759",
             fontweight="bold")

    # ── Panel 3: per-quadrant optimal rates ──────────────────────────────────
    ax3 = axes[2]
    q_names   = [q for q in QUADRANT_NAMES if q in per_q_sweep and per_q_sweep[q]]
    opt_rates = [min(per_q_sweep[q].values()) for q in q_names]
    opt_alphas= [min(per_q_sweep[q], key=per_q_sweep[q].get) for q in q_names]
    q_colors  = [colors[q] for q in q_names]

    bars3 = ax3.bar(range(len(q_names)), opt_rates, color=q_colors, alpha=0.85)
    for bar, rate, alpha_val in zip(bars3, opt_rates, opt_alphas):
        x = bar.get_x() + bar.get_width() / 2
        ax3.text(x, rate + 0.02, f"{rate:.3f}\nα={alpha_val:+.1f}",
                 ha="center", fontsize=8)

    ax3.axhline(universal_rate, color="black", linestyle="--", linewidth=1.5,
                label=f"Universal ({universal_rate:.3f})")
    ax3.set_xticks(range(len(q_names)))
    ax3.set_xticklabels([q.replace("_", "\n") for q in q_names], fontsize=8)
    ax3.set_ylabel("Hallucination rate at optimal α")
    ax3.set_title("Per-quadrant optimal α\n(independent optimisation)")
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Experiment 22c — Per-Quadrant Independent Alpha Optimisation")
    print("REQUIRES GPU. Run on Google Colab T4.")
    print("=" * 65)
    print("\nPer-quadrant alpha grids:")
    for q, alphas in QUADRANT_ALPHAS.items():
        print(f"  {q:<30} {alphas}")
    print()

    universal_sv, data, best_layer, features_df = load_prerequisites()
    print(f"Best layer: L{best_layer}")

    labels  = data["labels"].astype(int)
    prompts = data["prompts"].tolist()
    gts     = data["ground_truths"].tolist()
    pids    = data["prompt_ids"].tolist()

    # ── Assign quadrants ──────────────────────────────────────────────────────
    print("\nAssigning quadrants...")
    all_quadrants = assign_sample_quadrants(data, features_df)
    q_dist = {q: int((all_quadrants == q).sum()) for q in QUADRANT_NAMES}
    print("  Quadrant distribution (all 700):")
    for q, n in q_dist.items():
        print(f"    {q:<30} {n:3d}")

    # ── Load model ────────────────────────────────────────────────────────────
    from src.model.query_hf import load_model, cleanup
    model, tokenizer = load_model()

    # ── Compute per-quadrant steering vectors ─────────────────────────────────
    print("\nComputing per-quadrant steering vectors...")
    q_vectors = compute_quadrant_steering_vectors(
        model, tokenizer, prompts, labels,
        all_quadrants.tolist(), best_layer,
        min_samples=MIN_SAMPLES,
    )

    # ── Select ALL hallucinated samples ───────────────────────────────────────
    hall_idx = np.where(labels == 0)[0]
    n_total  = len(hall_idx)
    rng      = np.random.default_rng(SEED)
    selected = rng.permutation(hall_idx)      # shuffle but use all
    sel_q    = all_quadrants[selected]

    print(f"\nTotal hallucinated samples: {n_total}")
    sel_q_dist = {q: int((sel_q == q).sum()) for q in QUADRANT_NAMES}
    print("  Per-quadrant counts:")
    for q, cnt in sel_q_dist.items():
        print(f"    {q:<30} {cnt:3d}")

    # ── Phase 2: Per-quadrant independent alpha sweep ─────────────────────────
    # For each quadrant, sweep its alpha grid using only samples IN that quadrant.
    print("\n" + "=" * 65)
    print("Phase 2: Per-quadrant alpha sweep")
    print("=" * 65)

    per_q_sweep   = {q: {} for q in QUADRANT_NAMES}  # q -> {alpha: hall_rate}
    per_q_optimal = {}                                 # q -> best alpha

    for q_name in QUADRANT_NAMES:
        q_alphas  = QUADRANT_ALPHAS.get(q_name, [0.0])
        q_indices = [i for i in selected if all_quadrants[i] == q_name]
        n_q       = len(q_indices)

        if n_q < MIN_SAMPLES:
            print(f"\n  {q_name}: only {n_q} samples — skipping sweep")
            per_q_optimal[q_name] = 0.0
            continue

        print(f"\n  {q_name} (n={n_q})  alphas={q_alphas}")
        best_rate  = 1.0
        best_alpha = 0.0

        for alpha in q_alphas:
            corrected = 0
            wrong     = 0
            for idx in tqdm(q_indices, desc=f"    {q_name} α={alpha:+.1f}", leave=False):
                prompt   = str(prompts[idx])
                gt       = str(gts[idx])
                try:
                    response = generate_steered_adaptive(
                        model, tokenizer, prompt, q_name,
                        q_vectors, best_layer, alpha,
                        max_new_tokens=MAX_NEW_TOKENS,
                        fallback_to_universal=True,
                        universal_sv=universal_sv,
                    )
                    lbl = label_response(response, gt)
                    if lbl == 1:
                        corrected += 1
                    else:
                        wrong += 1
                except Exception as e:
                    print(f"\n    Error: {e}")
                    wrong += 1

            n_run = corrected + wrong
            rate  = wrong / n_run if n_run > 0 else float("nan")
            per_q_sweep[q_name][alpha] = round(rate, 4)
            print(f"    α={alpha:+.1f}  hall_rate={rate:.4f}  N={n_run}")

            if rate < best_rate:
                best_rate  = rate
                best_alpha = alpha

        per_q_optimal[q_name] = best_alpha
        print(f"  → Optimal for {q_name}: α={best_alpha:+.1f}  hall_rate={best_rate:.4f}")

    # ── Phase 3: Combined run with per-quadrant optimal alpha ─────────────────
    print("\n" + "=" * 65)
    print("Phase 3: Combined run — each sample gets its quadrant's optimal α")
    print(f"  Per-quadrant optimal alphas: {per_q_optimal}")
    print("=" * 65)

    outcomes = []   # 1=corrected, 0=still hallucinated
    per_q_combined = {q: {"corrected": 0, "wrong": 0} for q in QUADRANT_NAMES}

    for idx in tqdm(selected, desc="Combined optimal"):
        prompt   = str(prompts[idx])
        gt       = str(gts[idx])
        quadrant = all_quadrants[idx]
        alpha    = per_q_optimal.get(quadrant, 0.0)

        try:
            response = generate_steered_adaptive(
                model, tokenizer, prompt, quadrant,
                q_vectors, best_layer, alpha,
                max_new_tokens=MAX_NEW_TOKENS,
                fallback_to_universal=True,
                universal_sv=universal_sv,
            )
            lbl = label_response(response, gt)
            outcomes.append(lbl)
            if lbl == 1:
                per_q_combined[quadrant]["corrected"] += 1
            else:
                per_q_combined[quadrant]["wrong"] += 1
        except Exception as e:
            print(f"\n  Error: {e}")
            outcomes.append(0)
            per_q_combined[quadrant]["wrong"] += 1

    combined_hall = 1.0 - float(np.mean(outcomes))
    ci_lo, ci_hi  = bootstrap_ci(outcomes, N_BOOTSTRAP, SEED)

    print(f"\nCombined per-quadrant optimal result:")
    print(f"  hall_rate = {combined_hall:.4f}  (n={len(outcomes)})")
    print(f"  Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    for q in QUADRANT_NAMES:
        qc = per_q_combined[q]["corrected"]
        qw = per_q_combined[q]["wrong"]
        qt = qc + qw
        if qt > 0:
            print(f"    {q:<30} hall={qw/qt:.4f}  N={qt}  α={per_q_optimal.get(q, 0):+.1f}")

    cleanup()

    # ── Load comparison baselines ─────────────────────────────────────────────
    exp11_path = RESULTS_LOGS / "11_steering_results.json"
    universal_rate = 0.67  # fallback
    if exp11_path.exists():
        with open(exp11_path) as f:
            exp11 = json.load(f)
        opt_a = exp11.get("optimal_alpha", 0.0)
        universal_rate = exp11["results"][str(opt_a)]["hallucination_rate"]

    exp22_path = RESULTS_LOGS / "22_quadrant_steering_results.json"
    exp22_rate = 0.68  # fallback
    if exp22_path.exists():
        with open(exp22_path) as f:
            exp22 = json.load(f)
        exp22_rate = exp22.get("best_adaptive_hall", 0.68)

    improvement   = universal_rate - combined_hall
    improvement22 = exp22_rate    - combined_hall
    print(f"\nComparison:")
    print(f"  Universal steering (Exp 11):     {universal_rate:.4f}")
    print(f"  Shared-α adaptive (Exp 22):      {exp22_rate:.4f}")
    print(f"  Per-quadrant optimal (Exp 22c):  {combined_hall:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Improvement vs universal:        {improvement:+.4f}")
    print(f"  Improvement vs shared-α:         {improvement22:+.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    combined_result = {"hall_rate": combined_hall, "ci_lo": ci_lo, "ci_hi": ci_hi}
    plot_results(
        per_q_sweep, combined_result, universal_rate, exp22_rate,
        RESULTS_PLOTS / "22c_perquadrant_steering.png",
    )

    # ── Save log ──────────────────────────────────────────────────────────────
    log = {
        "experiment":           "22c_perquadrant_steering",
        "best_layer":           best_layer,
        "n_total_hallucinated": n_total,
        "entropy_threshold":    ENTROPY_THRESHOLD,
        "quadrant_alphas":      QUADRANT_ALPHAS,
        "per_q_sweep":          {q: {str(a): r for a, r in v.items()}
                                  for q, v in per_q_sweep.items()},
        "per_q_optimal":        per_q_optimal,
        "combined": {
            "hall_rate":       round(combined_hall, 4),
            "ci_lo":           round(ci_lo, 4),
            "ci_hi":           round(ci_hi, 4),
            "n":               len(outcomes),
            "per_q_breakdown": {
                q: {
                    "corrected": per_q_combined[q]["corrected"],
                    "wrong":     per_q_combined[q]["wrong"],
                    "hall_rate": round(
                        per_q_combined[q]["wrong"] /
                        max(1, per_q_combined[q]["corrected"] + per_q_combined[q]["wrong"]), 4),
                    "optimal_alpha": per_q_optimal.get(q, 0.0),
                }
                for q in QUADRANT_NAMES
            },
        },
        "comparison": {
            "universal":            round(universal_rate, 4),
            "exp22_shared_alpha":   round(exp22_rate, 4),
            "exp22c_per_quadrant":  round(combined_hall, 4),
            "improvement_vs_univ":  round(improvement, 4),
            "improvement_vs_exp22": round(improvement22, 4),
        },
    }

    log_path = RESULTS_LOGS / "22c_perquadrant_steering_results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved: {log_path.name}")
    print("\nExperiment 22c complete. Review results then run 23_final_evaluation.py")


if __name__ == "__main__":
    main()
