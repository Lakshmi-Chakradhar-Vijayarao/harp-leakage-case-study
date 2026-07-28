"""
14_sycophancy_analysis.py — Sycophancy vs epistemic failure signal comparison.

Phase 8, Experiment 14.
Uses the 30 sycophancy-type adversarial prompts.
Labels each response: sycophantic (agreed with false premise) vs corrective.
Compares entropy, top1_prob, and consistency_score distributions.

Key question: Is entropy LOWER for sycophantic responses?
  YES → model is confidently wrong; entropy alone cannot catch this.
       Consistency score (Group B) should catch it.
  NO  → entropy is a sufficient signal for sycophancy detection.

Produces:
  - Plot 9: entropy + consistency comparison (sycophantic vs corrective)
              results/plots/14_sycophancy_comparison.png
  - results/logs/14_sycophancy_results.json
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
from scipy import stats

DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_LOGS   = ROOT / "results" / "logs"
RESULTS_PLOTS  = ROOT / "results" / "plots"
RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)


def label_sycophantic(response: str, ground_truth: str) -> int:
    """
    Returns 1 if the model gave a sycophantic response (agreed with false premise),
    0 if it gave a corrective response (pushed back).

    Heuristic: if ground_truth words are NOT in response, the model probably
    agreed with the false framing rather than correcting it.
    A low label_response score = hallucinated = agreed with false premise.
    """
    from src.labeling.label_responses import label_response
    # For sycophancy prompts, a correct response pushes back (contains ground truth).
    # label_response returns 1=correct(pushback), 0=hallucinated(sycophantic).
    pushback = label_response(response, ground_truth)
    return 0 if pushback == 1 else 1  # flip: 1=sycophantic, 0=corrective


def plot_comparison(syco_vals, corr_vals, signal_name: str,
                    ax: plt.Axes) -> None:
    """Box + strip plot for one signal."""
    data = [syco_vals, corr_vals]
    bp = ax.boxplot(data, labels=["Sycophantic", "Corrective"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#f28e2b")
    bp["boxes"][1].set_facecolor("#4e79a7")
    for d, x_pos in zip(data, [1, 2]):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(d))
        ax.scatter(x_pos + jitter, d, alpha=0.5, s=20,
                   color=["#f28e2b", "#4e79a7"][x_pos - 1], zorder=3)
    ax.set_ylabel(signal_name)

    if len(syco_vals) >= 3 and len(corr_vals) >= 3:
        t, p = stats.mannwhitneyu(syco_vals, corr_vals, alternative="two-sided")
        ax.set_title(f"{signal_name}\np={p:.4f} (Mann-Whitney)")
    else:
        ax.set_title(signal_name)


def main():
    print("=" * 60)
    print("Experiment 14 — Sycophancy Analysis")
    print("=" * 60)

    adv_labeled = DATA_PROCESSED / "adversarial_labeled.csv"
    if not adv_labeled.exists():
        print(f"Missing: {adv_labeled}\nRun 03 first.")
        sys.exit(1)

    df = pd.read_csv(adv_labeled)

    if "prompt_type" not in df.columns:
        print("No prompt_type column. Cannot isolate sycophancy prompts.")
        sys.exit(1)

    syco_df = df[df["prompt_type"] == "sycophancy"].copy()
    print(f"Sycophancy prompts found: {len(syco_df)}")

    if len(syco_df) == 0:
        print("No sycophancy prompts in dataset. Check adversarial_prompts.json types.")
        sys.exit(1)

    # ── Re-label for sycophantic behaviour ───────────────────────────────────
    # label column from 03 = 1 if model was correct (= corrective pushback)
    # We relabel: sycophantic_label = 1 if model agreed with false premise
    syco_df["sycophantic"] = syco_df.apply(
        lambda r: label_sycophantic(
            str(r.get("response", "")),
            str(r.get("ground_truth", ""))
        ),
        axis=1,
    )

    n_syco = int(syco_df["sycophantic"].sum())
    n_corr = int((syco_df["sycophantic"] == 0).sum())
    print(f"Sycophantic responses: {n_syco} | Corrective: {n_corr}")

    if n_syco < 2 or n_corr < 2:
        print("Too few samples in one category for meaningful comparison.")

    # ── Extract signals ───────────────────────────────────────────────────────
    signal_cols = ["entropy", "top1_prob", "consistency_score"]
    avail = [c for c in signal_cols if c in syco_df.columns]

    results_signals = {}
    for sig in avail:
        syco_vals = syco_df[syco_df["sycophantic"] == 1][sig].dropna().tolist()
        corr_vals = syco_df[syco_df["sycophantic"] == 0][sig].dropna().tolist()

        mean_s = float(np.mean(syco_vals)) if syco_vals else float("nan")
        mean_c = float(np.mean(corr_vals)) if corr_vals else float("nan")
        std_s  = float(np.std(syco_vals))  if syco_vals else float("nan")
        std_c  = float(np.std(corr_vals))  if corr_vals else float("nan")

        if len(syco_vals) >= 3 and len(corr_vals) >= 3:
            _, p = stats.mannwhitneyu(syco_vals, corr_vals, alternative="two-sided")
        else:
            p = float("nan")

        results_signals[sig] = {
            "mean_sycophantic": mean_s,
            "mean_corrective":  mean_c,
            "std_sycophantic":  std_s,
            "std_corrective":   std_c,
            "p_value":          float(p) if not np.isnan(p) else None,
            "n_sycophantic":    len(syco_vals),
            "n_corrective":     len(corr_vals),
        }

        diff = mean_s - mean_c
        direction = "LOWER" if diff < 0 else "HIGHER"
        print(f"\n  {sig}:")
        print(f"    Sycophantic: {mean_s:.4f} ± {std_s:.4f}")
        print(f"    Corrective:  {mean_c:.4f} ± {std_c:.4f}")
        print(f"    Sycophantic is {direction} by {abs(diff):.4f}  (p={p:.4f})")

    # ── Key finding ───────────────────────────────────────────────────────────
    if "entropy" in results_signals:
        entropy_diff = (results_signals["entropy"]["mean_sycophantic"] -
                        results_signals["entropy"]["mean_corrective"])
        if entropy_diff < 0:
            print("\n>>> FINDING: Entropy is LOWER for sycophantic responses.")
            print("    Entropy alone cannot distinguish confident-correct from confident-sycophantic.")
            print("    Consistency score is needed to catch this failure mode.")
        else:
            print("\n>>> FINDING: Entropy is HIGHER for sycophantic responses.")
            print("    Entropy may partially help, but consistency provides additional signal.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_plots = len(avail)
    if n_plots > 0:
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for ax, sig in zip(axes, avail):
            syco_v = syco_df[syco_df["sycophantic"] == 1][sig].dropna().tolist()
            corr_v = syco_df[syco_df["sycophantic"] == 0][sig].dropna().tolist()
            plot_comparison(syco_v, corr_v, sig, ax)

        fig.suptitle(
            f"Sycophancy Analysis: Signal Distributions\n"
            f"n_sycophantic={n_syco}, n_corrective={n_corr}",
            fontsize=12
        )
        plt.tight_layout()
        plot_path = RESULTS_PLOTS / "14_sycophancy_comparison.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nPlot saved: {plot_path.name}")

    # ── Save log ──────────────────────────────────────────────────────────────
    log = {
        "experiment":       "14_sycophancy_analysis",
        "n_sycophancy_prompts": len(syco_df),
        "n_sycophantic":    n_syco,
        "n_corrective":     n_corr,
        "signals":          results_signals,
    }
    log_path = RESULTS_LOGS / "14_sycophancy_results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved: {log_path.name}")

    print("Experiment 14 complete. Next: run 15_calibration.py")


if __name__ == "__main__":
    main()
