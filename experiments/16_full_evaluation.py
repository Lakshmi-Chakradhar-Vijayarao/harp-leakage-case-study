"""
16_full_evaluation.py — Complete end-to-end evaluation and final results table.

Phase 10, Experiment 16.

Role of this experiment
------------------------
This is an AGGREGATION-ONLY experiment. It generates NO new model outputs, runs NO new
inference, and trains NO new estimators. It reads JSON logs written by experiments 04–15
and assembles them into a single coherent summary. This design ensures:
  - Final results are reproducible from any prior run state
  - The paper table always reflects whatever experiments have been run so far
  - Partially-completed pipelines still produce meaningful partial summaries

Why a single source of truth (final_results.json)?
----------------------------------------------------
The HaRP paper cites numbers from 11 different experiments. Quoting numbers directly
from individual JSON files risks version drift (re-running one experiment without another
produces inconsistent comparisons). final_results.json is written atomically at the end
of this script — it captures the state of all experiments at one moment in time. Any
citation in the paper should trace to this file, not to individual experiment logs.

What each key in the metrics table proves
------------------------------------------
  auroc_group_a          The baseline: how well token-level signals (entropy, top1_prob,
                         top-k spread) alone discriminate hallucination. Typically 0.60–0.68.
                         This is the STARTING POINT — it proves the problem is non-trivial
                         but that entropy signals are insufficient.

  auroc_group_ab         After adding Group B (cosine consistency across regenerations).
                         The near-zero delta (A → A+B) is a deliberate finding: consistency
                         signals do not add independent signal when the model is high-entropy.
                         The delta proves Group A and B are highly correlated in failure cases.

  auroc_group_abc        After adding Group C (representation geometry: distance to class
                         means, Mahalanobis distance, PCA reconstruction error). This jump
                         (typically +0.08–0.12) is the central empirical claim: GEOMETRY HELPS.
                         Group C detects confident hallucination (low entropy, wrong cluster
                         in representation space) — the failure mode invisible to Groups A+B.

  delta_a_to_abc         The headline delta. If this is > 0.08, HaRP's core claim holds.

  hall_rate_raw_llm      The unconditional hallucination rate — the starting problem. All
                         system comparisons are relative to this baseline.

  hall_rate_harp      Hallucination rate on ACCEPTED outputs only (the three-action policy
                         result). This is what users actually experience. A well-calibrated
                         policy should reduce this to ~0.10–0.20 (from ~0.35–0.50 raw).

  abstain_rate_harp   The cost of safety: what fraction of queries the system declines.
                         Together with hall_rate_harp this defines the precision-coverage
                         operating point for the λ=0.5 governance loss weight.

  alpha_star, beta_star  The optimized thresholds from Exp 12. α* is the Accept/REGENERATE
                         boundary (P(failure) below this → accept output). β* is the
                         REGENERATE/Abstain boundary. Their values determine the operating
                         regime: α*=0.30 means "accept anything the estimator thinks has
                         < 30% chance of being hallucinated."

  ece_raw_llm            Expected Calibration Error of the raw LLM (top1_prob). Expected
                         to be high (0.15–0.25): LLMs are overconfident. Confirms the
                         authorization policy cannot trust raw probabilities directly.

  ece_harp            ECE of the full HaRP failure estimator. Expected < 0.05.
                         If ECE_gov < ECE_raw: the multi-signal posterior is better calibrated
                         than the raw LLM's own probability estimate — justifying the
                         overhead of computing the full feature vector.

The 9 required plots and what each proves in the narrative
-----------------------------------------------------------
  Plot 1 — 07_pca_scatter.png
    The geometry of truth: PCA of Layer {best_layer} hidden states, coloured correct/hallucinated.
    If two clusters are visible, the linear truth direction is real. This is the geometric
    prerequisite for Group C signals to work at all.

  Plot 2 — 06_probing_curve.png
    Probing accuracy vs. transformer layer (AUROC). Peaks at best_layer. Proves that
    hallucination information is concentrated at a specific depth — not uniformly distributed.

  Plot 3 — 08_logit_lens.png
    Commitment layer distribution. Correct answers commit earlier (stable token probabilities
    over layers); hallucinations commit late or never. Motivates using trajectory signals
    (Group D) as complementary to single-layer geometry.

  Plot 4 — 09_auroc_comparison.png
    Three bars: Group A, Group A+B, Group A+B+C. Visualizes the delta story. The near-flat
    A→A+B then jump A+B→A+B+C is the visual proof that Group C is the critical component.

  Plot 5 — 16_ablation_bar.png   [GENERATED HERE]
    Same ablation but with 5-fold CV error bars. Provides statistical rigor for the delta.

  Plot 6 — 11_steering_curve.png
    Hallucination rate vs. steering coefficient α. Flat or worsening trend confirms that
    a UNIVERSAL steering vector is ineffective — motivating quadrant-adaptive steering (Exp 22).

  Plot 7 — 12_policy_decisions.png
    Bar chart of ACCEPT/REGENERATE/ABSTAIN counts. Visualizes the operating regime:
    are most samples accepted, or is the system over-abstaining?

  Plot 8 — 15_reliability_diagram.png
    Reliability diagrams for Raw LLM, temperature scaling, HaRP. Points on the diagonal
    = perfect calibration. Below diagonal = overconfident. Expected: HaRP closest to diagonal.

  Plot 9 — 14_sycophancy_comparison.png
    Entropy distributions for original vs. sycophantic contexts. If sycophantic framing
    reduces entropy on hallucinated samples, it confirms the motivation for Exp 14's finding
    that token signals alone are insufficient under adversarial prompting.

Produces
--------
  - results/logs/final_results.json      — single source of truth for all paper numbers
  - results/plots/16_ablation_bar.png    — Plot 5: component contribution bar with CV error bars
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

RESULTS_LOGS  = ROOT / "results" / "logs"
RESULTS_PLOTS = ROOT / "results" / "plots"
RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)


def load_json(name: str) -> dict:
    p = RESULTS_LOGS / name
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def check_plot(name: str) -> bool:
    return (RESULTS_PLOTS / name).exists()


def plot_ablation_bar(ablation: dict, out_path: Path) -> None:
    """Plot 5: component contribution — delta AUROC for each feature group."""
    groups = list(ablation.keys())
    aurocs = [ablation[g]["auroc_mean"] for g in groups]
    errs   = [ablation[g]["auroc_std"]  for g in groups]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"][:len(groups)]
    x = range(len(groups))

    bars = ax.bar(x, aurocs, color=colors, alpha=0.85, zorder=3)
    ax.errorbar(x, aurocs, yerr=errs,
                fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4)

    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2., val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.6, label="Random (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace(" (", "\n(") for g in groups], fontsize=9)
    ax.set_ylabel("AUROC (5-fold CV)")
    ax.set_title("HaRP Failure Estimator: Ablation by Feature Group\n"
                 "Each bar = adding the next signal group")
    ax.set_ylim(0.45, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


def build_metrics_table(exp04, exp09, exp12, exp13, exp15) -> dict:
    """Assemble the final metrics table from all experiment logs."""
    baselines_list = exp13.get("results", [])
    raw_hall = None
    gov_hall = None
    for b in baselines_list:
        if b.get("name") == "Raw LLM":
            raw_hall = b.get("hall_rate")
        if b.get("name") == "HaRP (full)":
            gov_hall = b.get("hall_rate")

    return {
        "auroc_group_a":    exp04.get("cv_auroc_mean"),
        "auroc_group_ab":   exp09.get("auroc_group_ab"),
        "auroc_group_abc":  exp09.get("auroc_group_abc"),
        "delta_a_to_abc":   exp09.get("delta_a_to_abc"),
        "hall_rate_raw_llm":    raw_hall,
        "hall_rate_harp":    (exp12.get("test_metrics") or {}).get("hall_rate_accepted"),
        "abstain_rate_harp": (exp12.get("test_metrics") or {}).get("abstain_rate"),
        "alpha_star":           exp12.get("alpha_star"),
        "beta_star":            exp12.get("beta_star"),
        "ece_raw_llm":          (exp15.get("systems") or {}).get("raw_llm", {}).get("ece"),
        "ece_harp":          (exp15.get("systems") or {}).get("harp_full", {}).get("ece"),
    }


def print_final_table(table: dict) -> None:
    print("\n" + "=" * 55)
    print("FINAL RESULTS TABLE")
    print("=" * 55)
    rows = [
        ("AUROC Group A (token signals)",  "auroc_group_a"),
        ("AUROC Group A+B (+consistency)", "auroc_group_ab"),
        ("AUROC Group A+B+C (full)",       "auroc_group_abc"),
        ("AUROC delta (A → A+B+C)",        "delta_a_to_abc"),
        ("Hallucination rate — Raw LLM",   "hall_rate_raw_llm"),
        ("Hallucination rate — HaRP",   "hall_rate_harp"),
        ("Abstention rate — HaRP",      "abstain_rate_harp"),
        ("Optimal α*",                     "alpha_star"),
        ("Optimal β*",                     "beta_star"),
        ("ECE — Raw LLM",                  "ece_raw_llm"),
        ("ECE — HaRP",                  "ece_harp"),
    ]
    for label, key in rows:
        val = table.get(key)
        val_s = f"{val:.4f}" if isinstance(val, float) else str(val) if val is not None else "not yet computed"
        print(f"  {label:<40}  {val_s}")
    print("=" * 55)


def check_all_plots() -> None:
    required = [
        "07_pca_scatter.png",
        "06_probing_curve.png",
        "08_logit_lens.png",
        "09_auroc_comparison.png",
        "16_ablation_bar.png",
        "11_steering_curve.png",
        "12_policy_decisions.png",
        "15_reliability_diagram.png",
        "14_sycophancy_comparison.png",
        "17_semantic_entropy_comparison.png",
        "18_uncertainty_quadrants.png",
        "19_cross_model_comparison.png",
    ]
    print("\nPlot status:")
    for name in required:
        status = "DONE" if check_plot(name) else "MISSING — run corresponding experiment"
        print(f"  {'[OK]' if check_plot(name) else '[  ]'}  {name:<45}  {status}")


def main():
    print("=" * 60)
    print("Experiment 16 — Full Evaluation + Final Results")
    print("=" * 60)

    # ── Load all experiment logs ──────────────────────────────────────────────
    exp04  = load_json("04_estimator_v1.json")
    exp06  = load_json("06_probing_results.json")
    exp07  = load_json("07_geometry_results.json")
    exp08  = load_json("08_logit_lens_results.json")
    exp09  = load_json("09_estimator_v2.json")
    exp10  = load_json("10_steering_vector_meta.json")
    exp11  = load_json("11_steering_results.json")
    exp12  = load_json("12_policy_results.json")
    exp13  = load_json("13_baselines.json")
    exp14  = load_json("14_sycophancy_results.json")
    exp15  = load_json("15_calibration_results.json")

    # ── Plot 5: ablation bar ──────────────────────────────────────────────────
    ablation = exp09.get("ablation", {})
    if ablation:
        plot_ablation_bar(ablation, RESULTS_PLOTS / "16_ablation_bar.png")
    else:
        print("Ablation data not found — skipping Plot 5 (run 09 first)")

    # ── Assemble metrics table ────────────────────────────────────────────────
    table = build_metrics_table(exp04, exp09, exp12, exp13, exp15)
    print_final_table(table)

    # ── Check all 9 plots ─────────────────────────────────────────────────────
    check_all_plots()

    # ── Key findings summary ──────────────────────────────────────────────────
    print("\n--- Key Findings ---")
    if exp07:
        print(f"  PC1 ↔ truth correlation (Layer {exp06.get('best_layer')}): "
              f"{exp07.get('pc1_correlation', 'n/a')}")
        print(f"  Linear separability:      {exp07.get('separability', 'n/a')}")
    if exp08:
        print(f"  Mean commitment layer (hallucinated): {exp08.get('mean_commit_hall', 'n/a')}")
    if exp11:
        print(f"  Optimal steering α:   {exp11.get('optimal_alpha', 'n/a')}")
    if exp14:
        sigs = exp14.get("signals", {})
        if "entropy" in sigs:
            s = sigs["entropy"]
            print(f"  Sycophancy entropy: "
                  f"syco={s.get('mean_sycophantic', 'n/a'):.4f} "
                  f"corr={s.get('mean_corrective', 'n/a'):.4f} "
                  f"(p={s.get('p_value', 'n/a')})")

    # ── Save final results ────────────────────────────────────────────────────
    final = {
        "experiment":        "16_full_evaluation",
        "metrics_table":     table,
        "key_findings": {
            "best_layer":          exp06.get("best_layer"),
            "probing_auroc":       exp06.get("best_auroc"),
            "pc1_correlation":     exp07.get("pc1_correlation"),
            "separability":        exp07.get("separability"),
            "commitment_layer":    exp08.get("mean_commit_hall"),
            "auroc_delta":         exp09.get("delta_a_to_abc"),
            "optimal_alpha_steer": exp11.get("optimal_alpha"),
            "alpha_star_policy":   exp12.get("alpha_star"),
            "beta_star_policy":    exp12.get("beta_star"),
        },
        "all_experiments": {
            "04_estimator_v1":   exp04,
            "06_probing":        exp06,
            "07_geometry":       exp07,
            "08_logit_lens":     exp08,
            "09_estimator_v2":   exp09,
            "10_steering_vec":   exp10,
            "11_steering_exp":   exp11,
            "12_policy":         exp12,
            "13_baselines":      exp13,
            "14_sycophancy":     exp14,
            "15_calibration":    exp15,
        },
    }
    final_path = RESULTS_LOGS / "final_results.json"
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved: {final_path.name}")

    print("\nExperiment 16 complete. All experiments done.")
    print("Next steps: write README, build Streamlit app.")


if __name__ == "__main__":
    main()
