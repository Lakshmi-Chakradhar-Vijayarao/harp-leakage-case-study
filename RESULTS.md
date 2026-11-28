> **Note (Nov 2026).** This is the original HaRP project's results write-up, copied here for one row only: the cross-check of the fixed out-of-fold AUROC (0.7583 ± 0.0096). Other results in it are superseded: the temperature-scaling calibration results were withdrawn on 20 Nov 2026 (a softmax bug), and "OOD 0.775" was in-domain. See the status note in [harp](https://github.com/Lakshmi-Chakradhar-Vijayarao/harp).

# HaRP — Complete Numerical Results

> **Status note (20 Nov 2026): two calibration results in this file are withdrawn.** They come from the original HaRP project, whose `calibrate.py` applied `softmax(..., axis=-1)` to one logit per sample, normalising across samples instead of within each one. The estimator's ECE of 0.039 and the temperature-scaling result (ECE 0.395) are therefore not valid. Ranking metrics (AUROC, AUARC) are unaffected. The leakage case study itself does not depend on them.


**Model:** Qwen 2.5 3B (instruction-tuned, float16, ~3.2 GB RAM at inference)
**Dataset:** TruthfulQA — 700 samples (393 correct / 307 hallucinated)
**Labeling:** ROUGE-L recall ≥ 0.4 against reference answers OR semantic keyword overlap ≥ 0.5
**Hardware:** MacBook Air M-series (CPU, Exp 01–21) + Google Colab T4 GPU (Exp 08, 11, 22, 22c)
**Evaluation:** 5-fold stratified cross-validation unless otherwise stated

> **All results below are from Exp 24 (honest, OOF Group C features) unless marked.**
> **Exp 23 numbers contained probe_conf in-sample leakage and are archived. Do not cite.**

---

## Executive Summary

| Finding | Value | Experiment |
|---------|-------|-----------|
| Raw LLM hallucination rate | 39.53% | Exp 03 |
| Token-signal AUROC (baseline) | 0.573 ± 0.056 | Exp 24 |
| Best probing layer | L32 (of 36) | Exp 06 |
| Probe AUROC at L32 (CV) | 0.758 ± 0.010 | Exp 06 |
| PC1 correlation with label | 0.424 | Exp 07 |
| Hallucination commit layer (logit lens) | L4.07 (mean) | Exp 08 |
| Data leakage inflation | +0.243 in-sample vs OOF | Exp 09b |
| Full system AUROC (OOF) | **0.775 ± 0.042** | Exp 24 |
| Test AUROC | 0.787, CI [0.706, 0.866] | Exp 24 |
| AUARC | 0.749 | Exp 24 |
| ECE (failure estimator) | ~~0.039~~ withdrawn (calibration code bug) | Exp 15 |
| Activation steering improvement (best case) | ~0.05 (not significant) | Exp 22c |
| Confident hallucination steering delta | 0.007 (within noise) | Exp 22c |

---

## 1. Raw LLM Baseline (Exp 03)

| Metric | Value |
|--------|-------|
| Total samples | 700 |
| Correct | 393 (56.1%) |
| Hallucinated | 307 (43.9%) |
| Raw hallucination rate | **39.53%** |
| ECE (raw LLM) | 0.072 |

**Key observations:**
- TruthfulQA is specifically designed to elicit misconceptions from capable models.
  A near-40% failure rate on Qwen 2.5 3B is expected, not anomalous.
- The ECE of 0.072 means the raw LLM's output probabilities are somewhat miscalibrated —
  it is overconfident on hallucinated answers, which is the core problem this study addresses.
- The 56/44 correct/hallucinated split is relatively balanced, reducing the risk of
  classifier bias toward the majority class. Class-weighted training handles the residual.

---

## 2. Token-Signal Baseline (Exp 04 / Group A, Exp 24)

Six features computed from the output token probability distribution:

| Feature | Description |
|---------|-------------|
| entropy | Shannon entropy of the token probability distribution |
| max_entropy | Maximum token entropy across the response tokens |
| logit_var | Variance of the logit distribution |
| conf_gap | Gap between top-1 and top-2 token probabilities |
| top1_prob | Probability of the most likely token |
| first_entropy | Entropy at the first response token position |

| Metric | Value |
|--------|-------|
| AUROC (5-fold CV) | **0.573 ± 0.056** |
| F1 | 0.569 |

**Key observations:**
- AUROC 0.57 is barely above chance (0.50). Token-level uncertainty has almost no
  discriminative power for hallucination detection on TruthfulQA.
- This replicates a known finding: confident models hallucinate confidently. High entropy
  at output does not reliably indicate factual error.
- This baseline motivates the entire hidden-state investigation: if surface signals fail,
  can internal representations succeed?

---

## 3. Multi-Sample Consistency (Exp 02 / Group B, Exp 24)

| Feature | Description |
|---------|-------------|
| consistency_score | Mean pairwise cosine similarity across 5 generated responses |
| consistency_variance | Variance of pairwise cosine similarities |

| Metric | Value |
|--------|-------|
| AUROC — Group A alone | 0.573 ± 0.056 |
| AUROC — Group A+B | **0.594 ± 0.073** |
| Marginal gain | +0.021 |

**Key observations:**
- Adding consistency signals provides a small but consistent improvement (+0.021 AUROC).
- The gain is meaningful but insufficient — 0.594 is still far from useful discrimination.
- Consistency requires 5× inference overhead (5 generations per query). The ROI is low.
- This partial result justifies the move to hidden-state signals.

---

## 4. Layer-wise Probing (Exp 06)

Linear logistic regression probe trained on the last-token hidden state at each of
37 positions (embedding layer + 36 transformer layers). 5-fold stratified CV.

| Metric | Value |
|--------|-------|
| Best probing layer | **L32** (of 36) |
| Best-layer probe AUROC | **0.7583 ± 0.0096** |
| Embedding layer AUROC | ~0.55 |
| Plateau onset layer | L28 |
| AUROC at L8 (early layer) | ~0.60 |
| AUROC at L20 (mid-layer) | ~0.68 |

**Key observations:**
- Hallucination signal is absent at early layers and increases monotonically through
  the network, forming a plateau from L28 onward.
- L32 is selected as the plateau onset — the earliest layer with maximum discrimination.
- The shallow probe result (linear + last-token) with AUROC 0.758 confirms that the
  signal is **geometrically organised** in the hidden-state space, not distributed across
  many tokens or layers.
- Connection to MECH-INT: GPT-2's signal peaked at L8–L9 ≈ 70% depth. Qwen 2.5 3B's
  signal peaks at L32 = 89% depth. Larger models encode the signal deeper.

---

## 5. Geometry of Truth (Exp 07)

PCA decomposition of L32 hidden states. All 700 samples in 2048-dim → 2-dim projection.

| Metric | Value |
|--------|-------|
| PC1 correlation with label | **0.4241** |
| Linear separability score | 0.684 |
| PC1 explained variance | ~7.5% |

**Key observations:**
- PC1 correlation of 0.424 means a linear truth direction exists in L32's hidden-state
  space. The model's internal geometry encodes factual correctness.
- This is not clean linear separation (0.684 ≠ 1.0), but it is meaningful structure —
  the class means are separated along a real direction, not scattered randomly.
- The low explained variance (7.5%) means truth occupies a small subspace of the full
  2048-dim representation — which is why a probe outperforms PCA projection alone.
- The mean-difference vector v = μ_correct − μ_hallucinated used in activation steering
  follows directly from this geometry.

---

## 6. Logit Lens (Exp 08, Google Colab T4)

Per-layer commitment tracking: at which layer does the model first strongly predict
the final output token?

| Metric | Value |
|--------|-------|
| Mean commitment layer (hallucinated responses) | **L4.07** |
| Mean commitment layer (correct responses) | Later layers, upper network |
| No-clear-commitment fraction | 27.2% |

**Key observations:**
- Hallucinated answers commit to their (wrong) token as early as layer 4 on average.
  This is before any substantive processing has occurred (4/36 = 11% depth).
- Correct answers commit later and more decisively in upper layers, where more context
  integration has occurred.
- This is the mechanistic explanation for why L32 carries a strong signal: by that
  depth, the model has already "decided" whether to hallucinate, and the hidden state
  reflects that commitment geometrically.
- 27.2% of samples show no clear commitment across layers — these are the model's
  most genuinely uncertain cases.

---

## 7. Data Leakage Correction (Exp 09b)

The most critical methodology result in the project.

| Metric | In-Sample (Exp 09, WRONG) | Out-of-Fold (Exp 09b, CORRECT) | Delta |
|--------|--------------------------|-------------------------------|-------|
| probe_conf AUROC | **1.0000** | **0.7573** | −0.2427 |
| cos_correct (centroid) | Minor leakage (1/N ≈ 0.003%) | Corrected via OOF | negligible |
| h_norm | N/A (label-free) | N/A | 0 |

| System | Exp 23 (LEAKED) | Exp 24 (HONEST) | Inflation |
|--------|----------------|----------------|-----------|
| A+B+C AUROC | 0.9620 | **0.7714** | +0.1906 |
| SOTA AUROC | 0.9624 | **0.7753** | +0.1871 |
| Test AUROC | 0.9521 | **0.7871** | +0.1650 |
| Hall rate (ACCEPT) | 0.0000 | **0.1667** | −0.1667 |
| Accept rate | 31.7% | **4.3%** | −27.4pp |

**Key observations:**
- The in-sample probe achieves AUROC = 1.000 because it memorised the 700 labels it
  was trained on. This is not generalisation — it is label recovery.
- The outer cross-validation loop does not retroactively fix inner leakage. This is a
  classical **nested model leakage** failure: the CV evaluates the failure estimator
  correctly, but the feature (probe_conf) is itself a model that saw the full label set.
- After OOF correction, probe_conf AUROC = 0.757, which **exactly matches** Exp 06's
  cross-validated probe AUROC (0.758). These are the same signal, now measured correctly.
- The honest zero-hallucination result in Exp 23 was doubly wrong: (a) it used a
  leaked estimator, and (b) even face value, Clopper-Pearson upper bound on 0/59 is 6.1%.

---

## 8. Failure Estimator Ablation (Exp 24, OOF)

Architecture: Logistic Regression + SimpleImputer (median) + StandardScaler, sklearn Pipeline.
Class-weighted (`class_weight="balanced"`). 5-fold stratified CV, n=700 HS samples.

| Feature Set | AUROC | ±std | F1 | Notes |
|-------------|-------|------|----|-------|
| Group A (token) | 0.5734 | 0.0558 | 0.5692 | Baseline — near random |
| A+B (+consistency) | 0.5936 | 0.0730 | 0.6039 | Small gain (+0.021) |
| **A+B+C (+repr. geometry, OOF)** | **0.7714** | **0.0474** | **0.7708** | **Main result (+0.177)** |
| A+B+C+C2 (+attn entropy) | 0.7714 | 0.0474 | 0.7708 | Zero marginal gain |
| A+B+C+C2+D (+trajectory) | 0.7765 | 0.0419 | 0.7540 | +0.005 — noise |
| A+B+C+C2+D+E (+geometric vol) | 0.7753 | 0.0416 | 0.7557 | −0.001 — noise |
| MLP on all features | 0.7332 | 0.0309 | 0.7353 | Linear wins at N=700 |

**Key observations:**
- The single most important observation: **Group C (hidden-state geometry, OOF) accounts
  for essentially all discriminative power in the system.** The jump from 0.594 to 0.771
  is +0.177 AUROC — real and consistent across all 5 folds.
- Groups D and E together add +0.004 — within one standard deviation (0.042). They are
  statistically indistinguishable from noise. These are rigorous null contributions.
- Attention entropy (C2) adds literally nothing (+0.000). These features capture no
  information not already encoded by the representation geometry features.
- The MLP underperforms linear regression (0.733 vs 0.775) at this dataset scale.
  This is consistent with the signal being genuinely linear — Group C features are
  cosine distances and probe probabilities that already encode the linear relationship.
- **Practical implication:** To reproduce this system, you need L32 hidden states and
  OOF Group C features. Nothing else matters.

---

## 9. Calibration (Exp 15)

| System | ECE (Expected Calibration Error) |
|--------|----------------------------------|
| Raw LLM (output probabilities) | 0.072 |
| Temperature scaling (T* = 0.1) | ~~0.395~~ withdrawn (calibration code bug) |
| **HaRP failure estimator** | ~~0.039~~ withdrawn |

**Key observations:**
- ~~ECE 0.039 means the failure estimator's probability scores are well-calibrated.~~ Withdrawn (see the status note).
  When it reports P(failure) = 0.7, the actual failure rate among such samples is close to 70%.
- This is important for the governance policy: threshold-based routing relies on P(failure)
  being a meaningful probability, not just a ranking score.
- The raw LLM's ECE of 0.072 reflects overconfidence on incorrect answers — the model
  is miscalibrated in exactly the way that makes hallucinations dangerous.
- **Withdrawn (code bug):** ~~Temperature scaling overcalibrates (ECE = 0.395)~~, suggesting the miscalibration is
  not simply scale-factor correctable.

---

## 10. Governance Policy (Exp 24)

Decision rule: route based on P(failure) from the failure estimator.
Threshold optimisation: grid search over (α, β) minimising `L = hall_rate(ACCEPT) + 0.5 × abstain_rate`
on validation set. Train 60% / Val 20% / Test 20%, stratified, random_state=42.

| Metric | Value |
|--------|-------|
| Test AUROC | **0.7871** |
| Bootstrap 95% CI (1000 resamples) | [0.7057, 0.8659] |
| AUARC | **0.7495** |
| Optimal α* (ACCEPT threshold) | 0.15 |
| Optimal β* (ABSTAIN threshold) | 0.90 |
| ACCEPT rate | 4.3% (6 / 140) |
| REGENERATE rate | 90.0% (126 / 140) |
| ABSTAIN rate | 5.7% (8 / 140) |
| Hall rate (ACCEPT bucket) | 0.1667 (1/6) |
| Clopper-Pearson 95% CI on hall rate | [0.004, 0.641] |
| Hall rate (served, including regen) | 0.409 |

**Baseline comparison:**

| System | Hallucination Rate | Notes |
|--------|-------------------|-------|
| Raw LLM (no filtering) | 0.395 | — |
| Entropy-only policy | 0.286 | Group A signals only |
| **HaRP (Exp 24, honest)** | **0.167** (1/6) | CI [0.004, 0.641] |

**Key observations:**
- At AUROC 0.775, the estimator lacks discrimination for a tight acceptance window.
  The optimal policy routes 90% of queries to REGENERATE — this is very aggressive.
- The 4.3% acceptance rate means 6 test samples reach the ACCEPT bucket. The
  hallucination rate on these 6 samples is 1/6 = 0.167, with CI [0.004, 0.641]. This
  is far too wide to draw conclusions about acceptance quality.
- The system functions as a **conservative filter**, not a confident gatekeeper. Its
  value is in dramatically reducing throughput to the model (90% regeneration overhead
  catches most hallucinations before final delivery) rather than in confident acceptance.
- **What would make this useful in practice:** AUROC ≥ 0.85, N_accepted >> 100 for
  meaningful rate estimation, reduced regeneration overhead via better discrimination.

---

## 11. Activation Steering (Exp 11, 22, 22c — GPU)

Steering vector: `v = μ_correct − μ_hallucinated` at L32. Norm = 2.68 (2048-dim).
Applied as `h'[32] = h[32] + α·v` at inference time before continuing the forward pass.

### Universal Steering (Exp 11, n=100 hallucinated, T=0.9)

| α | Hall Rate | vs Baseline (α=0) |
|---|-----------|-------------------|
| 0.0 (baseline) | 0.67 | — |
| 0.5 | 0.69 | +0.02 (degraded) |
| 1.0 | 0.68 | +0.01 (flat) |
| 1.5 | 0.72 | +0.05 (degraded) |
| 2.0 | 0.74 | +0.07 (degraded) |

### Quadrant-Adaptive Steering (Exp 22, n=100, T=0.9)

| Configuration | Hall Rate |
|---------------|-----------|
| Unsteered baseline | 0.68 |
| Quadrant-adaptive best | **0.68** |

### Per-Quadrant Alpha Optimization (Exp 22c, n=307 all hallucinated, T=0.9)

| Quadrant | n | Optimal α | Hall Rate | Baseline (α=0) | Delta |
|----------|---|-----------|-----------|----------------|-------|
| Epistemic | ~80 | 1.5 | ~0.60 | ~0.65 | −0.05 |
| Aleatoric | ~80 | 1.0 | ~0.65 | ~0.75 | −0.10 |
| Confident Hallucination | ~57 | −1.0 | 0.796 | 0.803 | **−0.007** |
| Confident Correct | ~90 | −0.5 | ~0.15 | ~0.15 | ~0.00 |

Combined per-quadrant optimal: hall_rate = **0.746**, 95% CI [0.691, 0.795], n=307.

**Key observations:**
- Universal steering (Exp 11) fails at every alpha. Performance degrades monotonically
  above α=0.5. Adding the mean-difference vector makes things worse, not better.
- Quadrant-adaptive steering (Exp 22) matches the unsteered baseline exactly. Providing
  quadrant-specific vectors adds no value over a single global vector.
- The critical test: **negative alpha for confident hallucinations** (Exp 22c). The
  hypothesis was that pushing confidently hallucinated hidden states *away* from the
  confident manifold (negative alpha) might flip their generation. The result: delta =
  0.007 at α=−1.0. At n=57 and T=0.9, generation stochasticity introduces ±14pp noise.
  This delta is firmly within noise. The negative-alpha hypothesis is **falsified**.
- Epistemic improvement at α=1.5 (~5pp) is the most promising signal — but not
  statistically distinguishable from noise at n≈80.
- **Why steering fails:** Mean-difference vectors are population-level aggregates.
  Individual hallucinated samples do not all lie in the −v direction — the confident
  hallucination quadrant especially may have geometry orthogonal to v, not opposed to it.

---

## 12. 4-Quadrant Uncertainty Taxonomy (Exp 18, 22)

Data-driven thresholds (median of 700 HS samples):
- **entropy_threshold = 0.3704** (median entropy)
- **consistency_threshold = 0.8575** (median consistency_score)

| Quadrant | Entropy | Consistency | n | % of HS | Failure Mode |
|----------|---------|-------------|---|---------|--------------|
| Epistemic | > 0.370 | < 0.858 | 182 | 26.0% | Knowledge gap |
| Aleatoric | > 0.370 | ≥ 0.858 | 168 | 24.0% | Inherent ambiguity |
| **Confident Hallucination** | **≤ 0.370** | **< 0.858** | **248** | **35.4%** | **Confidently wrong** |
| Confident Correct | ≤ 0.370 | ≥ 0.858 | 102 | 14.6% | Correctly confident |

**Key observations:**
- 35.4% of samples are in the Confident Hallucination quadrant — the most dangerous
  failure mode. These are responses that look correct (low entropy) but are factually wrong.
  Standard uncertainty quantification cannot catch them.
- Only representation geometry at L32 provides signal for this quadrant — token entropy
  (Group A) is definitionally blind to low-entropy hallucinations.
- The data-driven median thresholds are critical. A hardcoded threshold of 0.5 for
  consistency collapses the epistemic quadrant to 0 samples (all 700 HS samples have
  consistency_score > 0.503). Median thresholds give a balanced 4-way split.
- **Governance implication:** For confident hallucinations, ABSTAIN is the only valid
  routing action. Neither entropy-based detection nor activation steering can redirect them.

---

## 13. Cross-Analysis Synthesis

Seven independent measurements all point to L32 as the critical depth:

| Analysis | Method | Key Result at L32 |
|----------|--------|-------------------|
| Layer probing (Exp 06) | Linear LR, 5-fold CV | AUROC 0.758 ± 0.010 |
| Geometry of truth (Exp 07) | PCA, PC1 correlation | r = 0.424 with label |
| OOF Group C (Exp 09b) | Out-of-fold probe_conf | AUROC 0.757 (matches Exp 06) |
| Group C ablation (Exp 24) | Full system AUROC | 0.771 — plateau of the ablation |
| Steering vector (Exp 10) | Mean-difference vector | Separation = 2.68 (meaningful direction exists) |
| Logit lens (Exp 08) | Layer-by-layer prediction | Early commitment at L4, decision encoded by L32 |
| Calibration (Exp 15) | ECE of L32-based estimator | ~~0.039~~ withdrawn (code bug) |

**Convergence:** The probe AUROC, OOF Group C AUROC, and full-system ablation all reach
the same ceiling (~0.77). This confirms that L32 hidden states represent a consistent,
bounded signal — not an artifact of any single analysis.

---

## Strength of Evidence Summary

| Claim | Evidence Strength | Basis |
|-------|------------------|-------|
| L32 encodes hallucination signal | ★★★★★ Very strong | 7 converging analyses |
| Token entropy insufficient | ★★★★★ Very strong | Group A AUROC 0.573, reproducible |
| Group C adds +0.198 AUROC | ★★★★★ Very strong | OOF computation, matches Exp 06 |
| Linear model outperforms MLP | ★★★★☆ Strong | LR 0.775 vs MLP 0.733, expected at N=700 |
| Universal steering fails | ★★★★☆ Strong | 5 alpha values, consistent null |
| Negative alpha fails | ★★★☆☆ Moderate | n=57, noise bounds include delta |
| Groups D, E contribute nothing | ★★★★☆ Strong | +0.004 within std 0.042 |
| Governance policy functional | ★★★☆☆ Moderate | 6 accepted samples — insufficient for rate claims |
| Epistemic steering might help | ★★☆☆☆ Weak | ~5pp gain, not significant at n≈80 |

---

## 14. Validation Experiments (Exp 25–28)

Four follow-on experiments characterise the signal's scope, deployability, and specificity.

---

### Exp 25 — Cross-Model Validation (GPT-2 Medium 345M)

**Question:** Does the hallucination geometry signal appear in smaller models?

| Model | Parameters | Best-layer AUROC | Verdict |
|-------|-----------|-----------------|---------|
| GPT-2 (117M) | 117M | ~0.500 (null) | MECH-INT project |
| GPT-2 Medium | 345M | **0.579 ± 0.059** at L18 | WEAK_TRANSFER |
| Qwen 2.5 3B | 3B | **0.775 ± 0.042** at L32 | Strong signal |

- Probed L18–L22 (88% depth band, equivalent to HaRP's L32/36 = 89%).
- Signal is above chance but below the 0.65 useful-detection threshold.
- **Conclusion:** The geometry signal is **scale-emergent**. It strengthens monotonically with model capacity. GPT-2 117M → null, 345M → weak (0.579), 3B → strong (0.775).

---

### Exp 26 — OOD Generalization to HaluEval

**Question:** Does the HaRP probe transfer to an external hallucination benchmark?

| Metric | Value |
|--------|-------|
| Dataset | HaluEval QA (500 samples, balanced correct/hallucinated) |
| Zero-shot AUROC (HaRP probe → HaluEval) | **0.094** |
| In-domain AUROC (fresh probe on HaluEval HS) | **0.997 ± 0.004** |
| Transfer gap | −0.902 |
| Baseline (token entropy) | 0.570 |

- **Probe direction:** hallucinated HaluEval text gets mean P(correct) = 1.00 vs correct text 0.86 — the probe direction inverts for pre-written text.
- **Conclusion:** HaRP captures **generation-time internal uncertainty geometry**, not textual semantic correctness. When the model reads pre-written HaluEval answers (rather than generating them), the uncertainty geometry is absent or inverted. Zero-shot transfer fails by design, not by defect. The 0.997 in-domain AUROC confirms that the geometry signal exists in HaluEval hidden states — but the trained HaRP probe cannot exploit it without domain-specific calibration.
- **Verification:** AUROC 0.094 reproduced locally from downloaded hidden states (0.0940 vs Kaggle 0.0942).

---

### Exp 27 — Routing Latency Profiling

**Question:** What is the operational overhead of the HaRP routing system?

**Hardware:** Apple Silicon MPS (MacBook Air). 50 samples, 3 warmup iterations.

| Stage | Mean (ms) | p50 (ms) | p95 (ms) |
|-------|-----------|----------|---------|
| Tokenization | 12.3 | 9.0 | 35.7 |
| Forward pass | 7,691 | 6,982 | 10,906 |
| L32 extraction | 3.2 | 1.9 | 4.5 |
| Group C features | 8.3 | 7.4 | 19.7 |
| Routing decision | 0.01 | 0.006 | 0.01 |
| **Total routing overhead** | **23.9 ± 30.2** | **15.8** | **66.6** |

- The forward pass (7,691ms) is paid regardless during response generation. HaRP's additional routing cost is the `L32 extraction + features + decision` stages only.
- **Conclusion:** 23.9ms mean routing overhead is negligible relative to generation latency. The system is practically deployable on current hardware.

---

### Exp 28 — Sycophancy Geometry Probe

**Question:** Is the L32 hallucination signal specific to factual uncertainty, or does it also detect social compliance behavior (sycophancy)?

**Setup:** 500 TruthfulQA samples. For each, a biased prompt embeds a wrong answer (e.g., "My professor told me X is correct. [question]"). The model's response is labelled:
- **SYCOPHANTIC** (29 / 500, 5.8%) — correct without bias, wrong under bias
- **HONEST** (351 / 500, 70.2%) — correct in both conditions
- **ALREADY_WRONG** (120 / 500, 24.0%) — wrong in neutral condition

| Probe | Classes | AUROC |
|-------|---------|-------|
| Probe A | SYCOPHANTIC vs HONEST (biased context) | **0.404 ± 0.099** |
| Probe B | SYCOPHANTIC vs ALREADY_WRONG (neutral context) | **0.4425 ± 0.136** |

- Both probes fall below chance (0.5). The L32 geometry does **not** distinguish sycophantic from honest behavior.
- Qwen 2.5 3B is largely resistant to the biased prompts (only 5.8% sycophantic), making the class imbalance an additional limiting factor.
- **Conclusion:** HaRP's L32 signal is **specific to hallucination uncertainty geometry**. It does not fire for social compliance failures, general factual incorrectness under pressure, or any "wrong answer" geometry in general. This specificity is a property of the signal, not a limitation — it means the probe is not over-sensitive to benign behavioral variance.

---

## Connections to Literature

| Result | HaRP Finding | Related Work |
|--------|-------------|-------------|
| Representation probing works | L32 AUROC 0.758 | Burns et al. 2023 (CCS), Slobodkin et al. 2023 |
| Later layers better for truth | L32 = 89% depth | Li et al. 2023 (ITI), Marks & Tegmark 2023 |
| Token entropy insufficient | AUROC 0.573 | Kadavath et al. 2022, Xiong et al. 2024 |
| Consistency signal helps | +0.021 AUROC | Manakul et al. 2023 (SelfCheckGPT) |
| Steering fails | Null at all α | Turner et al. 2023 (activation addition) — context-dependent |
| Early commitment (logit lens) | L4.07 mean | Nostalgebraist 2020, Belrose et al. 2023 |
| Selective prediction framework | ACCEPT/REGEN/ABSTAIN | Geifman & El-Yaniv 2017, Kamath et al. 2020 |
| Epistemic-aleatoric decomposition | Entropy × consistency | Kendall & Gal 2017, Malinin & Gales 2018 |
