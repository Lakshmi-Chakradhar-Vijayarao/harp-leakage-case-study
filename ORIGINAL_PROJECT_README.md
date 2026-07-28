# Can a Language Model Know When It's Hallucinating?
## HaRP — Hallucination-aware Representation Probing

**Lakshmi Chakradhar Vijayarao**

Probing internal hidden-state geometry to detect hallucinations in Qwen 2.5 3B,
with a three-action uncertainty-aware output routing policy.

**Model:** Qwen 2.5 3B &nbsp;|&nbsp; **Dataset:** TruthfulQA (700 samples) &nbsp;|&nbsp; **Experiments:** 28 &nbsp;|&nbsp; **Hardware:** MacBook Air M-series + Google Colab T4 + Kaggle T4

[![Live Demo](https://img.shields.io/badge/Live%20Demo-harpfind.streamlit.app-7c4dff?logo=streamlit)](https://harpfind.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-harp-4527a0?logo=github)](https://github.com/Lakshmi-Chakradhar-Vijayarao/harp)

---

## The Problem

Language models hallucinate with confidence. Token entropy — the standard uncertainty
signal available from any LLM inference API — is near-random on factual QA:
**AUROC 0.57** on TruthfulQA. You cannot reliably tell whether a generated answer is
correct by looking at its surface probability distribution alone.

HaRP's central question: **does the model's internal representation "know" it's wrong,
even when its output looks confident?**

The answer is yes — but only if you look at the right layer, compute features honestly,
and report what the numbers actually say.

---

## The Three-Phase Pipeline

```
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │  PHASE 1 — DETECTION                                                                 │
 │  ─────────────────────────────────────────────────────────────────────────────────   │
 │                                                                                      │
 │   TruthfulQA (700 questions)                                                         │
 │        │                                                                             │
 │        ▼                                                                             │
 │   Qwen 2.5 3B  ──→  token log-probs  ──→  Group A signals (entropy, logit_var …)   │
 │        │                                                                             │
 │        ├──→  5× multi-sample generation  ──→  Group B signals (consistency)         │
 │        │                                                                             │
 │        └──→  ROUGE-L labeling  ──→  393 correct / 307 hallucinated                  │
 │                                                                                      │
 │  AUROC 0.573 (token signals alone — near random)                                     │
 └──────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │  PHASE 2 — LOCALISATION                                                              │
 │  ─────────────────────────────────────────────────────────────────────────────────   │
 │                                                                                      │
 │   Hidden states [700 × 37 layers × 2048 dim]  →  94 MB                              │
 │        │                                                                             │
 │        ├── Probe each layer (5-fold CV)  ──→  Best layer: L32 / AUROC 0.758         │
 │        │                                                                             │
 │        ├── PCA at L32  ──→  PC1 corr with label: 0.424  (linear truth direction)    │
 │        │                                                                             │
 │        ├── Logit lens (GPU)  ──→  Hallucination commits at L4.07 (too early)        │
 │        │                                                                             │
 │        └── OOF Group C features  ──→  cos_correct, cos_hall, h_norm, probe_conf     │
 │                                                                                      │
 │  AUROC 0.771 (hidden-state geometry, out-of-fold)  ← +0.198 over token signals      │
 └──────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │  PHASE 3 — ROUTING                                                                   │
 │  ─────────────────────────────────────────────────────────────────────────────────   │
 │                                                                                      │
 │   Failure estimator  ──→  P(failure) per query                                       │
 │        │                                                                             │
 │        ├── p_fail < α*  ──→  ACCEPT  (4.3% of queries)                              │
 │        ├── α* ≤ p_fail < β*  ──→  REGENERATE  (90.0%)                               │
 │        └── p_fail ≥ β*  ──→  ABSTAIN  (5.7%)                                        │
 │                                                                                      │
 │   ECE = 0.039 (well-calibrated)  |  Test AUROC 0.787  |  AUARC 0.749               │
 └──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Results Headlines

- **Representation geometry is real.** L32 hidden-state geometry (Group C, OOF) lifts AUROC
  from 0.573 → 0.771 — a **+0.198** gain. This is the entire discriminative contribution.
  Token entropy contributes almost nothing (0.57 ≈ random).

- **Groups D and E add nothing.** Cross-layer trajectory and geometric volume signals together
  add +0.004 AUROC over Group C — within measurement noise (std ≈ 0.05). Null contributions,
  rigorously documented.

- **Activation steering universally fails.** Five experiments (Exp 11, 22, 22c) test
  mean-difference steering at L32 across all alpha values and all four failure-mode quadrants.
  No improvement at any configuration, including negative alpha for confident hallucinations
  (delta = 0.007 — within generation noise).

- **35% of hallucinations are invisible to standard uncertainty.** The "Confident Hallucination"
  quadrant (low entropy, low consistency) cannot be caught by entropy alone. Only representation
  geometry at L32 flags these cases. ABSTAIN is the only viable routing decision.

- **Data leakage inflated results by +0.19 AUROC.** The original Exp 23 AUROC (0.962) was
  an artifact of in-sample probe_conf computation. Correct number after OOF fix: **0.775**.
  This transparency is part of the contribution.

- **The governance policy is risk-averse, not omniscient.** At AUROC 0.78, the system routes
  90% of queries to REGENERATE and accepts only 6 test samples. Hallucination CI on accepted
  queries: [0.004, 0.641]. Useful as a conservative filter — not yet a confident gatekeeper.

- **The signal is scale-emergent.** GPT-2 Medium (345M) achieves best AUROC 0.579 at the
  depth-equivalent layer — weak but above chance. GPT-2 117M shows null signal (MECH-INT).
  Qwen 2.5 3B achieves 0.775. The geometry strengthens with model scale. (Exp 25)

- **The signal is generation-time specific, not semantic.** Zero-shot transfer of the HaRP
  probe to HaluEval (pre-written QA pairs) yields AUROC 0.094 — anti-correlated. The probe
  captures internal uncertainty during generation, not textual semantic correctness. (Exp 26)

- **Routing overhead is 23.9ms.** Stage-by-stage profiling confirms HaRP adds 23.9ms ± 30.2ms
  overhead per inference (p95: 66.6ms). The forward pass is paid regardless for generation;
  the routing cost is negligible relative to total inference time. (Exp 27)

- **The hallucination geometry is specific.** A sycophancy probe at L32 (500 TruthfulQA
  samples, three behavioral classes) achieves AUROC 0.404 — below chance. The L32 signal
  detects hallucination-specific uncertainty geometry, not social compliance or general
  factual incorrectness. (Exp 28)

---

## 24-Experiment Index

| Exp | Question | Method | Key Result |
|-----|----------|--------|------------|
| 01 | What does Qwen 2.5 3B say on TruthfulQA? | Ollama local inference | 700 responses collected |
| 02 | How consistent are multi-sample outputs? | 5× generation + sentence-transformers cosine | consistency_score per sample |
| 03 | Which responses are hallucinated? | ROUGE-L ≥ 0.4 + semantic keyword overlap | hall_rate = **39.5%** |
| 04 | Can token signals alone detect failure? | Logistic regression on 6 entropy features | AUROC = **0.573** — near random |
| 05 | Where are the hidden states? | HuggingFace forward pass, all 37 layers | 700×37×2048 (94 MB) |
| 06 | Which layer encodes hallucination? | Linear probe at each layer, 5-fold CV | Best: **L32**, AUROC **0.758** |
| 07 | Is there a linear truth direction? | PCA on L32, PC1 vs label correlation | PC1 corr = **0.424**, sep = 0.684 |
| 08 | When does the model commit to wrong tokens? | Logit lens per-layer prediction tracking (GPU) | Hallucination commits at **L4.07** |
| 09 | [ARCHIVED] Can probe_conf be used in-sample? | Logistic regression with in-sample probe_conf | **Leakage detected** — AUROC 0.962 (false) |
| **09b** | **How do we fix probe_conf leakage?** | **OOF computation per 5-fold CV split** | **Honest AUROC = 0.757** |
| 10 | What is the steering direction at L32? | μ_correct − μ_hallucinated (mean-difference) | steering_vector.npy (norm = 2.68) |
| 11 | Does adding the steering vector reduce hallucinations? | h' = h + α·v at L32, α ∈ {0..2} (GPU) | **Null result** at all alpha |
| 12 | What thresholds minimise governance loss? | Grid search over α*, β* on validation set | α* = 0.15, β* = 0.90 |
| 13 | How does HaRP compare to baselines? | Raw LLM vs entropy-only vs full system | HaRP hall_rate: 0.167 vs 0.395 raw |
| 14 | Does sycophancy corrupt the signal? | Sycophancy prompt variants analysis | Failure mode identified, documented |
| 15 | Is the failure estimator well-calibrated? | ECE binned calibration | ECE = **0.039** (well-calibrated) |
| 16 | What is the intermediate ablation picture? | Feature group ablation table | Stepping stone to Exp 24 |
| 17 | Does NLI-based semantic entropy help? | Cluster entropy via NLI entailment | Group B2 — marginal contribution |
| 18 | Can we decompose uncertainty types? | Entropy × consistency 2×2 thresholding | 4-quadrant taxonomy defined |
| 19 | Does the signal generalise across models? | Signal extraction on alternate models | Generalization evidence |
| 20 | Do cross-layer dynamics add information? | ICR-inspired total variation and stabilization layer | Group D — +0.006 (noise) |
| 21 | Do geometric volume features help? | Granese simplex volume, centroid drift | Group E — −0.001 (null) |
| 22 | Does quadrant-adaptive steering work? | Per-quadrant steering vectors (GPU) | **Null result** — hall_rate = 0.68 |
| 22c | Does negative alpha fix confident hallucinations? | Per-quadrant α sweep incl. negative (GPU) | Delta = 0.007 — **generation noise** |
| **24** | **What are the honest final numbers?** | **5-fold CV, OOF features, n=700 HS samples** | **AUROC = 0.775, CI [0.706, 0.866]** |
| **25** | **Does the signal appear in smaller models?** | **OOF LR probe at depth-equivalent layers in GPT-2 Medium (345M)** | **Best AUROC = 0.579 — signal is scale-emergent** |
| **26** | **Does the HaRP probe transfer to an external benchmark?** | **Zero-shot transfer of HaRP probe to HaluEval (500 QA samples)** | **Zero-shot AUROC = 0.094 — signal is generation-time specific** |
| **27** | **What is the routing overhead of the HaRP system?** | **Stage-by-stage latency profiling on 50 samples (3 warmup)** | **23.9ms ± 30.2ms routing overhead (p95: 66.6ms)** |
| **28** | **Is the L32 signal specific to hallucination geometry?** | **Sycophancy probe: SYCOPHANTIC vs HONEST vs ALREADY_WRONG at L32** | **Probe AUROC = 0.404 — signal is hallucination-specific** |

---

## Signal Groups

| Group | Features | AUROC (5-fold CV, honest) | Interpretation |
|-------|----------|--------------------------|----------------|
| **A** | entropy, max_entropy, logit_var, conf_gap, top1_prob, first_entropy | 0.573 ± 0.056 | Surface signals — near random |
| **A+B** | + consistency_score, consistency_variance | 0.594 ± 0.073 | Multi-sample consistency adds 0.021 |
| **A+B+C** | + cos_correct, cos_hallucinated, h_norm, probe_conf (OOF) | **0.771 ± 0.047** | **Representation geometry — the signal** |
| A+B+C+C2 | + attn_entropy_mean, attn_entropy_std | 0.771 ± 0.047 | Attention entropy — no contribution |
| A+B+C+C2+D | + 5 trajectory signals | 0.777 ± 0.042 | Cross-layer dynamics — within noise |
| **SOTA (all)** | + simplex_volume, embedding_spread, centroid_drift | **0.775 ± 0.042** | No improvement over A+B+C |
| MLP on all | Full 22-feature set, multi-layer perceptron | 0.733 ± 0.031 | Linear model wins at N=700 |

**Group C (OOF) carries ~99% of the discriminative power. Everything else is noise.**

---

## The Data Leakage Bug — What Happened and What We Fixed

This is the most important methodological note in the project.

**The bug (Exp 09):** `probe_conf` was computed by fitting a probe on all 700 samples
and then evaluating it on those same 700 samples. In-sample evaluation of a probe that
was trained on the full dataset yields AUROC = 1.000 — perfect label memorisation. When
this feature entered the failure estimator's 5-fold cross-validation, the outer CV did
not retroactively fix the inner leakage. This is classical nested model leakage.

**Measured inflation:**
- probe_conf in-sample AUROC: 1.000 → OOF AUROC: 0.757
- Full system: Exp 23 (leaked): AUROC **0.962** → Exp 24 (honest): AUROC **0.775**
- Inflation: **+0.187**

**The fix (Exp 09b):** Out-of-fold (OOF) computation. For each of 5 CV folds, the probe
and class centroids are fitted on the training fold only, then applied to the held-out
test fold. The resulting `features_combined_oof.csv` has probe_conf AUROC = 0.757 —
consistent with Exp 06's cross-validated estimate (0.758).

```python
# WRONG (Exp 09) — in-sample leakage
probe.fit(X_all, y_all)
probe_conf = probe.predict_proba(X_all)[:, 1]  # ← evaluates on training data!

# CORRECT (Exp 09b) — out-of-fold
for train_idx, test_idx in StratifiedKFold(5).split(X, y):
    probe.fit(X[train_idx], y[train_idx])
    probe_conf_oof[test_idx] = probe.predict_proba(X[test_idx])[:, 1]
```

**Always use `features_combined_oof.csv` and Exp 24 results. Do not cite Exp 23.**

---

## Methodology in Plain English

**AUROC** — Area Under the ROC Curve. Probability that the failure estimator ranks a
random hallucinated sample higher than a random correct sample. 0.5 = random, 1.0 = perfect.

**AUARC** — Area Under the Accuracy-Rejection Curve. Sort samples by estimator confidence
and measure accuracy at each retention level. 1.0 = perfect identification of correct answers.

**OOF (Out-of-Fold)** — A technique to get honest out-of-sample predictions from models
that are themselves features. For each CV fold, the sub-model (probe, centroid) is fitted
only on training data and applied only to test data.

**Probe** — A simple logistic regression trained on the last-token hidden state at a specific
layer to predict correct vs hallucinated. Interpretable (coefficients reveal which dimensions
carry the signal) and appropriate at N=700.

**Activation Steering** — Modifying the hidden state at layer L at inference time by adding
a "truth direction" vector: h'[L] = h[L] + α·v. Tested to see if it pushes hallucinated
generations toward correct outputs.

**ECE** — Expected Calibration Error. Measures whether the estimator's probability scores
are accurate — not just well-ranked. ECE = 0.039 means the HaRP estimator is well-calibrated.

**Clopper-Pearson CI** — Exact binomial confidence interval for small-n rate estimation.
Used for hallucination rate among accepted queries (n=6). Normal approximation is invalid
when n < 30.

**4-Quadrant Taxonomy** — A decomposition of uncertainty type using entropy (token-level
uncertainty) and consistency (multi-sample agreement):
- High entropy + Low consistency → Epistemic (knowledge gap)
- High entropy + High consistency → Aleatoric (inherent ambiguity)
- Low entropy + Low consistency → Confident Hallucination (confidently wrong)
- Low entropy + High consistency → Confident Correct

---

## Architecture

```
src/
├── analysis/
│   ├── probing.py                   Layer-wise logistic regression probes (Exp 06)
│   ├── geometry_of_truth.py         PCA-based truth direction analysis (Exp 07)
│   ├── logit_lens.py                Token commitment layer analysis (Exp 08)
│   └── uncertainty_decomposition.py 4-quadrant taxonomy (Exp 18, 22)
│
├── signals/
│   ├── token_signals.py             Group A: 6 entropy/logit features
│   ├── consistency_signals.py       Group B: multi-sample cosine consistency
│   ├── semantic_entropy.py          Group B2: NLI-based semantic entropy
│   ├── representation_signals.py    Group C: hidden-state geometry (OOF critical)
│   ├── attention_signals.py         Group C2: attention entropy
│   ├── trajectory_signals.py        Group D: cross-layer dynamics (ICR)
│   └── geometric_signals.py         Group E: geometric volume (Granese)
│
├── authorization/
│   ├── failure_estimator.py         LR + imputer + scaler pipeline
│   └── policy.py                    3-action routing (ACCEPT/REGENERATE/ABSTAIN)
│
├── calibration/
│   └── calibrate.py                 ECE computation + temperature scaling
│
├── evaluation/
│   └── metrics.py                   AUROC, AUARC, Clopper-Pearson CI, bootstrap
│
├── steering/
│   └── activation_steering.py       Mean-difference vector intervention at L32
│
└── model/
    ├── query_ollama.py              Token signals (local, no hidden states needed)
    └── query_hf.py                  Hidden-state extraction (HuggingFace)
```

---

## Running the Pipeline

**Dependencies:**
```bash
pip install -r requirements.txt
ollama pull qwen2.5:3b   # for Exp 01–04, token signals
```

**Phase 1 — Data collection (CPU, Ollama):**
```bash
python experiments/01_collect_data.py
python experiments/02_consistency_check.py
python experiments/03_label_responses.py
python experiments/04_train_estimator_v1.py
```

**Phase 2 — Hidden states + analysis (HuggingFace, ~8 GB RAM):**
```bash
python experiments/05_extract_hidden_states.py
python experiments/06_probe_layers.py
python experiments/07_geometry_of_truth.py
python experiments/09b_oof_group_c.py        # MUST run before Exp 24
```

**Phase 3 — Feature engineering + policy (CPU):**
```bash
python experiments/10_compute_steering_vector.py
python experiments/12_run_policy.py
python experiments/13_run_baselines.py
python experiments/15_calibration.py
python experiments/20_cross_layer_dynamics.py
python experiments/21_geometric_volume_signals.py
```

**Phase 4 — Definitive evaluation:**
```bash
python experiments/24_honest_evaluation.py
```

**GPU experiments (Google Colab T4):** See `colab/colab_gpu_run2.ipynb`
Runs Exp 08 (logit lens), Exp 11 (universal steering), Exp 22, 22c (quadrant steering).

**Dashboard:**
```bash
streamlit run app.py
```

---

## Limitations

1. **Single in-domain benchmark.** TruthfulQA has a specific misconception structure. Exp 26
   confirms the HaRP probe does not transfer zero-shot to HaluEval (pre-written QA pairs) —
   AUROC 0.094. This is an interpretable result (generation-time vs text-level signal), but
   direct deployment to other benchmarks requires domain-specific re-training.

2. **Small test set for policy evaluation.** Only 6 queries reach the ACCEPT bucket in the
   140-sample test set. The Clopper-Pearson CI [0.004, 0.641] is too wide for policy claims.
   5,000+ samples would be needed for meaningful acceptance rate statistics.

3. **Scale dependency confirmed.** Exp 25 shows GPT-2 Medium (345M) achieves AUROC 0.579 —
   the signal weakens substantially at smaller scales. The geometry signal at 89% depth is
   strongest at 3B+ model sizes.

4. **Activation steering null result is mechanistically unexplained.** Why does the mean-
   difference vector fail to shift generation? Is it the coarse population-level direction?
   Token-level vs position-level intervention? This remains an open question.

5. **AUROC 0.775 is insufficient for production governance.** The policy accepts 4.3% of
   queries at the cost of 90% regeneration overhead. A useful research result, not a
   deployable filter.

---

## What Holds Up vs What Needs Qualification

| Claim | Status | Evidence |
|-------|--------|---------|
| L32 hidden states encode hallucination signal | ✅ Solid | AUROC 0.758, consistent across 5 folds |
| Token entropy alone is insufficient | ✅ Solid | AUROC 0.573 — near random, reproducible |
| Group C (hidden-state geometry) adds +0.198 AUROC | ✅ Solid | OOF computation, matches Exp 06 |
| Signal is linear (LR > MLP) | ✅ Solid | LR 0.775 vs MLP 0.733 at N=700 |
| Universal activation steering fails | ✅ Solid | 5 alpha values, all flat or degraded |
| Negative alpha fails for confident hallucinations | ✅ Solid | delta 0.007 — within noise bounds |
| Groups D and E add nothing | ✅ Solid | +0.004 — within std 0.05 |
| Signal is scale-emergent | ✅ Solid | GPT-2 117M null → 345M 0.579 → 3B 0.775 (Exp 25) |
| Signal is generation-time specific | ✅ Solid | HaluEval zero-shot AUROC 0.094 — anti-correlated (Exp 26) |
| Routing overhead is negligible | ✅ Solid | 23.9ms p50, 66.6ms p95 (Exp 27) |
| Signal is hallucination-specific, not sycophancy | ✅ Solid | Sycophancy probe AUROC 0.404 (Exp 28) |
| "Zero hallucination rate" | ❌ Artifact | Leaked estimator + 0/59 CI upper = 6.1% |
| "AUROC 0.962" | ❌ Corrected | Honest OOF value is 0.775 |
| "Formal Bayesian decomposition" | ⚠️ Overstated | Entropy + consistency proxy, not formal Bayes |
| "SOTA system" | ⚠️ No comparison | No empirical comparison to prior work |

---

## Future Directions

| Direction | What It Would Require |
|-----------|----------------------|
| In-domain HaluEval probe | Train fresh probe on HaluEval hidden states (in-domain AUROC = 0.997 confirmed in Exp 26) |
| Direct comparison to ICR Probe (Zhang et al. 2025) | Implement ICR, same 700 samples |
| Temperature=0 steering experiment | Re-run Exp 22c with deterministic generation |
| Larger sample size for policy claims | 5,000+ samples, multiple benchmark splits |
| Per-sample steering (not mean-difference) | Sample-specific intervention direction |
| Validate depth-fraction hypothesis at 7B+ | Check whether 89% depth holds at Mistral 7B / Llama 3 8B |

---

## Companion Project — MECH-INT & The Two-Part Arc

HaRP builds on the mechanistic foundation established in **[MECH-INT](https://github.com/Lakshmi-Chakradhar-Vijayarao/mech-int)**, a prior study that probed GPT-2 (117M) using 12 independent interpretability methods.

| | MECH-INT | HaRP |
|---|---|---|
| **Model** | GPT-2 (117M) | Qwen 2.5 3B |
| **Parameters** | 117M | 3B (25× larger) |
| **Goal** | Mechanistic understanding | Detection + governance |
| **Best AUROC** | 0.604 (L8 last-token probe) | 0.775 (honest OOF, 5-fold CV) |
| **Geometry signal** | Absent (SVD ~0.50 — scale too small) | Present at L32 (+0.198 lift) |
| **Token entropy** | AUROC 0.576 — near random | AUROC 0.573 — near random |
| **Primary failure mode** | FFN over-retrieval, 8/12 layers | 35% confident hallucinations |
| **Output** | 12-step interpretability pipeline | 28-experiment governance system |

### The Scale-Emergent Signal (Exp 25 — Cross-Model Validation)

HaRP Exp 25 probed GPT-2 Medium (345M) at the depth-equivalent layer, establishing the full scale curve:

| Model | Parameters | Best-Layer AUROC | Signal |
|---|---|---|---|
| GPT-2 (117M) — MECH-INT | 117M | ~0.500 (null) | Absent |
| GPT-2 Medium (345M) — Exp 25 | 345M | **0.579** | Weak |
| Qwen 2.5 3B — HaRP | 3B | **0.775** | Strong |

The depth fraction at peak is consistent across all three (≈88–89% depth), and the AUROC increases monotonically with scale.

### The Two-Part Argument

MECH-INT answers: **WHERE** is the signal? **WHAT** is the mechanism? **HOW** does it propagate layer by layer to specific heads?

HaRP answers: **HOW DO WE USE** the signal? **HOW DO WE GOVERN** it? **HOW DO WE CALIBRATE** it at production scale?

**In one sentence:** Hallucination is a parametric retrieval failure that writes itself into the residual stream at ≈89% model depth — invisible to output probabilities, detectable through hidden-state geometry, and manageable through a calibrated routing policy once you reach sufficient model scale.

**MECH-INT live dashboard:** [https://lakshmi-chakradhar-vijayarao-mech-int-app-3enga8.streamlit.app/](https://lakshmi-chakradhar-vijayarao-mech-int-app-3enga8.streamlit.app/)

---

## License

MIT
