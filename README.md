# HaRP — Leakage Case Study 1 (companion material for "Not All Leaks Are Equal")

This repository ships the specific code and result logs needed to independently
verify Case Study 1 (full-dataset fit-then-score feature leakage) from
"Not All Leaks Are Equal: A Taxonomy and Audit of Label-Information Leakage in
Hidden-State Hallucination Detection." HaRP itself is a separate,
Qwen2.5-3B-based hallucination-detection project; this repo contains the
subset of it relevant to that one leakage finding, not the full project
(see `ORIGINAL_PROJECT_README.md` for HaRP's own full scope).

## What to look at to verify the claim

The paper's claim: a probe feature (`probe_conf`) was fit on all 700 samples,
then that same probe scored those same 700 samples inside a later 5-fold CV
loop — nested-CV leakage. Fixing it (refit per fold, score only on held-out
indices) drops AUROC from 0.9620 (leaked) to 0.7714 (fixed), a +0.1906
inflation; `probe_conf` alone drops from 1.0000 (in-sample) to 0.7573 (proper
out-of-fold).

- `experiments/09_train_estimator_v2.py` and related `experiments/09b_*.py`
  scripts — the training scripts implementing both the leaked and fixed
  protocols.
- `src/authorization/failure_estimator.py` — the estimator whose CV loop
  the leak occurs in.
- `results/logs/09_estimator_v2.json`, `results/logs/09b_oof_group_c.json` —
  the result logs containing the exact AUROC numbers quoted above.
- `RESULTS.md` — the original project's own full results writeup, including
  an independent cross-check of the fixed AUROC (0.7583 ± 0.0096) against a
  separately-obtained cross-validated estimate, agreeing to within 0.001.

## What's excluded, and why

Raw hidden-state caches (`results/hidden_states/*.npz`, tens of MB) are
excluded — these are intermediate GPU-extraction outputs, not needed to
verify the leakage claim, which is entirely about the training/evaluation
protocol around already-cached features. Unrelated parts of the full HaRP
project (its own demo app, experiments on sycophancy/calibration/steering
that are outside the leakage-audit paper's scope) are also not included —
only the files relevant to Case Study 1.
