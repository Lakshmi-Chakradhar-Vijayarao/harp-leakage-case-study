"""
label_responses.py — Auto-label model responses as correct (1) or hallucinated (0).

Strategy (two-tier):
  1. Strict  — ground truth string appears inside response (exact substring)
  2. Lenient — keyword overlap between ground truth and response >= 0.5

A response is labeled 1 (correct) if EITHER condition holds.

Limitations:
  - TruthfulQA adversarial questions: correct answers sometimes share
    vocabulary with wrong answers. Spot-check 50 samples after labeling.
  - NaturalQuestions long-form: overlap threshold may need tuning.
  - Always verify class balance after labeling: target >= 30% hallucinated.

For the adversarial dataset (130 prompts), labels are provided manually
in data/raw/adversarial_prompts.json as ground_truth strings.
"""

import re
import json
import pandas as pd
from pathlib import Path


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def label_response(response: str, ground_truth: str) -> int:
    """
    Label a single response given its ground truth.

    Returns 1 (correct) or 0 (hallucinated).
    """
    r_norm = _normalise(response)
    g_norm = _normalise(ground_truth)

    # Tier 1: strict substring match
    if g_norm in r_norm:
        return 1

    # Tier 2: keyword overlap
    gt_words   = set(g_norm.split())
    resp_words = set(r_norm.split())
    if len(gt_words) == 0:
        return 0
    overlap = len(gt_words & resp_words) / len(gt_words)
    return 1 if overlap >= 0.5 else 0


def label_dataframe(df: pd.DataFrame,
                    response_col: str = "response",
                    ground_truth_col: str = "ground_truth") -> pd.DataFrame:
    """
    Apply label_response to every row of a DataFrame.

    Adds a 'label' column (1=correct, 0=hallucinated).
    Returns the DataFrame with the new column.
    """
    df = df.copy()
    df["label"] = df.apply(
        lambda row: label_response(
            str(row[response_col]),
            str(row[ground_truth_col])
        ),
        axis=1,
    )
    return df


def check_class_balance(labels: pd.Series) -> dict:
    """
    Report class distribution and warn if heavily imbalanced.

    Target: >= 30% hallucinated (label=0). If below, use class_weight='balanced'
    in all classifiers.
    """
    total       = len(labels)
    n_correct   = int((labels == 1).sum())
    n_hall      = int((labels == 0).sum())
    pct_hall    = round(100 * n_hall / total, 1) if total > 0 else 0

    status = "OK" if pct_hall >= 30 else "IMBALANCED — use class_weight='balanced'"

    print(f"Class balance: {n_correct} correct / {n_hall} hallucinated "
          f"({pct_hall}% hallucinated) — {status}")

    return {
        "total":       total,
        "n_correct":   n_correct,
        "n_hallucinated": n_hall,
        "pct_hallucinated": pct_hall,
        "balanced": pct_hall >= 30,
    }


def spot_check(df: pd.DataFrame,
               response_col: str = "response",
               ground_truth_col: str = "ground_truth",
               label_col: str = "label",
               n: int = 50,
               seed: int = 42) -> pd.DataFrame:
    """
    Return a random sample of n rows formatted for manual review.

    Save the output to a CSV and open it in Excel/Numbers to verify
    that the auto-labels are correct. Adjust the overlap threshold if
    you see systematic errors.
    """
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    review = sample[[response_col, ground_truth_col, label_col]].copy()
    review["manual_label"] = ""  # fill this column manually
    review["notes"]        = ""
    return review.reset_index(drop=True)
