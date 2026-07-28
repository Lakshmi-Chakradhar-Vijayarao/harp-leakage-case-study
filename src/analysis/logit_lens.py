"""
logit_lens.py — Track P(correct_token) and P(wrong_token) across all layers.

The logit lens (Nostalgebraist 2020) projects each layer's hidden state
through the model's unembedding matrix (LM head) to read off a vocabulary
distribution — as if that layer were the final layer.

This reveals WHEN the model "commits" to the wrong answer:
  Commitment layer = first layer where P(wrong) > P(correct) for >= 3 layers

Connects directly to MECH-INT: MECH-INT found layers 8-9 as the mechanistic
core. HaRP's logit lens should confirm whether commitment happens at the
same layers or downstream.

Runs on 100-sample subset only (memory constraint on 8GB Air).
"""

import numpy as np
from typing import List, Dict, Optional


def find_commitment_layer(
    trace: List[Dict],
    min_consecutive: int = 3,
) -> Optional[int]:
    """
    Find the first layer where P(wrong) > P(correct) for at least
    min_consecutive subsequent layers.

    Returns the layer index, or None if no commitment is found.
    """
    n = len(trace)
    for i in range(n - min_consecutive + 1):
        window = trace[i: i + min_consecutive]
        if all(t["p_wrong"] > t["p_correct"] for t in window):
            return trace[i]["layer"]
    return None


def aggregate_traces(
    traces: List[List[Dict]],
) -> List[Dict]:
    """
    Aggregate logit lens traces across multiple examples.

    Parameters
    ----------
    traces : list of traces, each trace = list of {layer, p_correct, p_wrong}

    Returns list of {layer, mean_p_correct, mean_p_wrong, std_p_correct, std_p_wrong}
    """
    if not traces:
        return []

    num_layers = len(traces[0])
    aggregated = []

    for layer_idx in range(num_layers):
        p_corrects = [t[layer_idx]["p_correct"] for t in traces]
        p_wrongs   = [t[layer_idx]["p_wrong"]   for t in traces]

        aggregated.append({
            "layer":           layer_idx,
            "mean_p_correct":  float(np.mean(p_corrects)),
            "mean_p_wrong":    float(np.mean(p_wrongs)),
            "std_p_correct":   float(np.std(p_corrects)),
            "std_p_wrong":     float(np.std(p_wrongs)),
        })

    return aggregated


def summarise_commitment_layers(
    traces: List[List[Dict]],
    labels: List[int],
    min_consecutive: int = 3,
) -> Dict:
    """
    Find commitment layers for all examples and report distribution.

    Parameters
    ----------
    traces : list of logit lens traces (one per example)
    labels : list of int — 1=correct, 0=hallucinated

    Returns dict:
        commitment_layers_hall  : list of layers for hallucinated examples
        commitment_layers_corr  : list of layers for correct examples
        mean_hall               : mean commitment layer (hallucinated)
        mean_corr               : mean commitment layer (correct)
        no_commitment_pct       : % of examples with no commitment found
    """
    hall_layers = []
    corr_layers = []
    no_commit   = 0

    for trace, label in zip(traces, labels):
        cl = find_commitment_layer(trace, min_consecutive)
        if cl is None:
            no_commit += 1
        elif label == 0:  # hallucinated
            hall_layers.append(cl)
        else:
            corr_layers.append(cl)

    total = len(traces)
    result = {
        "commitment_layers_hall": hall_layers,
        "commitment_layers_corr": corr_layers,
        "mean_hall":              float(np.mean(hall_layers)) if hall_layers else None,
        "mean_corr":              float(np.mean(corr_layers)) if corr_layers else None,
        "no_commitment_pct":      round(100 * no_commit / total, 1) if total else 0,
    }

    print(f"Logit Lens — Commitment Layer Analysis:")
    print(f"  Hallucinated: mean commit layer = {result['mean_hall']}")
    print(f"  Correct:      mean commit layer = {result['mean_corr']}")
    print(f"  No commit found: {result['no_commitment_pct']}% of examples")

    return result
