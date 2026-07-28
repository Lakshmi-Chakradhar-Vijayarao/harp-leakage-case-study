"""
attention_signals.py — Group C+: Attention-based uncertainty signals from HuggingFace.

Novel Group C extension: attention entropy as a hallucination signal.

Theoretical basis:
  Grounded generation → focused attention on relevant context tokens
  Hallucination       → diffuse attention across tokens (model is "grasping")

Empirically supported by:
  - Yuksekgonul et al. (2023): specific attention heads track factual accuracy
  - Li et al. (2023, ITI): attention heads have clear true/false distributions
  - Recent vision-language work (SAE, 2025): attention entropy correlates with hallucination

Our contribution:
  - First text-LLM study using attention entropy as a *feature* in a learned
    failure estimator (prior work uses it for detection directly, not in combination
    with token uncertainty + representation geometry)

Signals extracted:
  attn_entropy_mean    : mean entropy across all attention heads at the last token
  attn_entropy_max     : max entropy (most uncertain head)
  attn_entropy_std     : std across heads (head disagreement)
  attn_head_cv         : coefficient of variation of per-head entropies
  attn_top_k_mass      : fraction of attention mass in top-k positions (focus)

Requires HuggingFace mode (output_attentions=True).
"""

import numpy as np
from typing import List, Optional, Tuple


def compute_head_entropy(
    attn_weights: np.ndarray,
) -> float:
    """
    Shannon entropy of a single attention head's distribution.

    attn_weights: [seq_len] — attention distribution summing to 1
    Returns entropy in nats.
    """
    # Clip to avoid log(0)
    p = np.clip(attn_weights, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def compute_layer_attention_entropy(
    attention: np.ndarray,
    token_position: int = -1,
) -> np.ndarray:
    """
    Compute per-head entropy for one layer at the specified token position.

    Parameters
    ----------
    attention     : [num_heads, seq_len, seq_len] — raw attention weights
    token_position: which token's outgoing attention to use (-1 = last token)

    Returns
    -------
    np.ndarray [num_heads] — entropy per head
    """
    n_heads = attention.shape[0]
    entropies = np.zeros(n_heads)
    for h in range(n_heads):
        head_attn = attention[h, token_position, :]
        entropies[h] = compute_head_entropy(head_attn)
    return entropies


def extract_attention_signals(
    attentions: List[np.ndarray],
    token_position: int = -1,
    top_k: int = 5,
    target_layers: Optional[List[int]] = None,
) -> dict:
    """
    Extract attention-based uncertainty signals from all layers.

    Parameters
    ----------
    attentions    : list of [num_heads, seq_len, seq_len] per layer
                    (from model output with output_attentions=True)
    token_position: position to query (default -1 = last token)
    top_k         : number of top-attended positions for focus metric
    target_layers : which layers to use (default: last 25% of layers)

    Returns
    -------
    dict with keys:
        attn_entropy_mean   : mean entropy across all heads in target layers
        attn_entropy_max    : max entropy (most diffuse head)
        attn_entropy_std    : std across heads (disagreement)
        attn_head_cv        : coefficient of variation
        attn_top_k_mass     : mean attention mass in top-k positions
    """
    n_layers = len(attentions)

    # Default: use last 25% of layers (where factual decisions are made)
    if target_layers is None:
        start = max(0, int(0.75 * n_layers))
        target_layers = list(range(start, n_layers))

    all_entropies = []
    all_top_k_mass = []

    for layer_idx in target_layers:
        if layer_idx >= n_layers:
            continue
        layer_attn = attentions[layer_idx]

        # Handle batch dim if present: [1, n_heads, seq, seq] → [n_heads, seq, seq]
        if layer_attn.ndim == 4:
            layer_attn = layer_attn[0]

        head_entropies = compute_layer_attention_entropy(layer_attn, token_position)
        all_entropies.append(head_entropies)

        # Top-k attention mass per head
        for h in range(layer_attn.shape[0]):
            head_row = layer_attn[h, token_position, :]
            sorted_attn = np.sort(head_row)[::-1]
            k = min(top_k, len(sorted_attn))
            all_top_k_mass.append(float(sorted_attn[:k].sum()))

    if not all_entropies:
        return {
            "attn_entropy_mean": float("nan"),
            "attn_entropy_max":  float("nan"),
            "attn_entropy_std":  float("nan"),
            "attn_head_cv":      float("nan"),
            "attn_top_k_mass":   float("nan"),
        }

    all_ent_flat = np.concatenate(all_entropies)
    mean_ent = float(all_ent_flat.mean())
    std_ent  = float(all_ent_flat.std())

    return {
        "attn_entropy_mean": mean_ent,
        "attn_entropy_max":  float(all_ent_flat.max()),
        "attn_entropy_std":  std_ent,
        "attn_head_cv":      float(std_ent / (mean_ent + 1e-12)),
        "attn_top_k_mass":   float(np.mean(all_top_k_mass)),
    }


def extract_attention_from_hf_output(
    model_output,
    token_position: int = -1,
    top_k: int = 5,
    target_layers: Optional[List[int]] = None,
) -> dict:
    """
    Convenience wrapper: extract attention signals from a HuggingFace model output.

    Handles the conversion from torch tensors to numpy.

    Usage:
        outputs = model(input_ids, output_attentions=True)
        signals = extract_attention_from_hf_output(outputs)
    """
    if not hasattr(model_output, "attentions") or model_output.attentions is None:
        return {
            "attn_entropy_mean": float("nan"),
            "attn_entropy_max":  float("nan"),
            "attn_entropy_std":  float("nan"),
            "attn_head_cv":      float("nan"),
            "attn_top_k_mass":   float("nan"),
        }

    attentions_np = []
    for layer_attn in model_output.attentions:
        # Convert torch tensor to numpy (CPU, float32)
        attn_np = layer_attn.detach().float().cpu().numpy()
        attentions_np.append(attn_np)

    return extract_attention_signals(
        attentions_np,
        token_position=token_position,
        top_k=top_k,
        target_layers=target_layers,
    )


def find_high_entropy_heads(
    attentions: List[np.ndarray],
    token_position: int = -1,
    top_n: int = 10,
) -> List[Tuple[int, int, float]]:
    """
    Find the attention heads with highest entropy.

    Returns list of (layer_idx, head_idx, entropy) tuples, sorted descending.
    Useful for mechanistic analysis: which heads are most uncertain?
    """
    results = []
    for layer_idx, layer_attn in enumerate(attentions):
        if layer_attn.ndim == 4:
            layer_attn = layer_attn[0]
        for h in range(layer_attn.shape[0]):
            head_row = layer_attn[h, token_position, :]
            ent = compute_head_entropy(head_row)
            results.append((layer_idx, h, ent))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]
