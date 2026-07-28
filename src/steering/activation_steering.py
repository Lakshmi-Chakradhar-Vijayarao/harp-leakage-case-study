"""
activation_steering.py — Activation steering via residual stream intervention.

Two steering modes:

1. UNIVERSAL STEERING (original)
   Computes a single steering vector = μ_correct - μ_hallucinated at best_layer.
   Exp 11 showed this yields α=0 as optimal — the single vector averages across
   fundamentally different failure modes and cancels itself out.

2. QUADRANT-ADAPTIVE STEERING (new — HaRP SOTA)
   Computes 4 steering vectors, one per uncertainty quadrant:
     epistemic              → μ_correct_epistemic - μ_hall_epistemic
     aleatoric              → μ_correct_aleatoric - μ_hall_aleatoric
     confident_hallucination→ μ_correct_confhall  - μ_hall_confhall
     confident_correct      → identity (should not need steering)

   During REGENERATE:
     1. Classify the sample into a quadrant (from its signal features)
     2. Apply the quadrant-specific steering vector

   Theoretical motivation:
     Adaptive Activation Steering (ACT, WWW 2025) shows category-specific
     vectors dramatically outperform universal vectors (↑142% on LLaMA).
     HaRP's 4-quadrant taxonomy provides a mechanistically-motivated
     categorisation superior to ACT's unsupervised clustering.

   Why α=0 was optimal in Exp 11:
     The universal vector averages across all four failure modes. Epistemic
     failures (low entropy, low consistency) live in a different subspace than
     confident hallucinations (low entropy, low consistency + geometry mismatch).
     Mixing them produces a near-zero net correction direction (cosine_sim=0.989
     between class means in the aggregated space).

   Quadrant-adaptive vectors decompose this into separable correction problems.
"""

import numpy as np
import torch
from typing import Optional


def compute_steering_vector(
    model,
    tokenizer,
    correct_prompts: list[str],
    hallucinated_prompts: list[str],
    target_layer: int,
) -> np.ndarray:
    """
    Compute the mean-difference steering vector at target_layer.

    Steering vector = μ_correct - μ_hallucinated at the last token position.

    Parameters
    ----------
    correct_prompts      : list of prompts where model answered correctly
    hallucinated_prompts : list of prompts where model hallucinated
    target_layer         : transformer layer index (0-indexed)

    Returns np.ndarray of shape [hidden_dim] (float32)
    """
    def get_mean_hidden(prompts: list[str]) -> np.ndarray:
        vecs = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[target_layer]  # npz index k = output of transformer layer k-1
            h_last = h[:, -1, :].squeeze().cpu().float().numpy()
            vecs.append(h_last)
        return np.mean(np.stack(vecs), axis=0)

    print(f"Computing steering vector at layer {target_layer}...")
    mu_correct      = get_mean_hidden(correct_prompts)
    mu_hallucinated = get_mean_hidden(hallucinated_prompts)

    steering_vec = (mu_correct - mu_hallucinated).astype(np.float32)
    norm = np.linalg.norm(steering_vec)
    print(f"  Steering vector norm: {norm:.4f}")
    print(f"  Shape: {steering_vec.shape}")

    return steering_vec


def apply_steering_hook(
    model,
    target_layer: int,
    steering_vec: np.ndarray,
    alpha: float,
):
    """
    Register a forward hook that adds alpha * steering_vec to the residual stream.

    The hook intercepts the output of model.model.layers[target_layer - 1],
    which corresponds to outputs.hidden_states[target_layer] in the npz indexing
    convention (index 0 = embedding layer, index k = output of transformer layer k-1).

    Parameters
    ----------
    target_layer : int — npz layer index (matches probe training and steering vector)
    steering_vec : np.ndarray [hidden_dim] — the steering direction
    alpha        : float — intervention strength (0.0 = no intervention)

    Returns
    -------
    hook_handle — call handle.remove() to detach the hook after generation.
    """
    sv_tensor = torch.tensor(steering_vec, dtype=torch.float16)

    def hook_fn(module, input, output):
        if alpha == 0.0:
            return output
        # Transformers 4.4x+: Qwen2 decoder layer returns hidden states
        # directly as a Tensor (DynamicCache updated in-place, not returned).
        # Older transformers return a tuple (hidden_states, past_kv, ...).
        if isinstance(output, torch.Tensor):
            intervention = alpha * sv_tensor.to(output.device, output.dtype)
            return output + intervention
        hidden = output[0]
        intervention = alpha * sv_tensor.to(hidden.device, hidden.dtype)
        new_hidden = hidden + intervention
        rest = output[1:]
        return (new_hidden, rest) if isinstance(rest, torch.Tensor) else (new_hidden,) + rest

    handle = model.model.layers[target_layer - 1].register_forward_hook(hook_fn)
    return handle


# ── Quadrant-adaptive steering ────────────────────────────────────────────────

QUADRANT_NAMES = [
    "epistemic",
    "aleatoric",
    "confident_hallucination",
    "confident_correct",
]


def compute_quadrant_steering_vectors(
    model,
    tokenizer,
    prompts: list,
    labels,
    quadrants: list,
    target_layer: int,
    min_samples: int = 5,
) -> dict:
    """
    Compute one steering vector per uncertainty quadrant.

    For each quadrant q:
      steering_vec_q = μ_correct_q - μ_hallucinated_q at target_layer

    Only computed for quadrants with at least min_samples examples in
    both the correct and hallucinated classes.

    Parameters
    ----------
    prompts      : list[str] — input prompts
    labels       : array-like [N] — 1=correct, 0=hallucinated
    quadrants    : list[str] [N] — quadrant assignment per sample
    target_layer : int — layer index for hidden state extraction
    min_samples  : int — minimum samples per class per quadrant to compute a vector

    Returns
    -------
    dict mapping quadrant_name → np.ndarray [hidden_dim] | None
    None means insufficient data for that quadrant.
    """
    import numpy as np

    labels_arr = np.array(labels)
    quadrants_arr = np.array(quadrants)

    vectors = {}
    print(f"\nComputing quadrant-adaptive steering vectors at layer {target_layer}...")

    for q_name in QUADRANT_NAMES:
        q_mask    = quadrants_arr == q_name
        q_correct = q_mask & (labels_arr == 1)
        q_hall    = q_mask & (labels_arr == 0)

        n_c = int(q_correct.sum())
        n_h = int(q_hall.sum())
        print(f"  {q_name:<28} correct={n_c:3d}  hallucinated={n_h:3d}", end="")

        if n_c < min_samples or n_h < min_samples:
            print("  → SKIPPED (insufficient data)")
            vectors[q_name] = None
            continue

        # Compute class means at target_layer
        correct_prompts = [prompts[i] for i in range(len(prompts)) if q_correct[i]]
        hall_prompts    = [prompts[i] for i in range(len(prompts)) if q_hall[i]]

        sv = compute_steering_vector(model, tokenizer,
                                     correct_prompts, hall_prompts,
                                     target_layer)
        vectors[q_name] = sv
        norm = float(np.linalg.norm(sv))
        print(f"  → sv norm={norm:.4f}")

    n_computed = sum(1 for v in vectors.values() if v is not None)
    print(f"  Computed {n_computed}/{len(QUADRANT_NAMES)} quadrant vectors.")
    return vectors


def generate_steered_adaptive(
    model,
    tokenizer,
    prompt: str,
    quadrant: str,
    steering_vectors: dict,
    target_layer: int,
    alpha: float,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    fallback_to_universal: bool = True,
    universal_sv: Optional[np.ndarray] = None,
) -> str:
    """
    Generate a response using the quadrant-specific steering vector.

    If no vector is available for this quadrant (insufficient training data),
    falls back to the universal steering vector (if provided) or plain
    regeneration at temperature=0.9.

    Parameters
    ----------
    quadrant          : str — uncertainty quadrant of this sample
    steering_vectors  : dict — output of compute_quadrant_steering_vectors()
    fallback_to_universal: bool — use universal_sv if quadrant vector unavailable
    universal_sv      : np.ndarray | None — universal fallback vector

    Returns
    -------
    Generated text string.
    """
    sv = steering_vectors.get(quadrant)

    if sv is None:
        if fallback_to_universal and universal_sv is not None:
            sv = universal_sv
        else:
            # No steering — plain regeneration
            import torch
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(new_ids, skip_special_tokens=True)

    return generate_steered(
        model, tokenizer, prompt, sv, target_layer, alpha,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )


def generate_steered(
    model,
    tokenizer,
    prompt: str,
    steering_vec: np.ndarray,
    target_layer: int,
    alpha: float,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
) -> str:
    """
    Generate a response with the steering hook active.

    Used during REGENERATE action: same prompt, higher temperature,
    steering vector nudging hidden states toward correct representation.

    Returns the generated text string.
    """
    handle = apply_steering_hook(model, target_layer, steering_vec, alpha)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the new tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True)
    finally:
        handle.remove()  # always remove hook, even on exception
