"""
query_hf.py — Load Qwen 2.5 3B via HuggingFace and extract hidden states.

IMPORTANT — Memory Rules:
  1. NEVER import this module in the same Python process as query_ollama.py.
     The HF model takes ~3.2 GB. Ollama is another ~2.5 GB. Together = OOM.
  2. Always call cleanup() after you're done with the model.
  3. Run experiments that use this file in isolation:
       python experiments/05_extract_hidden_states.py
     Close Ollama (Ctrl+C in its terminal) before running.

Device selection:
  - Apple Silicon (M1/M2/M3): uses MPS backend automatically
  - CPU fallback if MPS unavailable
"""

import gc
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

_model     = None
_tokenizer = None


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str = MODEL_ID) -> tuple:
    """
    Load the model and tokenizer into module-level globals.
    Safe to call multiple times — returns cached instance if already loaded.

    Returns (model, tokenizer).
    """
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    device = _get_device()
    print(f"Loading {model_id} on {device} ...")

    _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        output_hidden_states=True,
        device_map="auto",        # respects MPS on Apple Silicon
        trust_remote_code=True,
    )
    _model.eval()
    print(f"Model loaded — layers: {_model.config.num_hidden_layers}, "
          f"hidden_dim: {_model.config.hidden_size}")
    return _model, _tokenizer


def cleanup():
    """
    Delete model from memory and empty MPS/CUDA cache.
    Call this after all HF experiments are done.
    """
    global _model, _tokenizer
    if _model is not None:
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("Model unloaded and cache cleared.")


def get_hidden_states(
    prompt: str,
    layer_indices: Optional[list] = None,
    return_attentions: bool = False,
) -> dict:
    """
    Run a forward pass and return hidden states at requested layers.

    Parameters
    ----------
    prompt            : str — the input prompt
    layer_indices     : list[int] | None — layers to return (None = all layers)
    return_attentions : bool — if True, also return attention weights for Group C2

    Returns
    -------
    dict with keys:
        hidden_states    : np.ndarray [num_layers, seq_len, hidden_dim]
        last_token_hs    : np.ndarray [num_layers, hidden_dim]
        num_tokens       : int
        layer_indices    : list[int]
        attentions       : list[np.ndarray] | None  — per-layer [n_heads, seq, seq]
    """
    model, tokenizer = load_model()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    num_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=return_attentions,
        )

    all_hs = outputs.hidden_states  # tuple: embed + N transformer layers

    if layer_indices is None:
        layer_indices = list(range(len(all_hs)))

    hs_list = []
    for idx in layer_indices:
        h = all_hs[idx][0].cpu().float().numpy()
        hs_list.append(h)

    hidden_states = np.stack(hs_list)
    last_token_hs = hidden_states[:, -1, :]

    attentions = None
    if return_attentions and outputs.attentions is not None:
        attentions = [
            a[0].cpu().float().numpy()   # [n_heads, seq_len, seq_len]
            for a in outputs.attentions
        ]

    return {
        "hidden_states": hidden_states,
        "last_token_hs": last_token_hs,
        "num_tokens":    num_tokens,
        "layer_indices": layer_indices,
        "attentions":    attentions,
    }


def get_logit_lens_trace(
    prompt: str,
    correct_token: str,
    wrong_token: str,
) -> list[dict]:
    """
    Track P(correct_token) and P(wrong_token) at every layer via logit lens.

    Applies the model's LM head (unembedding matrix) to each layer's
    last-token hidden state to get a vocabulary distribution per layer.

    Parameters
    ----------
    correct_token : str — the correct answer token (e.g. "Paris")
    wrong_token   : str — the hallucinated token   (e.g. "London")

    Returns
    -------
    list of dicts: [{layer, p_correct, p_wrong}, ...]
    """
    model, tokenizer = load_model()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    all_hs = outputs.hidden_states  # tuple: (embed + N layers)

    # Tokenise the answer tokens (add leading space as models use subword BPE)
    def get_token_id(text: str) -> int:
        ids = tokenizer.encode(" " + text.strip(), add_special_tokens=False)
        return ids[0] if ids else tokenizer.unk_token_id

    c_id = get_token_id(correct_token)
    w_id = get_token_id(wrong_token)

    trace = []
    for layer_idx, h in enumerate(all_hs):
        # Project hidden state through the LM head
        last_h = h[:, -1, :].to(model.lm_head.weight.dtype)
        logits = model.lm_head(last_h)                    # [1, vocab_size]
        probs  = torch.softmax(logits.float(), dim=-1)[0] # [vocab_size]

        trace.append({
            "layer":     layer_idx,
            "p_correct": float(probs[c_id].item()),
            "p_wrong":   float(probs[w_id].item()),
        })

    return trace
