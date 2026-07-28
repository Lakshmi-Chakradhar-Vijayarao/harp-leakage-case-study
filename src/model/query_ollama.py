"""
query_ollama.py — Query Qwen 2.5 3B via Ollama local API.

Two public functions:
  query_with_logprobs(prompt, model)  → response text + per-token log-probs
  query_consistency(prompt, model, n) → n responses at temp=0.7 for Group B

Ollama endpoint: http://localhost:11434
Model name:      qwen2.5:3b

LOGPROB FALLBACK:
  Ollama's logprob support is backend-dependent. If logprobs are absent
  from the response, query_with_logprobs returns token_logprobs=None and
  top_logprobs=None. The calling code MUST check for None and fall back
  to consistency-based entropy (Group B) when logprobs are unavailable.
"""

import requests
from typing import Optional

OLLAMA_URL  = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"


def query_with_logprobs(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> dict:
    """
    Send a prompt to Ollama and return the response with token log-probs.

    Returns
    -------
    dict with keys:
        response_text  : str   — the model's text output
        token_logprobs : list[float] | None — per-token log-prob (ln scale)
        top_logprobs   : list[dict]  | None — top-k token alternatives per step
        logprobs_available : bool   — False if Ollama didn't return logprobs

    Notes
    -----
    - temperature=0.0  for deterministic Group A feature extraction
    - Set num_predict=-1 to let the model finish naturally
    - stream=False so we get a single complete JSON response
    """
    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "options": {
            "temperature": temperature,
            "top_p":       1.0,
            "num_predict": 128,
        },
        "logprobs": True,  # request token-level logprobs
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    response_text = data.get("response", "")

    # Ollama may return logprobs under different keys depending on version.
    # Try both known locations.
    raw_logprobs = (
        data.get("logprobs")
        or data.get("token_logprobs")
        or None
    )

    token_logprobs: Optional[list] = None
    top_logprobs:   Optional[list] = None
    logprobs_available = False

    if raw_logprobs is not None:
        logprobs_available = True
        # Normalise to a flat list of floats (log-prob per token)
        if isinstance(raw_logprobs, list):
            if len(raw_logprobs) > 0 and isinstance(raw_logprobs[0], (int, float)):
                token_logprobs = [float(x) for x in raw_logprobs]
            elif len(raw_logprobs) > 0 and isinstance(raw_logprobs[0], dict):
                # Some Ollama versions return list of {token, logprob, bytes} dicts
                token_logprobs = [float(d['logprob']) if 'logprob' in d else list(d.values())[0]
                                  for d in raw_logprobs]
                top_logprobs   = raw_logprobs
        elif isinstance(raw_logprobs, dict):
            # Structured format: {"tokens": [...], "token_logprobs": [...]}
            token_logprobs = raw_logprobs.get("token_logprobs")
            top_logprobs   = raw_logprobs.get("top_logprobs")

    return {
        "response_text":      response_text,
        "token_logprobs":     token_logprobs,
        "top_logprobs":       top_logprobs,
        "logprobs_available": logprobs_available,
    }


def query_consistency(
    prompt: str,
    model: str = DEFAULT_MODEL,
    n: int = 3,
    temperature: float = 0.7,
) -> list[str]:
    """
    Query the model n times at temperature=0.7 and return a list of responses.

    Used for Group B (consistency signal). High temperature introduces
    variability — if all n responses agree, the model is epistemically
    confident. If they diverge, it is uncertain (or sycophantically unstable).

    Returns
    -------
    list[str] of length n — one response text per run
    """
    responses = []
    payload_base = {
        "model":   model,
        "stream":  False,
        "options": {
            "temperature": temperature,
            "top_p":       0.9,
            "num_predict": 128,
        },
    }
    for _ in range(n):
        payload = {**payload_base, "prompt": prompt}
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        responses.append(resp.json().get("response", ""))
    return responses
