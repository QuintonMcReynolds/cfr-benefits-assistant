"""LLM backends behind one interface.

mlx     (default) local open-weights model on Apple silicon; no API keys, no data leaves the machine
openai  any OpenAI-compatible /v1/chat/completions server (vLLM, Ollama, a hosted API), chosen with
        CFR_LLM_BACKEND=openai, CFR_LLM_BASE_URL, CFR_LLM_API_KEY. This is what the Linux container uses.
"""
from __future__ import annotations

import time

_CACHE: dict = {}


def LLM(model_id: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"):  # noqa: N802 - factory keeps call sites unchanged
    import os

    if os.environ.get("CFR_LLM_BACKEND", "mlx") == "openai":
        return OpenAICompatible(os.environ.get("CFR_LLM_MODEL", model_id))
    return MLXLLM(model_id)


class OpenAICompatible:
    def __init__(self, model_id: str):
        import os

        self.model_id = model_id
        self.base = os.environ.get("CFR_LLM_BASE_URL", "http://localhost:8001/v1").rstrip("/")
        self.key = os.environ.get("CFR_LLM_API_KEY", "")

    def chat(self, messages: list[dict], max_tokens: int = 400, temperature: float = 0.0) -> dict:
        import httpx

        t0 = time.perf_counter()
        r = httpx.post(f"{self.base}/chat/completions", timeout=120,
                       headers={"Authorization": f"Bearer {self.key}"} if self.key else {},
                       json={"model": self.model_id, "messages": messages, "max_tokens": max_tokens, "temperature": temperature})
        r.raise_for_status()
        j = r.json()
        u = j.get("usage", {})
        return {"text": j["choices"][0]["message"]["content"].strip(), "prompt_tokens": u.get("prompt_tokens", 0),
                "completion_tokens": u.get("completion_tokens", 0), "latency_s": time.perf_counter() - t0}


class MLXLLM:
    def __init__(self, model_id: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"):
        from mlx_lm import load

        if model_id not in _CACHE:
            _CACHE[model_id] = load(model_id)
        self.model, self.tok = _CACHE[model_id]
        self.model_id = model_id

    def chat(self, messages: list[dict], max_tokens: int = 400, temperature: float = 0.0) -> dict:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        n_in = len(self.tok.encode(prompt))
        t0 = time.perf_counter()
        out = generate(self.model, self.tok, prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temperature),
                       verbose=False)
        return {"text": out.strip(), "prompt_tokens": n_in, "completion_tokens": len(self.tok.encode(out)),
                "latency_s": time.perf_counter() - t0}
