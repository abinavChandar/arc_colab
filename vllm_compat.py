# vllm_compat.py
"""
Drop-in replacement for the OllamaClient used in your ARC scripts,
but backed by a vLLM OpenAI-compatible server.

Usage in your scripts:
    from vllm_compat import SyncClient as OllamaClient
    client = OllamaClient()
    text = client.generate(model="Qwen/Qwen2.5-7B-Instruct", prompt=..., temperature=0.2)

Environment:
    OPENAI_BASE_URL (e.g., http://127.0.0.1:8000/v1)
    OPENAI_API_KEY  (any string; vLLM ignores it, but header is required)
"""

from __future__ import annotations
import os, json, time
import httpx
from typing import Optional

BASE = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

# vLLM ignores the API key content, but the header must be present.
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

class SyncClient:
    """Synchronous client with the same .generate(...) shape your code expects."""
    def __init__(self, timeout: int = 180):
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        max_tokens: int = 1024,
    ) -> str:
        """
        match OllamaClient.generate(...) return behavior:
        returns raw text emitted by the model (not streamed).
        """
        payload = {
            "model": model,  # must match the model you launched vLLM with
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        # If you need deterministic-ish behavior, vLLM supports seed in some builds; if not, it's safely ignored.
        if seed is not None:
            payload["seed"] = int(seed)

        r = self._client.post(
            f"{BASE}/chat/completions",
            headers=HEADERS,
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass
