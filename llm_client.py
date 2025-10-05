import os
import asyncio
import aiohttp
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("MODEL", "qwen2.5-coder:7b"))
# Gate concurrent requests hitting the GPU. Tune to your A100 headroom.
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "6"))
# Optional on-disk cache to avoid duplicate requests during a run.
LLM_CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", "./.llm_cache"))
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# A single-process semaphore to limit concurrent LLM calls.
_sem = asyncio.Semaphore(LLM_CONCURRENCY)


def _cache_key(model: str, prompt: str, options: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update(prompt.encode())
    h.update(b"\x00")
    h.update(json.dumps(options, sort_keys=True).encode())
    return h.hexdigest()


def _cache_path(key: str) -> Path:
    return LLM_CACHE_DIR / f"{key}.json"


async def generate(prompt: str, *,
                   model: Optional[str] = None,
                   max_new_tokens: int = 512,
                   temperature: float = 0.2,
                   stop: Optional[list] = None,
                   stream: bool = False,
                   ) -> str:
    """Generate text from Ollama with concurrency gating and basic caching.

    Args:
        prompt: The input prompt.
        model: Override model name; defaults to LLM_MODEL.
        max_new_tokens: Limit tokens for speed and VRAM safety.
        temperature: Sampling temperature.
        stop: Optional stop tokens.
        stream: Always False here (batch-friendly, simpler).
    Returns:
        Response string from the model.
    """
    mdl = model or LLM_MODEL
    opts = {"num_predict": max_new_tokens, "temperature": temperature}
    if stop:
        opts["stop"] = stop

    key = _cache_key(mdl, prompt, opts)
    cpath = _cache_path(key)
    if cpath.exists():
        try:
            return json.loads(cpath.read_text(encoding="utf-8"))['response']
        except Exception:
            pass

    async with _sem:  # GPU gate
        async with aiohttp.ClientSession() as sess:
            url = f"{LLM_URL}/api/generate"
            payload = {"model": mdl, "prompt": prompt, "stream": False, "options": opts}
            async with sess.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as r:
                r.raise_for_status()
                data = await r.json()
                # Cache minimal payload
                try:
                    cpath.write_text(json.dumps({"response": data.get("response", "")}), encoding="utf-8")
                except Exception:
                    pass
                return data.get("response", "")