# vllm_compat.py
import os, json, time, requests
from typing import Optional

class SyncClient:
    def __init__(self, timeout: int = 120):
        self.base = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        self.key  = os.getenv("OPENAI_API_KEY", "dummy")
        self.timeout = timeout
        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: dict):
        url = f"{self.base}{path}"
        r = self.session.post(url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
        if r.status_code >= 400:
            raise requests.HTTPError(f"Client error '{r.status_code} {r.reason}' for url '{url}'\n{r.text}")
        return r.json()

    def generate(self, model: str, prompt: str, temperature: float = 0.2, seed: Optional[int] = None) -> str:
        # 1) Try Chat Completions
        chat_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
        }
        if seed is not None:
            # Many servers accept 'seed'; if not, they'll ignore it.
            chat_payload["seed"] = int(seed)

        try:
            data = self._post("/v1/chat/completions", chat_payload)
            msg = (data.get("choices") or [{}])[0].get("message", {})
            content = msg.get("content", "")
            if content:
                return content.strip()
        except requests.HTTPError as e:
            # Fall through to /v1/completions on typical chat route errors
            if not any(code in str(e) for code in ["404", "405", "400"]):
                raise

        # 2) Fallback to Completions
        comp_payload = {
            "model": model,
            "prompt": prompt,
            "temperature": float(temperature),
        }
        if seed is not None:
            comp_payload["seed"] = int(seed)

        data = self._post("/v1/completions", comp_payload)
        text = (data.get("choices") or [{}])[0].get("text", "")
        return (text or "").strip()
