# save as ollama_compat.py and run: python ollama_compat.py
from flask import Flask, request, Response, jsonify
import json, requests, time

# point to vLLM OpenAI-compatible endpoint
OPENAI_BASE = "http://127.0.0.1:8000/v1"
MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"

app = Flask(__name__)

@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True) or {}
    model = data.get("model", MODEL_DEFAULT)
    prompt = data.get("prompt", "")
    stream = bool(data.get("stream", True))
    opts = data.get("options", {}) or {}
    temperature = float(opts.get("temperature", 0.2))
    max_tokens = int(opts.get("max_tokens", 512))

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    headers = {"Authorization": "Bearer EMPTY", "Content-Type": "application/json"}

    if not stream:
        r = requests.post(f"{OPENAI_BASE}/chat/completions", headers=headers, json=payload, timeout=600)
        if r.status_code != 200:
            return jsonify({"error": f"upstream {r.status_code}: {r.text}"}), 500
        text = r.json()["choices"][0]["message"]["content"]
        return jsonify({"model": model, "created_at": time.time(), "response": text, "done": True})

    def sse():
        with requests.post(f"{OPENAI_BASE}/chat/completions", headers=headers, json=payload, stream=True, timeout=600) as r:
            if r.status_code != 200:
                yield json.dumps({"error": f"upstream {r.status_code}: {r.text}"}) + "\n"
                return
            buffer = ""
            for chunk in r.iter_lines(decode_unicode=True):
                if not chunk:
                    continue
                if chunk.startswith("data: "):
                    chunk = chunk[6:]
                if chunk == "[DONE]":
                    yield json.dumps({"done": True}) + "\n"
                    break
                try:
                    obj = json.loads(chunk)
                    delta = obj["choices"][0]["delta"].get("content", "")
                except Exception:
                    # some servers send plain JSON lines without "data: "
                    try:
                        obj = json.loads(chunk)
                        delta = obj["choices"][0]["message"]["content"]
                    except Exception:
                        delta = ""
                if delta:
                    yield json.dumps({"response": delta, "done": False}) + "\n"
            # safety finalizer
            yield json.dumps({"done": True}) + "\n"

    return Response(sse(), mimetype="application/x-ndjson")

@app.route("/api/tags", methods=["GET"])
def tags():
    # minimal implementation used by some tools
    return jsonify({"models": [{"name": MODEL_DEFAULT}]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=11434, threaded=True)
