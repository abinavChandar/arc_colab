import asyncio, httpx, time, json, statistics as stats

BASE = "http://127.0.0.1:8000/v1/chat/completions"
HEAD = {"Authorization": "Bearer EMPTY", "Content-Type": "application/json"}

def payload(txt):
    return {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": txt}],
        "max_tokens": 128,
        "temperature": 0.2
    }

async def one(client, data):
    t0 = time.time()
    r = await client.post(BASE, headers=HEAD, json=data, timeout=120)
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"]
    dt = time.time() - t0
    return out, dt

async def burst(n=16):
    prompts = [f"Req {i}: list three short Python perf tips." for i in range(n)]
    datas = [payload(p) for p in prompts]
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[one(client, d) for d in datas])
    wall = time.time() - t0
    per_req = [dt for _, dt in results]
    print(f"Burst size: {n}")
    print(f"Wall-time: {wall:.2f}s")
    print(f"Per-request lat (median): {stats.median(per_req):.2f}s")
    print(f"Speedup vs sequential (≈median*n/wall): { (stats.median(per_req)*n)/wall :.1f}×")
    print("Sample output:", results[0][0][:120], "...")
    return wall

asyncio.run(burst(16))  # try 8, 16, 24, 32 and see scaling
