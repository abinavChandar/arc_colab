import os
import sys
import json
import time
import glob
import argparse
import asyncio
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

# local import
import llm_client


# --------- Timing utilities ---------
class Timer:
    def __init__(self):
        self._t0 = None
        self.elapsed = 0.0  # ensure attribute always exists
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        if self._t0 is not None:
            self.elapsed = time.perf_counter() - self._t0


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%S")


# --------- Minimal task loader (adjust to your repo) ---------
def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def find_tasks(root: Path, split: str, limit: Optional[int] = None, pattern: Optional[str] = None) -> List[Path]:
    # In ARC-AGI-2, tasks live under e.g. data/training/*.json
    base = root / split
    pat = pattern or "*.json"
    paths = sorted(Path(base).glob(pat))
    return paths[:limit] if limit else paths


# --------- SOLVER INTEGRATION ---------
"""
You have two options:

A) Adapter import:
   - If you already have a solve_task(task_json_path: Path) -> dict, import and call it here.

B) Inline adapter:
   - Implement solve_task() below to call into your pipeline. The only hard requirement is:
       return a dict with keys like {"solved": bool, "score": float, ...}

Below we provide a thin example that expects your per-task solver to:
   1) Read the task JSON.
   2) Use llm_client.generate(...) whenever it needs the LLM.
   3) Produce a result dict.

Replace this stub with your actual logic or import your module.
"""


async def solve_task(task_path: Path, compile_timer: Callable[[str], None]) -> Dict[str, Any]:
    """Example async solver stub.
    - Call llm_client.generate() wherever you previously hit the LLM.
    - Use compile_timer(label) to record compilation/codegen/verify timings.
    """
    # Load task
    with Timer() as t_load:
        task = load_json(task_path)
    # Example: plan NL → code (placeholder prompts)
    with Timer() as t_plan:
        plan = await llm_client.generate(
            f"You are an ARC planner. Produce concise bullet steps to transform the input grid to the output.\nTask id: {task_path.stem}\nKeep under 10 bullets.",
            max_new_tokens=256,
            temperature=0.2,
        )
    compile_timer(f"plan_nl:{t_plan.elapsed:.3f}s")

    # Example: codegen
    with Timer() as t_codegen:
        code = await llm_client.generate(
            f"Turn these steps into a Python function transform(grid) that returns the output grid.\nSteps:\n{plan}\nReturn only code.",
            max_new_tokens=512,
            temperature=0.2,
            stop=["```"],
        )
    compile_timer(f"codegen:{t_codegen.elapsed:.3f}s")

    # Example: compile & verify (pseudo)
    solved = False
    score = 0.0
    diagnostics = ""
    with Timer() as t_compile:
        try:
            # VERY simplified: exec the function safely (you should sandbox in real runs)
            ns: Dict[str, Any] = {}
            exec(code, ns, ns)
            transform = ns.get("transform")
            if callable(transform):
                # Here you'd run through train pairs and compute accuracy
                # We mark a stubbed success to demonstrate timing
                solved = True
                score = 1.0
            else:
                diagnostics = "No transform() found in code."
        except Exception as e:
            diagnostics = f"compile_error: {e}"
    compile_timer(f"compile:{t_compile.elapsed:.3f}s")

    # Example: verification timing stub
    with Timer() as t_verify:
        time.sleep(0.01)
    compile_timer(f"verify:{t_verify.elapsed:.3f}s")

    return {
        "task_id": task_path.stem,
        "solved": solved,
        "score": score,
        "diagnostics": diagnostics,
        "timings": {
            "load": t_load.elapsed,
            "plan": t_plan.elapsed,
            "codegen": t_codegen.elapsed,
            "compile": t_compile.elapsed,
            "verify": t_verify.elapsed,
        },
        "plan": plan,
        "code": code,
    }


# --------- Orchestrator with live ETA ---------
@dataclass
class RunConfig:
    root: Path
    split: str
    max_tasks: Optional[int]
    task_concurrency: int
    llm_concurrency: int
    model: str
    pattern: Optional[str]
    out_dir: Path
    resume_dir: Optional[Path]


def ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    return out_dir


class CompilationTimeline:
    """Collects per-task compilation/phase stamps for debugging performance."""
    def __init__(self):
        self.events: List[str] = []
    def record(self, label: str):
        self.events.append(label)
    def as_str(self) -> str:
        return " | ".join(self.events)


class RunStats:
    def __init__(self, total: int):
        self.total = total
        self.start = time.monotonic()
        self.done = 0
        self.times: List[float] = []
        self._lock = asyncio.Lock()
    async def add(self, dt: float):
        async with self._lock:
            self.done += 1
            self.times.append(dt)
    def snapshot(self):
        now = time.monotonic()
        elapsed = now - self.start
        avg = (sum(self.times) / len(self.times)) if self.times else 0.0
        remaining = self.total - self.done
        eta = remaining * avg
        return {
            "done": self.done,
            "total": self.total,
            "elapsed_s": elapsed,
            "avg_task_s": avg,
            "eta_s": eta,
        }


async def worker(task_path: Path, cfg: RunConfig, out_fh, lock: asyncio.Lock, stats: RunStats):
    timeline = CompilationTimeline()
    def _record(label: str):
        timeline.record(label)

    with Timer() as t_all:
        try:
            result = await solve_task(task_path, _record)
            status = "ok"
        except Exception as e:
            result = {"task_id": task_path.stem, "error": str(e)}
            status = "error"
    total_time = t_all.elapsed

    await stats.add(total_time)

    # finalize record
    result["total_time"] = total_time
    result["timeline"] = timeline.as_str()

    line = json.dumps({"status": status, **result}, ensure_ascii=False)
    async with lock:
        out_fh.write(line + "\n")
        out_fh.flush()
    print(line)


async def ticker(stats: RunStats, interval: float = 5.0):
    while stats.done < stats.total:
        await asyncio.sleep(interval)
        s = stats.snapshot()
        def fmt(sec: float) -> str:
            return time.strftime("%H:%M:%S", time.gmtime(max(0, sec)))
        print(
            f"[progress] {s['done']}/{s['total']} | elapsed {fmt(s['elapsed_s'])} | "
            f"avg/task {s['avg_task_s']:.1f}s | ETA {fmt(s['eta_s'])}"
        )


async def run(cfg: RunConfig):
    os.environ["LLM_MODEL"] = cfg.model
    os.environ["LLM_CONCURRENCY"] = str(cfg.llm_concurrency)

    tasks = find_tasks(cfg.root, cfg.split, cfg.max_tasks, cfg.pattern)
    if not tasks:
        raise SystemExit(f"No tasks found under {cfg.root}/{cfg.split} (pattern={cfg.pattern})")

    out_dir = ensure_out_dir(cfg.out_dir)
    results_path = out_dir / "results.jsonl"

    # Resume: skip completed task_ids if resuming into an existing results file.
    done_ids = set()
    if cfg.resume_dir:
        resume_file = cfg.resume_dir / "results.jsonl"
        if resume_file.exists():
            for line in resume_file.read_text(encoding="utf-8").splitlines():
                try:
                    obj = json.loads(line)
                    if obj.get("task_id"):
                        done_ids.add(obj.get("task_id"))
                except Exception:
                    pass
            print(f"[resume] found {len(done_ids)} completed tasks; they will be skipped.")

    lock = asyncio.Lock()
    sem_tasks = asyncio.Semaphore(cfg.task_concurrency)

    def eligible(path: Path) -> bool:
        return path.stem not in done_ids

    selected = [p for p in tasks if eligible(p)]
    print(f"Planning to run {len(selected)} tasks (of {len(tasks)} found)…")

    # Open results file once.
    results_fh = open(results_path, "a", encoding="utf-8")

    stats = RunStats(total=len(selected))
    tick_task = asyncio.create_task(ticker(stats, 5.0))

    async def _run_one(p: Path):
        async with sem_tasks:
            await worker(p, cfg, results_fh, lock, stats)

    try:
        # Launch tasks and consume as they complete to keep live stats accurate
        await asyncio.gather(*[ _run_one(p) for p in selected ])
    finally:
        tick_task.cancel()
        results_fh.close()
        s = stats.snapshot()
        print(json.dumps({
            "summary": {
                "total": s["total"],
                "completed": s["done"],
                "elapsed_s": s["elapsed_s"],
                "avg_task_s": s["avg_task_s"],
                "eta_s": s["eta_s"],
                "out": str(results_path),
            }
        }))
        print(f"Wrote results to: {results_path}")


def parse_args(argv: Optional[List[str]] = None) -> RunConfig:
    parser = argparse.ArgumentParser(description="ARC batch runner with GPU-throttled LLM, timers, live ETA")
    parser.add_argument("--root", type=str, required=True, help="Path to ARC-AGI-2/data")
    parser.add_argument("--split", type=str, default="training", help="training|evaluation|test")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--task-concurrency", type=int, default=12, help="Concurrent tasks (CPU/async)")
    parser.add_argument("--llm-concurrency", type=int, default=6, help="Concurrent LLM calls (GPU gate)")
    parser.add_argument("--model", type=str, default="qwen2.5-coder:7b", help="Ollama model name")
    parser.add_argument("--pattern", type=str, default=None, help="Glob like 00*.json to filter tasks")
    parser.add_argument("--out-dir", type=str, default=None, help="Output dir; default out/arc_runs/<ts>")
    parser.add_argument("--resume-dir", type=str, default=None, help="Existing run dir to skip completed tasks")

    a = parser.parse_args(argv)
    root = Path(a.root)
    out_dir = Path(a.out_dir) if a.out_dir else Path("out/arc_runs") / stamp()
    resume_dir = Path(a.resume_dir) if a.resume_dir else None

    return RunConfig(
        root=root,
        split=a.split,
        max_tasks=a.max_tasks,
        task_concurrency=a.task_concurrency,
        llm_concurrency=a.llm_concurrency,
        model=a.model,
        pattern=a.pattern,
        out_dir=out_dir,
        resume_dir=resume_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    # Colab-friendly: ensure event loop
    try:
        asyncio.run(run(cfg))
    except RuntimeError:
        # If running inside an environment that already has a loop (e.g., Jupyter), fall back
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run(cfg))