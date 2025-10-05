#!/usr/bin/env python3
# fanout.py — run arc_nl_pipeline.py N times in parallel and print a TOTAL at the end
import argparse, os, subprocess, sys, time
from pathlib import Path

def hms(seconds: int) -> str:
    return f"{seconds//3600:02d}:{(seconds%3600)//60:02d}:{seconds%60:02d}"

def main():
    ap = argparse.ArgumentParser(description="Parallel launcher for arc_nl_pipeline.py")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", type=str, default="training")
    ap.add_argument("--pattern", type=str, default="*.json", help='Glob under <root>/<split> (e.g., "00*.json")')
    ap.add_argument("--max-tasks", type=int, default=20, help="How many tasks to launch total")
    ap.add_argument("--concurrency", type=int, default=12, help="How many arc_nl_pipeline.py processes at once")
    ap.add_argument("--ollama-num-parallel", type=int, default=6, help="Gate concurrent LLM requests to GPU/Metal")
    ap.add_argument("--ollama-max-loaded-models", type=int, default=1, help="Keep N models resident on GPU")
    # passthrough knobs to your pipeline (tweak as you like)
    ap.add_argument("--num-candidates", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()

    # env for all children
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["OLLAMA_NUM_PARALLEL"] = str(args.ollama_num_parallel)
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = str(args.ollama_max_loaded_models)

    # discover task IDs
    tasks = sorted((args.root / args.split).glob(args.pattern))[: args.max_tasks]
    if not tasks:
        print("No tasks found. Check --root/--split/--pattern.", file=sys.stderr)
        sys.exit(2)

    print(f"[fanout] matched {len(tasks)} tasks; launching up to {args.concurrency} at a time")

    procs = []
    t0 = time.time()

    def start_task(p: Path):
        tid = p.stem
        cmd = [
            sys.executable, "-u", "arc_nl_pipeline.py",
            "--root", str(args.root), "--split", args.split,
            "--task-id", tid,
            "--num-candidates", str(args.num_candidates),
            "--rounds", str(args.rounds),
            "--parallel", str(args.parallel),
        ]
        print(f"[start] {tid}")
        return tid, subprocess.Popen(cmd)

    # launch respecting concurrency
    it = iter(tasks)
    # prime up to concurrency
    for _ in range(min(args.concurrency, len(tasks))):
        tid, pr = start_task(next(it))
        procs.append((tid, pr))

    # as each finishes, start the next
    results = {}
    remaining = len(tasks) - len(procs)
    while procs:
        for i, (tid, pr) in enumerate(list(procs)):
            rc = pr.poll()
            if rc is None:
                continue
            # finished
            print(f"[done]  {tid} (rc={rc})")
            results[tid] = rc
            procs.pop(i)
            # launch next if any left
            if remaining > 0:
                tid2, pr2 = start_task(next(it))
                procs.append((tid2, pr2))
                remaining -= 1
        time.sleep(0.2)

    total_secs = int(time.time() - t0)
    failures = sum(1 for rc in results.values() if rc != 0)
    print(f"[SUMMARY] total={len(results)} ok={len(results)-failures} fail={failures}")
    print(f"[TOTAL] elapsed {hms(total_secs)}")

    sys.exit(0 if failures == 0 else 1)

if __name__ == "__main__":
    main()
