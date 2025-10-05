#!/usr/bin/env python3
"""
ARC-AGI Natural-Language Evaluator (cell-wise + complete accuracy)

What it does
------------
Given the instruction candidates produced by arc_nl_describer.py, this script:
1) For each task and each candidate instruction:
   - Prompts an Ollama model to APPLY the instructions to every training input.
   - Parses the predicted output grid (strict JSON [[int,...],...]).
   - Compares to the ground-truth grid.

2) Scores:
   - Secondary (cell-wise) accuracy: sum(correct_cells) / sum(total_cells) across pairs
     *Only accumulated when sizes match; otherwise 0 for that pair.*
   - Primary (complete): number of training pairs solved perfectly.
   - Also logs parse_rate and size_ok_rate diagnostics.

3) Writes results:
   - eval_outputs/<split>/<task_id>/cand_<k>.json      (per-candidate details)
   - eval_outputs/<split>/<task_id>/task_summary.json  (ranked candidates)

Ranking rule
------------
primary_perfect DESC, then secondary_pct DESC, then parse_rate DESC, size_ok_rate DESC.

Usage
-----
python3 arc_nl_evaluator.py \
  --root /path/to/ARC-AGI-2/data \
  --split training \
  --model qwen2.5-coder:7b \
  --parallel 6 \
  --attempts 1 \
  --temperature 0.0 \
  --max-tasks 20
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import http.client
import sys

# -------------------------
# ARC dataset helpers
# -------------------------

def load_arc_task(task_path: Path) -> Dict[str, Any]:
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_tasks(root: Path, split: str) -> List[Path]:
    pattern = str(root / split / "*.json")
    paths = [Path(p) for p in glob.glob(pattern)]
    paths.sort()
    return paths

def grid_shape(grid: List[List[int]]) -> Tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)

def cell_accuracy(pred: List[List[int]], truth: List[List[int]]) -> Tuple[bool, bool, int, int]:
    """
    Returns (exact, size_ok, correct_cells, total_cells).
    Secondary (cell-wise) credit only accrues when size_ok is True.
    """
    if not truth:
        return False, False, 0, 0
    Ht, Wt = grid_shape(truth)
    Hp, Wp = grid_shape(pred)
    size_ok = (Ht == Hp and Wt == Wp)
    if not size_ok:
        return False, False, 0, Ht * Wt
    correct = 0
    for r in range(Ht):
        for c in range(Wt):
            if pred[r][c] == truth[r][c]:
                correct += 1
    exact = (correct == Ht * Wt)
    return exact, True, correct, Ht * Wt

# -------------------------
# Ollama client
# -------------------------

class OllamaClient:
    def __init__(self, host: str = "localhost", port: int = 11434, timeout: int = 120):
        self.host = host
        self.port = port
        self.timeout = timeout

    def generate(self, model: str, prompt: str, temperature: float = 0.0, seed: int | None = None) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": float(temperature)},
        }
        if seed is not None:
            payload["options"]["seed"] = int(seed)
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            body = json.dumps(payload)
            conn.putrequest("POST", "/api/generate")
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(len(body)))
            conn.endheaders()
            conn.send(body.encode("utf-8"))
            chunks: List[str] = []
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f"Ollama HTTP {resp.status}: {resp.reason}")
            for raw_line in resp.read().splitlines():
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line.decode("utf-8"))
                except Exception:
                    try:
                        obj = json.loads(raw_line)
                    except Exception:
                        continue
                if "error" in obj:
                    raise RuntimeError(f"Ollama error: {obj['error']}")
                if obj.get("done"):
                    break
                token = obj.get("response", "")
                if token:
                    chunks.append(token)
            return "".join(chunks).strip()
        finally:
            conn.close()

# -------------------------
# Prompting + parsing
# -------------------------

APPLY_SYSTEM = (
    "You are an ARC grid transformer. APPLY the given instructions to the INPUT grid.\n"
    "Return ONLY the OUTPUT grid as strict JSON like [[0,1],[2,3]] — integers 0–9, no prose, no code fences.\n"
    "The OUTPUT grid size must follow the transformation pattern implied by the instructions and training pattern. "
    "If the rule entails resizing, compute the new dimensions accordingly; otherwise preserve size only if the rule implies it."
)

def build_apply_prompt(task_id: str, instructions: str, input_grid: List[List[int]]) -> str:
    return (
        f"Task: {task_id}\n"
        f"Instructions (natural language, to APPLY):\n{instructions}\n\n"
        f"INPUT grid (H={len(input_grid)}, W={len(input_grid[0]) if input_grid else 0}):\n"
        f"{json.dumps(input_grid, separators=(',', ':'))}\n\n"
        f"{APPLY_SYSTEM}"
    )

def is_valid_grid(obj: Any) -> bool:
    if not isinstance(obj, list) or not obj:
        return False
    row_len = None
    for row in obj:
        if not isinstance(row, list) or not row:
            return False
        if row_len is None:
            row_len = len(row)
        if len(row) != row_len:
            return False
        for v in row:
            if not isinstance(v, int) or v < 0 or v > 9:
                return False
    return True

def try_extract_json_grid(text: str) -> List[List[int]] | None:
    # 1) direct
    try:
        obj = json.loads(text)
        if is_valid_grid(obj):
            return obj
    except Exception:
        pass
    # 2) fenced or noisy
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("` \n")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1].strip()
    try:
        start = cleaned.index('[')
        end = cleaned.rindex(']') + 1
        snippet = cleaned[start:end]
        obj = json.loads(snippet)
        if is_valid_grid(obj):
            return obj
    except Exception:
        return None

# -------------------------
# IO: describer/eval artifacts
# -------------------------

def load_describer_bundle(outputs_root: Path, split: str, task_id: str) -> Dict[str, Any] | None:
    p = outputs_root / split / f"{task_id}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_eval_dirs(eval_root: Path, split: str, task_id: str):
    task_dir = eval_root / split / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir

# -------------------------
# Evaluation core
# -------------------------

def evaluate_candidate(
    client: OllamaClient,
    model: str,
    task_id: str,
    task: Dict[str, Any],
    candidate_id: int,
    instructions: str,
    attempts: int,
    temperature: float,
    parallel: int,
) -> Dict[str, Any]:
    train_pairs = task.get("train", [])
    pair_results: List[Dict[str, Any]] = []
    primary = 0
    secondary_cells = 0
    secondary_possible = 0
    parse_success = 0
    size_ok_count = 0

    def eval_one(i: int, inp: List[List[int]], truth: List[List[int]]):
        nonlocal primary, secondary_cells, secondary_possible, parse_success, size_ok_count
        prompt = build_apply_prompt(task_id, instructions, inp)
        pred_grid, raw_text, err = None, None, None
        for k in range(attempts):
            try:
                raw_text = client.generate(model=model, prompt=prompt, temperature=temperature)
                pred_grid = try_extract_json_grid(raw_text)
                if pred_grid is not None:
                    parse_success += 1
                    break
            except Exception as e:
                err = str(e)
                time.sleep(0.5 * (k + 1))

        if pred_grid is None:
            exact, size_ok, correct, total = False, False, 0, len(truth) * (len(truth[0]) if truth else 0)
        else:
            exact, size_ok, correct, total = cell_accuracy(pred_grid, truth)
            if size_ok:
                size_ok_count += 1

        if exact:
            primary += 1
        else:
            secondary_cells += correct
            secondary_possible += total

        pair_results.append({
            "index": i,
            "size_ok": size_ok,
            "exact": exact,
            "correct_cells": correct,
            "total_cells": total,
            "accuracy": (correct / total) if total else 0.0,
            "pred": pred_grid,
            "error": err,
        })

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = []
            for i, pair in enumerate(train_pairs):
                futures.append(ex.submit(eval_one, i, pair["input"], pair["output"]))
            for fut in as_completed(futures):
                fut.result()
    else:
        for i, pair in enumerate(train_pairs):
            eval_one(i, pair["input"], pair["output"])

    pair_results.sort(key=lambda r: r["index"])
    num_pairs = len(train_pairs)

    summary = {
        "task_id": task_id,
        "candidate_id": candidate_id,
        "primary_perfect": primary,
        "primary_pct": (primary / num_pairs) if num_pairs else 0.0,
        "secondary_cells": secondary_cells,
        "secondary_possible": secondary_possible,
        "secondary_pct": (secondary_cells / secondary_possible) if secondary_possible else 0.0,
        "num_pairs": num_pairs,
        "parse_rate": (parse_success / num_pairs) if num_pairs else 0.0,
        "size_ok_rate": (size_ok_count / num_pairs) if num_pairs else 0.0,
    }

    return {
        "ts": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task_id": task_id,
        "candidate_id": candidate_id,
        "instructions": instructions,
        "evaluation": {
            "summary": summary,
            "pairs": pair_results,
        },
    }

# -------------------------
# Ranking
# -------------------------

def rank_candidates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # primary DESC, then secondary_pct DESC, then parse_rate, then size_ok_rate
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r.get("primary_perfect", 0),
            r.get("secondary_pct", 0.0),
            r.get("parse_rate", 0.0),
            r.get("size_ok_rate", 0.0),
        ),
        reverse=True,
    )
    return rows_sorted

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="ARC-AGI dataset root")
    ap.add_argument("--split", type=str, choices=["training", "evaluation"], default="training")
    ap.add_argument("--outputs", type=Path, default=Path("outputs"), help="Describer outputs root")
    ap.add_argument("--eval-outputs", type=Path, default=Path("eval_outputs"), help="Where to write evaluation artifacts")
    ap.add_argument("--model", type=str, default="qwen2.5-coder:7b")
    ap.add_argument("--temperature", type=float, default=0.0, help="Apply temperature (keep low for determinism)")
    ap.add_argument("--attempts", type=int, default=1, help="Retries per apply call")
    ap.add_argument("--parallel", type=int, default=4, help="Workers per candidate")
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=0, help="Optional cap per task (0=all)")
    args = ap.parse_args()

    client = OllamaClient()

    task_paths = list_tasks(args.root, args.split)
    if args.max_tasks and args.max_tasks > 0:
        task_paths = task_paths[: args.max_tasks]

    print(f"Evaluating {len(task_paths)} tasks with model={args.model}")

    for t_idx, task_path in enumerate(task_paths, 1):
        task_id = task_path.stem
        task = load_arc_task(task_path)

        bundle = load_describer_bundle(args.outputs, args.split, task_id)
        if not bundle:
            print(f"[{t_idx}/{len(task_paths)}] {task_id}: missing describer outputs — skipping")
            continue

        candidates = bundle.get("candidates", [])
        if args.max_candidates and args.max_candidates > 0:
            candidates = candidates[: args.max_candidates]

        task_dir = ensure_eval_dirs(args.eval_outputs, args.split, task_id)

        ranked_rows: List[Dict[str, Any]] = []

        for cand in candidates:
            cand_id = int(cand.get("candidate_id", 0))
            instr   = cand.get("nl_description") or ""
            if not instr.strip():
                print(f"  - cand {cand_id}: empty instruction text; skipping.")
                continue

            res = evaluate_candidate(
                client=client,
                model=args.model,
                task_id=task_id,
                task=task,
                candidate_id=cand_id,
                instructions=instr,
                attempts=args.attempts,
                temperature=args.temperature,
                parallel=args.parallel,
            )

            # write per-candidate file
            with open(task_dir / f"cand_{cand_id}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)

            s = res["evaluation"]["summary"]
            ranked_rows.append({
                "candidate_id": cand_id,
                "task_id": task_id,
                "primary_perfect": s["primary_perfect"],
                "primary_pct": s["primary_pct"],
                "secondary_cells": s["secondary_cells"],
                "secondary_possible": s["secondary_possible"],
                "secondary_pct": s["secondary_pct"],
                "num_pairs": s["num_pairs"],
                "parse_rate": s["parse_rate"],
                "size_ok_rate": s["size_ok_rate"],
            })

            print(f"  - cand {cand_id}: primary={s['primary_perfect']} ({s['primary_pct']*100:.1f}%) "
                  f"secondary={s['secondary_cells']}/{s['secondary_possible']} ({s['secondary_pct']*100:.1f}%) "
                  f"parse={s['parse_rate']*100:.0f}% size_ok={s['size_ok_rate']*100:.0f}%")

        # rank and write summary
        ranked = rank_candidates(ranked_rows)
        summary_payload = {
            "task_id": task_id,
            "split": args.split,
            "ranked_candidates": ranked,
            "ranking_rule": ["primary_perfect DESC", "secondary_pct DESC", "parse_rate DESC", "size_ok_rate DESC"],
        }
        with open(task_dir / "task_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)

        if ranked:
            top = ranked[0]
            print(f"[{t_idx}/{len(task_paths)}] {task_id}: best cand={top['candidate_id']} "
                  f"primary={top['primary_perfect']} ({top['primary_pct']*100:.1f}%) "
                  f"secondary={top['secondary_cells']}/{top['secondary_possible']} ({top['secondary_pct']*100:.1f}%)")
        else:
            print(f"[{t_idx}/{len(task_paths)}] {task_id}: no valid candidates evaluated.")

    print("Done.")

if __name__ == "__main__":
    main()
