#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC-AGI NL — Pooled Revision Phase (Screenshot-Style Prompt)
Robust to revision files where 'evaluation' is either:
  A) {'summary': {...}, 'pairs': [...]}
  B) {...flat summary fields...}  (no 'summary' key)

If B, we wrap it as {'summary': <that dict>, 'pairs': []} and (optionally) quick-evaluate
to populate pairs for nicer per-example accuracy lines.

See usage example at the bottom of this file.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import http.client
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------- ARC I/O -------------------------
def load_arc_task(task_path: Path) -> Dict[str, Any]:
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_tasks(root: Path, split: str) -> List[Path]:
    pattern = str(root / split / "*.json")
    paths = [Path(p) for p in glob.glob(pattern)]
    paths.sort()
    return paths

def grid_shape(g: List[List[int]]) -> Tuple[int, int]:
    return len(g), (len(g[0]) if g else 0)

def grid_to_ascii(g: List[List[int]]) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in g)

# ------------------------- Base candidates & revisions -------------------------
def load_base_candidates(outputs_root: Path, split: str, task_id: str) -> Optional[Dict[str, Any]]:
    p = outputs_root / split / f"{task_id}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def list_round_files(rev_root: Path, split: str, task_id: str, cand_id: int) -> List[Path]:
    pat = str(rev_root / split / task_id / f"cand_{cand_id}_round*.json")
    paths = [Path(p) for p in glob.glob(pat)]
    paths.sort(key=lambda p: p.stem)  # round0, round1, ...
    return paths

def load_latest_round(rev_root: Path, split: str, task_id: str, cand_id: int) -> Optional[Dict[str, Any]]:
    files = list_round_files(rev_root, split, task_id, cand_id)
    if not files:
        return None
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------------- JSON grid parsing -------------------------
def _is_valid_grid(x: Any) -> bool:
    if not isinstance(x, list) or not x:
        return False
    w = None
    for row in x:
        if not isinstance(row, list) or not row:
            return False
        if w is None:
            w = len(row)
        if len(row) != w:
            return False
        for v in row:
            if not isinstance(v, int) or v < 0 or v > 9:
                return False
    return True

def try_extract_json_grid(text: str) -> Optional[List[List[int]]]:
    # 1) direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "grid" in obj and _is_valid_grid(obj["grid"]):
            return obj["grid"]
        if _is_valid_grid(obj):
            return obj
    except Exception:
        pass
    # 2) fenced/noisy: object slice
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("` \n")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1].strip()
    try:
        start = cleaned.index('{'); end = cleaned.rindex('}') + 1
        snippet = cleaned[start:end]
        obj = json.loads(snippet)
        if isinstance(obj, dict) and "grid" in obj and _is_valid_grid(obj["grid"]):
            return obj["grid"]
    except Exception:
        pass
    # 3) bare array slice
    try:
        start = cleaned.index('['); end = cleaned.rindex(']') + 1
        snippet = cleaned[start:end]
        obj = json.loads(snippet)
        if _is_valid_grid(obj):
            return obj
    except Exception:
        pass
    return None

# ------------------------- Ollama client -------------------------
class OllamaClient:
    def __init__(self, host="localhost", port=11434, timeout=120):
        self.host = host
        self.port = port
        self.timeout = timeout

    def generate(self, model: str, prompt: str, temperature: float = 0.2, seed: Optional[int] = None) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "30m",
            "options": {
                "temperature": float(temperature),
                "num_thread": 10,
                "num_ctx": 2048,
                "num_batch": 128,
                "num_predict": 1024,
            },
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

# ------------------------- Constrained apply + scoring -------------------------
APPLY_SYSTEM = (
    "You are an ARC grid transformer. APPLY the given instructions to the INPUT grid.\n"
    "STRICT OUTPUT FORMAT:\n"
    "Return a JSON object with exactly one key 'grid' whose value is the 2D integer array, e.g. {\"grid\": [[0,1],[2,3]]}.\n"
    "Include nothing else: no prose, no code fences, no trailing text.\n"
    "If the rule resizes, compute the new dimensions accordingly; if not, preserve size."
)

def build_apply_prompt(task_id: str, instructions: str, input_grid: List[List[int]],
                       target_shape: Optional[Tuple[int,int]]=None) -> str:
    tgt = ""
    if target_shape:
        th, tw = target_shape
        tgt = (
            "\nREQUIREMENT: The OUTPUT grid MUST have this exact shape derived from the rule: "
            f"H_out={th}, W_out={tw}. Adjust your operations to produce exactly this size.\n"
        )
    return (
        f"Task: {task_id}\n"
        f"Instructions (natural language):\n{instructions}\n\n"
        f"INPUT grid (H={len(input_grid)}, W={len(input_grid[0]) if input_grid else 0}):\n"
        f"{json.dumps(input_grid, separators=(',', ':'))}\n"
        f"{tgt}\n"
        f"{APPLY_SYSTEM}"
    )

def cell_accuracy(pred: Optional[List[List[int]]], truth: List[List[int]]) -> Tuple[bool, bool, int, int]:
    if truth is None or len(truth) == 0:
        return False, False, 0, 0
    Ht, Wt = grid_shape(truth)
    if pred is None:
        return False, False, 0, Ht * Wt
    Hp, Wp = grid_shape(pred)
    size_ok = (Hp == Ht and Wp == Wt)
    if not size_ok:
        return False, False, 0, Ht * Wt
    correct = 0
    total = Ht * Wt
    for r in range(Ht):
        for c in range(Wt):
            if pred[r][c] == truth[r][c]:
                correct += 1
    return (correct == total), True, correct, total

def try_apply_constrained(client: OllamaClient, apply_model: str, task_id: str,
                          instructions: str, inp: List[List[int]], truth: List[List[int]],
                          temperature: float, parse_attempts: int = 1, shape_attempts: int = 3) -> Tuple[Optional[List[List[int]]], str]:
    th, tw = grid_shape(truth)
    prompt = build_apply_prompt(task_id, instructions, inp, target_shape=(th, tw))
    pred, note = None, ""
    err = None
    # parse attempts
    for k in range(max(1, parse_attempts)):
        try:
            raw = client.generate(model=apply_model, prompt=prompt, temperature=temperature)
            g = try_extract_json_grid(raw)
            if g is not None:
                pred = g
                break
        except Exception as e:
            err = str(e)
            time.sleep(0.2 * (k + 1))
    # shape nudges
    if pred is not None:
        Hp, Wp = grid_shape(pred)
        tries = 0
        while (Hp, Wp) != (th, tw) and tries < max(1, shape_attempts):
            tries += 1
            nudge = (
                "\nYour previous output had the wrong shape "
                f"({Hp}x{Wp}). Produce a JSON object {{\"grid\": ...}} whose grid is EXACTLY {th}x{tw}. "
                "Use only integers 0–9; no prose."
            )
            try:
                raw2 = client.generate(model=apply_model, prompt=prompt + nudge, temperature=temperature)
                g2 = try_extract_json_grid(raw2)
                if g2 is not None:
                    pred = g2
                    Hp, Wp = grid_shape(pred)
                    continue
            except Exception:
                pass
            time.sleep(0.1 * tries)
        if (Hp, Wp) != (th, tw):
            note = f"shape_mismatch_after_retries: got {Hp}x{Wp}, want {th}x{tw}"
    if pred is None and err:
        note = f"parse_error: {err}"
    return pred, note

def evaluate_instructions(client: OllamaClient, apply_model: str, task_id: str, task: Dict[str, Any],
                          instructions: str, temperature: float, attempts: int, parallel: int,
                          shape_attempts: int) -> Dict[str, Any]:
    train_pairs = task.get("train", [])
    pair_results: List[Dict[str, Any]] = []
    primary = 0
    secondary_cells = 0
    secondary_possible = 0
    parse_success = 0
    size_ok_constrained_count = 0

    def work(i: int, inp: List[List[int]], truth: List[List[int]]):
        nonlocal primary, secondary_cells, secondary_possible, parse_success, size_ok_constrained_count
        pred, _ = try_apply_constrained(client, apply_model, task_id, instructions, inp, truth,
                                        temperature=temperature, parse_attempts=attempts, shape_attempts=shape_attempts)
        if pred is not None:
            parse_success += 1
        exact, size_ok, correct, total = cell_accuracy(pred, truth)
        if size_ok:
            size_ok_constrained_count += 1
        if exact:
            primary += 1
        pair_results.append({
            "index": i,
            "pred": pred,
            "exact": exact,
            "accuracy": (correct / total) if total else 0.0,
            "correct_cells": correct,
            "total_cells": total,
            "size_ok": size_ok,
        })
        return correct, total

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = [ex.submit(work, i, p["input"], p["output"]) for i, p in enumerate(train_pairs)]
            for f in as_completed(futs):
                corr, tot = f.result()
                secondary_cells += corr
                secondary_possible += tot
    else:
        for i, p in enumerate(train_pairs):
            corr, tot = work(i, p["input"], p["output"])
            secondary_cells += corr
            secondary_possible += tot

    pair_results.sort(key=lambda r: r["index"])
    n = len(train_pairs)
    summary = {
        "task_id": task_id,
        "primary_perfect": primary,
        "primary_pct": (primary / n) if n else 0.0,
        "secondary_cells": secondary_cells,
        "secondary_possible": secondary_possible,
        "secondary_pct": (secondary_cells / secondary_possible) if secondary_possible else 0.0,
        "num_pairs": n,
        "parse_rate": (parse_success / n) if n else 0.0,
        "size_ok_rate": (size_ok_constrained_count / n) if n else 0.0,
    }
    return {"summary": summary, "pairs": pair_results}

# ------------------------- Ranking -------------------------
def rank_candidates(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(c: Dict[str, Any]):
        summ = c.get("evaluation", {}).get("summary", {})
        return (
            -(summ.get("primary_perfect", 0)),
            -(summ.get("secondary_pct", 0.0)),
            -(summ.get("parse_rate", 0.0)),
        )
    return sorted(cands, key=key)

# ------------------------- NORMALIZERS (fix for KeyError) -------------------------
def normalize_eval_struct(e: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts either:
      A) {'summary': {...}, 'pairs': [...]}
      B) {...flat summary...}  (no 'summary' key)
    Returns {'summary': {...}, 'pairs': list}
    """
    if e is None:
        return {"summary": {}, "pairs": []}
    if "summary" in e:
        # Already normalized
        summ = e.get("summary", {}) or {}
        pairs = e.get("pairs", []) or []
        return {"summary": summ, "pairs": pairs}
    # Flat summary → wrap
    summ = dict(e)  # shallow copy
    pairs = summ.pop("pairs", []) if isinstance(summ.get("pairs"), list) else []
    return {"summary": summ, "pairs": pairs if isinstance(pairs, list) else []}

# ------------------------- Screenshot-Style Pooled Prompt -------------------------
def build_pooled_prompt_screenshot_style(task_id: str,
                                         task: Dict[str, Any],
                                         top_cands: List[Dict[str, Any]]) -> str:
    intro = (
        "Pooled Revision Prompt Example\n\n"
        "User:\n"
        "You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\n"
        "Find the common pattern that transforms each input grid into its corresponding output grid, based on\n"
        "the training examples below.\n\n"
        "Your task is to write clear instructions that describe this transformation pattern. These instructions must:\n"
        "- Apply consistently to ALL training examples (the same rule works for every input→output pair)\n"
        "- Be general enough to work on new test cases\n"
        "- Be intuitive and easy to understand\n"
        "- Describe the pattern without referencing specific example numbers or positions\n\n"
        "The transformation pattern should be simple and logical - these puzzles are designed to have elegant,\n"
        "intuitive solutions that humans can readily grasp.\n\n"
        "Write your instructions as a clear, step-by-step process that someone could follow to transform any\n"
        "input grid into the correct output grid.\n\n"
        "Here are the training examples and test input grids:\n"
        "--Training Examples--\n"
    )

    # Training examples
    train_lines: List[str] = []
    for i, pair in enumerate(task.get("train", [])):
        inp, out = pair["input"], pair["output"]
        train_lines += [
            f"Training Example {i+1}",
            "Input:",
            grid_to_ascii(inp),
            "Output:",
            grid_to_ascii(out),
            ""
        ]
    train_block = "\n".join(train_lines).rstrip()

    # Test inputs (REQUIRED by user)
    test_lines: List[str] = ["--End of Training Examples--\n", "\n--Test Inputs--"]
    tests = task.get("test", [])
    if not tests:
        test_lines += ["(No explicit test inputs provided in this task file)"]
    else:
        for i, ex in enumerate(tests):
            test_lines += [
                f"Test Input {i+1}",
                grid_to_ascii(ex["input"]),
                ""
            ]
    test_lines += ["--End of Test Inputs--\n"]
    test_block = "\n".join(test_lines).rstrip()

    # Multi-attempt explanation
    expl = (
        "\nMultiple expert puzzle solvers have attempted to describe the transformation pattern for these grids.\n"
        "Each attempt captured some aspects correctly but failed in other ways.\n\n"
        "Below you'll find:\n"
        "- Each set of proposed instructions\n"
        "- The outputs produced when following those instructions\n"
        "- How those outputs differ from the correct answers\n\n"
        "Your task is to analyze why each approach partially failed and synthesize a complete, correct set of\n"
        "instructions.\n\n"
        "By examining multiple flawed attempts, you can:\n"
        "- Identify what each approach got right\n"
        "- Understand what each approach missed\n"
        "- Recognize common misconceptions about the pattern\n"
        "- Build comprehensive instructions that avoid all these pitfalls\n\n"
        "Study the patterns of success and failure across all attempts, then write instructions that correctly\n"
        "describe the complete transformation rule that works for ALL training examples.\n\n"
        "Your final instructions should be clear, intuitive, and capture the true underlying pattern.\n"
    )

    parts: List[str] = [intro, train_block, test_block, expl]

    # Top-K blocks: <instructions_i> ... <scores_from_instructions_i>
    for idx, cand in enumerate(top_cands, 1):
        instr = (cand.get("nl_description") or cand.get("revised_instructions") or "").strip()
        parts.append(f"<instructions_{idx}>")
        parts.append(instr if instr else "(empty)")
        parts.append(f"</instructions_{idx}>")

        parts.append(f"<scores_from_instructions_{idx}>")
        eval_obj = cand.get("evaluation", {})
        eval_norm = normalize_eval_struct(eval_obj)
        pairs = eval_norm.get("pairs", [])
        summ  = eval_norm.get("summary", {}) or {}

        # Prefer per-pair lines; fallback to summary secondary pct if necessary
        if pairs:
            pairs = sorted(pairs, key=lambda r: r.get("index", 0))
            for pr in pairs:
                i = pr.get("index", 0)
                acc_pct = float(pr.get("accuracy", 0.0)) * 100.0
                parts.append(f"The human got the grid for Training Example {i+1} {acc_pct:.0f}% correct with these instructions.")
        else:
            sec_pct = float(summ.get("secondary_pct", 0.0)) * 100.0
            num_pairs = int(summ.get("num_pairs", 0) or 0)
            if num_pairs <= 0:  # still show at least one line so the format matches screenshot
                num_pairs = 1
            for i in range(num_pairs):
                parts.append(f"The human got the grid for Training Example {i+1} {sec_pct:.0f}% correct with these instructions.")

        parts.append(f"</scores_from_instructions_{idx}>\n")

    parts += [
        "Assistant:",
        "Synthesize new instructions that:\n"
        "- Fix the specific errors shown across attempts\n"
        "- Still work correctly for ALL training examples\n"
        "- Remain clear, intuitive, and general\n\n"
        "Return ONLY:\n"
        "1) A one-sentence rule summary.\n"
        "2) A numbered, step-by-step procedure for transforming ANY new input for this task.\n"
        "3) Brief notes for edge-cases/ambiguities.\n"
    ]
    return "\n".join(parts)

# ------------------------- FS outputs -------------------------
def ensure_pooled_dirs(base: Path, split: str) -> Tuple[Path, Path]:
    d = base / split
    d.mkdir(parents=True, exist_ok=True)
    (d / "pooled_candidates.jsonl").touch(exist_ok=True)
    return d, (d / "pooled_candidates.jsonl")

def write_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def save_task_bundle(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def ensure_eval_dirs(base: Path, split: str, task_id: str) -> Path:
    d = base / split / task_id
    d.mkdir(parents=True, exist_ok=True)
    return d

# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", type=str, choices=["training", "evaluation"], default="training")

    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--revisions", type=Path, default=Path("revisions"))
    ap.add_argument("--pooled-out", type=Path, default=Path("pooled"))

    ap.add_argument("--task-id", type=str, default="")
    ap.add_argument("--max-tasks", type=int, default=0)

    # Models
    ap.add_argument("--reviser-model", type=str, default="qwen2.5-coder:7b",
                    help="Model to synthesize pooled NEW instructions.")
    ap.add_argument("--apply-model", type=str, default="qwen2.5:1.5b-instruct",
                    help="Model to APPLY instructions to grids for scoring (fast JSON-friendly).")

    # Generation/Eval params
    ap.add_argument("--num-new", type=int, default=5, help="How many new candidates to generate from the pool.")
    ap.add_argument("--top-k", type=int, default=5, help="How many top candidates to pool.")
    ap.add_argument("--temperature", type=float, default=0.2, help="Reviser model temperature.")
    ap.add_argument("--apply-temperature", type=float, default=0.05, help="Apply temperature for scoring.")
    ap.add_argument("--attempts", type=int, default=1, help="Parse attempts per pair before shape nudges.")
    ap.add_argument("--shape-attempts", type=int, default=3, help="Retries to fix shape on constrained apply.")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--evaluate-new", action="store_true", help="Immediately evaluate new pooled candidates.")
    ap.add_argument("--print-grids", action="store_true", help="Print PREDICTED vs TRUTH for new pooled candidates when evaluating.")

    args = ap.parse_args()

    tasks = list_tasks(args.root, args.split)
    if args.task_id:
        tasks = [p for p in tasks if p.stem == args.task_id]
        if not tasks:
            print(f"No task found for --task-id {args.task_id}", file=sys.stderr)
            sys.exit(2)
    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]

    apply_client = OllamaClient()
    reviser_client = OllamaClient()

    pooled_dir, pooled_jsonl = ensure_pooled_dirs(args.pooled_out, args.split)
    eval_pooled_dir_base = Path("eval_outputs_pooled")

    print(f"Pooled revision for {len(tasks)} task(s) | top_k={args.top_k} num_new={args.num_new} reviser={args.reviser_model}")

    for ti, task_path in enumerate(tasks, 1):
        task_id = task_path.stem
        task = load_arc_task(task_path)

        base_bundle = load_base_candidates(args.outputs, args.split, task_id)
        if not base_bundle or not base_bundle.get("candidates"):
            print(f"[{ti}/{len(tasks)}] {task_id}: no base candidates in outputs/ — skipping")
            continue

        # Collect each candidate's latest evaluation; normalize structure
        enriched: List[Dict[str, Any]] = []
        for c in base_bundle["candidates"]:
            cid = int(c.get("candidate_id", 0))
            latest = load_latest_round(args.revisions, args.split, task_id, cid)
            if latest:
                # Prefer revised text if present
                instr_text = (c.get("nl_description") or latest.get("revised_instructions") or "").strip()
                eval_field = latest.get("evaluation")
                if eval_field is not None:
                    eval_norm = normalize_eval_struct(eval_field)
                    enriched.append({
                        "candidate_id": cid,
                        "nl_description": instr_text,
                        "evaluation": eval_norm,
                    })
                    continue
            # If no latest revision OR no eval inside: quick-evaluate the base instructions so we can rank and show per-pair lines
            instr_text = (c.get("nl_description") or "").strip()
            if not instr_text:
                continue
            eval_res = evaluate_instructions(
                apply_client, args.apply_model, task_id, task, instr_text,
                temperature=args.apply_temperature, attempts=args.attempts, parallel=args.parallel,
                shape_attempts=args.shape_attempts
            )
            enriched.append({
                "candidate_id": cid,
                "nl_description": instr_text,
                "evaluation": eval_res,  # already in normalized shape
            })

        if not enriched:
            print(f"[{ti}/{len(tasks)}] {task_id}: nothing to rank — skipping")
            continue

        ranked = rank_candidates(enriched)
        top_cands = ranked[:args.top_k]

        # Build EXACT screenshot-style pooled prompt
        pooled_prompt = build_pooled_prompt_screenshot_style(task_id, task, top_cands)

        # Generate N new instruction sets from the pooled prompt
        new_records: List[Dict[str, Any]] = []
        for k in range(args.num_new):
            seed_k = k + 17
            temp_k = args.temperature
            text = None
            err = None
            try:
                text = reviser_client.generate(model=args.reviser_model, prompt=pooled_prompt,
                                               temperature=temp_k, seed=seed_k)
            except Exception as e:
                err = str(e)
            ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            rec = {
                "pooled_id": k,
                "ts": ts,
                "task_id": task_id,
                "model": args.reviser_model,
                "temperature": temp_k,
                "seed": seed_k,
                "nl_description": (text or ""),
                "error": err,
                "source_top_k": [c["candidate_id"] for c in top_cands],
                "pooled_prompt_len": len(pooled_prompt),
            }
            new_records.append(rec)
            write_jsonl(pooled_jsonl, rec)

        # Save pooled bundle for the task
        save_task_bundle(pooled_dir / f"{task_id}.json", {
            "task_id": task_id,
            "split": args.split,
            "source_top_k": [c["candidate_id"] for c in top_cands],
            "pooled_prompt": pooled_prompt,
            "num_new": args.num_new,
            "candidates": new_records,
        })
        print(f"[{ti}/{len(tasks)}] {task_id}: wrote {len(new_records)} pooled candidates")

        # Optional: evaluate new pooled candidates immediately
        if args.evaluate_new:
            eval_dir = ensure_eval_dirs(eval_pooled_dir_base, args.split, task_id)
            print(f"  Evaluating pooled candidates with apply_model={args.apply_model} …")
            for rec in new_records:
                instr = (rec.get("nl_description") or "").strip()
                if not instr:
                    print(f"    - pooled {rec['pooled_id']}: empty text; skipping eval")
                    continue
                eval_res = evaluate_instructions(
                    apply_client, args.apply_model, task_id, task, instr,
                    temperature=args.apply_temperature, attempts=args.attempts, parallel=args.parallel,
                    shape_attempts=args.shape_attempts
                )
                # save one file per pooled candidate
                with open(eval_dir / f"pooled_cand_{rec['pooled_id']}.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "task_id": task_id,
                        "pooled_id": rec["pooled_id"],
                        "instructions": instr,
                        "evaluation": eval_res,
                    }, f, ensure_ascii=False, indent=2)

                s = eval_res["summary"]
                print(f"    - pooled {rec['pooled_id']}: primary={s['primary_perfect']} ({s['primary_pct']*100:.1f}%) "
                      f"secondary={s['secondary_cells']}/{s['secondary_possible']} ({s['secondary_pct']*100:.1f}%) "
                      f"parse={s['parse_rate']*100:.0f}% size_ok={s['size_ok_rate']*100:.0f}%")

                if args.print_grids:
                    train_pairs = task.get("train", [])
                    print(f"      --- pooled {rec['pooled_id']} — Predicted vs Truth per pair ---")
                    for pr in eval_res["pairs"]:
                        i = pr["index"]
                        truth = train_pairs[i]["output"]
                        pred  = pr.get("pred")
                        th, tw = grid_shape(truth)
                        ph, pw = grid_shape(pred) if pred else (0, 0)
                        acc = pr.get("accuracy", 0.0)
                        corr, tot = pr.get("correct_cells", 0), pr.get("total_cells", th*tw)
                        print(f"        pair {i}: pred_shape={ph}x{pw} truth_shape={th}x{tw} "
                              f"size_ok={pr.get('size_ok', False)} exact={pr.get('exact', False)} "
                              f"acc={acc*100:.1f}% ({corr}/{tot})")
                        print("        PREDICTED:")
                        print("          (parse failure)" if pred is None else "          " + "\n          ".join(grid_to_ascii(pred).splitlines()))
                        print("        TRUTH:")
                        print("          " + "\n          ".join(grid_to_ascii(truth).splitlines()))
                    print("")

    print("Done.")

if __name__ == "__main__":
    main()
