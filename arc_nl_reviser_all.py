#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC-AGI NL Instruction Reviser — multi-round revisions + proper secondary accuracy

• Loads NL candidates from ./outputs/<split>/<task_id>.json
• For each candidate, repeats for --rounds:
   1) Evaluate current instructions on training pairs:
        - Constrained apply (with ground-truth shape hint) → defines size_ok, exact, secondary (cell-wise)
        - Unconstrained apply (no hint) → sanity flag (size_ok_unconstrained, optional)
   2) Build a detailed revision prompt (INPUT, PREDICTED, TRUTH, ASCII diffs + shape notes) and
      ask a *reviser model* to rewrite the instructions.
   3) Replaces instructions with the revision and loops.

• Prints per-pair grids and cell-wise accuracy (percent + counts) each round.
• Saves each round’s revised instructions to:
    revisions/<split>/<task_id>/cand_<cid>_round<r>.json

Important flags:
--rounds N                    number of revision rounds (default 1)
--reviser-model               LLM used to rewrite instructions (default qwen2.5-coder:7b)
--model                       LLM used to APPLY instructions to grids (default qwen2.5:1.5b-instruct)
--strict-shape                also run an unconstrained shape check (reports size_ok_unconstrained)
--strict-affects-secondary    if set, zero secondary when unconstrained shape fails (old behavior)

Example:
python3 arc_nl_reviser_all.py \
  --root /path/to/ARC-AGI-2/data \
  --split training \
  --task-id 00576224 \
  --reviser-model qwen2.5-coder:7b \
  --model qwen2.5:1.5b-instruct \
  --rounds 3 \
  --shape-attempts 3 \
  --strict-shape \
  --print-train \
  --print-grids
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import http.client

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

# ------------------------- Candidates bundle -------------------------
def load_describer_bundle(outputs_root: Path, split: str, task_id: str) -> Optional[Dict[str, Any]]:
    p = outputs_root / split / f"{task_id}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------------- JSON parsing -------------------------
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
    # 2) fenced/noisy → prefer object with "grid"
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
    # 3) fallback: bare array
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

    def generate(self, model: str, prompt: str, temperature: float = 0.1, seed: Optional[int] = None) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "30m",
            "format": "json",  # request valid JSON when possible (ignored for plain text models)
            "options": {
                "temperature": float(temperature),
                "num_thread": 10,   # tune to your CPU
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

# ------------------------- Apply prompts -------------------------
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

def build_apply_prompt_unconstrained(task_id: str, instructions: str, input_grid: List[List[int]]) -> str:
    return (
        f"Task: {task_id}\n"
        f"Instructions (natural language):\n{instructions}\n\n"
        f"INPUT grid (H={len(input_grid)}, W={len(input_grid[0]) if input_grid else 0}):\n"
        f"{json.dumps(input_grid, separators=(',', ':'))}\n\n"
        f"{APPLY_SYSTEM}"
    )

# ------------------------- Scoring -------------------------
def cell_accuracy(pred: Optional[List[List[int]]], truth: List[List[int]]) -> Tuple[bool, bool, int, int]:
    """Returns (exact, size_ok, correct_cells, total_cells) for the given pred vs truth."""
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
    exact = (correct == total)
    return exact, True, correct, total

# ------------------------- Apply (constrained & unconstrained) -------------------------
def apply_constrained(client: OllamaClient, model: str, task_id: str,
                      instructions: str, inp: List[List[int]], truth: List[List[int]],
                      temperature: float, parse_attempts: int, shape_attempts: int) -> Tuple[Optional[List[List[int]]], str]:
    th, tw = grid_shape(truth)
    prompt = build_apply_prompt(task_id, instructions, inp, target_shape=(th, tw))
    pred, note, err = None, "", None

    # initial parse tries
    for k in range(max(1, parse_attempts)):
        try:
            raw = client.generate(model=model, prompt=prompt, temperature=temperature)
            g = try_extract_json_grid(raw)
            if g is not None:
                pred = g
                break
        except Exception as e:
            err = str(e)
            time.sleep(0.2 * (k + 1))

    # corrective nudge if parse failed
    if pred is None:
        nudge = "\nReturn ONLY a JSON object with a single key 'grid', for example: {\"grid\": [[0,1],[2,3]]}"
        try:
            raw2 = client.generate(model=model, prompt=prompt + nudge, temperature=temperature)
            g2 = try_extract_json_grid(raw2)
            if g2 is not None:
                pred = g2
        except Exception as e:
            err = str(e)

    # shape retries to force exact (th, tw)
    if pred is not None:
        Hp, Wp = grid_shape(pred)
        tries = 0
        while (Hp, Wp) != (th, tw) and tries < max(1, shape_attempts):
            tries += 1
            # NOTE: doubled braces for a literal {"grid": ...}
            nudge2 = (
                "\nYour previous output had the wrong shape "
                f"({Hp}x{Wp}). Produce a JSON object {{\"grid\": ...}} whose grid is EXACTLY {th}x{tw}. "
                "Use only integers 0–9; no prose."
            )
            try:
                raw3 = client.generate(model=model, prompt=prompt + nudge2, temperature=temperature)
                g3 = try_extract_json_grid(raw3)
                if g3 is not None:
                    pred = g3
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

def apply_unconstrained(client: OllamaClient, model: str, task_id: str,
                        instructions: str, inp: List[List[int]], temperature: float) -> Tuple[Optional[List[List[int]]], str]:
    prompt = build_apply_prompt_unconstrained(task_id, instructions, inp)
    pred, note, err = None, "", None
    for k in range(2):
        try:
            raw = client.generate(model=model, prompt=prompt, temperature=temperature)
            g = try_extract_json_grid(raw)
            if g is not None:
                pred = g
                break
        except Exception as e:
            err = str(e)
            time.sleep(0.2 * (k + 1))
    if pred is None and err:
        note = f"parse_error: {err}"
    return pred, note

# ------------------------- Evaluation -------------------------
def evaluate_instructions(client: OllamaClient, model: str, task_id: str, task: Dict[str, Any],
                          instructions: str, temperature: float, attempts: int, parallel: int,
                          shape_attempts: int, strict_shape: bool, strict_affects_secondary: bool=False) -> Dict[str, Any]:
    train_pairs = task.get("train", [])
    pair_results: List[Dict[str, Any]] = []

    primary = 0
    secondary_cells = 0
    secondary_possible = 0
    parse_success = 0
    size_ok_constrained_count = 0
    size_ok_unconstrained_count = 0

    def work(i: int, inp: List[List[int]], truth: List[List[int]]):
        nonlocal primary, secondary_cells, secondary_possible, parse_success
        nonlocal size_ok_constrained_count, size_ok_unconstrained_count

        # Constrained apply: defines size_ok & secondary
        pred_c, note_c = apply_constrained(client, model, task_id, instructions, inp, truth,
                                           temperature, attempts, shape_attempts)
        if pred_c is not None:
            parse_success += 1
        exact_c, size_ok_c, correct_c, total = cell_accuracy(pred_c, truth)
        if size_ok_c:
            size_ok_constrained_count += 1

        # Unconstrained apply: sanity only
        size_ok_u: Optional[bool] = None
        if strict_shape:
            pred_u, _ = apply_unconstrained(client, model, task_id, instructions, inp, temperature)
            th, tw = grid_shape(truth)
            ph_u, pw_u = grid_shape(pred_u) if pred_u else (0, 0)
            size_ok_u = (ph_u == th and pw_u == tw)
            if size_ok_u:
                size_ok_unconstrained_count += 1

        if exact_c:
            primary += 1

        if not (strict_shape and strict_affects_secondary and size_ok_u is False):
            secondary_cells += correct_c
            secondary_possible += total

        pair_results.append({
            "index": i,
            "pred": pred_c,  # constrained prediction (printed)
            "note": note_c,
            "exact": exact_c,
            "accuracy": (correct_c / total) if total else 0.0,
            "correct_cells": correct_c,
            "total_cells": total,
            "size_ok": size_ok_c,                       # constrained
            "size_ok_unconstrained": size_ok_u,         # separate flag
        })

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = [ex.submit(work, i, p["input"], p["output"]) for i, p in enumerate(train_pairs)]
            for f in as_completed(futs):
                f.result()
    else:
        for i, p in enumerate(train_pairs):
            work(i, p["input"], p["output"])

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
        "size_ok_rate_unconstrained": (size_ok_unconstrained_count / n) if (n and strict_shape) else None,
    }
    return {"summary": summary, "pairs": pair_results}

# ------------------------- Printing -------------------------
def print_training_pairs(task_id: str, task: Dict[str, Any]):
    print("    === Training Examples (INPUT → OUTPUT) ===")
    for i, pair in enumerate(task.get("train", [])):
        ih, iw = grid_shape(pair["input"])
        oh, ow = grid_shape(pair["output"])
        print(f"      Example {i+1}: IN({ih}x{iw}) → OUT({oh}x{ow})")
        print("      INPUT:")
        print("        " + "\n        ".join(grid_to_ascii(pair["input"]).splitlines()))
        print("      OUTPUT:")
        print("        " + "\n        ".join(grid_to_ascii(pair["output"]).splitlines()))
        print("")

def print_eval_pairs(task: Dict[str, Any], eval_pairs: List[Dict[str, Any]], round_idx: int, cand_id: int):
    train_pairs = task.get("train", [])
    print(f"    --- Candidate {cand_id} | Round {round_idx} — Predicted vs Truth per pair ---")
    for rec in eval_pairs:
        i = rec["index"]
        truth = train_pairs[i]["output"]
        pred  = rec.get("pred")
        th, tw = grid_shape(truth)
        ph, pw = grid_shape(pred) if pred else (0, 0)
        size_ok = rec.get("size_ok", False)
        size_ok_u = rec.get("size_ok_unconstrained", None)
        exact   = rec.get("exact", False)
        acc     = rec.get("accuracy", 0.0)
        corr    = rec.get("correct_cells", 0)
        tot     = rec.get("total_cells", th * tw)

        print(f"      pair {i}: pred_shape={ph}x{pw} truth_shape={th}x{tw} "
              f"size_ok={size_ok} size_ok_unconstrained={size_ok_u} "
              f"exact={exact} acc={acc*100:.1f}% ({corr}/{tot})")
        print("      PREDICTED:")
        print("        (parse failure)" if pred is None else "        " + "\n        ".join(grid_to_ascii(pred).splitlines()))
        print("      TRUTH:")
        print("        " + "\n        ".join(grid_to_ascii(truth).splitlines()))
        print("")

# ------------------------- ASCII diff for reviser prompt -------------------------
def ascii_diff(pred: Optional[List[List[int]]], truth: List[List[int]]) -> str:
    """Produces an ASCII cell-by-cell diff when shapes match; otherwise a shape note."""
    if pred is None:
        return "(no prediction parsed)"
    Hp, Wp = grid_shape(pred)
    Ht, Wt = grid_shape(truth)
    if (Hp, Wp) != (Ht, Wt):
        return f"(shape mismatch: pred {Hp}x{Wp} vs truth {Ht}x{Wt})"
    lines: List[str] = []
    for r in range(Ht):
        row_bits = []
        for c in range(Wt):
            a, b = pred[r][c], truth[r][c]
            if a == b:
                row_bits.append(f"{a}")
            else:
                row_bits.append(f"{a}->{b}")
        lines.append(" ".join(row_bits))
    return "\n".join(lines)

def build_revision_prompt(task_id: str,
                          task: Dict[str, Any],
                          prev_instructions: str,
                          eval_pairs: List[Dict[str, Any]]) -> str:
    """Detailed prompt showing INPUT, PREDICTED, TRUTH, and ASCII diffs per pair."""
    header = (
        "You are revising natural-language instructions for an ARC task.\n"
        "Goal: rewrite the instructions so that applying them to each INPUT produces the TRUTH output.\n"
        "Focus on fixing mismatches shown below (shape, tiling factors, row/col alternation, etc.).\n"
        "Return ONLY:\n"
        "1) A one-sentence rule summary.\n"
        "2) A numbered, step-by-step procedure that a human could follow to transform ANY new input for this task.\n"
        "3) Brief notes for edge-cases/ambiguities.\n"
        "Be precise about resizing rules (e.g., H_out=k_h*H_in, W_out=k_w*W_in) when applicable.\n"
    )
    prev = f"Previous Instructions:\n{prev_instructions}\n\n"

    blocks: List[str] = []
    train_pairs = task.get("train", [])
    for rec in eval_pairs:
        i = rec["index"]
        inp   = train_pairs[i]["input"]
        truth = train_pairs[i]["output"]
        pred  = rec.get("pred")
        th, tw = grid_shape(truth)
        ph, pw = grid_shape(pred) if pred else (0, 0)
        acc    = rec.get("accuracy", 0.0)
        corr   = rec.get("correct_cells", 0)
        tot    = rec.get("total_cells", th * tw)
        size_ok = rec.get("size_ok", False)

        block = [
            f"-- Training Pair {i} --",
            f"INPUT (H={len(inp)}, W={len(inp[0]) if inp else 0}):",
            grid_to_ascii(inp),
            "",
            f"PREDICTED (H={ph}, W={pw}):",
            "(parse failure)" if pred is None else grid_to_ascii(pred),
            "",
            f"TRUTH (H={th}, W={tw}):",
            grid_to_ascii(truth),
            "",
            "ASCII DIFF (cell mismatches shown as a->b):",
            ascii_diff(pred, truth),
            "",
            f"PAIR STATS: size_ok={size_ok}  cell_acc={acc*100:.1f}% ({corr}/{tot})",
            "",
        ]
        blocks.append("\n".join(block))

    footer = (
        "Rewrite the instructions to fix the observed errors across ALL pairs, without overfitting to specific indices.\n"
        "If outputs resize, state the exact size rule clearly. Avoid vague language; be operational and testable.\n"
    )
    return f"{header}{prev}" + "\n".join(blocks) + footer

# ------------------------- FS helpers -------------------------
def ensure_rev_out_dirs(base: Path, split: str, task_id: str) -> Path:
    d = base / split / task_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_round_revision(out_dir: Path, cand_id: int, round_idx: int, instructions: str, eval_summary: Dict[str, Any]) -> None:
    p = out_dir / f"cand_{cand_id}_round{round_idx}.json"
    payload = {
        "candidate_id": cand_id,
        "round": round_idx,
        "revised_instructions": instructions,
        "evaluation": eval_summary,
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", type=str, choices=["training", "evaluation"], default="training")
    ap.add_argument("--describer-outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--revisions-out", type=Path, default=Path("revisions"))

    # Models
    ap.add_argument("--reviser-model", type=str, default="qwen2.5-coder:7b",
                    help="Model used to REWRITE the instructions each round (NL).")
    ap.add_argument("--model", type=str, default="qwen2.5:1.5b-instruct",
                    help="Model used to APPLY instructions to grids (JSON).")

    # Behaviors
    ap.add_argument("--temperature", type=float, default=0.05, help="Apply temperature (keep low for JSON).")
    ap.add_argument("--revise-temperature", type=float, default=0.2, help="Reviser model temperature.")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=4, help="Workers for per-pair apply")
    ap.add_argument("--attempts", type=int, default=1, help="Parse attempts per pair (before shape retries)")
    ap.add_argument("--shape-attempts", type=int, default=3, help="Corrective retries to hit target shape (per pair).")
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--task-id", type=str, default="")
    ap.add_argument("--print-grids", action="store_true", default=True,
                    help="Print predicted vs truth per pair after each round.")
    ap.add_argument("--print-train", action="store_true", default=True,
                    help="Print all training input/output grids at task start.")
    ap.add_argument("--strict-shape", action="store_true", default=True,
                    help="Also run an unconstrained apply and report size_ok_unconstrained.")
    ap.add_argument("--strict-affects-secondary", action="store_true", default=False,
                    help="If set, unconstrained shape failure zeros secondary accuracy (old behavior).")

    args = ap.parse_args()

    apply_client   = OllamaClient()
    reviser_client = OllamaClient()

    tasks = list_tasks(args.root, args.split)
    if args.task_id:
        tasks = [p for p in tasks if p.stem == args.task_id]
        if not tasks:
            print(f"No task found matching --task-id {args.task_id}", file=sys.stderr)
            sys.exit(2)
    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    print(f"Revising ALL candidates for {len(tasks)} task{'s' if len(tasks)!=1 else ''} | rounds={args.rounds} apply_model={args.model} reviser_model={args.reviser_model}")

    for t_idx, task_path in enumerate(tasks, 1):
        task_id = task_path.stem
        task = load_arc_task(task_path)
        bundle = load_describer_bundle(args.describer_outputs, args.split, task_id)
        if not bundle or not bundle.get("candidates"):
            print(f"[{t_idx}/{len(tasks)}] {task_id}: no NL candidates found — skipping")
            continue

        cand_list = bundle["candidates"]
        out_dir = ensure_rev_out_dirs(args.revisions_out, args.split, task_id)
        print(f"[{t_idx}/{len(tasks)}] {task_id}: revising {len(cand_list)} candidates")

        if args.print_train:
            print("    === Training Examples (INPUT → OUTPUT) ===")
            for i, pair in enumerate(task.get("train", [])):
                ih, iw = grid_shape(pair["input"])
                oh, ow = grid_shape(pair["output"])
                print(f"      Example {i+1}: IN({ih}x{iw}) → OUT({oh}x{ow})")
                print("      INPUT:")
                print("        " + "\n        ".join(grid_to_ascii(pair["input"]).splitlines()))
                print("      OUTPUT:")
                print("        " + "\n        ".join(grid_to_ascii(pair["output"]).splitlines()))
                print("")

        for cand in cand_list:
            cand_id = int(cand.get("candidate_id", 0))
            instructions = (cand.get("nl_description") or "").strip()
            if not instructions:
                print(f"  - cand {cand_id}: empty instructions; skipping")
                continue

            for r in range(args.rounds):
                # 1) Evaluate current instructions
                eval_res = evaluate_instructions(
                    apply_client, args.model, task_id, task, instructions,
                    temperature=args.temperature, attempts=args.attempts, parallel=args.parallel,
                    shape_attempts=args.shape_attempts, strict_shape=args.strict_shape,
                    strict_affects_secondary=args.strict_affects_secondary
                )

                if args.print_grids:
                    print_eval_pairs(task, eval_res["pairs"], round_idx=r, cand_id=cand_id)

                s = eval_res["summary"]
                print(f"  - cand {cand_id} | round {r}: primary={s['primary_perfect']} ({s['primary_pct']*100:.1f}%) "
                      f"secondary={s['secondary_cells']}/{s['secondary_possible']} ({s['secondary_pct']*100:.1f}%) "
                      f"parse={s['parse_rate']*100:.0f}% size_ok={s['size_ok_rate']*100:.0f}% "
                      f"size_ok_unconstrained={'N/A' if s['size_ok_rate_unconstrained'] is None else f'{s['size_ok_rate_unconstrained']*100:.0f}%'}")

                # Save current round (even round 0) for traceability
                save_round_revision(out_dir, cand_id, r, instructions, s)

                # 2) If this was the last round, stop; otherwise rewrite instructions
                if r == args.rounds - 1:
                    break

                # Build revision prompt from current eval
                revision_prompt = build_revision_prompt(task_id, task, instructions, eval_res["pairs"])

                # Ask reviser model (plain text expected). We want natural language, not JSON.
                # Temporarily override "format":"json" by not relying on it for reviser instructions;
                # we reuse the client but the endpoint will ignore format for many text models anyway.
                try:
                    revised = reviser_client.generate(
                        model=args.reviser_model,
                        prompt=revision_prompt,
                        temperature=args.revise_temperature
                    )
                except Exception as e:
                    print(f"    ! reviser error (cand {cand_id}, round {r}): {e}")
                    # If reviser fails, keep previous instructions
                    revised = instructions

                # Basic sanitation: keep a bounded-length revision
                revised = (revised or "").strip()
                if not revised:
                    revised = instructions  # fallback

                # Update instructions for next round
                instructions = revised

        print(f"[{t_idx}/{len(tasks)}] {task_id}: revisions complete for candidate {cand_id}")

    print("Done.")

if __name__ == "__main__":
    main()
