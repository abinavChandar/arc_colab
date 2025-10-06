#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC-AGI Natural-Language Task Describer (multi-candidate, SHAPE-VALIDATED)
vLLM-ready version

Generates *multiple* plain-English instruction candidates per task using an
OpenAI-compatible endpoint (e.g., vLLM). After generating a candidate, we APPLY
those instructions to each training input and ENFORCE the ground-truth OUTPUT
SHAPE. If any pair fails to match the exact shape (even after corrective
nudges), the candidate is rejected and we regenerate a new one (up to
--candidate-tries per candidate slot).

Environment (picked up by the compat client):
  OPENAI_BASE_URL (default http://127.0.0.1:8000/v1)
  OPENAI_API_KEY  (value content ignored by vLLM but header required)

Outputs:
  - ./outputs/<split>/<task_id>.json            (per-task bundle of shape-valid candidates)
  - ./outputs/<split>/nl_descriptions.jsonl     (one line per accepted candidate)

Example:
  python3 arc_nl_describer.py \
    --root /path/to/ARC-AGI-2/data \
    --split training \
    --max-tasks 25 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-candidates 10

Flags:
  --apply-model            Use a different model for APPLY checks (default: same as --model)
  --apply-temperature      Temperature for APPLY (default: 0.1)
  --shape-attempts         Corrective retries to hit target shape per pair (default: 4)
  --candidate-tries        Max regenerations per candidate slot until one passes shape (default: 6)
  --print-shape-check      Print per-pair pred vs truth shapes during validation (default: off)
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# IMPORTANT: use the vLLM/OpenAI-compatible client shim
# (you already have/added this file as per earlier instructions)
from vllm_compat import SyncClient as OllamaClient  # noqa: N813 (we intentionally alias)

# -------------------------
# ARC I/O helpers
# -------------------------

def load_arc_task(task_path: Path) -> Dict[str, Any]:
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_tasks(root: Path, split: str) -> List[Path]:
    pattern = str(root / split / "*.json")
    paths = [Path(p) for p in glob.glob(pattern)]
    paths.sort()
    return paths

def grid_to_ascii(grid: List[List[int]]) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in grid)

def grid_shape(grid: List[List[int]]) -> Tuple[int, int]:
    return len(grid), (len(grid[0]) if grid else 0)

def format_training_examples_for_prompt(task: Dict[str, Any]) -> str:
    lines: List[str] = []
    train_pairs = task.get("train", [])
    for i, pair in enumerate(train_pairs):
        inp = pair["input"]; out = pair["output"]
        ih, iw = grid_shape(inp); oh, ow = grid_shape(out)
        lines.append(f"Training Example {i+1}")
        lines.append(f"Input (H={ih}, W={iw}):")
        lines.append(grid_to_ascii(inp))
        lines.append("Output:")
        lines.append(grid_to_ascii(out))
        lines.append("")
    return "\n".join(lines).strip()

def observed_shape_mapping(task: Dict[str, Any]) -> str:
    """A tiny shape map to steer the model toward explicit resizing rules."""
    lines = ["--Observed Shape Mapping--"]
    ok = True
    sh: Optional[int] = None
    sw: Optional[int] = None
    for i, pair in enumerate(task.get("train", [])):
        ih, iw = grid_shape(pair["input"])
        oh, ow = grid_shape(pair["output"])
        lines.append(f"Example {i+1}: {ih}x{iw} → {oh}x{ow}  (ratios: H×{oh}/{ih}, W×{ow}/{iw})")
        if ih == 0 or iw == 0 or oh % ih != 0 or ow % iw != 0:
            ok = False
        else:
            rh = oh // ih; rw = ow // iw
            sh = rh if sh is None else (sh if sh == rh else None)
            sw = rw if sw is None else (sw if sw == rw else None)
            if sh is None or sw is None:
                ok = False
    if ok and sh is not None and sw is not None:
        lines.append(f"Inferred consistent tiling factor: {sh}×{sw} (rows×cols)")
    lines.append("--End of Observed Shape Mapping--")
    return "\n".join(lines)

# -------------------------
# Prompts (Describer & Apply)
# -------------------------

SYSTEM_GUIDE = (
    "You are an expert ARC (Abstraction & Reasoning Corpus) analyst. "
    "Your job is to infer a GENERAL transformation rule that maps each input grid to its output grid. "
    "Explain the rule in clear, plain English. DO NOT write code. "
    "Describe patterns, numbers (0-9 colors), shapes, repetition/tiling, symmetry, copying, rotation/reflection, "
    "painting/masking, and connected components as needed. "
    "Generalize across ALL examples; be concise and unambiguous."
)

OUTPUT_FORMAT_GUIDE = (
    "Return ONLY:\n"
    "1) A one-sentence rule summary.\n"
    "2) A numbered, step-by-step procedure to transform ANY new input for this task.\n"
    "3) Brief notes for edge-cases/ambiguities."
)

def build_describer_prompt(task_id: str, task: Dict[str, Any]) -> str:
    examples = format_training_examples_for_prompt(task)
    shape_map = observed_shape_mapping(task)
    # Make the size requirement explicit, but without leaking exact target sizes outside the examples.
    return (
        f"{SYSTEM_GUIDE}\n\n"
        f"Task: {task_id}\n"
        "You will see training input→output pairs. Infer the single transformation rule that explains ALL pairs.\n\n"
        f"{examples}\n\n"
        f"{shape_map}\n\n"
        "IMPORTANT:\n"
        "- If outputs differ in size from inputs, you MUST specify the exact resizing/tiling rule and the formula for the new size "
        "(e.g., H_out = k * H_in, W_out = k * W_in, with the correct k's).\n"
        "- When repeating/tiling, preserve the row/column alternation patterns visible in the outputs (avoid grouping entire blocks by value).\n"
        "- Be clear, general, and avoid mentioning specific example indices.\n\n"
        f"{OUTPUT_FORMAT_GUIDE}\n"
    )

APPLY_SYSTEM = (
    "You are an ARC grid transformer. APPLY the given instructions to the INPUT grid.\n"
    "Return ONLY the OUTPUT grid as JSON like [[0,1],[2,3]] — integers 0–9, no prose, no code fences.\n"
    "If the rule resizes, compute the new dimensions accordingly; if not, preserve size."
)

def build_apply_prompt(task_id: str,
                       instructions: str,
                       input_grid: List[List[int]],
                       target_shape: Tuple[int, int] | None) -> str:
    tgt = ""
    if target_shape:
        th, tw = target_shape
        tgt = (
            "\nREQUIREMENT: The OUTPUT grid MUST have this exact shape derived from the training rule: "
            f"H_out={th}, W_out={tw}. If your earlier idea implied a different size, correct it to match this shape.\n"
        )
    return (
        f"Task: {task_id}\n"
        "Instructions (natural language):\n"
        f"{instructions}\n\n"
        f"INPUT grid (H={len(input_grid)}, W={len(input_grid[0]) if input_grid else 0}):\n"
        f"{json.dumps(input_grid, separators=(',', ':'))}\n"
        f"{tgt}\n"
        f"{APPLY_SYSTEM}\n"
    )

# -------------------------
# Parsing helpers
# -------------------------

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

def try_extract_json_grid(text: str) -> Optional[List[List[int]]]:
    # Direct parse
    try:
        obj = json.loads(text)
        if is_valid_grid(obj):
            return obj
    except Exception:
        pass
    # Fenced/noisy
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
    return None

# -------------------------
# Shape validation (apply)
# -------------------------

def apply_with_shape(client: OllamaClient,
                     model: str,
                     task_id: str,
                     instructions: str,
                     inp: List[List[int]],
                     target_shape: Tuple[int, int],
                     temperature: float,
                     parse_attempts: int = 1,
                     shape_attempts: int = 4) -> Tuple[Optional[List[List[int]]], str]:
    """
    Apply 'instructions' to 'inp' and force the output shape to equal target_shape.
    Returns (grid_or_None, note). We keep nudging until shape matches or shape_attempts exhausted.
    """
    th, tw = target_shape
    prompt = build_apply_prompt(task_id, instructions, inp, target_shape=(th, tw))
    pred: Optional[List[List[int]]] = None
    note = ""
    err: Optional[str] = None

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
            time.sleep(0.25 * (k + 1))

    # shape loop
    if pred is not None:
        Hp, Wp = grid_shape(pred)
        tries = 0
        while (Hp, Wp) != (th, tw) and tries < max(1, shape_attempts):
            tries += 1
            nudge = (
                "\nYour previous output had the wrong shape "
                f"({Hp}×{Wp}). Produce a JSON grid with EXACT shape {th}×{tw}. "
                "Use only integers 0–9; no prose."
            )
            try:
                raw2 = client.generate(model=model, prompt=prompt + nudge, temperature=temperature)
                g2 = try_extract_json_grid(raw2)
                if g2 is not None:
                    pred = g2
                    Hp, Wp = grid_shape(pred)
                    continue
            except Exception as e:
                err = str(e)
            time.sleep(0.15 * tries)

        if (Hp, Wp) != (th, tw):
            note = f"shape_mismatch_after_retries: got {Hp}x{Wp}, want {th}x{tw}"

    if pred is None and err:
        note = f"parse_error: {err}"
    return pred, note

def candidate_passes_shape(client: OllamaClient,
                           model: str,
                           task_id: str,
                           task: Dict[str, Any],
                           instructions: str,
                           temperature: float,
                           shape_attempts: int,
                           print_checks: bool = False) -> bool:
    """
    Returns True iff for *all* training pairs we can get an output grid in the exact ground-truth shape.
    """
    ok_all = True
    for i, pair in enumerate(task.get("train", [])):
        truth = pair["output"]; th, tw = grid_shape(truth)
        pred, note = apply_with_shape(
            client=client,
            model=model,
            task_id=task_id,
            instructions=instructions,
            inp=pair["input"],
            target_shape=(th, tw),
            temperature=temperature,
            parse_attempts=1,
            shape_attempts=shape_attempts,
        )
        ph, pw = grid_shape(pred) if pred is not None else (0, 0)
        size_ok = (ph == th and pw == tw)
        if print_checks:
            print(f"      [shape-check] pair {i}: pred_shape={ph}x{pw} truth_shape={th}x{tw} size_ok={size_ok} {('(note='+note+')' if note else '')}")
        if not size_ok:
            ok_all = False
    return ok_all

# -------------------------
# Output writers
# -------------------------

def ensure_out_dirs(base: Path, split: str) -> Tuple[Path, Path]:
    split_dir = base / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "nl_descriptions.jsonl").touch(exist_ok=True)
    return split_dir, split_dir / "nl_descriptions.jsonl"

def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def save_per_task_json(out_dir: Path, task_id: str, payload: Dict[str, Any]) -> None:
    with open(out_dir / f"{task_id}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# -------------------------
# Runner
# -------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="ARC-AGI natural-language describer (vLLM/OpenAI-compatible) — multi-candidate, shape-validated")
    p.add_argument("--root", type=Path, required=True, help="Path to ARC-AGI dataset root (contains training/ evaluation/)")
    p.add_argument("--split", type=str, choices=["training", "evaluation"], default="training")
    p.add_argument("--max-tasks", type=int, default=0, help="Optional cap on number of tasks (0 = all)")

    # Defaults updated to HF IDs that match vLLM launches
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model for NL description (describer)")
    p.add_argument("--temperature", type=float, default=0.2, help="Temperature for describer")

    p.add_argument("--apply-model", type=str, default="", help="Model for APPLY shape checks (default: same as --model)")
    p.add_argument("--apply-temperature", type=float, default=0.1, help="Temperature for APPLY")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=int, default=120, help="Request timeout seconds (handled by compat client)")
    p.add_argument("--outputs", type=Path, default=Path("outputs"))
    p.add_argument("--overwrite", action="store_true", help="Regenerate even if per-task JSON exists")
    p.add_argument("--num-candidates", type=int, default=10, help="Number of NL candidates per task")

    # Hard shape gating
    p.add_argument("--shape-attempts", type=int, default=4, help="Corrective retries to hit target shape per pair")
    p.add_argument("--candidate-tries", type=int, default=6, help="Regenerations per candidate slot until one passes shape")
    p.add_argument("--print-shape-check", action="store_true", help="Print per-pair pred vs truth shapes during validation")
    p.add_argument("--task-id", type=str, default="", help="Only run this task id (e.g. 00576224)")

    args = p.parse_args()

    random.seed(args.seed or None)

    tasks = list_tasks(args.root, args.split)

    if args.task_id:
        tasks = [p for p in tasks if p.stem == args.task_id]
        if not tasks:
            print(f"No task found matching --task-id {args.task_id}", file=sys.stderr)
            sys.exit(2)

    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        print("No tasks found. Check --root and --split.", file=sys.stderr)
        sys.exit(2)

    out_dir, jsonl_path = ensure_out_dirs(args.outputs, args.split)

    # The compat client reads OPENAI_BASE_URL / OPENAI_API_KEY from env.
    client = OllamaClient(timeout=args.timeout)

    print(f"Found {len(tasks)} tasks. Generating up to {args.num_candidates} SHAPE-VALID NL candidates per task…", flush=True)

    for idx, task_path in enumerate(tasks, 1):
        task_id = task_path.stem
        per_task_path = out_dir / f"{task_id}.json"
        if per_task_path.exists() and not args.overwrite:
            print(f"[{idx}/{len(tasks)}] {task_id}: exists, skip (use --overwrite to regenerate)")
            continue

        try:
            task = load_arc_task(task_path)
        except Exception as e:
            print(f"[{idx}/{len(tasks)}] {task_id}: ERROR loading: {e}")
            continue

        apply_model = args.apply_model or args.model
        accepted: List[Dict[str, Any]] = []

        print(f"[{idx}/{len(tasks)}] {task_id}: target candidates={args.num_candidates}")

        # Keep generating until we have num-candidates shape-valid candidates (or we hit a large global cap)
        for k in range(args.num_candidates):
            # For each slot, we may need several tries to get a shape-valid instruction set
            got = False
            for tnum in range(1, args.candidate_tries + 1):
                seed_k = (args.seed or 0) + (k * 13 + tnum)  # vary seed per try
                temp_k = max(0.0, min(1.0, args.temperature + 0.05 * ((tnum % 3) - 1)))

                # Build describer prompt anew each try
                prompt = build_describer_prompt(task_id, task)

                # Generate instructions
                response_text = None
                last_err: Optional[str] = None
                try:
                    response_text = client.generate(
                        model=args.model, prompt=prompt, temperature=temp_k, seed=seed_k
                    )
                except Exception as e:
                    last_err = str(e)

                if not response_text:
                    print(f"  - cand {k} try {tnum}/{args.candidate_tries}: describer error: {last_err or 'unknown'}")
                    continue

                # SHAPE VALIDATION: must match truth shapes on ALL training pairs
                print(f"  - cand {k} try {tnum}/{args.candidate_tries}: validating shapes…")
                ok = candidate_passes_shape(
                    client=client,
                    model=apply_model,
                    task_id=task_id,
                    task=task,
                    instructions=response_text,
                    temperature=args.apply_temperature,
                    shape_attempts=args.shape_attempts,
                    print_checks=args.print_shape_check,
                )

                if ok:
                    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    rec = {
                        "candidate_id": k,
                        "ts": ts,
                        "prompt": prompt,
                        "nl_description": response_text,
                        "error": None,
                        "model": args.model,
                        "temperature": temp_k,
                        "seed": seed_k,
                        "shape_validated": True,
                        "shape_attempts": args.shape_attempts,
                        "candidate_try_index": tnum,
                    }
                    accepted.append(rec)
                    write_jsonl(jsonl_path, {
                        "ts": ts,
                        "task_id": task_id,
                        "split": args.split,
                        "candidate_id": k,
                        "model": args.model,
                        "temperature": temp_k,
                        "seed": seed_k,
                        "nl_description": response_text,
                        "error": None,
                        "shape_validated": True,
                        "candidate_try_index": tnum,
                    })
                    print(f"    ✓ cand {k}: ACCEPTED (shape-valid on all training pairs)")
                    got = True
                    break
                else:
                    print(f"    ✗ cand {k}: rejected (failed shape on at least one training pair)")

            if not got:
                print(f"  - cand {k}: FAILED after {args.candidate_tries} tries — leaving this slot empty")

        bundle = {
            "task_id": task_id,
            "split": args.split,
            "num_candidates_requested": args.num_candidates,
            "num_candidates_accepted": len(accepted),
            "candidates": accepted,
            "note": "Only candidates that passed shape validation on ALL training pairs are saved.",
        }
        save_per_task_json(out_dir, task_id, bundle)
        print(f"[{idx}/{len(tasks)}] {task_id}: wrote {len(accepted)} shape-valid candidates")

    print("Done.")

if __name__ == "__main__":
    main()
