#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC-AGI NL — End-to-End Pipeline
================================

Runs the full flow in one command:

  (A) Candidate generation (NL describer)
  (B) Individual revisions + evaluation
  (C) Pooled revision + evaluation (if no perfect solutions)
  (D) Side-by-side summary (individual vs pooled)

This script shells out to your existing tools:
  - arc_nl_describer.py            (your shape-validated candidate generator)
  - arc_nl_reviser_all.py          (individual revision engine)
  - arc_nl_pooled_reviser.py       (pooled revision phase)
  - arc_compare_revisions.py       (side-by-side accuracy summary)

It also inspects revisions/ JSON to decide whether to proceed to pooled revision.

Example (fast local models):
----------------------------
python3 arc_nl_pipeline.py \
  --root /Users/abi/Documents/dual-arc/ARC-AGI-2/data \
  --split training \
  --task-id 00576224 \
  --num-candidates 10 \
  --rounds 2 \
  --describer-model qwen2.5:1.5b-instruct \
  --reviser-model qwen2.5:1.5b-instruct \
  --apply-model qwen2.5:1.5b-instruct \
  --top-k 5 --num-new 5

If you prefer higher-quality reviser for pooled:
  --reviser-model qwen2.5-coder:7b
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- ANSI ----------
RESET = "\x1b[0m"; BOLD = "\x1b[1m"; DIM = "\x1b[2m"
GREEN = "\x1b[32m"; YEL = "\x1b[33m"; RED = "\x1b[31m"; CYAN = "\x1b[36m"; GREY = "\x1b[90m"

def c(s: str, col: str) -> str:
    return f"{col}{s}{RESET}"

# ---------- file helpers ----------
def normalize_eval(e: Dict[str, Any]) -> Dict[str, Any]:
    """Accept either {'summary': {...}, 'pairs': [...]} or flat summary dict."""
    if not e:
        return {"summary": {}, "pairs": []}
    if "summary" in e:
        return {"summary": e.get("summary", {}) or {}, "pairs": e.get("pairs", []) or []}
    # flat
    flat = dict(e)
    pairs = flat.pop("pairs", []) if isinstance(flat.get("pairs"), list) else []
    return {"summary": flat, "pairs": pairs if isinstance(pairs, list) else []}

def load_latest_revision_objects(revisions_root: Path, split: str, task_id: str) -> List[Dict[str, Any]]:
    """Pick the latest round for each candidate: revisions/<split>/<task_id>/cand_{id}_round*.json"""
    out: List[Dict[str, Any]] = []
    tdir = revisions_root / split / task_id
    if not tdir.exists():
        return out
    # Map cand_id -> latest path
    latest: Dict[int, Path] = {}
    for p in sorted(tdir.glob("cand_*_round*.json"), key=lambda q: q.stem):
        stem = p.stem  # cand_{id}_round{n}
        try:
            cid = int(stem.split("_")[1])
            latest[cid] = p  # sorted => later rounds replace earlier
        except Exception:
            continue
    for cid, path in sorted(latest.items()):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            obj["_cand_id"] = cid
            out.append(obj)
        except Exception:
            pass
    return out

def has_perfect_solution(revisions_root: Path, split: str, task_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    revs = load_latest_revision_objects(revisions_root, split, task_id)
    for obj in revs:
        ev = normalize_eval(obj.get("evaluation", {}))
        s = ev["summary"]
        n = int(s.get("num_pairs", 0) or 0)
        pp = int(s.get("primary_perfect", 0) or 0)
        if n > 0 and pp == n:
            return True, obj
    return False, None

def ensure_file_exists(path: Path, nice_name: str):
    if not path.exists():
        print(c(f"[ERROR] Missing {nice_name}: {path}", RED), file=sys.stderr)
        sys.exit(2)

# ---------- shell helper ----------
def run(cmd: List[str]) -> int:
    print(c("➤ " + " ".join(cmd), CYAN))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(c(f"[ERROR] Command failed with exit code {proc.returncode}", RED), file=sys.stderr)
    return proc.returncode

# ---------- main pipeline ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", type=str, choices=["training", "evaluation"], default="training")
    ap.add_argument("--task-id", type=str, required=True)

    # Phase A: describer
    ap.add_argument("--num-candidates", type=int, default=10)
    ap.add_argument("--describer-model", type=str, default="qwen2.5-coder:7b")
    ap.add_argument("--describer-temp", type=float, default=0.2)
    ap.add_argument("--describer-shape-attempts", type=int, default=4)
    ap.add_argument("--describer-candidate-tries", type=int, default=6)
    ap.add_argument("--describer-print-shape", action="store_true")
    ap.add_argument("--overwrite-describer", action="store_true")

    # Phase B: individual reviser
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--reviser-model", type=str, default="qwen2.5-coder:7b")
    ap.add_argument("--apply-model", type=str, default="qwen2.5:1.5b-instruct")
    ap.add_argument("--apply-temp", type=float, default=0.05)
    ap.add_argument("--shape-attempts", type=int, default=3)
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--print-grids", action="store_true")
    ap.add_argument("--strict-shape", action="store_true")

    # Phase C: pooled
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--num-new", type=int, default=5)
    ap.add_argument("--pooled-reviser-model", type=str, default="qwen2.5-coder:7b")
    ap.add_argument("--pooled-apply-model", type=str, default="qwen2.5:1.5b-instruct")
    ap.add_argument("--pooled-reviser-temp", type=float, default=0.2)
    ap.add_argument("--pooled-apply-temp", type=float, default=0.05)

    # Paths
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--revisions", type=Path, default=Path("revisions"))
    ap.add_argument("--pooled-out", type=Path, default=Path("pooled"))
    ap.add_argument("--pooled-evals", type=Path, default=Path("eval_outputs_pooled"))

    args = ap.parse_args()

    # --------- sanity: scripts present ----------
    here = Path(__file__).parent.resolve()
    describer_py = here / "arc_nl_describer.py"      # your shape-validated generator
    reviser_all_py = here / "arc_nl_reviser_all.py"  # your individual reviser (with constrained apply)
    pooled_py = here / "arc_nl_pooled_reviser.py"    # pooled reviser (screenshot-style prompt)
    compare_py = here / "arc_compare_revisions.py"   # side-by-side summary

    ensure_file_exists(describer_py, "describer")
    ensure_file_exists(reviser_all_py, "reviser_all")
    ensure_file_exists(pooled_py, "pooled_reviser")
    ensure_file_exists(compare_py, "compare script")

    # --------- Phase A: Generate candidates (shape-validated) ----------
    print(c("\n[A] Generating shape-validated candidates", BOLD))
    cmdA = [
        sys.executable, str(describer_py),
        "--root", str(args.root),
        "--split", args.split,
        "--task-id", args.task-id if False else args.task_id,  # just to highlight param name similar to your scripts
        "--model", args.describer_model,
        "--temperature", str(args.describer_temp),
        "--num-candidates", str(args.num_candidates),
        "--shape-attempts", str(args.describer_shape_attempts),
        "--candidate-tries", str(args.describer_candidate_tries),
    ]
    if args.describer_print_shape:
        cmdA.append("--print-shape-check")
    if args.overwrite_describer:
        cmdA.append("--overwrite")
    rc = run(cmdA)
    if rc != 0:
        sys.exit(rc)

    # --------- Phase B: Individual revisions ----------
    print(c("\n[B] Individual revisions + evaluation", BOLD))
    cmdB = [
        sys.executable, str(reviser_all_py),
        "--root", str(args.root),
        "--split", args.split,
        "--task-id", args.task_id,

        # IMPORTANT: map our pipeline flags to reviser script flags
        "--reviser-model", args.reviser_model,      # model that rewrites instructions
        "--model", args.apply_model,                 # model that applies instructions to grids

        "--revise-temperature", str(args.describer_temp if False else args.describer_temp),  # or set your own reviser temp
        "--temperature", str(args.apply_temp),       # temperature for APPLY/execution

        "--rounds", str(args.rounds),
        "--attempts", str(args.attempts),
        "--parallel", str(args.parallel),
        "--shape-attempts", str(args.shape_attempts),
    ]
    if args.print_grids:
        cmdB.append("--print-grids")
    if args.strict_shape:
        cmdB.append("--strict-shape")

    rc = run(cmdB)
    if rc != 0:
        sys.exit(rc)

    # --------- Check for perfect solution ----------
    print(c("\n[✓] Checking for perfect solutions after individual revisions…", CYAN))
    perfect, best_obj = has_perfect_solution(args.revisions, args.split, args.task_id)
    if perfect:
        print(c("Found a PERFECT solution in individual revisions — skipping pooled phase.", GREEN))
    else:
        print(c("No perfect solutions found — proceeding to pooled revision.", YEL))

        # ----- Phase C: Pooled revision -----
        print(c("\n[C] Pooled revision (top-K merge → new candidates → evaluate)", BOLD))
        cmdC = [
            sys.executable, str(pooled_py),
            "--root", str(args.root),
            "--split", args.split,
            "--task-id", args.task_id,
            "--reviser-model", args.pooled_reviser_model,
            "--apply-model", args.pooled_apply_model,
            "--top-k", str(args.top_k),
            "--num-new", str(args.num_new),
            "--temperature", str(args.pooled_reviser_temp),
            "--apply-temperature", str(args.pooled_apply_temp),
            "--evaluate-new",
        ]
        if args.print_grids:
            cmdC.append("--print-grids")
        rc = run(cmdC)
        if rc != 0:
            sys.exit(rc)

    # --------- Phase D: Side-by-side summary ----------
    print(c("\n[D] Individual vs Pooled — side-by-side summary", BOLD))
    cmdD = [
        sys.executable, str(compare_py),
        "--split", args.split,
        "--task-id", args.task_id,
        "--revisions", str(args.revisions),
        "--pooled-evals", str(args.pooled_evals),
    ]
    run(cmdD)

    print(c("\nDone.", GREEN))


if __name__ == "__main__":
    main()
