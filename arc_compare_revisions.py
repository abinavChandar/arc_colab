#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC-AGI NL — Individual vs Pooled Revision: Side-by-Side Comparison

Reads:
- Individual revision results:   revisions/<split>/<task_id>/cand_{ID}_round*.json
- Pooled revision eval results:  eval_outputs_pooled/<split>/<task_id>/pooled_cand_{ID}.json
  (created by arc_nl_pooled_reviser.py when run with --evaluate-new)

Outputs a colorized table comparing:
  * Best (primary DESC, then secondary_pct DESC)
  * Average (across candidates)

If pooled eval files are missing, the pooled columns will show "N/A".
(You can re-run pooled reviser with --evaluate-new to produce them.)

Example:
python3 arc_compare_revisions.py \
  --split training \
  --task-id 00576224
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- ANSI helpers ----------
RESET = "\x1b[0m"
BOLD  = "\x1b[1m"
DIM   = "\x1b[2m"
GREEN = "\x1b[32m"
YEL   = "\x1b[33m"
RED   = "\x1b[31m"
CYAN  = "\x1b[36m"
GREY  = "\x1b[90m"

def pct_color(p: float) -> str:
    if p >= 0.90: return GREEN
    if p >= 0.50: return YEL
    return RED

def fmt_pct(p: Optional[float]) -> str:
    if p is None: return GREY + "N/A" + RESET
    return f"{p*100:.1f}%"

def fmt_rate(p: Optional[float]) -> str:
    return fmt_pct(p)

def cfmt(label: str, color: str) -> str:
    return f"{color}{label}{RESET}"

# ---------- normalization ----------
def normalize_eval(e: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either:
      A) {'summary': {...}, 'pairs': [...]}
      B) {...flat summary fields...}
    Return {'summary': {...}, 'pairs': list}
    """
    if e is None:
        return {"summary": {}, "pairs": []}
    if "summary" in e:
        summ = e.get("summary", {}) or {}
        pairs = e.get("pairs", []) or []
        return {"summary": summ, "pairs": pairs}
    summ = dict(e)
    pairs = summ.pop("pairs", []) if isinstance(summ.get("pairs"), list) else []
    return {"summary": summ, "pairs": pairs if isinstance(pairs, list) else []}

# ---------- loaders ----------
def load_individual_revisions(rev_root: Path, split: str, task_id: str) -> List[Dict[str, Any]]:
    """Return list of latest-round JSON objects for each candidate."""
    task_dir = rev_root / split / task_id
    if not task_dir.exists():
        return []
    # group by cand id
    files = sorted(task_dir.glob("cand_*_round*.json"), key=lambda p: p.stem)
    latest_by_cid: Dict[int, Path] = {}
    for p in files:
        # name: cand_{id}_round{n}.json
        stem = p.stem
        try:
            cid = int(stem.split("_")[1])
            # Always overwrite; sorted ensures later rounds replace earlier
            latest_by_cid[cid] = p
        except Exception:
            continue
    out = []
    for cid, p in sorted(latest_by_cid.items()):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            obj["_cand_id"] = cid
            out.append(obj)
        except Exception:
            pass
    return out

def load_pooled_evals(eval_root: Path, split: str, task_id: str) -> List[Dict[str, Any]]:
    """Return list of eval JSON objects saved by pooled reviser (--evaluate-new)."""
    task_dir = eval_root / split / task_id
    if not task_dir.exists():
        return []
    paths = sorted(task_dir.glob("pooled_cand_*.json"), key=lambda p: p.stem)
    out = []
    for p in paths:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            # store pooled_id for display
            stem = p.stem
            pid = int(stem.split("_")[-1])
            obj["_pooled_id"] = pid
            out.append(obj)
        except Exception:
            pass
    return out

# ---------- metrics helpers ----------
def pick_best(evals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not evals:
        return None
    def keyfn(o: Dict[str, Any]):
        ev = normalize_eval(o.get("evaluation", {}))
        s = ev["summary"]
        return (
            -int(s.get("primary_perfect", 0)),
            -float(s.get("secondary_pct", 0.0)),
            -float(s.get("parse_rate", 0.0)),
        )
    return sorted(evals, key=keyfn)[0]

def average_summary(evals: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not evals:
        return None
    prims: List[float] = []
    prim_den: List[int] = []
    secs: List[float] = []
    parses: List[float] = []
    sizes: List[float] = []
    for o in evals:
        ev = normalize_eval(o.get("evaluation", {}))
        s = ev["summary"]
        # primary as fraction of pairs
        n = int(s.get("num_pairs", 0) or 0)
        pp = int(s.get("primary_perfect", 0) or 0)
        prims.append((pp / n) if n else 0.0)
        prim_den.append(n)
        secs.append(float(s.get("secondary_pct", 0.0)))
        parses.append(float(s.get("parse_rate", 0.0)))
        sizes.append(float(s.get("size_ok_rate", 0.0)))
    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0
    return {
        "primary_pct": avg(prims),
        "secondary_pct": avg(secs),
        "parse_rate": avg(parses),
        "size_ok_rate": avg(sizes),
    }

def extract_summary_row(label: str, obj: Dict[str, Any]) -> Tuple[str, float, float, float, float]:
    ev = normalize_eval(obj.get("evaluation", {}))
    s = ev["summary"]
    primary_pct = (int(s.get("primary_perfect", 0)) / int(s.get("num_pairs", 1) or 1))
    return (
        label,
        float(primary_pct),
        float(s.get("secondary_pct", 0.0)),
        float(s.get("parse_rate", 0.0)),
        float(s.get("size_ok_rate", 0.0)),
    )

# ---------- table printing ----------
def print_header(task_id: str):
    print(f"{BOLD}Task {task_id}{RESET}")
    print("-" * 72)
    print(f"{BOLD}Source{' ' * 11}Primary    Secondary   Parse      Size OK{RESET}")
    print(f"{DIM}{'-'*72}{RESET}")

def print_row(name: str, primary: Optional[float], secondary: Optional[float],
              parse: Optional[float], size_ok: Optional[float]):
    pc = pct_color(primary or 0.0) if primary is not None else GREY
    sc = pct_color(secondary or 0.0) if secondary is not None else GREY
    rc = pct_color(parse or 0.0) if parse is not None else GREY
    zc = pct_color(size_ok or 0.0) if size_ok is not None else GREY
    print(
        f"{name:<16}"
        f"{cfmt(fmt_pct(primary), pc):>10}  "
        f"{cfmt(fmt_pct(secondary), sc):>10}  "
        f"{cfmt(fmt_rate(parse), rc):>8}  "
        f"{cfmt(fmt_rate(size_ok), zc):>8}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, choices=["training", "evaluation"], default="training")
    ap.add_argument("--task-id", type=str, required=True)
    ap.add_argument("--revisions", type=Path, default=Path("revisions"))
    ap.add_argument("--pooled-evals", type=Path, default=Path("eval_outputs_pooled"))
    args = ap.parse_args()

    # Load individual (latest round per candidate)
    indiv = load_individual_revisions(args.revisions, args.split, args.task_id)

    # Load pooled (eval files produced by pooled reviser)
    pooled = load_pooled_evals(args.pooled_evals, args.split, args.task_id)

    # Build rows
    print_header(args.task_id)

    # Individual — Best / Average
    if indiv:
        best_i = pick_best(indiv)
        avg_i  = average_summary(indiv)
        if best_i:
            name, p, s, r, z = extract_summary_row("Individual Best", best_i)
            print_row(name, p, s, r, z)
        if avg_i:
            print_row("Individual Avg", avg_i["primary_pct"], avg_i["secondary_pct"],
                      avg_i["parse_rate"], avg_i["size_ok_rate"])
    else:
        print_row("Individual Best", None, None, None, None)
        print_row("Individual Avg",  None, None, None, None)

    # Pooled — Best / Average
    if pooled:
        best_p = pick_best(pooled)
        avg_p  = average_summary(pooled)
        if best_p:
            name, p, s, r, z = extract_summary_row("Pooled Best", best_p)
            print_row(name, p, s, r, z)
        if avg_p:
            print_row("Pooled Avg", avg_p["primary_pct"], avg_p["secondary_pct"],
                      avg_p["parse_rate"], avg_p["size_ok_rate"])
    else:
        print_row("Pooled Best", None, None, None, None)
        print_row("Pooled Avg",  None, None, None, None)

    print(f"{DIM}{'-'*72}{RESET}")
    note = "Tip: run pooled reviser with --evaluate-new to populate eval_outputs_pooled/."
    print(f"{GREY}{note}{RESET}")

if __name__ == "__main__":
    main()
