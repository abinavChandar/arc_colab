#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC-AGI — Codegen from Best Instruction (prints winning instructions, full code,
OPS/DSL JSON, CONSTRAINTS JSON, and per-pair INPUT/PRED/TRUTH)

Adds constraints behavior on top of your previous code:
- Asks model for THREE fenced blocks: ```python (module)```, ```json {"ops":[...]}```, and
  a second ```json {"constraints": {...}}```.
- If JSON blocks are missing, retries once with a stricter prompt.
- If constraints are still missing, derives fallback constraints from training pairs.
- Prints OPS and CONSTRAINTS to terminal and saves to:
    generated/dsl/<task_id>_ops.json
    generated/constraints/<task_id>_constraints.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import http.client

Grid = List[List[int]]

# -------------------------
# Files & selection helpers
# -------------------------

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def has_pooled_evals(split: str, task_id: str) -> bool:
    d = Path("eval_outputs_pooled") / split / task_id
    return d.exists() and any(d.glob("*.json"))

def list_latest_individual(rev_root: Path, split: str, task_id: str) -> List[Path]:
    tdir = rev_root / split / task_id
    if not tdir.exists():
        return []
    files = sorted(tdir.glob("cand_*_round*.json"))
    latest_by_cid: Dict[int, Path] = {}
    for p in files:
        stem = p.stem  # cand_{id}_round{n}
        try:
            cid = int(stem.split("_")[1])
            prev = latest_by_cid.get(cid)
            if prev is None or p.name > prev.name:
                latest_by_cid[cid] = p
        except Exception:
            continue
    return [latest_by_cid[k] for k in sorted(latest_by_cid.keys())]

def normalize_eval(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Accept either {'summary': {...}, 'pairs': [...]} or flat summary dict."""
    if not obj:
        return {"summary": {}, "pairs": []}
    if "summary" in obj:
        return {"summary": obj.get("summary", {}) or {}, "pairs": obj.get("pairs", []) or []}
    flat = dict(obj)
    pairs = flat.pop("pairs", []) if isinstance(flat.get("pairs"), list) else []
    return {"summary": flat, "pairs": pairs if isinstance(pairs, list) else []}

@dataclass
class ScoredCandidate:
    source: str            # 'pooled' | 'individual' | 'describer'
    path: Optional[Path]
    candidate_id: int
    instructions: str
    primary_perfect: int
    num_pairs: int
    secondary_pct: float
    parse_rate: float
    size_ok_rate: float

def extract_best_from_pooled(split: str, task_id: str) -> List[ScoredCandidate]:
    base = Path("eval_outputs_pooled") / split / task_id
    out: List[ScoredCandidate] = []
    if not base.exists():
        return out
    for p in sorted(base.glob("*.json"), key=lambda q: q.stem):
        obj = load_json(p)
        instructions = obj.get("nl_description") or obj.get("instructions") or ""
        ev = normalize_eval(obj.get("evaluation", obj))
        s = ev["summary"]
        out.append(
            ScoredCandidate(
                source="pooled",
                path=p,
                candidate_id=int(obj.get("candidate_id", obj.get("_pooled_id", 0))),
                instructions=instructions.strip(),
                primary_perfect=int(s.get("primary_perfect", 0) or 0),
                num_pairs=int(s.get("num_pairs", 0) or 0),
                secondary_pct=float(s.get("secondary_pct", 0.0) or 0.0),
                parse_rate=float(s.get("parse_rate", 0.0) or 0.0),
                size_ok_rate=float(s.get("size_ok_rate", 0.0) or 0.0),
            )
        )
    return out

def extract_best_from_individual(split: str, task_id: str) -> List[ScoredCandidate]:
    roots = Path("revisions")
    paths = list_latest_individual(roots, split, task_id)
    out: List[ScoredCandidate] = []
    for p in paths:
        obj = load_json(p)
        instr = obj.get("revised_instructions") or obj.get("instructions") or obj.get("nl_description") or ""
        ev = normalize_eval(obj.get("evaluation", {}))
        s = ev["summary"]
        cid = int(obj.get("candidate_id", obj.get("_cand_id", 0)))
        out.append(
            ScoredCandidate(
                source="individual",
                path=p,
                candidate_id=cid,
                instructions=instr.strip(),
                primary_perfect=int(s.get("primary_perfect", 0) or 0),
                num_pairs=int(s.get("num_pairs", 0) or 0),
                secondary_pct=float(s.get("secondary_pct", 0.0) or 0.0),
                parse_rate=float(s.get("parse_rate", 0.0) or 0.0),
                size_ok_rate=float(s.get("size_ok_rate", 0.0) or 0.0),
            )
        )
    return out

def extract_describer_fallback(describer_outputs: Path, split: str, task_id: str) -> List[ScoredCandidate]:
    path = describer_outputs / split / f"{task_id}.json"
    if not path.exists():
        return []
    obj = load_json(path)
    out: List[ScoredCandidate] = []
    for c in obj.get("candidates", []):
        out.append(
            ScoredCandidate(
                source="describer",
                path=path,
                candidate_id=int(c.get("candidate_id", len(out))),
                instructions=(c.get("nl_description") or "").strip(),
                primary_perfect=0,
                num_pairs=0,
                secondary_pct=0.0,
                parse_rate=0.0,
                size_ok_rate=0.0,
            )
        )
    return out

def pick_best(scored: List[ScoredCandidate]) -> Optional[ScoredCandidate]:
    if not scored:
        return None
    def key(c: ScoredCandidate):
        return (
            -(c.primary_perfect),
            -(c.secondary_pct),
            -(c.parse_rate),
            -(c.size_ok_rate),
            c.candidate_id,
        )
    return sorted(scored, key=key)[0]

# -------------------------
# Minimal Ollama client for codegen
# -------------------------

class OllamaClient:
    def __init__(self, host: str = "localhost", port: int = 11434, timeout: int = 180):
        self.host = host
        self.port = port
        self.timeout = timeout

    def generate(self, model: str, prompt: str, temperature: float = 0.1, seed: int | None = None) -> str:
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
            for raw in resp.read().splitlines():
                if not raw:
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                if "error" in obj:
                    raise RuntimeError(f"Ollama error: {obj['error']}")
                if obj.get("done"):
                    break
                tok = obj.get("response", "")
                if tok:
                    chunks.append(tok)
            return "".join(chunks).strip()
        finally:
            conn.close()

# -------------------------
# Task helpers & scoring
# -------------------------

def grid_shape(g: Grid) -> Tuple[int, int]:
    return len(g), (len(g[0]) if g else 0)

def to_ascii(g: Grid) -> str:
    return "\n".join(" ".join(str(x) for x in r) for r in g)

def load_task(root: Path, split: str, task_id: str) -> Dict[str, Any]:
    return load_json(root / split / f"{task_id}.json")

def score_transform(fn, task: Dict[str, Any], print_grids: bool = False) -> Dict[str, Any]:
    pairs = task.get("train", [])
    n = len(pairs)
    primary = 0
    sec_corr = 0
    sec_tot = 0
    details = []
    for i, pair in enumerate(pairs):
        inp = pair["input"]; out = pair["output"]
        try:
            pred = fn(inp)
        except Exception:
            pred = None
        th, tw = grid_shape(out)
        ph, pw = grid_shape(pred) if pred is not None else (0, 0)
        size_ok = (ph == th and pw == tw)
        if not size_ok or pred is None:
            corr = 0
            tot = th * tw
            exact = False
        else:
            corr = sum(1 for r in range(th) for c in range(tw) if pred[r][c] == out[r][c])
            tot = th * tw
            exact = (corr == tot)
        if exact:
            primary += 1
        sec_corr += corr; sec_tot += tot
        if print_grids:
            print(f"  pair {i}: pred_shape={ph}x{pw} truth_shape={th}x{tw} "
                  f"size_ok={size_ok} exact={exact} acc={(corr/tot*100 if tot else 0):.1f}% ({corr}/{tot})")
            print("  INPUT:")
            print("    " + "\n    ".join(to_ascii(inp).splitlines()))
            print("  PRED:")
            if pred is None:
                print("    (error)")
            else:
                print("    " + "\n    ".join(to_ascii(pred).splitlines()))
            print("  TRUTH:")
            print("    " + "\n    ".join(to_ascii(out).splitlines()))
        details.append({
            "index": i, "size_ok": size_ok, "exact": exact,
            "correct_cells": corr, "total_cells": tot, "accuracy": (corr / tot) if tot else 0.0
        })
    summary = {
        "primary_perfect": primary,
        "primary_pct": (primary / n) if n else 0.0,
        "secondary_cells": sec_corr,
        "secondary_possible": sec_tot,
        "secondary_pct": (sec_corr / sec_tot) if sec_tot else 0.0,
        "num_pairs": n,
    }
    return {"summary": summary, "pairs": details}

# -------------------------
# Fenced block extraction (code / json)
# -------------------------

def extract_fenced_block(text: str, lang: str) -> Optional[str]:
    """
    Extract content from a ```<lang> ... ``` fenced block.
    Returns None if not found.
    """
    t = text
    needle = f"```{lang}"
    start = t.find(needle)
    if start != -1:
        start = start + len(needle)
        end = t.find("```", start)
        if end != -1:
            return t[start:end].strip()
    return None

def extract_first_fenced_block(text: str) -> Optional[str]:
    t = text
    first = t.find("```")
    if first == -1:
        return None
    end = t.find("```", first + 3)
    if end == -1:
        return None
    return t[first+3:end].strip()

def extract_python_block(text: str) -> str:
    code = extract_fenced_block(text, "python")
    if code:
        return code
    # fallback: first fenced block
    fb = extract_first_fenced_block(text)
    return fb if fb else text.strip()

def extract_json_ops(text: str) -> Optional[Dict[str, Any]]:
    js = extract_fenced_block(text, "json")
    if js is None:
        fb = extract_first_fenced_block(text)
        if fb and fb.strip().startswith("{"):
            js = fb
    if js is None:
        return None
    try:
        obj = json.loads(js)
        if isinstance(obj, dict) and "ops" in obj and isinstance(obj["ops"], list):
            return obj
        if isinstance(obj, list):
            return {"ops": obj}
        return obj
    except Exception:
        return None

def extract_second_json_constraints(text: str) -> Optional[Dict[str, Any]]:
    """
    Find the SECOND ```json fenced block and parse it as constraints.
    Strategy: remove the first json block we already used for ops, then search again.
    """
    t = text
    needle = "```json"
    idx = t.find(needle)
    if idx == -1:
        return None
    end1 = t.find("```", idx + len(needle))
    if end1 == -1:
        return None
    remainder = t[:idx] + t[end1+3:]
    idx2 = remainder.find(needle)
    if idx2 == -1:
        return None
    end2 = remainder.find("```", idx2 + len(needle))
    if end2 == -1:
        return None
    js2 = remainder[idx2 + len(needle): end2].strip()
    try:
        obj = json.loads(js2)
        if isinstance(obj, dict) and "constraints" in obj:
            return obj
        if isinstance(obj, dict):
            return {"constraints": obj}
        return None
    except Exception:
        return None

def import_from_string(module_name: str, source: str) -> types.ModuleType:
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    exec(source, module.__dict__)
    sys.modules[module_name] = module
    return module

# -------------------------
# Instruction parsing → fallback ops
# -------------------------

_step_re = re.compile(r"^\s*(\d+)[\.)]\s+(.*)", re.M)

def split_numbered_steps(instructions: str) -> List[str]:
    """Extract numbered steps from the instruction text."""
    steps: List[str] = []
    for m in _step_re.finditer(instructions or ""):
        step_text = m.group(2).strip()
        if step_text:
            steps.append(step_text)
    if not steps:
        raw = [x.strip("-• ").strip() for x in (instructions or "").split("\n") if x.strip()]
        steps = [s for s in raw if len(s) > 3]
    return steps

def infer_op_name(text: str) -> str:
    t = text.lower()
    if "tile" in t or "repeat" in t or "duplicate" in t:
        return "repeat"
    if "checker" in t or "alternat" in t:
        return "checker_tile"
    if "mirror" in t or "reflect" in t:
        return "mirror"
    if "rotate" in t:
        return "rotate"
    if "transpose" in t or "swap rows and columns" in t:
        return "transpose"
    if "flip" in t and "vertical" in t:
        return "flip_v"
    if "flip" in t and "horizontal" in t:
        return "flip_h"
    if "flip" in t:
        return "flip"
    if "copy" in t:
        return "copy"
    if "mask" in t or "paint" in t:
        return "paint"
    if "translate" in t or "shift" in t:
        return "shift"
    if "map" in t or "recolor" in t or "palette" in t:
        return "color_map"
    return "step"

def build_fallback_ops(instructions: str) -> Dict[str, Any]:
    steps = split_numbered_steps(instructions)
    ops = []
    for i, s in enumerate(steps, 1):
        ops.append({"op": infer_op_name(s), "step_index": i, "description": s})
    return {"ops": ops, "_source": "fallback-from-steps"}

# -------------------------
# Constraints derivation (fallback)
# -------------------------

def derive_constraints_from_examples(task: Dict[str, Any]) -> Dict[str, Any]:
    trains = task.get("train", [])
    def shape(g): return (len(g), len(g[0]) if g else 0)

    rel = "same"
    kh = kw = 1
    ok_mul = True
    for p in trains:
        (hi, wi), (ho, wo) = shape(p["input"]), shape(p["output"])
        if hi == ho and wi == wo:
            continue
        if hi == 0 or wi == 0 or ho % hi or wo % wi:
            ok_mul = False
            break
        kh_i, kw_i = ho // hi, wo // wi
        if kh == 1 and kw == 1:
            kh, kw = kh_i, kw_i
        elif (kh, kw) != (kh_i, kw_i):
            ok_mul = False
            break
    if ok_mul and (kh, kw) != (1, 1):
        rel = "multiply"

    in_colors = set(); out_colors = set()
    for p in trains:
        in_colors |= {c for row in p["input"] for c in row}
        out_colors |= {c for row in p["output"] for c in row}
    palette_type = "subset_of_input" if out_colors.issubset(in_colors) else "unconstrained"

    return {
        "constraints": {
            "shape": {"relation": rel, "kh": kh, "kw": kw},
            "palette": {"type": palette_type},
            "periodicity": None,
            "symmetry": None,
            "color_count": {"preserve": False},
            "divisibility": {
                "H_out_mod": 0 if rel == "multiply" else None,
                "W_out_mod": 0 if rel == "multiply" else None
            }
        }
    }

# -------------------------
# Prompt for code + ops + constraints
# -------------------------

def build_codegen_prompt(task_id: str, instructions: str, task: Dict[str, Any],
                         shape_guard: bool, strict_blocks: bool = False) -> str:
    lines = []
    lines.append("You will produce THREE artifacts for this ARC task:")
    lines.append("")
    lines.append("1) A single, clean Python module implementing:")
    lines.append("       def transform(grid: list[list[int]]) -> list[list[int]]")
    lines.append("   Constraints:")
    lines.append("   - Cells are integers 0–9; grid is rectangular (equal-length rows).")
    lines.append("   - No external libraries; only Python stdlib. No I/O. No randomness. No LLM calls.")
    if shape_guard:
        lines.append("   - If the rule implies a resize, compute that size exactly and deterministically.")
    lines.append("")
    lines.append("2) A compact, machine-readable OPERATION LIST (emergent DSL).")
    lines.append('   • JSON shape:  {"ops": [ {"op":"<name>", ...params }, ... ] }')
    lines.append("")
    lines.append("3) HARD CONSTRAINTS that must hold for any correct solution.")
    lines.append('   • JSON shape: {"constraints": { ... }}')
    lines.append('   • Possible fields (include only if justified):')
    lines.append('     - "shape": {"relation":"same"|"multiply","kh":<int>,"kw":<int>}')
    lines.append('     - "palette": {"type":"subset_of_input"|"unconstrained"}')
    lines.append('     - "periodicity": {"rows":<int|null>, "cols":<int|null>, "pattern":"alternating"|null}')
    lines.append('     - "symmetry": "h"|"v"|null')
    lines.append('     - "color_count": {"preserve":true|false}')
    lines.append('     - "divisibility": {"H_out_mod":<int|null>, "W_out_mod":<int|null>}')
    lines.append("")
    lines.append(f"Task ID: {task_id}")
    lines.append("Implement these natural-language instructions:")
    lines.append(instructions.strip())
    lines.append("")
    lines.append("Training examples (Input → Output), for grounding only:")
    for i, pair in enumerate(task.get("train", [])):
        inp = pair["input"]; out = pair["output"]
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        lines.append(f"- Example {i+1}  IN({ih}x{iw}) → OUT({oh}x{ow})")
        lines.append("  INPUT:")
        for r in inp: lines.append("    " + " ".join(map(str, r)))
        lines.append("  OUTPUT:")
        for r in out: lines.append("    " + " ".join(map(str, r)))
        lines.append("")
    if not strict_blocks:
        lines.append("Return EXACTLY THREE fenced blocks in this order, with nothing else:")
    else:
        lines.append("Return EXACTLY THREE fenced blocks in this order (all REQUIRED), with nothing else:")
    lines.append("  (A) ```python")
    lines.append("      # full module below")
    lines.append("      def transform(grid: list[list[int]]) -> list[list[int]]:")
    lines.append("          ...")
    lines.append("      ```")
    lines.append("  (B) ```json")
    lines.append('      {"ops": [ {"op":"..."} ] }')
    lines.append("      ```")
    lines.append("  (C) ```json")
    lines.append('      {"constraints": { "shape": {...}, "palette": {...} }}')
    lines.append("      ```")
    if strict_blocks:
        lines.append("If you failed to include the JSON blocks previously, return ONLY blocks (B) and (C) now.")
    return "\n".join(lines)

# -------------------------
# Main
# -------------------------

class GenerationFailure(Exception):
    pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", type=str, choices=["training","evaluation"], default="training")
    ap.add_argument("--task-id", type=str, required=True)
    ap.add_argument("--describer-outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--code-model", type=str, default="qwen2.5-coder:7b")
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print-grids", action="store_true")
    ap.add_argument("--apply-shape-guard", action="store_true")
    ap.add_argument("--save-dir", type=Path, default=Path("generated/pyprog"))
    ap.add_argument("--ops-dir", type=Path, default=Path("generated/dsl"))
    ap.add_argument("--constraints-dir", type=Path, default=Path("generated/constraints"))
    args = ap.parse_args()

    # 1) Gather scored candidates (pooled > individual > describer)
    scored: List[ScoredCandidate] = []
    if has_pooled_evals(args.split, args.task_id):
        scored.extend(extract_best_from_pooled(args.split, args.task_id))
    if not scored:
        scored.extend(extract_best_from_individual(args.split, args.task_id))
    if not scored:
        scored.extend(extract_describer_fallback(args.describer_outputs, args.split, args.task_id))
    if not scored:
        print("No candidates found for this task. Run the pipeline first.", file=sys.stderr)
        sys.exit(2)

    best = pick_best(scored)
    if best is None or not best.instructions.strip():
        print("Could not locate a non-empty best instruction.", file=sys.stderr)
        sys.exit(2)

    # Print the TOP instruction set
    print("="*80)
    print(f"Selected BEST instruction set for task {args.task_id}")
    print(f"  source={best.source}  cand_id={best.candidate_id}  "
          f"primary={best.primary_perfect}/{best.num_pairs}  "
          f"secondary={best.secondary_pct*100:.1f}%  parse={best.parse_rate*100:.1f}%  size_ok={best.size_ok_rate*100:.1f}%")
    print("-"*80)
    print(best.instructions.strip())
    print("="*80)

    # 2) Load task for grounding
    task = load_task(args.root, args.split, args.task_id)

    # 3) Codegen via model (code + ops + constraints)
    client = OllamaClient()
    prompt = build_codegen_prompt(args.task_id, best.instructions, task,
                                  shape_guard=args.apply_shape_guard, strict_blocks=False)
    raw = client.generate(model=args.code_model, prompt=prompt,
                          temperature=args.temperature, seed=args.seed)

    code = extract_python_block(raw)
    ops_obj = extract_json_ops(raw)
    constraints_obj = extract_second_json_constraints(raw)

    # Strict retry if blocks missing
    if ops_obj is None or constraints_obj is None:
        strict_prompt = build_codegen_prompt(args.task_id, best.instructions, task,
                                             shape_guard=args.apply_shape_guard, strict_blocks=True)
        raw2 = client.generate(model=args.code_model, prompt=strict_prompt,
                               temperature=max(0.01, args.temperature-0.02), seed=(args.seed or 0)+1)
        code2 = extract_python_block(raw2)
        ops2 = extract_json_ops(raw2)
        cons2 = extract_second_json_constraints(raw2)
        if ops2 is not None:
            ops_obj = ops2
        if cons2 is not None:
            constraints_obj = cons2
        if "def transform(" in code2:
            code = code2

    # Fallbacks
    if ops_obj is None:
        ops_obj = build_fallback_ops(best.instructions)
    if constraints_obj is None:
        constraints_obj = derive_constraints_from_examples(task)

    # Ensure code exists
    if "def transform(" not in code:
        print("Generated code did not contain a 'transform' function. Dumping last output for debugging:\n", file=sys.stderr)
        print(raw, file=sys.stderr)
        sys.exit(3)

    # Print the FULL generated code
    print("\nGenerated Python module:")
    print("-"*80)
    print(code)
    print("-"*80)

    # Always print the OPS/DSL JSON
    print("\nEmergent OPS / DSL (machine-readable):")
    print("-"*80)
    try:
        print(json.dumps(ops_obj, indent=2, ensure_ascii=False))
    except Exception:
        print(ops_obj)
    print("-"*80)

    # Always print the HARD CONSTRAINTS JSON
    print("\nHard CONSTRAINTS (machine-readable):")
    print("-"*80)
    try:
        print(json.dumps(constraints_obj, indent=2, ensure_ascii=False))
    except Exception:
        print(constraints_obj)
    print("-"*80)

    # 4) Save code, ops, constraints
    out_dir = args.save_dir
    ops_dir = args.ops_dir
    cons_dir = args.constraints_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ops_dir.mkdir(parents=True, exist_ok=True)
    cons_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{args.task_id}_best.py"
    out_path.write_text(code, encoding="utf-8")
    print(f"Wrote Python program to: {out_path}")

    ops_path = ops_dir / f"{args.task_id}_ops.json"
    ops_path.write_text(json.dumps(ops_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote OPS JSON to: {ops_path}")

    cons_path = cons_dir / f"{args.task_id}_constraints.json"
    cons_path.write_text(json.dumps(constraints_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote CONSTRAINTS JSON to: {cons_path}")

    # 5) Dynamic import & self-test (prints INPUT/PRED/TRUTH per pair if --print-grids)
    mod = import_from_string(f"arc_best_{args.task_id}", code)
    if not hasattr(mod, "transform"):
        print("The module lacks a 'transform' function.", file=sys.stderr)
        sys.exit(4)
    fn = getattr(mod, "transform")

    result = score_transform(fn, task, print_grids=args.print_grids)
    s = result["summary"]
    print(f"\nSelf-test on training pairs:")
    print(f"  primary={s['primary_perfect']}/{s['num_pairs']} ({s['primary_pct']*100:.1f}%)  "
          f"secondary={s['secondary_cells']}/{s['secondary_possible']} ({s['secondary_pct']*100:.1f}%)")

if __name__ == "__main__":
    main()
