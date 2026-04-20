# gen_event_g4.py

import re
import ast
import numpy as np
from datasets import load_dataset
from collections import defaultdict

_G4_LOOKUP = {}   # (vtx_x, vtx_y, vtx_z, px, py, pz) → {"x": [...], "y": [...], "z": [...]}

def _make_key(vtx, mom):
    return (
        round(vtx[0], 1), round(vtx[1], 1), round(vtx[2], 1),
        round(mom[0], 1), round(mom[1], 1), round(mom[2], 1),
    )

# gen_event_g4.py — return the dataset from load_g4_lookup so training can reuse it

def load_g4_lookup(hf_repo: str, hf_token: str, split: str = "train"):
    global _G4_LOOKUP

    print(f"Loading G4 dataset from {hf_repo} ...")
    dataset = load_dataset(hf_repo, split=split, token=hf_token)
    print(f"  Loaded {len(dataset):,} rows")

    partial = defaultdict(dict)

    for row in dataset:
        target   = row["target"]
        hits_raw = row["input"]

        vtx_m  = re.search(r'vtx=\[([^\]]+)\]', target)
        mom_m  = re.search(r'mom=\[([^\]]+)\]', target)
        axis_m = re.search(r"axis='([xyz])'",   target)

        if not (vtx_m and mom_m and axis_m):
            continue

        vtx  = ast.literal_eval(f"[{vtx_m.group(1)}]")
        mom  = ast.literal_eval(f"[{mom_m.group(1)}]")
        axis = axis_m.group(1)

        try:
            hits = ast.literal_eval(hits_raw)
        except Exception:
            continue

        key = _make_key(vtx, mom)
        partial[key][axis] = hits

    complete = 0
    for key, axes in partial.items():
        if all(a in axes for a in ("x", "y", "z")):
            _G4_LOOKUP[key] = axes
            complete += 1

    print(f"  Complete events (all 3 axes): {complete:,}")
    print(f"  Lookup ready — {len(_G4_LOOKUP):,} events indexed")

    # No return needed since we use the lookup directly

def gen_event(vtx=None, mom=None, axis=None):
    if vtx is None or mom is None or axis is None:
        return []
    key  = _make_key(vtx, mom)
    hits = _G4_LOOKUP.get(key, {}).get(axis, [])
    return hits if len(hits) > 5 else []


def eval_gen_event(code_string: str):
    if not code_string.startswith("gen_event"):
        return []
    try:
        params = parse_gen_event_params(code_string)
        return gen_event(**params)
    except Exception:
        return []


def is_valid_gen_event(text: str) -> bool:
    pattern = r"""
    ^gen_event\(
        vtx=\[
            -?\d+(\.\d+)?,\s*
            -?\d+(\.\d+)?,\s*
            -?\d+(\.\d+)?
        \],
        \s*
        mom=\[
            -?\d+(\.\d+)?,\s*
            -?\d+(\.\d+)?,\s*
            -?\d+(\.\d+)?
        \],
        \s*
        axis='[xyz]'
    \)$
    """
    return bool(re.fullmatch(pattern, text, re.VERBOSE))


def parse_gen_event_params(func_string: str):
    match = re.search(r'gen_event\(([^)]+)\)', func_string)
    if not match:
        raise ValueError(f"Invalid gen_event format: {func_string!r}")
    params     = match.group(1)
    vtx_match  = re.search(r'vtx=\[([^\]]+)\]', params)
    mom_match  = re.search(r'mom=\[([^\]]+)\]', params)
    axis_match = re.search(r"axis='([xyz])'",   params)
    vtx  = ast.literal_eval(f"[{vtx_match.group(1)}]") if vtx_match  else []
    mom  = ast.literal_eval(f"[{mom_match.group(1)}]") if mom_match  else []
    axis = axis_match.group(1)                          if axis_match else None
    return {"vtx": vtx, "mom": mom, "axis": axis}