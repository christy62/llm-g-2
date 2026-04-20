import re
import ast
from hit_details import HelixFromCSV

#  Detector model
helix = HelixFromCSV(B=3.0)


#  Forward physics model

def forward_gen_event(vtx, mom, axis="x", step=0.3, max_steps=5000):
   
    pts, _, _, _, cumE, _ = helix.compute_helix(
        decay_pos=vtx,
        decay_momentum=mom,
        step=step,
        max_steps=max_steps,
    )

    E_THRESHOLD = 0.03   # MeV — ignore hits with negligible energy deposition

    hits = []
    for i in range(len(pts)):
        if cumE[i] <= E_THRESHOLD:
            continue
        coord = {"x": pts[i][0], "y": pts[i][1], "z": pts[i][2]}.get(axis)
        if coord is not None:
            hits.append(round(float(coord), 1))

    return hits


def gen_event(vtx=None, mom=None, axis=None):
    """
    Public interface called by the LLM.
    Returns [] for missing args or fewer than 5 hits (too sparse to be useful).
    """
    if vtx is None or mom is None or axis is None:
        return []
    hits = forward_gen_event(vtx, mom, axis)
    return hits if len(hits) > 5 else []


#  Validation 
def is_valid_gen_event(text):
    """Return True iff `text` exactly matches the gen_event call signature."""
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


#  Parameter parser

def parse_gen_event_params(func_string):
    """Extract vtx, mom, axis from a gen_event(...) string."""
    pattern = r'gen_event\(([^)]+)\)'
    match   = re.search(pattern, func_string)
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


#  Safe evaluator 

def eval_gen_event(code_string):
    """
    FIX: replaced bare eval() with safe parser + explicit function call.
    bare eval() allowed arbitrary code execution from model-generated strings.
    """
    if not code_string.startswith("gen_event"):
        return []
    try:
        params = parse_gen_event_params(code_string)
        return gen_event(**params)
    except Exception:
        return []


#  Similarity metrics 

def get_similarity(true_hits, pred_hits):
    """Recall: fraction of true hits that appear in predicted hits."""
    true_set = set(true_hits)
    if len(true_set) == 0:
        return 0.0
    return len(true_set & set(pred_hits)) / len(true_set)


def get_dsimilarity(true_hits, pred_hits):
    """False-discovery rate: fraction of predicted hits not in true hits."""
    pred_set = set(pred_hits)
    if len(pred_set) == 0:
        return 1.0
    return len(pred_set - set(true_hits)) / len(pred_set)


#  test 

if __name__ == "__main__":
    vtx = [-266.9, -200.1, 0.0]
    mom = [-32.4,   230.2, -23.7]

    print("\nTesting forward model\n")
    for ax in ("x", "y", "z"):
        hits = gen_event(vtx=vtx, mom=mom, axis=ax)
        label = f"{ax.upper()} axis"
        if hits:
            print(f"{label} valid  | hits: {len(hits):3d} | first 10: {hits[:10]}")
        else:
            print(f"{label} rejected (≤5 hits)")