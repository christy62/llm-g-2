import re
import ast
from hit_details import HelixFromCSV

# --------------------------------------------
# Detector model
# --------------------------------------------
helix = HelixFromCSV(B=3.0)


# --------------------------------------------
# Forward physics model
# --------------------------------------------
def forward_gen_event(vtx, mom, axis="x", step=0.3, max_steps=5000):

    pts, _, _, _, cumE, times = helix.compute_helix(
        decay_pos=vtx,
        decay_momentum=mom,
        step=step,
        max_steps=max_steps
    )

    E_THRESHOLD = 0.03

    hits = []

    for i in range(len(pts)):

        x, y, z = pts[i]

        if cumE[i] <= E_THRESHOLD:
            continue

        if axis == "x":
            hits.append(round(float(x), 1))

        elif axis == "y":
            hits.append(round(float(y), 1))

        elif axis == "z":
            hits.append(round(float(z), 1))

    return hits


# --------------------------------------------
# Symbolic function predicted by LLM
# --------------------------------------------
def gen_event(vtx=None, mom=None, axis=None):

    if vtx is None or mom is None or axis is None:
        return []

    hits = forward_gen_event(vtx, mom, axis)

    # keep only events with >5 hits
    if len(hits) <= 5:
        return []

    return hits


# --------------------------------------------
# Validation
# --------------------------------------------
def is_valid_gen_event(text):

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


# --------------------------------------------
# Parse parameters
# --------------------------------------------
def parse_gen_event_params(func_string):

    pattern = r'gen_event\(([^)]+)\)'
    match = re.search(pattern, func_string)

    if not match:
        raise ValueError("Invalid gen_event format")

    params = match.group(1)

    vtx_match = re.search(r'vtx=\[([^\]]+)\]', params)
    mom_match = re.search(r'mom=\[([^\]]+)\]', params)
    axis_match = re.search(r"axis='([xyz])'", params)

    vtx = ast.literal_eval(f"[{vtx_match.group(1)}]") if vtx_match else []
    mom = ast.literal_eval(f"[{mom_match.group(1)}]") if mom_match else []
    axis = axis_match.group(1) if axis_match else None

    return {"vtx": vtx, "mom": mom, "axis": axis}


# --------------------------------------------
# Safe evaluator
# --------------------------------------------
def eval_gen_event(code_string):

    if not code_string.startswith("gen_event"):
        return []

    try:
        return eval(code_string)
    except:
        return []


# --------------------------------------------
# Similarity metrics
# --------------------------------------------
def get_similarity(true_hits, pred_hits):

    true_set = set(true_hits)
    pred_set = set(pred_hits)

    if len(true_set) == 0:
        return 0

    return len(true_set & pred_set) / len(true_set)


def get_dsimilarity(true_hits, pred_hits):

    true_set = set(true_hits)
    pred_set = set(pred_hits)

    if len(pred_set) == 0:
        return 1

    return len(pred_set - true_set) / len(pred_set)


# ------------------------------------------------------------
# MAIN — validity test
# ------------------------------------------------------------
if __name__ == "__main__":

    vtx = [-266.9, -200.1, 0.0]
    mom = [-32.4, 230.2, -23.7]

    print("\nTesting forward model\n")

    hits_x = gen_event(vtx=vtx, mom=mom, axis="x")
    hits_y = gen_event(vtx=vtx, mom=mom, axis="y")
    hits_z = gen_event(vtx=vtx, mom=mom, axis="z")

    if len(hits_x) > 5:
        print("X axis valid")
        print("First 10 X hits:", hits_x[:10])
        print("Total X hits:", len(hits_x))
    else:
        print("X axis rejected (<5 hits)")

    print()

    if len(hits_y) > 5:
        print("Y axis valid")
        print("First 10 Y hits:", hits_y[:10])
        print("Total Y hits:", len(hits_y))
    else:
        print("Y axis rejected (<5 hits)")

    print()

    if len(hits_z) > 5:
        print("Z axis valid")
        print("First 10 Z hits:", hits_z[:10])
        print("Total Z hits:", len(hits_z))
    else:
        print("Z axis rejected (<5 hits)")