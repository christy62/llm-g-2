import os, re, ast, math
import torch
import numpy as np
from collections import defaultdict
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import gen_event as gevt

#  Config 
MODEL_NAME = "t5-small"
HF_REPO    = "christyelsa/g2-train-05"
HF_SPLIT   = "train"
curr_ver   = "/kaggle/working/t5-small-00"
next_ver   = "/kaggle/working/t5-small-01"
model_name = "t5-small"

BASE_DIR   = "./runs"
OUTPUT_DIR = os.path.join(BASE_DIR, f"{model_name}-{curr_ver}")
SAVE_DIR   = os.path.join(BASE_DIR, f"{model_name}-{next_ver}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,   exist_ok=True)

USE_CUDA   = torch.cuda.is_available()
DEVICE     = torch.device("cuda" if USE_CUDA else "cpu")
BATCH_SIZE = 32 if USE_CUDA else 4
NUM_EPOCHS = 100
LR         = 1.5e-4

COORD_RANGE = 300.0   # mm
MOM_RANGE   = 300.0   # MeV
DELTA_RANGE = 50.0    # mm  — step displacement
CURV_RANGE  = 0.05    # mm⁻¹ — signed curvature
PITCH_RANGE = 50.0    # mm/rad — helix pitch

MAX_INPUT_LEN  = 256
MAX_TARGET_LEN = 64

#  Helpers 
def _fmt(v: float, scale: float) -> str:
    return f"{max(-1.0, min(1.0, v / scale)):+.3f}"

def compute_kappa(p0, p1, p2) -> float:
    """Signed Menger curvature in the xy-plane for three 2-D points."""
    area  = 0.5 * ((p1[0]-p0[0])*(p2[1]-p0[1])
                  -(p2[0]-p0[0])*(p1[1]-p0[1]))
    d01   = np.linalg.norm(p1 - p0)
    d12   = np.linalg.norm(p2 - p1)
    d20   = np.linalg.norm(p0 - p2)
    denom = d01 * d12 * d20
    return float(2.0 * area / denom) if denom > 1e-9 else 0.0

def compute_helix_features(pts: list) -> list:
    """
    Given N hit points [[x0,y0,z0], ...] return normalised string tokens
    encoding full 3-D helix geometry.

    Per-hit tokens (10 per hit):
      x, y, z          — raw position
      dx, dy, dz        — step from previous hit (tangent direction)
      dr                — transverse step magnitude
      kappa             — signed Menger curvature (1/R ∝ 1/pT)
      pitch             — dz/dphi  (∝ pz/pT)
      arc_pos           — cumulative arc position 0→1

    Prepended summary token (1):
      arc_tot           — total transverse arc length
    """
    n   = len(pts)
    arr = np.array(pts, dtype=np.float64)   # (N, 3)

    tokens = []

    # cumulative transverse arc length
    dxy     = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)  # (N-1,)
    arc_cum = np.concatenate([[0.0], np.cumsum(dxy)])               # (N,)
    arc_tot = arc_cum[-1] if arc_cum[-1] > 0 else 1.0

    tokens.append(_fmt(arc_tot, COORD_RANGE))   # summary token

    for i in range(n):
        # raw position
        tokens += [_fmt(arr[i,0], COORD_RANGE),
                   _fmt(arr[i,1], COORD_RANGE),
                   _fmt(arr[i,2], COORD_RANGE)]

        # step vector
        if i < n - 1:
            dx, dy, dz = arr[i+1] - arr[i]
        else:
            dx, dy, dz = arr[i] - arr[i-1]

        dr = math.hypot(dx, dy)
        tokens += [_fmt(dx, DELTA_RANGE),
                   _fmt(dy, DELTA_RANGE),
                   _fmt(dz, DELTA_RANGE),
                   _fmt(dr, DELTA_RANGE)]

        # signed curvature — use best available triplet at boundaries
        if n >= 3:
            if i == 0:
                kappa = compute_kappa(arr[0,:2], arr[1,:2], arr[2,:2])
            elif i == n - 1:
                kappa = compute_kappa(arr[-3,:2], arr[-2,:2], arr[-1,:2])
            else:
                kappa = compute_kappa(arr[i-1,:2], arr[i,:2], arr[i+1,:2])
        else:
            kappa = 0.0

        tokens.append(_fmt(kappa, CURV_RANGE))

        # pitch = dz / dphi,  dphi ≈ dr * |kappa|
        dphi  = dr * abs(kappa)
        pitch = float(dz / dphi) if dphi > 1e-9 else 0.0
        tokens.append(_fmt(pitch, PITCH_RANGE))

        # normalised arc position along track
        tokens.append(_fmt(arc_cum[i] / arc_tot, 1.0))

    return tokens

#  Dataset merging 
def merge_axes(raw_dataset) -> Dataset:
    """
    The raw dataset has 3 rows per physical event (axis='x', 'y', 'z').
    Each row's input is a 1-D hit list for that axis projection.
    Since hits are paired (hit i in x == hit i in y == hit i in z),
    we merge them into proper 3-D (x,y,z) triplets per hit.

    Before:  3 rows  →  input=[z1,z2,...],  target=gen_event(...,axis='z')
    After:   1 row   →  input=[[x1,y1,z1],[x2,y2,z2],...] (flat),
                         target=gen_event(vtx=[...], mom=[...])
    """
    groups = defaultdict(dict)

    for row in raw_dataset:
        target = row["target"]
        hits   = ast.literal_eval(row["input"])

        # group key = target without the axis field
        key = re.sub(r",\s*axis='[xyz]'", "", target).strip()

        axis_match = re.search(r"axis='([xyz])'", target)
        if not axis_match:
            continue
        axis = axis_match.group(1)

        groups[key][axis] = hits

    merged = {"input": [], "target": []}

    for key, axes in groups.items():
        if not all(a in axes for a in ("x", "y", "z")):
            continue   # skip incomplete events

        xs, ys, zs = axes["x"], axes["y"], axes["z"]

        if not (len(xs) == len(ys) == len(zs)):
            continue   # skip mismatched events

        # interleave into flat [x1,y1,z1, x2,y2,z2, ...]
        flat = []
        for x, y, z in zip(xs, ys, zs):
            flat.extend([x, y, z])

        merged["input"].append(str(flat))
        merged["target"].append(key + ")")

    return Dataset.from_dict(merged)

# Feature normalisation 
def normalise_input(raw_str: str) -> str:
    """Parse flat [x1,y1,z1,...] string → helix feature tokens."""
    try:
        vals = ast.literal_eval(raw_str)
    except Exception:
        return raw_str

    if len(vals) % 3 != 0 or len(vals) == 0:
        return " ".join(_fmt(v, COORD_RANGE) for v in vals)

    pts    = [vals[i:i+3] for i in range(0, len(vals), 3)]
    tokens = compute_helix_features(pts)
    return " ".join(tokens)

def normalise_target(raw_str: str) -> str:
    """Normalise vtx and mom values in the target string."""
    def replace_list(m, scale):
        vals = ast.literal_eval(f"[{m.group(1)}]")
        return "[" + ", ".join(_fmt(v, scale) for v in vals) + "]"

    s = re.sub(r'vtx=\[([^\]]+)\]',
               lambda m: "vtx=" + replace_list(m, COORD_RANGE), raw_str)
    s = re.sub(r'mom=\[([^\]]+)\]',
               lambda m: "mom=" + replace_list(m, MOM_RANGE),   s)
    return s

#  Sanity check 
print("Checking gen_event import …")
try:
    result = gevt.eval_gen_event(
        "gen_event(vtx=[262.9, 205.3, 0.0], mom=[121.7, 51.6, 14.5], axis='x')"
    )
    print(f"  gen_event OK — {len(result)} hits returned")
except Exception as e:
    print(f"  ERROR: {e}")
    raise SystemExit(1)

#  Load, merge & split dataset
print(f"\nLoading dataset from {HF_REPO}/{HF_SPLIT} …")
raw    = load_dataset(HF_REPO, split=HF_SPLIT)
print(f"  Raw rows   : {len(raw):,}")

print("Merging x/y/z axes into 3-D events …")
merged = merge_axes(raw)
print(f"  Merged events: {len(merged):,}  (expected ~{len(raw)//3:,})")

train_ds = merged
eval_ds  = merged
# Uncomment below for a proper train/eval split:
# split    = merged.train_test_split(test_size=0.05, seed=42)
# train_ds = split["train"]
# eval_ds  = split["test"]

#  Tokeniser & model
print(f"\nLoading tokeniser and model ({MODEL_NAME}) …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
print(f"  Device: {DEVICE}  |  Vocab size: {len(tokenizer)}")

# Tokenisation 
def tokenize(examples):
    inputs  = [normalise_input(t)  for t in examples["input"]]
    targets = [normalise_target(t) for t in examples["target"]]

    model_inputs = tokenizer(inputs,  max_length=MAX_INPUT_LEN,  truncation=True)
    labels       = tokenizer(text_target=targets,
                             max_length=MAX_TARGET_LEN, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenising …")
tok_train = train_ds.map(tokenize, batched=True, remove_columns=["input", "target"])
tok_eval  = eval_ds.map( tokenize, batched=True, remove_columns=["input", "target"])
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

#  Sample logger callback 
class SampleLoggerCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, log_every=200, n_samples=3):
        self.tokenizer    = tokenizer
        self.eval_dataset = eval_dataset
        self.log_every    = log_every
        self.n_samples    = n_samples

    def on_step_end(self, args, state: TrainerState,
                    control: TrainerControl, **kwargs):
        if state.global_step % self.log_every != 0 or state.global_step == 0:
            return

        model = kwargs["model"]
        model.eval()
        batch = self.eval_dataset.select(range(self.n_samples))

        with torch.no_grad():
            ids  = torch.nn.utils.rnn.pad_sequence(
                       [torch.tensor(x) for x in batch["input_ids"]],
                       batch_first=True,
                       padding_value=self.tokenizer.pad_token_id).to(DEVICE)
            mask = torch.nn.utils.rnn.pad_sequence(
                       [torch.tensor(x) for x in batch["attention_mask"]],
                       batch_first=True, padding_value=0).to(DEVICE)
            gen  = model.generate(ids, attention_mask=mask,
                                  max_length=MAX_TARGET_LEN)

        print(f"\n{'='*60}")
        print(f"Step {state.global_step} — sample predictions")
        for i in range(self.n_samples):
            pred = self.tokenizer.decode(gen[i], skip_special_tokens=True)
            lbl  = self.tokenizer.decode(
                [t for t in batch["labels"][i] if t != -100],
                skip_special_tokens=True)
            print(f"  Target: {lbl}")
            print(f"  Pred  : {pred}")
            print(f"  Valid : {gevt.is_valid_gen_event(pred)}")
        print('='*60)
        model.train()

#  Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    learning_rate               = LR,
    weight_decay                = 0.01,
    warmup_steps                = 500,
    predict_with_generate       = True,
    generation_max_length       = MAX_TARGET_LEN,
    fp16                        = USE_CUDA,
    dataloader_pin_memory       = USE_CUDA,
    gradient_accumulation_steps = 1,
    gradient_checkpointing      = True,
    eval_strategy               = "steps",
    eval_steps                  = 200,
    save_strategy               = "steps",
    save_steps                  = 400,
    save_total_limit            = 3,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    logging_steps               = 20,
    remove_unused_columns       = False,
    report_to                   = "none",
)

#  Trainer 
trainer = Seq2SeqTrainer(
    model            = model,
    args             = training_args,
    train_dataset    = tok_train,
    eval_dataset     = tok_eval,
    data_collator    = data_collator,
    processing_class = tokenizer,
    callbacks        = [
        SampleLoggerCallback(tokenizer    = tokenizer,
                             eval_dataset = tok_eval,
                             log_every    = 200,
                             n_samples    = 3)
    ],
)

#  Final pre-train check
print("\nFinal pre-train check …")
try:
    gevt.gen_event(vtx=[262.9, 205.3, 0.0], mom=[121.7, 51.6, 14.5], axis="x")
    print("  gen_event OK")
except Exception as e:
    print(f"  ERROR: {e}")
    raise SystemExit(1)

#  Train 
print("\nStarting training …")
trainer.train()

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nModel saved → {SAVE_DIR}")

print("\n=== Final evaluation ===")
final = trainer.evaluate()
print(f"  eval_loss : {final['eval_loss']:.4f}")