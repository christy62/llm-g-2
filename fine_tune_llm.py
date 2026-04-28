import os, re, ast, math
import torch
import matplotlib.pyplot as plt
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

HF_REPO    = "christyelsa/llm-g4"
HF_SPLIT   = "train"

curr_ver   = "00"
next_ver   = "01"

def get_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda x: int(x.split("-")[-1]))

MODEL_NAME = "christyelsa/llm-g2-12"

OUTPUT_DIR = f"./t5-small-{curr_ver}"
SAVE_DIR   = f"./t5-small-{next_ver}"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,   exist_ok=True)

USE_CUDA   = torch.cuda.is_available()
DEVICE     = torch.device("cuda" if USE_CUDA else "cpu")
BATCH_SIZE = 96 if USE_CUDA else 4
NUM_EPOCHS = 500
LR         = 5e-5      
COORD_RANGE = 333.564
MOM_RANGE   = 300.0
DELTA_RANGE = 50.0
CURV_RANGE  = 0.05
PITCH_RANGE = 50.0

MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 64

# ── Physics penalty controls 
# Phase 1 (steps 0 → CE_ONLY_STEPS): pure cross-entropy, no physics penalty
# Phase 2 (steps > CE_ONLY_STEPS): physics penalty kicks in gently
CE_ONLY_STEPS  = 3000       # no physics until CE loss is healthy (~0.4)
PHYSICS_WEIGHT = 0.3        # kept small — never overwhelm CE loss
GENERATE_EVERY = 20         # only run generate() every N steps for speed
HIT_TOL_EVAL   = 0.5        # tight tolerance used in callback logging

# ── Helpers

def _fmt(v: float, scale: float) -> str:
    """Normalise v by scale and hard-clip to [-1, 1]."""
    v = round(v, 1)
    x = max(-1.0, min(1.0, v / scale))   # hard clip — prevents >1 outputs
    return f"{x:+.4f}"

def compute_kappa(p0, p1, p2) -> float:
    area  = 0.5 * ((p1[0]-p0[0])*(p2[1]-p0[1])
                  -(p2[0]-p0[0])*(p1[1]-p0[1]))
    d01   = np.linalg.norm(p1 - p0)
    d12   = np.linalg.norm(p2 - p1)
    d20   = np.linalg.norm(p0 - p2)
    denom = d01 * d12 * d20
    return float(2.0 * area / denom) if denom > 1e-9 else 0.0

def compute_helix_features(pts: list) -> list:
    n   = len(pts)
    arr = np.array(pts, dtype=np.float64)
    tokens = []
    dxy     = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
    arc_cum = np.concatenate([[0.0], np.cumsum(dxy)])
    arc_tot = arc_cum[-1] if arc_cum[-1] > 0 else 1.0
    tokens.append(_fmt(arc_tot, COORD_RANGE))
    for i in range(n):
        tokens += [_fmt(arr[i,0], COORD_RANGE),
                   _fmt(arr[i,1], COORD_RANGE),
                   _fmt(arr[i,2], COORD_RANGE)]
        if i < n - 1:
            dx, dy, dz = arr[i+1] - arr[i]
        else:
            dx, dy, dz = arr[i] - arr[i-1]
        dr = math.hypot(dx, dy)
        tokens += [_fmt(dx, DELTA_RANGE), _fmt(dy, DELTA_RANGE),
                   _fmt(dz, DELTA_RANGE), _fmt(dr, DELTA_RANGE)]
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
        dphi  = dr * abs(kappa)
        pitch = float(dz / dphi) if dphi > 1e-9 else 0.0
        tokens.append(_fmt(pitch, PITCH_RANGE))
        tokens.append(_fmt(arc_cum[i] / arc_tot, 1.0))
    return tokens

# ── Dataset merging

def merge_axes(raw_dataset) -> Dataset:
    groups = defaultdict(dict)
    for row in raw_dataset:
        target = row["target"]
        hits   = ast.literal_eval(row["input"])
        key    = re.sub(r",\s*axis='[xyz]'", "", target).strip()
        axis_match = re.search(r"axis='([xyz])'", target)
        if not axis_match:
            continue
        groups[key][axis_match.group(1)] = hits

    merged = {"input": [], "target": []}
    for key, axes in groups.items():
        if not all(a in axes for a in ("x", "y", "z")):
            continue
        xs, ys, zs = axes["x"], axes["y"], axes["z"]
        if not (len(xs) == len(ys) == len(zs)):
            continue
        flat = []
        for x, y, z in zip(xs, ys, zs):
            flat.extend([x, y, z])
        merged["input"].append(str(flat))
        merged["target"].append(key + ")")
    return Dataset.from_dict(merged)

# ── Feature normalisation 

def normalise_input(raw_str: str) -> str:
    try:
        vals = ast.literal_eval(raw_str)
    except Exception:
        return raw_str
    if len(vals) % 3 != 0 or len(vals) == 0:
        return " ".join(_fmt(v, COORD_RANGE) for v in vals)
    pts = [vals[i:i+3] for i in range(0, len(vals), 3)]
    return " ".join(compute_helix_features(pts))

def safe_parse_list(s):
            try:
                return ast.literal_eval(f"[{s}]")
            except Exception:
                # fallback: extract numbers manually
                import re
                nums = re.findall(r'[-+]?\d*\.?\d+', s)
                return [float(x) for x in nums] if nums else []
            
def normalise_target(raw_str: str) -> str:
    def replace_list(m, scale):
        vals = safe_parse_list(m.group(1))
        return "[" + ", ".join(_fmt(v, scale) for v in vals) + "]"
    s = re.sub(r'vtx=\[([^\]]+)\]',
                lambda m: "vtx=" + replace_list(m, COORD_RANGE), raw_str)
    s = re.sub(r'mom=\[([^\]]+)\]',
                lambda m: "mom=" + replace_list(m, MOM_RANGE), s)
    return s

# ── Metric helpers 

def get_similarity(tgt_array, pred_array, tol=HIT_TOL_EVAL):
    """Fraction of TARGET hits that have a PRED hit within tol."""
    if len(tgt_array) == 0 or len(pred_array) == 0:
        return 0.0
    tgt_arr  = np.array(tgt_array,  dtype=float)
    pred_arr = np.array(pred_array, dtype=float)
    matches = sum(
        1 for hit in tgt_arr
        if np.linalg.norm(pred_arr - hit, axis=1).min() <= tol
    )
    return matches / len(tgt_arr)

def get_dsimilarity(tgt_array, pred_array, tol=HIT_TOL_EVAL):
    """Fraction of PRED hits that have NO target hit within tol (spurious)."""
    if len(pred_array) == 0:
        return 1.0
    if len(tgt_array) == 0:
        return 1.0
    tgt_arr  = np.array(tgt_array,  dtype=float)
    pred_arr = np.array(pred_array, dtype=float)
    spurious = sum(
        1 for hit in pred_arr
        if np.linalg.norm(tgt_arr - hit, axis=1).min() > tol
    )
    return spurious / len(pred_arr)

def denorm_target(pred_str: str) -> str:
    """
    Undo _fmt() normalisation back to physical units.
    Also clamps model output to [-1, 1] before rescaling to prevent
    out-of-detector coordinates that make gen_event return 0 hits.
    """
    def rescale(m, scale):
        vals = safe_parse_list(m.group(1))
        if not vals:
            return "[0.0, 0.0, 0.0]"  # fallback

        vals = [round(max(-1.0, min(1.0, v)) * scale, 1) for v in vals]
        return "[" + ", ".join(f"{v:.1f}" for v in vals) + "]"

    s = re.sub(r'vtx=\[([^\]]+)\]', lambda m: "vtx=" + rescale(m, COORD_RANGE), pred_str)
    s = re.sub(r'mom=\[([^\]]+)\]', lambda m: "mom=" + rescale(m, MOM_RANGE),   s)

    # ensure axis='x' is present (required by is_valid_gen_event)
    if "axis=" not in s:
        s = s.rstrip(")") + ", axis='x')"
    else:
        s = re.sub(r"axis='[xyz]'", "axis='x'", s)
    return s

def compression_ratio(raw_input_str: str) -> float:
    try:
        vals = ast.literal_eval(raw_input_str)
        L_i  = len(vals) // 3
        if L_i == 0:
            return float('nan')
        return (L_i - 6) / L_i * 100.0
    except Exception:
        return float('nan')

def hits_from_physical(physical_str: str):
    """Run gen_event for all 3 axes and return list of [x,y,z] triplets."""
    base = re.sub(r",\s*axis='[xyz]'", "", physical_str).rstrip(")")
    hits = []
    for axis in ("x", "y", "z"):
        hits += gevt.eval_gen_event(base + f", axis='{axis}')")
    return [hits[j:j+3] for j in range(0, len(hits), 3)]

# ── Sanity check 

print("Checking gen_event import …")
try:
    result = gevt.eval_gen_event(
        "gen_event(vtx=[262.9, 205.3, 0.0], mom=[121.7, 51.6, 14.5], axis='x')"
    )
    print(f"  gen_event OK — {len(result)} hits returned")
except Exception as e:
    print(f"  ERROR: {e}")
    raise SystemExit(1)

# ── Load & merge dataset 
print(f"\nLoading dataset from {HF_REPO}/{HF_SPLIT} …")
raw    = load_dataset(HF_REPO, split=HF_SPLIT)
print(f"  Raw rows   : {len(raw):,}")

print("Merging x/y/z axes into 3-D events …")
merged = merge_axes(raw)
print(f"  Merged events: {len(merged):,}")

raw_inputs  = [merged[i]["input"]  for i in range(len(merged))]
raw_targets = [merged[i]["target"] for i in range(len(merged))]

# ── Tokeniser & model 

print(f"\nLoading tokeniser and model ({MODEL_NAME}) …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
print(f"  Device: {DEVICE}  |  Vocab size: {len(tokenizer)}")

# ── Tokenisation 

def tokenize(examples):
    inputs  = [normalise_input(t)  for t in examples["input"]]
    targets = [normalise_target(t) for t in examples["target"]]
    model_inputs = tokenizer(inputs,  max_length=MAX_INPUT_LEN,  truncation=True)
    labels       = tokenizer(text_target=targets,
                             max_length=MAX_TARGET_LEN, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenising …")
tok_train     = merged.map(tokenize, batched=True, remove_columns=["input", "target"])
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# ── Physics-aware trainer

class PhysicsTrainer(Seq2SeqTrainer):
    """
    Phase 1 (step < CE_ONLY_STEPS): pure cross-entropy only.
        → model re-learns correct scale mapping first
    Phase 2 (step >= CE_ONLY_STEPS): adds physics penalty every GENERATE_EVERY steps.
        → gently guides predictions toward physically correct hit positions

    total_loss = CE + PHYSICS_WEIGHT*(1-S) + PHYSICS_WEIGHT*D
    Physics penalty is capped at CE loss so it never dominates.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._phys_step = 0

    

# ── Metrics callback 

class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, tokenizer, train_dataset, raw_inputs, raw_targets,
                 log_every=200, n_samples=64):
        self.tokenizer     = tokenizer
        self.train_dataset = train_dataset
        self.raw_inputs    = raw_inputs
        self.raw_targets   = raw_targets
        self.log_every     = log_every
        self.n_samples     = n_samples
        self.steps         = []
        self.s_all         = []
        self.d_all         = []
        self.c_all         = []
        self.c_pred_all    = []

    def on_step_end(self, args, state: TrainerState,
                    control: TrainerControl, **kwargs):
        if state.global_step % self.log_every != 0 or state.global_step == 0:
            return

        model = kwargs["model"]
        model.eval()

        indices     = np.random.choice(len(self.train_dataset), self.n_samples, replace=False)
        batch       = self.train_dataset.select(indices)
        raw_in_batch = [self.raw_inputs[i] for i in indices]

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

        s_batch, d_batch, c_batch, c_pred_batch = [], [], [], []

        print(f"\n{'='*60}")
        print(f"Step {state.global_step} — metrics on {self.n_samples} samples")

        for i, (raw_in, label_ids) in enumerate(zip(raw_in_batch, batch["labels"])):
            pred_norm = self.tokenizer.decode(gen[i], skip_special_tokens=True)
            tgt_norm  = self.tokenizer.decode(
                [t for t in label_ids if t != -100], skip_special_tokens=True)
            true_target = self.raw_targets[indices[i]]

            try:
                pred_physical = denorm_target(pred_norm)
                tgt_physical  = denorm_target(tgt_norm)
            except Exception:
                s_batch.append(0.0)
                d_batch.append(1.0)
                continue

            is_valid = gevt.is_valid_gen_event(pred_physical)

            # print first 3 samples for quick visual check
            if i < 3:
                print(f"  [{i}] TRUE  : {true_target}")
                print(f"  [{i}] PRED  : {pred_physical}  valid={is_valid}")
                print(f"  [{i}] TARGET: {tgt_physical}")

            if not is_valid:
                s_batch.append(0.0)
                d_batch.append(1.0)
                continue

            c_batch.append(compression_ratio(raw_in))

            try:
                pred_hits  = hits_from_physical(pred_physical)
                L_i_pred   = len(pred_hits)
                if L_i_pred > 0:
                    c_pred_batch.append((L_i_pred - 6) / L_i_pred * 100.0)
            except Exception:
                pass

            try:
                tgt_array  = hits_from_physical(tgt_physical)
                pred_array = hits_from_physical(pred_physical)
                s_batch.append(get_similarity(tgt_array,  pred_array))
                d_batch.append(get_dsimilarity(tgt_array, pred_array))
            except Exception as e:
                print(f"  [{i}] eval error: {e}")
                s_batch.append(0.0)
                d_batch.append(1.0)

        if s_batch:
            mean_s = np.mean(s_batch)
            mean_d = np.mean(d_batch)
            mean_c = np.nanmean(c_batch) if c_batch else float('nan')
            print(f"  Mean S={mean_s:.4f}  D={mean_d:.4f}  C={mean_c:.1f}")

            self.steps.append(state.global_step)
            MAX_STORE = 5000
            self.s_all      = (self.s_all      + s_batch)[-MAX_STORE:]
            self.d_all      = (self.d_all      + d_batch)[-MAX_STORE:]
            self.c_all      = (self.c_all      + c_batch)[-MAX_STORE:]
            self.c_pred_all = (self.c_pred_all + c_pred_batch)[-MAX_STORE:]

            # save plot + CSV every log step
            self.plot(save_path=os.path.join(SAVE_DIR, "eval.png"))

            import pandas as pd
            pd.DataFrame({
                "step": self.steps,
            }).to_csv(os.path.join(SAVE_DIR, "steps.csv"), index=False)
            pd.DataFrame({
                "S": self.s_all,
                "D": self.d_all,
            }).to_csv(os.path.join(SAVE_DIR, "metrics.csv"), index=False)

        print('='*60)
        model.train()

    

#  Training arguments 

training_args = Seq2SeqTrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    learning_rate               = LR,
    weight_decay                = 0.01,
    warmup_steps                = 500,
    predict_with_generate       = True,
    generation_max_length       = MAX_TARGET_LEN,
    fp16                        = USE_CUDA,
    dataloader_pin_memory       = USE_CUDA,
    gradient_accumulation_steps = 2,
    gradient_checkpointing      = True,
    eval_strategy               = "no",
    save_strategy               = "steps",
    save_steps                  = 400,
    save_total_limit            = 3,
    logging_steps               = 20,
    remove_unused_columns       = False,
    report_to                   = "none",
)

#  Callback & trainer

metrics_cb = MetricsLoggerCallback(
    tokenizer     = tokenizer,
    train_dataset = tok_train,
    raw_inputs    = raw_inputs,
    raw_targets   = raw_targets,
    log_every     = 200,
    n_samples     = 64,
)

trainer = PhysicsTrainer(
    model            = model,
    args             = training_args,
    train_dataset    = tok_train,
    eval_dataset     = None,
    data_collator    = data_collator,
    processing_class = tokenizer,
    callbacks        = [metrics_cb],
)

# set LR explicitly (overrides any checkpoint-loaded optimizer state)
trainer.create_optimizer_and_scheduler(num_training_steps=training_args.max_steps)
for param_group in trainer.optimizer.param_groups:
    param_group["lr"] = LR

#  Train 

print("\nStarting training …")
print(f"  Phase 1: CE only  (steps 0 → {CE_ONLY_STEPS})")
print(f"  Phase 2: CE + physics penalty (steps {CE_ONLY_STEPS} → end)")

checkpoint = get_latest_checkpoint(OUTPUT_DIR)

if checkpoint is not None:
    print(f"Resuming from checkpoint: {checkpoint}")
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    trainer.train()
else:
    print("No checkpoint found, starting fresh.")
    trainer.train()

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nModel saved → {SAVE_DIR}")

metrics_cb.plot(save_path=os.path.join(SAVE_DIR, "eval_final.png"))