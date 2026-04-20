
import os, re, ast
import torch
import numpy as np
from datasets import load_dataset
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
MODEL_NAME    = "t5-small"
HF_REPO       = "christyelsa/g2-train-05"
HF_SPLIT      = "train"
curr_ver = "00"
next_ver = "01"
model_name = "t5-small"

BASE_DIR = "./runs"   # or any folder you like

OUTPUT_DIR = os.path.join(BASE_DIR, f"{model_name}-{curr_ver}")
SAVE_DIR   = os.path.join(BASE_DIR, f"{model_name}-{next_ver}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

USE_CUDA      = torch.cuda.is_available()
DEVICE        = torch.device("cuda" if USE_CUDA else "cpu")
BATCH_SIZE    = 32 if USE_CUDA else 4
NUM_EPOCHS    = 100
LR            = 2e-4



COORD_RANGE  = 300.0   # mm  — covers vtx and hit coordinates
MOM_RANGE    = 300.0   # MeV — covers typical momenta

def _fmt(v, scale):
    """Normalise v to [-1,1] and format as a fixed-width 6-char string."""
    return f"{v / scale:+.3f}"

def normalise_input(raw_str):
    """
    Convert a raw hit-list string like "[213.9,190.2,...]" into a
    space-separated sequence of normalised tokens: "+0.713 +0.634 ..."
    """
    try:
        vals = ast.literal_eval(raw_str)
    except Exception:
        return raw_str   # fall back to original if parsing fails
    return " ".join(_fmt(v, COORD_RANGE) for v in vals)

def normalise_target(raw_str):
    """
    Normalise all float values inside a gen_event(...) target string.
    vtx and mom coordinates are scaled separately.
    The axis token ('x','y','z') is left unchanged.
    """
    def replace_list(m, scale):
        vals = ast.literal_eval(f"[{m.group(1)}]")
        return "[" + ", ".join(_fmt(v, scale) for v in vals) + "]"

    s = re.sub(
        r'vtx=\[([^\]]+)\]',
        lambda m: "vtx=" + replace_list(m, COORD_RANGE),
        raw_str,
    )
    s = re.sub(
        r'mom=\[([^\]]+)\]',
        lambda m: "mom=" + replace_list(m, MOM_RANGE),
        s,
    )
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

#  Load & split dataset 
print(f"\nLoading dataset from {HF_REPO}/{HF_SPLIT} …")
raw = load_dataset(HF_REPO, split=HF_SPLIT)
train_ds = raw
eval_ds  = raw   # just reuse same dataset
# split = raw.train_test_split(test_size=0.05, seed=42)
# train_ds = split["train"]
# eval_ds  = split["test"]
print(f"  Train: {len(train_ds):,}  |  Eval: {len(eval_ds):,}")

# Tokeniser & model 
print(f"\nLoading tokeniser and model ({MODEL_NAME}) …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
print(f"  Device: {DEVICE}  |  Vocab size: {len(tokenizer)}")

#  Tokenisation with float normalisation 
def tokenize(examples):
    inputs  = [normalise_input(t)  for t in examples["input"]]
    targets = [normalise_target(t) for t in examples["target"]]

    model_inputs = tokenizer(inputs,  max_length=256, truncation=True)
    labels       = tokenizer(text_target=targets, max_length=256, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenising …")
tok_train = train_ds.map(tokenize, batched=True, remove_columns=["input", "target"])
tok_eval  = eval_ds.map( tokenize, batched=True, remove_columns=["input", "target"])

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

#  Periodic sample logger (callback, zero training overhead)
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
            # FIX: pad sequences to the same length before stacking into a tensor
            input_ids_list = [torch.tensor(x) for x in batch["input_ids"]]
            mask_list      = [torch.tensor(x) for x in batch["attention_mask"]]

            ids  = torch.nn.utils.rnn.pad_sequence(
                       input_ids_list, batch_first=True,
                       padding_value=self.tokenizer.pad_token_id
                   ).to(DEVICE)
            mask = torch.nn.utils.rnn.pad_sequence(
                       mask_list, batch_first=True,
                       padding_value=0
                   ).to(DEVICE)

            gen = model.generate(ids, attention_mask=mask, max_length=256)

        print(f"\n{'='*60}")
        print(f"Step {state.global_step} — sample predictions")
        for i in range(self.n_samples):
            inp  = self.tokenizer.decode(batch["input_ids"][i],  skip_special_tokens=True)
            pred = self.tokenizer.decode(gen[i],                  skip_special_tokens=True)
            lbl  = self.tokenizer.decode(
                [t for t in batch["labels"][i] if t != -100],
                skip_special_tokens=True,
            )
            print(f"  Input : {inp[:80]}")
            print(f"  Target: {lbl}")
            print(f"  Pred  : {pred}")
            print(f"  Valid format: {gevt.is_valid_gen_event(pred)}")
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
    warmup_steps                = 100,
    predict_with_generate       = True,
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

# Trainer 
trainer = Seq2SeqTrainer(
    model         = model,
    args          = training_args,
    train_dataset = tok_train,
    eval_dataset  = tok_eval,
    data_collator = data_collator,
    processing_class = tokenizer,
    callbacks     = [
        SampleLoggerCallback(
            tokenizer    = tokenizer,
            eval_dataset = tok_eval,
            log_every    = 200,
            n_samples    = 3,
        )
    ],
)

#  Second sanity check before long run
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

# Save 
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nModel saved → {SAVE_DIR}")

print("\n=== Final evaluation ===")
final = trainer.evaluate()
print(f"  eval_loss : {final['eval_loss']:.4f}")