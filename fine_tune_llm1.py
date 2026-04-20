#%%
from transformers import (
    T5ForConditionalGeneration,  # Better Seq2Seq architecture
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    T5Tokenizer
)

import ast
import gen_event as gevt
#import icaltracks as ical
from datasets import Dataset
from datasets import load_dataset
#from peft import LoraConfig, get_peft_model, TaskType
import torch
import re
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

print("testing if functions are predefined...")
try:
    print(gevt.eval_gen_event("gen_event(vtx=[262.9, 205.3, 0.0],mom=[121.7, 51.6, 14.5],axis='x')"))
    print("gen_event function defined!")
except:
    print("Error: gen_event function not defined")
    exit()
#%%
curr_ver = "00"
next_ver ="01"
model_name = "t5-small"
new_model_name= f"{model_name}-{next_ver}"
output_dir=f"{model_name}-{curr_ver}"

hf_repo=  "christyelsa/g2-train-05" # arrays encoded as strings
#hf_repo=  "deepaksamuel-cuk/cleantracks-str" # arrays encoded as strings
hf_repo_split = "train"
train_dataset = load_dataset(hf_repo,split=hf_repo_split)

print(f"Training dataset loaded from {hf_repo}/{hf_repo_split} with {len(train_dataset)} samples.")

tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer loaded from {model_name} with {len(tokenizer)} tokens.")
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Detect CUDA and move model to the appropriate device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")
model.to(device)


#%%
# 3. Tokenize dataset with proper Seq2Seq formatting
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input"],
        max_length=512,
        truncation=True
    )
    # with tokenizer.as_target_tokenizer():
    labels = tokenizer(
        text_target=examples["target"],
        max_length=512,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['input', 'target']
)



data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)



training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_steps=100,
    logging_steps=10,
    save_steps=500, # 100
    # Choose sensible batch sizes depending on whether a GPU is available.
    per_device_train_batch_size=(250 if torch.cuda.is_available() else 1),  # Further reduced to 2
    per_device_eval_batch_size=(250 if torch.cuda.is_available() else 1),
    num_train_epochs=5000,  # Reduced from 2000 to 1000
    learning_rate=2e-4,  # Trying higher LR, initially at 3e-4
    weight_decay=0.01,
    warmup_steps=100,   #changed from 500 to 100
    predict_with_generate=True,  # Critical for Seq2Seq
    # fp16=True,
    # Only enable FP16 and pinned memory when CUDA is available to avoid
    # warnings/errors on CPU-only environments.
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=torch.cuda.is_available(),
    remove_unused_columns=False,  # ← ADD THIS
    # fp16=torch.cuda.is_available(),
    #metric_for_best_model="train_loss",  # Use validation loss for best model
    metric_for_best_model=None,  # Use validation loss for best model
    gradient_accumulation_steps=1,  # Increased to compensate
    gradient_checkpointing=True,  # Added to save memory
    report_to="none",
    # load_best_model_at_end=True,  # Save best model based on eval loss
    # eval_strategy="steps",  # Enable evaluation or epoch
    # save_strategy="steps",
    greater_is_better=False,  # Lower loss is better
    save_total_limit=3,  # Limit the total amount of checkpoints. Deletes the older checkpoints.
)

# 6. Custom trainer with format enforcement
class CodeGenerationTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_validation_loss = float('inf')
    def compute_loss(self, model, inputs,num_items_in_batch=0, return_outputs=False):
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
        )

        loss = outputs.loss

        if model.training or not model.training:
            # Generate predictions for format checking
            

            input_texts = self.processing_class.batch_decode(inputs["input_ids"],skip_special_tokens=True)
            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512  # Reduced from 2048 to save memory
            )

            format_penalty=0
            similarity=0
            dsimilarity=1
           
            for i, (input_text, pred) in enumerate(zip(input_texts, generated)):
                pred_text = self.processing_class.decode(pred, skip_special_tokens=True)

                if self.state.global_step % 200 == 0 and i < 5:
                    print("\n====== SAMPLE PREDICTION ======")
                    print("Input hits:", input_text[:200])
                    print("Predicted event:", pred_text)
                
                if  gevt.is_valid_gen_event(pred_text):
                    try:
                        format_penalty += 0.0
                        in_array = ast.literal_eval(input_text)
                        #in_array = gevt.eval_gen_event(input_text)
                        pred_array = gevt.eval_gen_event(pred_text)
                        similarity += gevt.get_similarity(in_array, pred_array) # how many in the input array are in output array
                        dsimilarity += gevt.get_dsimilarity(in_array, pred_array) # how many in the output array are not in input array
                    except:
                        format_penalty +=100
                        dsimilarity +=1
                        print(len(in_array), len(pred_array))
                        print("Error while evaluating, though formatting is correct...")
                        print(f"in: {input_text}")
                        print(f"out: {pred_text}")
                else:
                    format_penalty += 100
                    dsimilarity += 1
            if dsimilarity >1:
                dsimilarity = dsimilarity-1
            
            similarity = similarity/len(generated)
            dsimilarity = dsimilarity/len(generated)
            format_penalty = format_penalty/len(generated)
            # print(f'loss:{loss} format_penalty:{format_penalty:.2f} similarity:{similarity:.2f} dsimilarity:{dsimilarity:.2f} len:{len(generated)}')
            total_loss = loss + 5*(1-similarity) + 5*dsimilarity
            # print(total_loss, len(outputs))
            return (total_loss, outputs) if return_outputs else total_loss
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to track validation loss"""
        eval_output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Track best validation loss
        current_val_loss = eval_output.get("eval_loss", float('inf'))
        if current_val_loss < self.best_validation_loss:
            self.best_validation_loss = current_val_loss
            print(f"🎉 New best validation loss: {current_val_loss:.4f}")
        
        return eval_output

# 7. Create and run trainer
trainer = CodeGenerationTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    # eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    processing_class=tokenizer
)

print("testing if functions are predefined...")
try:
    gevt.gen_event(vtx=[262.9, 205.3, 0.0],mom=[121.7, 51.6, 14.5],axis="x")
    print("gen_event function defined!")
except:
    print("Error: gen_event function not defined")
    exit()

print("Starting training...")
trainer.train()

# 8. Save model
model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)
print(f"model saved to {new_model_name}")

print("\n=== Final Evaluation ===")
final_eval = trainer.evaluate()
print(f"Final validation loss: {final_eval['eval_loss']:.4f}")
print(f"Best validation loss: {trainer.best_validation_loss:.4f}")



#%%
