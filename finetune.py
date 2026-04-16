"""
Fine-tuning Script — Llama 3.2 3B with LoRA
Trains 3 separate models, one per corpus (A, B, C)
Configured for Apple Silicon (MPS) with 48GB unified memory
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# --- Config ---
MODEL_ID = "meta-llama/Llama-3.2-3B"
MAX_LENGTH = 512        # Max tokens per training sample
BATCH_SIZE = 4          # Safe for 48GB MPS
GRAD_ACCUM = 4          # Effective batch size = 16
EPOCHS = 3
LEARNING_RATE = 2e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SYNTHETIC_RATIOS = [0, 10, 25, 50, 75, 90, 100]
CORPORA = [
    (f"mix_{pct}", f"training_sets/mix_{pct}/mix_{pct}.txt", f"models/model_mix_{pct}")
    for pct in SYNTHETIC_RATIOS
]

# LoRA config — efficient fine-tuning without touching base weights
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # Rank — higher = more expressive, more memory
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.05,
    target_modules=[         # Which layers to adapt
        "q_proj", "v_proj",
        "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none"
)


def load_corpus(path):
    """Load corpus .txt file and return list of {title, abstract} dicts."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    records = []
    for block in text.strip().split("\n---\n"):
        block = block.strip()
        if not block:
            continue
        title, abstract = "", ""
        for line in block.split("\n"):
            if line.startswith("Title: "):
                title = line[len("Title: "):]
            elif line.startswith("Abstract: "):
                abstract = line[len("Abstract: "):]
        if title and abstract:
            records.append({"title": title, "abstract": abstract})
    return records


def format_sample(record):
    """
    Format each record as a prompt/completion pair for causal LM training.
    Task: given a title, generate the abstract.
    """
    return f"### Title:\n{record['title']}\n\n### Abstract:\n{record['abstract']}"


def build_dataset(records, tokenizer):
    """Tokenize and prepare dataset for training."""
    texts = [format_sample(r) for r in records]

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds.set_format(type="torch")
    return ds


def train_model(corpus_name, corpus_path, output_dir):
    print(f"\n{'='*55}")
    print(f"Training Model {corpus_name} — {corpus_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {DEVICE}")
    print(f"{'='*55}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,   # float16 is well supported on MPS
        device_map={"": DEVICE},
    )

    # Attach LoRA adapters
    print("Attaching LoRA adapters...")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # Build dataset
    print("Building dataset...")
    records = load_corpus(corpus_path)
    print(f"  Loaded {len(records)} records")
    dataset = build_dataset(records, tokenizer)

    # Split 90/10 train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # Training arguments — MPS-specific settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        fp16=False,              # MPS doesn't support fp16 training
        bf16=False,              # bf16 also not supported on MPS
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",        # Disable wandb/tensorboard
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False                # Causal LM, not masked
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    # Train
    print(f"\nStarting training...")
    trainer.train()

    # Save LoRA adapter weights + tokenizer
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metadata
    metadata = {
        "corpus": corpus_name,
        "corpus_path": corpus_path,
        "base_model": MODEL_ID,
        "num_records": len(records),
        "epochs": EPOCHS,
        "lora_r": LORA_CONFIG.r,
        "lora_alpha": LORA_CONFIG.lora_alpha,
        "max_length": MAX_LENGTH,
        "device": DEVICE,
    }
    with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model {corpus_name} done.\n")

    # Free memory before next model
    del model
    torch.mps.empty_cache()

    return trainer.state.log_history


def main():
    print("Llama 3.2 3B LoRA Fine-tuning")
    print(f"Device: {DEVICE}")
    print(f"Training sets: {[c[0] for c in CORPORA]}")
    print(f"Output: models/model_mix_{{0..100}}\n")

    if DEVICE == "cpu":
        print("Warning: MPS not available, falling back to CPU. Training will be slow.")

    all_logs = {}

    for corpus_name, corpus_path, output_dir in CORPORA:
        if not os.path.exists(corpus_path):
            print(f"Skipping Corpus {corpus_name} — {corpus_path} not found")
            continue
        logs = train_model(corpus_name, corpus_path, output_dir)
        all_logs[corpus_name] = logs

    # Save combined training logs
    with open("models/training_logs.json", "w") as f:
        json.dump(all_logs, f, indent=2)

    print("\n" + "="*55)
    print(f"All {len(all_logs)} models trained successfully.")
    print(f"Saved to: models/model_mix_{{0..100}}")
    print("Next step: run evaluate.py")
    print("="*55)


if __name__ == "__main__":
    main()
