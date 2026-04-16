"""
Evaluation Script — Compare 7 LoRA-tuned models (mix_0 through mix_100)
Generates abstracts from held-out titles, computes automated metrics,
and produces a blind review file for manual comparison.
"""

import os
import json
import math
import random
from collections import Counter

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge_score import rouge_scorer

# --- Config ---
MODEL_ID = "meta-llama/Llama-3.2-3B"
DEVICE = "cpu"
MAX_NEW_TOKENS = 200
NUM_TEST_SAMPLES = 200
SEED = 42
OUTPUT_DIR = "evaluation"

SYNTHETIC_RATIOS = [0, 10, 25, 50, 75, 90, 100]
MODELS = [
    (f"mix_{pct}", f"models/model_mix_{pct}")
    for pct in SYNTHETIC_RATIOS
]

CORPUS_A_PATH = "corpus_a/corpus_a.txt"


def load_corpus_txt(path):
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


def build_test_set(records, n, seed):
    """Select n random records as the shared test set."""
    rng = random.Random(seed)
    if n > len(records):
        n = len(records)
    return rng.sample(records, n)


def load_model(adapter_path):
    """Load base model with LoRA adapter merged in."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map={"": DEVICE},
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_abstract(model, tokenizer, title):
    """Generate an abstract given a title, using the same prompt format as training."""
    prompt = f"### Title:\n{title}\n\n### Abstract:\n"
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=256
    ).to(DEVICE)

    torch.manual_seed(SEED)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    marker = "### Abstract:\n"
    if marker in full_text:
        return full_text.split(marker, 1)[1].strip()
    return full_text[len(prompt):].strip()


def compute_perplexity(model, tokenizer, text):
    """Compute perplexity of a text under the model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return math.exp(outputs.loss.item())


def compute_rouge(generated, reference):
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}


def compute_distinct_n(texts, n):
    """Compute distinct-n: ratio of unique n-grams to total n-grams across all texts."""
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        all_ngrams.extend(zip(*[tokens[i:] for i in range(n)]))
    if not all_ngrams:
        return 0.0
    return round(len(set(all_ngrams)) / len(all_ngrams), 4)


def evaluate_model(model_name, adapter_path, test_set):
    """Run full evaluation for one model."""
    print(f"\n{'='*55}")
    print(f"Evaluating Model {model_name} — {adapter_path}")
    print(f"{'='*55}")

    model, tokenizer = load_model(adapter_path)

    generations = []
    rouge_scores = []
    perplexities = []

    for i, sample in enumerate(test_set):
        title = sample["title"]
        reference = sample["abstract"]

        print(f"  [{i+1}/{len(test_set)}] Generating for: {title[:70]}...")

        generated = generate_abstract(model, tokenizer, title)
        generations.append(generated)

        rouge = compute_rouge(generated, reference)
        rouge_scores.append(rouge)

        formatted = f"### Title:\n{title}\n\n### Abstract:\n{reference}"
        ppl = compute_perplexity(model, tokenizer, formatted)
        perplexities.append(ppl)

    gen_lengths = [len(g.split()) for g in generations]

    results = {
        "model": model_name,
        "adapter_path": adapter_path,
        "num_samples": len(test_set),
        "perplexity": {
            "mean": round(float(np.mean(perplexities)), 2),
            "median": round(float(np.median(perplexities)), 2),
            "std": round(float(np.std(perplexities)), 2),
        },
        "rouge": {
            "rouge1": round(float(np.mean([s["rouge1"] for s in rouge_scores])), 4),
            "rouge2": round(float(np.mean([s["rouge2"] for s in rouge_scores])), 4),
            "rougeL": round(float(np.mean([s["rougeL"] for s in rouge_scores])), 4),
        },
        "diversity": {
            "distinct_1": compute_distinct_n(generations, 1),
            "distinct_2": compute_distinct_n(generations, 2),
        },
        "generation_length": {
            "mean": round(float(np.mean(gen_lengths)), 1),
            "median": round(float(np.median(gen_lengths)), 1),
            "min": int(np.min(gen_lengths)),
            "max": int(np.max(gen_lengths)),
        },
    }

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results, generations


def build_blind_review(test_set, all_generations, model_names, seed):
    """
    Produce a shuffled, anonymized review file.
    Each entry shows the title and labeled outputs (S1, S2, ... S7)
    with a hidden key mapping labels to models.
    """
    rng = random.Random(seed)
    review_entries = []

    labels = [f"S{i+1}" for i in range(len(model_names))]

    for i, sample in enumerate(test_set):
        perm = list(range(len(model_names)))
        rng.shuffle(perm)
        shuffled_labels = {labels[j]: model_names[perm[j]] for j in range(len(model_names))}

        entry = {
            "id": i + 1,
            "title": sample["title"],
            "reference_abstract": sample["abstract"],
            "outputs": {},
            "_key": shuffled_labels,
        }
        for j, label in enumerate(labels):
            entry["outputs"][label] = all_generations[perm[j]][i]

        review_entries.append(entry)

    return review_entries


def print_comparison_table(all_results):
    """Print a summary comparison table."""
    col_width = 12
    num_models = len(all_results)
    table_width = 25 + col_width * num_models + 4

    print(f"\n{'=' * table_width}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * table_width}")

    header = f"{'Metric':<25}"
    for r in all_results:
        header += f" {r['model']:>{col_width}}"
    print(header)
    print("-" * table_width)

    rows = [
        ("Perplexity (mean)", "perplexity", "mean"),
        ("Perplexity (median)", "perplexity", "median"),
        ("ROUGE-1", "rouge", "rouge1"),
        ("ROUGE-2", "rouge", "rouge2"),
        ("ROUGE-L", "rouge", "rougeL"),
        ("Distinct-1", "diversity", "distinct_1"),
        ("Distinct-2", "diversity", "distinct_2"),
        ("Avg gen length", "generation_length", "mean"),
    ]

    for label, category, key in rows:
        line = f"{label:<25}"
        for r in all_results:
            line += f" {r[category][key]:>{col_width}}"
        print(line)

    print(f"{'=' * table_width}\n")


def main():
    print("Model Evaluation — Llama 3.2 3B LoRA")
    print(f"Device: {DEVICE}")
    print(f"Test samples: {NUM_TEST_SAMPLES}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading test set from Corpus A (human-written references)...")
    corpus_a = load_corpus_txt(CORPUS_A_PATH)
    test_set = build_test_set(corpus_a, NUM_TEST_SAMPLES, SEED)
    print(f"Selected {len(test_set)} test samples\n")

    with open(os.path.join(OUTPUT_DIR, "test_set.json"), "w") as f:
        json.dump(test_set, f, indent=2)

    all_results = []
    all_generations = []
    model_names = []

    for model_name, adapter_path in MODELS:
        if not os.path.exists(adapter_path):
            print(f"Skipping Model {model_name} — {adapter_path} not found")
            continue
        results, generations = evaluate_model(model_name, adapter_path, test_set)
        all_results.append(results)
        all_generations.append(generations)
        model_names.append(model_name)

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print_comparison_table(all_results)

    print("Building blind review file...")
    blind_review = build_blind_review(test_set, all_generations, model_names, SEED)
    with open(os.path.join(OUTPUT_DIR, "blind_review.json"), "w") as f:
        json.dump(blind_review, f, indent=2)
    print(f"Saved {len(blind_review)} entries to {OUTPUT_DIR}/blind_review.json")

    print(f"\nAll results saved to {OUTPUT_DIR}/")
    print("Files:")
    print(f"  results.json       — Full metrics for all models")
    print(f"  test_set.json      — The {len(test_set)} titles/references used")
    print(f"  blind_review.json  — Anonymized outputs for manual review")


if __name__ == "__main__":
    main()
