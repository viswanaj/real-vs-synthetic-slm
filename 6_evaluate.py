"""
Evaluation Script — Evaluate one LoRA-tuned model at a time (mix_0 … mix_100).
Generates abstracts from held-out titles, computes automated metrics,
and optionally produces a blind review file when comparing multiple models
(use a separate workflow or merge outputs for that).
"""

import argparse
import os
import json
import math
import random

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

CORPUS_A_PATH = "corpus_a/corpus_a.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a single LoRA adapter on the shared test set."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Which adapter to evaluate: synthetic ratio "
            f"{min(SYNTHETIC_RATIOS)}–{max(SYNTHETIC_RATIOS)} (e.g. 25), "
            "or name like mix_25."
        ),
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Override adapter directory (default: models/model_mix_<pct>).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for test-set sampling and generation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_TEST_SAMPLES,
        help="Number of held-out titles to evaluate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Max tokens generated per sample.",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip perplexity to speed up CPU evaluation.",
    )
    parser.add_argument(
        "--ppl-samples",
        type=int,
        default=None,
        help="If set, compute perplexity on only this many samples (after generation).",
    )
    return parser.parse_args()


def resolve_model(selector, adapter_override=None):
    """
    Return (model_name, adapter_path) for one run.
    selector: e.g. "25", "mix_25", "90"
    """
    s = selector.strip().lower()
    if s.startswith("mix_"):
        s = s[4:]
    try:
        pct = int(s)
    except ValueError:
        raise SystemExit(
            f"Invalid --model {selector!r}; use an integer from {SYNTHETIC_RATIOS} "
            "or a name like mix_25."
        )
    if pct not in SYNTHETIC_RATIOS:
        raise SystemExit(
            f"--model {pct} is not in configured ratios {SYNTHETIC_RATIOS}."
        )
    name = f"mix_{pct}"
    path = adapter_override if adapter_override else f"models/model_mix_{pct}"
    return name, path


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


def generate_abstract(model, tokenizer, title, seed, max_new_tokens):
    """Generate an abstract given a title, using the same prompt format as training."""
    prompt = f"### Title:\n{title}\n\n### Abstract:\n"
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=256
    ).to(DEVICE)

    torch.manual_seed(seed)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
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


def evaluate_model(
    model_name,
    adapter_path,
    test_set,
    seed,
    max_new_tokens,
    skip_perplexity=False,
    ppl_samples=None,
):
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

        generated = generate_abstract(model, tokenizer, title, seed, max_new_tokens)
        generations.append(generated)

        rouge = compute_rouge(generated, reference)
        rouge_scores.append(rouge)

        if not skip_perplexity:
            if ppl_samples is None or i < ppl_samples:
                formatted = f"### Title:\n{title}\n\n### Abstract:\n{reference}"
                ppl = compute_perplexity(model, tokenizer, formatted)
                perplexities.append(ppl)

    gen_lengths = [len(g.split()) for g in generations]

    results = {
        "model": model_name,
        "adapter_path": adapter_path,
        "num_samples": len(test_set),
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
    if skip_perplexity:
        results["perplexity"] = None
    elif perplexities:
        results["perplexity"] = {
            "mean": round(float(np.mean(perplexities)), 2),
            "median": round(float(np.median(perplexities)), 2),
            "std": round(float(np.std(perplexities)), 2),
            "computed_on_samples": len(perplexities),
        }
    else:
        results["perplexity"] = None

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
            if category == "perplexity" and not r.get("perplexity"):
                val = "n/a"
            else:
                val = r[category][key]
            line += f" {val:>{col_width}}"
        print(line)

    print(f"{'=' * table_width}\n")


def main():
    args = parse_args()
    model_name, adapter_path = resolve_model(args.model, args.adapter)

    print("Model Evaluation — Llama 3.2 3B LoRA (single adapter)")
    print(f"Model: {model_name} — {adapter_path}")
    print(f"Device: {DEVICE}")
    print(f"Seed: {args.seed}")
    print(f"Test samples: {args.num_samples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Skip perplexity: {args.skip_perplexity}")
    if args.ppl_samples is not None:
        print(f"Perplexity samples: {args.ppl_samples}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(adapter_path):
        raise SystemExit(f"Adapter path not found: {adapter_path}")

    print("Loading test set from Corpus A (human-written references)...")
    corpus_a = load_corpus_txt(CORPUS_A_PATH)
    test_set = build_test_set(corpus_a, args.num_samples, args.seed)
    print(f"Selected {len(test_set)} test samples\n")

    test_set_file = os.path.join(OUTPUT_DIR, f"test_set_seed_{args.seed}.json")
    with open(test_set_file, "w") as f:
        json.dump(test_set, f, indent=2)

    results, _ = evaluate_model(
        model_name,
        adapter_path,
        test_set,
        args.seed,
        args.max_new_tokens,
        skip_perplexity=args.skip_perplexity,
        ppl_samples=args.ppl_samples,
    )
    all_results = [results]

    results_file = os.path.join(OUTPUT_DIR, f"results_{model_name}_seed_{args.seed}.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print_comparison_table(all_results)

    print(
        "\nSkipping blind_review.json (single-model run). "
        "Merge generations from multiple runs if you need a combined blind review."
    )

    print(f"\nResults saved under {OUTPUT_DIR}/")
    print("Files:")
    print(f"  {os.path.basename(results_file)} — Metrics for {model_name}")
    print(f"  {os.path.basename(test_set_file)} — The {len(test_set)} titles/references used")


if __name__ == "__main__":
    main()
