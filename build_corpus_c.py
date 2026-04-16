"""
Training Set Builder — Variable Synthetic Ratios
Blends Corpus A (human) and Corpus B (AI-generated) into 7 training sets
with 0%, 10%, 25%, 50%, 75%, 90%, and 100% synthetic content.
Each training set has 1000 total abstracts.
"""

import json
import os
import random

CORPUS_A_PATH = "corpus_a/corpus_a.json"
CORPUS_B_PATH = "corpus_b/corpus_b.json"
OUTPUT_DIR = "training_sets"
RANDOM_SEED = 42
TARGET_SIZE = 1000

SYNTHETIC_RATIOS = [0, 10, 25, 50, 75, 90, 100]


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_mix(corpus_a, corpus_b, synthetic_pct, target_size, rng):
    """Build a training set with the given percentage of synthetic content."""
    ai_count = round(target_size * synthetic_pct / 100)
    human_count = target_size - ai_count

    if human_count > len(corpus_a):
        print(f"  Warning: need {human_count} human records but only {len(corpus_a)} available")
        human_count = len(corpus_a)
    if ai_count > len(corpus_b):
        print(f"  Warning: need {ai_count} AI records but only {len(corpus_b)} available")
        ai_count = len(corpus_b)

    sample_a = rng.sample(corpus_a, human_count)
    sample_b = rng.sample(corpus_b, ai_count)

    for r in sample_a:
        r = dict(r)
    for r in sample_b:
        r = dict(r)

    mixed = []
    for r in sample_a:
        rec = dict(r)
        rec["corpus"] = f"mix_{synthetic_pct}"
        rec["corpus_source"] = "A_human"
        mixed.append(rec)

    for r in sample_b:
        rec = dict(r)
        rec["corpus"] = f"mix_{synthetic_pct}"
        rec["corpus_source"] = "B_ai"
        mixed.append(rec)

    rng.shuffle(mixed)
    return mixed, human_count, ai_count


def save_training_set(records, synthetic_pct, human_count, ai_count, output_dir):
    """Save a training set as JSON, TXT, and manifest."""
    set_dir = os.path.join(output_dir, f"mix_{synthetic_pct}")
    os.makedirs(set_dir, exist_ok=True)

    json_path = os.path.join(set_dir, f"mix_{synthetic_pct}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(set_dir, f"mix_{synthetic_pct}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"Title: {r['title']}\n")
            f.write(f"Abstract: {r['abstract']}\n")
            f.write("\n---\n\n")

    manifest = {
        "synthetic_pct": synthetic_pct,
        "total": len(records),
        "human_count": human_count,
        "ai_count": ai_count,
        "human_fraction": round(human_count / len(records), 4),
        "ai_fraction": round(ai_count / len(records), 4),
        "random_seed": RANDOM_SEED,
        "corpus_a_source": CORPUS_A_PATH,
        "corpus_b_source": CORPUS_B_PATH,
    }
    with open(os.path.join(set_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return set_dir


def main():
    print("=" * 55)
    print("Training Set Builder — Variable Synthetic Ratios")
    print(f"Ratios: {SYNTHETIC_RATIOS}% synthetic")
    print(f"Target size: {TARGET_SIZE} abstracts per set")
    print("=" * 55)

    for path in [CORPUS_A_PATH, CORPUS_B_PATH]:
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run previous steps first.")
            return

    corpus_a = load(CORPUS_A_PATH)
    corpus_b = load(CORPUS_B_PATH)
    print(f"Loaded Corpus A: {len(corpus_a)} records (human)")
    print(f"Loaded Corpus B: {len(corpus_b)} records (AI-generated)")

    if len(corpus_a) < TARGET_SIZE or len(corpus_b) < TARGET_SIZE:
        print(f"\nWarning: corpora have fewer than {TARGET_SIZE} records.")
        print(f"  Some training sets may be smaller than target.\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    print()
    for pct in SYNTHETIC_RATIOS:
        print(f"Building mix_{pct} ({pct}% synthetic, {100-pct}% human)...")
        records, human_count, ai_count = build_mix(
            corpus_a, corpus_b, pct, TARGET_SIZE, rng
        )
        set_dir = save_training_set(records, pct, human_count, ai_count, OUTPUT_DIR)
        print(f"  → {set_dir}/ ({human_count} human + {ai_count} AI = {len(records)} total)")

    print(f"\n{'='*55}")
    print(f"Built {len(SYNTHETIC_RATIOS)} training sets in {OUTPUT_DIR}/")
    print("Next step: run finetune.py to train one model per set.")
    print("=" * 55)


if __name__ == "__main__":
    main()
