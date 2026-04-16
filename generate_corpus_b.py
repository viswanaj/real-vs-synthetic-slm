"""
Corpus B Builder — AI-Generated Text
Takes topics from Corpus A and generates synthetic equivalents
using Claude. Same domain, same structure, different provenance.
"""

import json
import time
import os
import re
import requests
from datetime import datetime

# --- Config ---
CORPUS_A_PATH = "corpus_a/corpus_a.json"
OUTPUT_DIR = "corpus_b"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")  # Set via env variable
BATCH_DELAY = 1.5  # Seconds between API calls to avoid rate limits


def load_corpus_a(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_synthetic_abstract(title, journal, year, api_key):
    """
    Ask Claude to write a synthetic abstract on the same topic.
    Crucially: same domain, same structure — only provenance differs.
    """
    prompt = f"""You are a neuropharmacology researcher writing a scientific abstract.

Write a complete, realistic research abstract on the following topic for a neuropharmacology paper.
The abstract should be similar in length and structure to a real PubMed abstract (150-300 words).
It should include: background/context, methods, results, and conclusions.
Write it as if it were published in {journal} around {year}.

Topic (derived from paper title): {title}

Write ONLY the abstract text. No title, no authors, no labels like "Abstract:". Just the paragraph(s)."""

    response = requests.post(
        CLAUDE_API_URL,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-dangerous-direct-browser-access": "true"
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    response.raise_for_status()
    data = response.json()
    return data["content"][0]["text"].strip()


def build_corpus_b(corpus_a, api_key, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    records = []
    failed = []
    total = len(corpus_a)

    print(f"Generating {total} synthetic abstracts...")

    for i, record in enumerate(corpus_a):
        print(f"  [{i+1}/{total}] {record['title'][:70]}...")

        try:
            synthetic_abstract = generate_synthetic_abstract(
                title=record["title"],
                journal=record["journal"],
                year=record["year"],
                api_key=api_key
            )

            records.append({
                "pmid": f"synthetic_{record['pmid']}",
                "title": record["title"],  # Same title/topic as Corpus A
                "abstract": synthetic_abstract,
                "year": record["year"],
                "journal": record["journal"],
                "source_pmid": record["pmid"],  # Link back to human original
                "corpus": "B",
                "provenance": "ai_generated",
                "model": "claude-sonnet-4-20250514",
                "generated_at": datetime.utcnow().isoformat()
            })

            time.sleep(BATCH_DELAY)

        except Exception as e:
            print(f"    Failed: {e}")
            failed.append(record["pmid"])
            time.sleep(2)
            continue

        # Save checkpoint every 50 records
        if (i + 1) % 50 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{i+1}.json")
            with open(checkpoint_path, "w") as f:
                json.dump(records, f, indent=2)
            print(f"  Checkpoint saved at {i+1} records")

    # Final save
    json_path = os.path.join(output_dir, "corpus_b.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(output_dir, "corpus_b.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"Title: {r['title']}\n")
            f.write(f"Abstract: {r['abstract']}\n")
            f.write("\n---\n\n")

    print(f"\n--- Corpus B Summary ---")
    print(f"  Generated:  {len(records)} records")
    print(f"  Failed:     {len(failed)}")
    print(f"  Saved JSON: {json_path}")
    print(f"  Saved TXT:  {txt_path}")

    if failed:
        print(f"  Failed PMIDs: {failed}")


def main():
    print("=" * 50)
    print("Corpus B Builder — AI-Generated (Claude)")
    print("=" * 50)

    if not CLAUDE_API_KEY:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Run: export ANTHROPIC_API_KEY=your_key_here")
        return

    if not os.path.exists(CORPUS_A_PATH):
        print(f"Error: Corpus A not found at {CORPUS_A_PATH}")
        print("Run fetch_corpus_a.py first.")
        return

    corpus_a = load_corpus_a(CORPUS_A_PATH)
    print(f"Loaded {len(corpus_a)} records from Corpus A")

    build_corpus_b(corpus_a, CLAUDE_API_KEY, OUTPUT_DIR)

    print("\nDone! Corpus B is ready.")
    print("Next step: run build_corpus_c.py to create the mixed corpus.")


if __name__ == "__main__":
    main()
