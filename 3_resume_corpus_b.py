"""
Resume Corpus B generation from the last checkpoint.
Run this instead of generate_corpus_b.py if it crashed mid-run.
"""

import json
import time
import os
import requests
from datetime import datetime

CORPUS_A_PATH = "corpus_a/corpus_a.json"
CORPUS_B_DIR = "corpus_b"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BATCH_DELAY = 1.5


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_checkpoint(corpus_b_dir):
    """Find the most recent checkpoint file."""
    files = [f for f in os.listdir(corpus_b_dir) if f.startswith("checkpoint_") and f.endswith(".json")]
    if not files:
        # Check if final file exists
        final = os.path.join(corpus_b_dir, "corpus_b.json")
        if os.path.exists(final):
            return final, True
        return None, False

    # Sort by record number
    files.sort(key=lambda f: int(f.replace("checkpoint_", "").replace(".json", "")))
    latest = os.path.join(corpus_b_dir, files[-1])
    print(f"  Found checkpoints: {[f for f in files]}")
    print(f"  Resuming from: {files[-1]}")
    return latest, False


def get_completed_pmids(checkpoint_path):
    """Return set of source PMIDs already processed."""
    data = load_json(checkpoint_path)
    return {r["source_pmid"] for r in data}, data


def generate_synthetic_abstract(title, journal, year, api_key):
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


def main():
    print("=" * 50)
    print("Corpus B — Resume from Checkpoint")
    print("=" * 50)

    if not CLAUDE_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set.")
        print("Run: export ANTHROPIC_API_KEY=your_key_here")
        return

    if not os.path.exists(CORPUS_B_DIR):
        print(f"No corpus_b directory found. Run generate_corpus_b.py instead.")
        return

    # Find latest checkpoint
    checkpoint_path, already_done = find_latest_checkpoint(CORPUS_B_DIR)

    if already_done:
        print("corpus_b.json already exists and appears complete!")
        data = load_json(checkpoint_path)
        print(f"Records: {len(data)}")
        return

    if not checkpoint_path:
        print("No checkpoints found. Run generate_corpus_b.py from scratch.")
        return

    # Load what we have
    completed_pmids, existing_records = get_completed_pmids(checkpoint_path)
    print(f"  Already completed: {len(existing_records)} records")

    # Load Corpus A and find what's missing
    corpus_a = load_json(CORPUS_A_PATH)
    remaining = [r for r in corpus_a if r["pmid"] not in completed_pmids]
    print(f"  Remaining to generate: {len(remaining)} records")

    if not remaining:
        print("All records already generated!")
    else:
        records = list(existing_records)
        failed = []

        for i, record in enumerate(remaining):
            print(f"  [{len(records)+1}/{len(corpus_a)}] {record['title'][:70]}...")
            try:
                synthetic_abstract = generate_synthetic_abstract(
                    title=record["title"],
                    journal=record["journal"],
                    year=record["year"],
                    api_key=CLAUDE_API_KEY
                )
                records.append({
                    "pmid": f"synthetic_{record['pmid']}",
                    "title": record["title"],
                    "abstract": synthetic_abstract,
                    "year": record["year"],
                    "journal": record["journal"],
                    "source_pmid": record["pmid"],
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

            # Checkpoint every 50
            if (i + 1) % 50 == 0:
                cp = os.path.join(CORPUS_B_DIR, f"checkpoint_{len(records)}.json")
                with open(cp, "w") as f:
                    json.dump(records, f, indent=2)
                print(f"  Checkpoint saved: {len(records)} records")

    # Save final output
    json_path = os.path.join(CORPUS_B_DIR, "corpus_b.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(CORPUS_B_DIR, "corpus_b.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"Title: {r['title']}\n")
            f.write(f"Abstract: {r['abstract']}\n")
            f.write("\n---\n\n")

    print(f"\n--- Corpus B Summary ---")
    print(f"  Generated:  {len(records)} records")
    print(f"  Failed:     {len(failed)}")
    print(f"  Saved:      {json_path}")
    print("\nDone! Run 4_build_mix_corpora.py next.")


if __name__ == "__main__":
    main()
