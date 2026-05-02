"""
Corpus A Builder — Human-Generated Text
Fetches neuropharmacology abstracts from PubMed (pre-2022)
Saves as JSON + plain text for fine-tuning
"""

import requests
import json
import time
import os
import re
import html
from datetime import datetime

# --- Config ---
# Broader query to pull closer to 500 records with abstracts
DOMAIN_QUERY = (
    "neuropharmacology[MeSH Terms] OR "
    "neurotransmitter[MeSH Terms] OR "
    "synaptic pharmacology[tiab] OR "
    "neurochemistry[MeSH Terms] OR "
    "psychopharmacology[MeSH Terms] OR "
    "receptor pharmacology[tiab] OR "
    "drug brain[tiab]"
)
MAX_PAPERS = 1600         # Fetch more to account for records without abstracts
DATE_MIN = "1990/01/01"
DATE_MAX = "2021/12/31"   # Hard cutoff — 2022 papers will also be filtered post-fetch
OUTPUT_DIR = "corpus_a"
ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Optional: add your NCBI API key here for higher rate limits (free at ncbi.nlm.nih.gov)
NCBI_API_KEY = ""


def search_pubmed(query, max_results, date_min, date_max):
    """Search PubMed and return list of PMIDs."""
    print(f"Searching PubMed for up to {max_results} papers...")
    params = {
        "db": "pubmed",
        "term": query,
        "datetype": "pdat",
        "mindate": date_min,
        "maxdate": date_max,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(f"{ENTREZ_BASE}/esearch.fcgi", params=params)
    resp.raise_for_status()
    data = resp.json()
    pmids = data["esearchresult"]["idlist"]
    print(f"  Found {len(pmids)} PMIDs")
    return pmids


def fetch_abstracts(pmids, batch_size=100):
    """Fetch full records for a list of PMIDs in batches."""
    records = []
    total = len(pmids)

    for i in range(0, total, batch_size):
        batch = pmids[i:i + batch_size]
        print(f"  Fetching records {i+1}–{min(i+batch_size, total)} of {total}...")

        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        resp = requests.get(f"{ENTREZ_BASE}/efetch.fcgi", params=params)
        resp.raise_for_status()

        parsed = parse_xml_records(resp.text)
        records.extend(parsed)

        time.sleep(0.4 if NCBI_API_KEY else 1.0)

    return records


def parse_xml_records(xml_text):
    """Parse PubMed XML into clean record dicts."""
    records = []
    articles = re.findall(r'<PubmedArticle>(.*?)</PubmedArticle>', xml_text, re.DOTALL)

    for article in articles:
        try:
            pmid_match = re.search(r'<PMID[^>]*>(\d+)</PMID>', article)
            pmid = pmid_match.group(1) if pmid_match else ""

            title_match = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', article, re.DOTALL)
            title = clean_text(title_match.group(1)) if title_match else ""

            abstract_parts = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', article, re.DOTALL)
            abstract = " ".join(clean_text(p) for p in abstract_parts)

            if not abstract or len(abstract) < 100:
                continue

            year_match = re.search(r'<PubDate>.*?<Year>(\d{4})</Year>', article, re.DOTALL)
            year = year_match.group(1) if year_match else ""

            journal_match = re.search(r'<Title>(.*?)</Title>', article, re.DOTALL)
            journal = clean_text(journal_match.group(1)) if journal_match else ""

            author_matches = re.findall(
                r'<LastName>(.*?)</LastName>.*?<ForeName>(.*?)</ForeName>',
                article, re.DOTALL
            )
            authors = [f"{last}, {first}" for last, first in author_matches[:5]]

            records.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "journal": journal,
                "authors": authors,
                "corpus": "A",
                "provenance": "human",
                "fetched_at": datetime.utcnow().isoformat()
            })

        except Exception as e:
            print(f"  Warning: failed to parse one article — {e}")
            continue

    return records


def clean_text(text):
    """Remove XML tags, decode HTML entities, normalize whitespace."""
    text = re.sub(r'<[^>]+>', '', text)
    text = html.unescape(text)           # Fixes &amp; &lt; &gt; etc.
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_and_filter(records, max_year=2021, target=500):
    """
    Post-processing:
    - Strip records with year > max_year
    - Deduplicate by PMID
    - Trim to target size
    """
    print(f"\nCleaning and filtering...")
    original_count = len(records)

    # Filter out anything past max_year
    records = [r for r in records if r.get("year", "").isdigit() and int(r["year"]) <= max_year]
    print(f"  After year filter (≤{max_year}): {len(records)} records (removed {original_count - len(records)})")

    # Deduplicate by PMID
    seen = set()
    deduped = []
    for r in records:
        if r["pmid"] not in seen:
            seen.add(r["pmid"])
            deduped.append(r)
    print(f"  After deduplication: {len(deduped)} records")

    # Trim to target
    final = deduped[:target]
    print(f"  Final corpus size: {len(final)} records")
    return final


def save_corpus(records, output_dir):
    """Save corpus as JSON and plain text for fine-tuning."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "corpus_a.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(records)} records → {json_path}")

    txt_path = os.path.join(output_dir, "corpus_a.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"Title: {r['title']}\n")
            f.write(f"Abstract: {r['abstract']}\n")
            f.write("\n---\n\n")
    print(f"  Saved plain text → {txt_path}")

    years = [int(r["year"]) for r in records if r.get("year", "").isdigit()]
    journals = list(set(r["journal"] for r in records if r["journal"]))

    print(f"\n--- Corpus A Summary ---")
    print(f"  Total records:    {len(records)}")
    if years:
        print(f"  Year range:       {min(years)}–{max(years)}")
    print(f"  Unique journals:  {len(journals)}")
    print(f"  Sample journals:  {', '.join(journals[:5])}")
    avg_len = sum(len(r["abstract"].split()) for r in records) / len(records)
    print(f"  Avg abstract len: {avg_len:.0f} words")


def main():
    print("=" * 50)
    print("Corpus A Builder — Neuropharmacology (Human)")
    print("=" * 50)

    pmids = search_pubmed(DOMAIN_QUERY, MAX_PAPERS, DATE_MIN, DATE_MAX)

    if not pmids:
        print("No PMIDs found. Check your query or network connection.")
        return

    print(f"\nFetching abstracts...")
    records = fetch_abstracts(pmids)

    if not records:
        print("No records parsed. Something went wrong.")
        return

    # Clean, filter, deduplicate
    records = clean_and_filter(records, max_year=2021, target=1000)

    print(f"\nSaving corpus...")
    save_corpus(records, OUTPUT_DIR)

    print("\nDone! Corpus A is ready.")
    print("Next step: run generate_corpus_b.py to create AI-generated equivalents.")


if __name__ == "__main__":
    main()
