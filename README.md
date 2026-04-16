# Corpus Builder — Neuropharmacology SLM Study

Builds training corpora and fine-tunes models to study the effect of synthetic training data ratios on SLM quality.

## Experimental Design

Two source corpora (1000 abstracts each) are blended into 7 training sets with varying ratios of synthetic content:

| Training Set | Human % | Synthetic % | Total |
|-------------|---------|-------------|-------|
| mix_0       | 100%    | 0%          | 1000  |
| mix_10      | 90%     | 10%         | 1000  |
| mix_25      | 75%     | 25%         | 1000  |
| mix_50      | 50%     | 50%         | 1000  |
| mix_75      | 25%     | 75%         | 1000  |
| mix_90      | 10%     | 90%         | 1000  |
| mix_100     | 0%      | 100%        | 1000  |

## Source Corpora

| Corpus | Provenance | Size |
|--------|-----------|------|
| A | Human-written (PubMed pre-2022) | ~1000 papers |
| B | AI-generated (Claude, same topics) | ~1000 papers |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install requests
```

## Run Order

### Step 1 — Fetch Corpus A (human text from PubMed)
```bash
python fetch_corpus_a.py
```
No API key needed. NCBI allows free access.
Optional: get a free NCBI API key at ncbi.nlm.nih.gov/account for faster fetching.

### Step 2 — Generate Corpus B (AI text via Claude)
```bash
export ANTHROPIC_API_KEY=your_key_here
python generate_corpus_b.py
```
Checkpoints every 50 records so you can resume if interrupted.
Use `resume_corpus_b.py` to continue from the last checkpoint.

### Step 3 — Build training sets (variable synthetic ratios)
```bash
python build_corpus_c.py
```
No API needed — blends A and B at 7 different ratios.

### Step 4 — Fine-tune models
```bash
python finetune.py
```
Trains 7 LoRA-adapted Llama 3.2 3B models, one per training set.

### Step 5 — Evaluate
```bash
python evaluate.py
```
Generates 200 abstracts per model, computes ROUGE, perplexity, and diversity metrics.
Produces a blind review file for manual comparison.

## Output Structure

```
corpus_a/                     # Human-written source corpus
corpus_b/                     # AI-generated source corpus
training_sets/
  mix_0/                      # 100% human
  mix_10/                     # 90% human, 10% AI
  mix_25/                     # 75% human, 25% AI
  mix_50/                     # 50/50
  mix_75/                     # 25% human, 75% AI
  mix_90/                     # 10% human, 90% AI
  mix_100/                    # 100% AI
models/
  model_mix_0/ ... model_mix_100/   # LoRA adapters per training set
evaluation/
  results.json                # Full metrics for all models
  test_set.json               # The 200 titles/references used
  blind_review.json           # Anonymized outputs for manual review
```
