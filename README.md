# Synthetic Ratio Study for Neuropharmacology Abstracts

This repository studies how training-data provenance changes generation behavior:
fine-tuned LoRA adapters are trained on blends of human PubMed abstracts (Corpus A)
and AI-generated abstracts (Corpus B), then evaluated on held-out human references.

## Experiment Design

Seven training sets are built with fixed total size and varying synthetic ratio:

| Training Set | Human % | Synthetic % | Total |
| --- | --- | --- | --- |
| `mix_0` | 100 | 0 | 1000 |
| `mix_10` | 90 | 10 | 1000 |
| `mix_25` | 75 | 25 | 1000 |
| `mix_50` | 50 | 50 | 1000 |
| `mix_75` | 25 | 75 | 1000 |
| `mix_90` | 10 | 90 | 1000 |
| `mix_100` | 0 | 100 | 1000 |

## Repo Layout

- `fetch_corpus_a.py`: fetch PubMed abstracts (human corpus)
- `generate_corpus_b.py`: generate synthetic corpus via Anthropic API
- `resume_corpus_b.py`: continue corpus B generation from latest checkpoint
- `build_corpus_c.py`: build ratio-based training sets
- `finetune.py`: train one LoRA adapter per ratio
- `evaluate.py`: evaluate **one** adapter per run (`--model 50`, etc.)
- `generate_abstracts_all_mixes.py`: one abstract per mix model for a title
- `evaluation/plot_mix_summary.py`: summary figure from `results_mix_*.json`

## Quick Start

### 1) Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Data pipeline

```bash
python fetch_corpus_a.py
export ANTHROPIC_API_KEY=your_key_here
python generate_corpus_b.py
python build_corpus_c.py
```

### 3) Train adapters

```bash
python finetune.py
```

### 4) Evaluate (one mix at a time)

```bash
python evaluate.py --model 50 --seed 42
```

CPU-friendly:

```bash
python evaluate.py --model 50 --seed 42 --num-samples 40 --max-new-tokens 120 --skip-perplexity
```

Outputs: `evaluation/results_mix_<ratio>_seed_<seed>.json`, `evaluation/test_set_seed_<seed>.json`.

### 5) Plot sweep

```bash
python evaluation/plot_mix_summary.py
```

## GitHub / large files

- `.gitignore` excludes `venv/`, `models/`, and common generated artifacts.
- **Do not commit** adapter weights or checkpoints; use [Git LFS](https://git-lfs.com/) or external storage if you need them on the remote.
- After rewriting history to drop large blobs, update the remote with:  
  `git push --force-with-lease origin main`  
  (only if you intend to replace the remote history.)

## Secrets

Use environment variables (e.g. `ANTHROPIC_API_KEY`); never commit API keys.
