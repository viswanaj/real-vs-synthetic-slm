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

## Repo Layout (numbered pipeline)

Scripts are ordered roughly by run order:

| Script | Purpose |
| --- | --- |
| `1_fetch_corpus_a.py` | Fetch PubMed abstracts (human corpus A) |
| `2_generate_corpus_b.py` | Generate synthetic corpus B via Anthropic API |
| `3_resume_corpus_b.py` | Resume corpus B from the latest checkpoint |
| `4_build_mix_corpora.py` | Blend A+B into `training_sets/mix_*` at 7 ratios |
| `5_finetune.py` | Train one LoRA adapter per mix (`models/model_mix_*`) |
| `6_evaluate.py` | Evaluate **one** adapter per run (e.g. `--model 50`) |
| `7_generate_abstracts_all_mixes.py` | Generate one abstract per mix model for a title |
| `evaluation/plot_mix_summary.py` | Summary figure from `results_mix_*.json` |

## Quick Start

### 1) Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Data pipeline

```bash
python 1_fetch_corpus_a.py
export ANTHROPIC_API_KEY=your_key_here
python 2_generate_corpus_b.py
python 4_build_mix_corpora.py
```

If corpus B generation stops midway, continue with:

```bash
python 3_resume_corpus_b.py
```

Then run `4_build_mix_corpora.py` again if needed.

### 3) Train adapters

```bash
python 5_finetune.py
```

### 4) Evaluate (one mix at a time)

```bash
python 6_evaluate.py --model 50 --seed 42
```

CPU-friendly:

```bash
python 6_evaluate.py --model 50 --seed 42 --num-samples 40 --max-new-tokens 120 --skip-perplexity
```

Outputs: `evaluation/results_mix_<ratio>_seed_<seed>.json`, `evaluation/test_set_seed_<seed>.json`.

### 5) Plot sweep

```bash
python evaluation/plot_mix_summary.py
```

### Optional — compare generations across mixes

```bash
python 7_generate_abstracts_all_mixes.py
```

## GitHub / large files

- `.gitignore` excludes `venv/`, `models/`, and common generated artifacts.
- **Do not commit** adapter weights or checkpoints; use [Git LFS](https://git-lfs.com/) or external storage if you need them on the remote.
- After rewriting history to drop large blobs, update the remote with:  
  `git push --force-with-lease origin main`  
  (only if you intend to replace the remote history.)

## Secrets

Use environment variables (e.g. `ANTHROPIC_API_KEY`); never commit API keys.
