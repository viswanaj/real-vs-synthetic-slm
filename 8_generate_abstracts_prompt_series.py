#!/usr/bin/env python3
"""
Generate abstracts for many prompts (titles), each run through every mix adapter.

Loads each model once, then generates for all prompts — much faster than reloading
per prompt on CPU. Output is one JSON file with structure:
  prompts[].title + prompts[].generations[mix_*]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("_evaluate_module", _ROOT / "6_evaluate.py")
assert _SPEC and _SPEC.loader
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate abstracts for N titles × every mix adapter (efficient load order)."
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        default=20,
        help="How many distinct titles to use (when sampling from corpus A).",
    )
    p.add_argument(
        "--prompt-seed",
        type=int,
        default=42,
        help="RNG seed for sampling titles from corpus A (no effect if --titles-file is set).",
    )
    p.add_argument(
        "--titles-file",
        type=str,
        default=None,
        help="Optional UTF-8 text file: one paper title per line (overrides sampling).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=os.path.join(ev.OUTPUT_DIR, "abstracts_prompt_series.json"),
        help="Where to write JSON results.",
    )
    p.add_argument(
        "--ratios",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of mix percentages. "
            "Default: all configured ratios."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=ev.SEED,
        help="Base RNG seed for generation; actual seed per prompt is seed + prompt_index.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=ev.MAX_NEW_TOKENS,
        help="Max new tokens per abstract.",
    )
    return p.parse_args()


def parse_ratios(spec: str | None) -> list[int]:
    if spec is None:
        return list(ev.SYNTHETIC_RATIOS)
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v not in ev.SYNTHETIC_RATIOS:
            raise SystemExit(f"Ratio {v} not in allowed {ev.SYNTHETIC_RATIOS}")
        out.append(v)
    if not out:
        raise SystemExit("No valid ratios in --ratios")
    return out


def sample_titles(n: int, rng_seed: int) -> list[str]:
    records = ev.load_corpus_txt(ev.CORPUS_A_PATH)
    if not records:
        raise SystemExit(f"No records in {ev.CORPUS_A_PATH}")
    if len(records) < n:
        raise SystemExit(
            f"Need {n} distinct titles but corpus_a has only {len(records)} records."
        )
    rng = random.Random(rng_seed)
    picked = rng.sample(records, n)
    return [r["title"] for r in picked]


def titles_from_file(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"No non-empty lines in {path}")
    return lines


def main() -> None:
    args = parse_args()
    ratios = parse_ratios(args.ratios)

    if args.titles_file:
        titles = titles_from_file(args.titles_file)
        if args.num_prompts and len(titles) > args.num_prompts:
            titles = titles[: args.num_prompts]
    else:
        titles = sample_titles(args.num_prompts, args.prompt_seed)

    n = len(titles)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    print(f"Device: {ev.DEVICE}")
    print(f"Prompts: {n}  |  Adapters: {ratios}")
    print(f"Base generation seed: {args.seed} (per prompt: base + index)")
    print(f"max_new_tokens: {args.max_new_tokens}\n")

    # results[prompt_index][mix_name] = abstract text
    rows: list[dict] = []
    for i in range(n):
        rows.append({"id": i + 1, "title": titles[i], "generations": {}})

    adapter_paths: dict[str, str] = {}

    for pct in ratios:
        name, adapter_path = ev.resolve_model(str(pct))
        adapter_paths[name] = adapter_path
        if not os.path.isdir(adapter_path):
            raise SystemExit(f"Missing adapter: {adapter_path}")

        print(f"Loading {name} ({adapter_path}) …")
        model, tokenizer = ev.load_model(adapter_path)

        for i, title in enumerate(titles):
            g_seed = args.seed + i
            abstract = ev.generate_abstract(
                model, tokenizer, title, g_seed, args.max_new_tokens
            )
            rows[i]["generations"][name] = abstract
            prev = abstract.replace("\n", " ")[:80]
            print(f"  [{i + 1}/{n}] {prev}{'…' if len(abstract) > 80 else ''}")

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print()

    payload = {
        "base_model": ev.MODEL_ID,
        "num_prompts": n,
        "prompt_seed": args.prompt_seed if not args.titles_file else None,
        "titles_file": args.titles_file,
        "generation_seed_base": args.seed,
        "generation_seed_note": "per prompt: generation_seed_base + prompt_index",
        "max_new_tokens": args.max_new_tokens,
        "ratios": ratios,
        "prompts": rows,
        "adapter_paths": adapter_paths,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
