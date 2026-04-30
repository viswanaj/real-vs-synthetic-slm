#!/usr/bin/env python3
"""
Generate one abstract per mix adapter (mix_0 … mix_100) for a single title,
using the same prompt format and decoding settings as evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import evaluate as ev


def parse_args():
    p = argparse.ArgumentParser(
        description="Run title→abstract generation for every trained mix adapter."
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Paper title to condition on (default: first title from corpus A).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=os.path.join(ev.OUTPUT_DIR, "abstracts_all_mixes.json"),
        help="Where to write JSON results.",
    )
    p.add_argument(
        "--ratios",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of mix percentages, e.g. '0,50,100'. "
            "Default: all configured ratios."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=ev.SEED,
        help="Random seed for generation (same as evaluate.py).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=ev.MAX_NEW_TOKENS,
        help="Max tokens per abstract (same as evaluate.py).",
    )
    return p.parse_args()


def default_title() -> str:
    records = ev.load_corpus_txt(ev.CORPUS_A_PATH)
    if not records:
        raise SystemExit(f"No records in {ev.CORPUS_A_PATH}")
    return records[0]["title"]


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
            raise SystemExit(
                f"Ratio {v} not in allowed {ev.SYNTHETIC_RATIOS}"
            )
        out.append(v)
    if not out:
        raise SystemExit("No valid ratios in --ratios")
    return out


def main() -> None:
    args = parse_args()
    title = args.title or default_title()
    ratios = parse_ratios(args.ratios)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    print(f"Device: {ev.DEVICE}")
    print(f"Seed: {args.seed}  max_new_tokens: {args.max_new_tokens}")
    print(f"Title ({len(title)} chars):\n  {title}\n")
    print(f"Adapters: {ratios}\n")

    generations: dict[str, str] = {}
    meta: dict[str, str] = {}

    for pct in ratios:
        name, adapter_path = ev.resolve_model(str(pct))
        if not os.path.isdir(adapter_path):
            raise SystemExit(f"Missing adapter: {adapter_path}")

        print(f"Loading {name} ({adapter_path}) …")
        model, tokenizer = ev.load_model(adapter_path)
        abstract = ev.generate_abstract(
            model, tokenizer, title, args.seed, args.max_new_tokens
        )
        generations[name] = abstract
        meta[name] = adapter_path

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        preview = abstract.replace("\n", " ")[:120]
        print(f"  → {preview}{'…' if len(abstract) > 120 else ''}\n")

    payload = {
        "base_model": ev.MODEL_ID,
        "title": title,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "generations": generations,
        "adapter_paths": meta,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
