#!/usr/bin/env python3
"""Convert evaluation/abstracts_prompt_series.json to a readable .md file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "abstracts_prompt_series.json",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Default: <input_stem>.md",
    )
    args = p.parse_args()
    inp = args.input
    if not inp.is_file():
        raise SystemExit(f"Not found: {inp}")
    out = args.output or inp.with_suffix(".md")

    data = json.loads(inp.read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append("# Abstract prompt series\n")
    lines.append("Generated from `abstracts_prompt_series.json`.\n")
    lines.append("## Run metadata\n")
    lines.append(f"- **Base model:** `{data.get('base_model', '')}`")
    lines.append(f"- **Prompts:** {data.get('num_prompts', '')}")
    if data.get("prompt_seed") is not None:
        lines.append(f"- **Prompt sample seed:** {data['prompt_seed']}")
    lines.append(f"- **Titles file:** {data.get('titles_file')!r}")
    lines.append(f"- **Generation seed base:** {data.get('generation_seed_base', '')}")
    if data.get("generation_seed_note"):
        lines.append(f"- **Seeds:** {data['generation_seed_note']}")
    lines.append(f"- **Max new tokens:** {data.get('max_new_tokens', '')}")
    ratios = data.get("ratios", [])
    lines.append(f"- **Mix ratios:** {ratios}")
    lines.append("")

    adapter_paths = data.get("adapter_paths") or {}
    if adapter_paths:
        lines.append("### Adapter paths\n")
        for name in sorted(adapter_paths.keys(), key=lambda x: int(x.split("_")[1]) if "_" in x else 0):
            lines.append(f"- `{name}` → `{adapter_paths[name]}`")
        lines.append("")

    for row in data.get("prompts", []):
        pid = row.get("id", "")
        title = row.get("title", "").strip()
        lines.append(f"## Prompt {pid}\n")
        lines.append(f"**Title:** {title}\n")
        gens = row.get("generations") or {}
        # Stable order by mix ratio
        keys = sorted(gens.keys(), key=lambda k: int(k.split("_")[1]))
        for mk in keys:
            body = gens[mk].strip()
            lines.append(f"### {mk}\n")
            lines.append("```text")
            lines.append(body)
            lines.append("```\n")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
