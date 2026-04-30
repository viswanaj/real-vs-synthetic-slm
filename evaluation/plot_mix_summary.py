#!/usr/bin/env python3
"""Build a summary figure from evaluation/results_mix_*.json (mix 0–100)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_mix_results(evaluation_dir: Path) -> tuple[list[int], dict[str, np.ndarray]]:
    pattern = re.compile(r"results_mix_(\d+)\.json$")
    rows: list[tuple[int, dict]] = []
    for path in sorted(evaluation_dir.glob("results_mix_*.json")):
        m = pattern.match(path.name)
        if not m:
            continue
        pct = int(m.group(1))
        if not 0 <= pct <= 100:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data:
            continue
        rows.append((pct, data[0]))
    rows.sort(key=lambda x: x[0])
    if not rows:
        raise SystemExit(f"No results_mix_*.json files under {evaluation_dir}")

    mixes = np.array([r[0] for r in rows], dtype=float)
    ppl_mean = np.array([r[1]["perplexity"]["mean"] for r in rows])
    ppl_std = np.array([r[1]["perplexity"]["std"] for r in rows])
    r1 = np.array([r[1]["rouge"]["rouge1"] for r in rows])
    r2 = np.array([r[1]["rouge"]["rouge2"] for r in rows])
    rL = np.array([r[1]["rouge"]["rougeL"] for r in rows])
    d1 = np.array([r[1]["diversity"]["distinct_1"] for r in rows])
    d2 = np.array([r[1]["diversity"]["distinct_2"] for r in rows])
    gen_mean = np.array([r[1]["generation_length"]["mean"] for r in rows])

    metrics = {
        "ppl_mean": ppl_mean,
        "ppl_std": ppl_std,
        "rouge1": r1,
        "rouge2": r2,
        "rougeL": rL,
        "distinct_1": d1,
        "distinct_2": d2,
        "gen_mean": gen_mean,
    }
    return list(map(int, mixes)), metrics


def main() -> None:
    evaluation_dir = Path(__file__).resolve().parent
    out_path = evaluation_dir / "mix_summary_figure.png"

    mixes, m = load_mix_results(evaluation_dir)
    x = np.array(mixes, dtype=float)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    fig.suptitle("Evaluation vs. corpus B mix (0–100%)", fontsize=14, fontweight="semibold")

    ax = axes[0, 0]
    ax.errorbar(
        x,
        m["ppl_mean"],
        yerr=m["ppl_std"],
        fmt="o-",
        capsize=3,
        color="#1f77b4",
        ecolor="#1f77b4",
        alpha=0.85,
    )
    ax.set_xlabel("Mix (% corpus B)")
    ax.set_ylabel("Perplexity (mean ± std)")
    ax.set_xlim(-5, 105)

    ax = axes[0, 1]
    ax.plot(x, m["rouge1"], "o-", label="ROUGE-1", color="#2ca02c")
    ax.plot(x, m["rouge2"], "s-", label="ROUGE-2", color="#ff7f0e")
    ax.plot(x, m["rougeL"], "^-", label="ROUGE-L", color="#9467bd")
    ax.set_xlabel("Mix (% corpus B)")
    ax.set_ylabel("ROUGE F1")
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim(-5, 105)

    ax = axes[1, 0]
    ax.plot(x, m["distinct_1"], "o-", label="Distinct-1", color="#d62728")
    ax.plot(x, m["distinct_2"], "s-", label="Distinct-2", color="#17becf")
    ax.set_xlabel("Mix (% corpus B)")
    ax.set_ylabel("Distinct-n")
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim(-5, 105)

    ax = axes[1, 1]
    ax.plot(x, m["gen_mean"], "o-", color="#8c564b")
    ax.set_xlabel("Mix (% corpus B)")
    ax.set_ylabel("Mean generation length (tokens)")
    ax.set_xlim(-5, 105)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
