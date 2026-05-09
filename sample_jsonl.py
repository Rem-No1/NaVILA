#!/usr/bin/env python
"""Sample records from a JSONL file into a new JSONL file."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def write_selected_lines(input_path: Path, output_path: Path, selected: set[int]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_idx, line in enumerate(src):
            if line_idx in selected:
                dst.write(line)
                written += 1
    return written


def first_n_indices(total: int, n: int) -> set[int]:
    return set(range(min(total, n)))


def random_indices(total: int, n: int, seed: int) -> set[int]:
    n = min(total, n)
    rng = random.Random(seed)
    return set(rng.sample(range(total), n))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="qwenTrain/r2r_rxr_openai.jsonl", type=Path, help="Input JSONL file")
    parser.add_argument("--output", default="qwenTrain/r2r_rxr_openai_2w.jsonl", type=Path, help="Output sampled JSONL file")
    parser.add_argument("--num-samples", type=int, default=20000, help="Number of lines to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument(
        "--mode",
        choices=("random", "first"),
        default="random",
        help="Use random sampling or simply take the first N lines",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    total = count_lines(args.input)
    if args.mode == "first":
        selected = first_n_indices(total, args.num_samples)
    else:
        selected = random_indices(total, args.num_samples, args.seed)

    written = write_selected_lines(args.input, args.output, selected)
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"mode={args.mode} seed={args.seed}")
    print(f"total_lines={total} requested={args.num_samples} written={written}")


if __name__ == "__main__":
    main()

