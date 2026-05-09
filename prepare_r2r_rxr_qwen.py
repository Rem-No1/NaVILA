#!/usr/bin/env python
"""Convert NaVILA R2R/RxR annotations to OpenAI-style or Qwen JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from navila_qwen_data import convert_navila_annotations, shuffle_records, write_jsonl
except ImportError:
    from qwenTrain.navila_qwen_data import convert_navila_annotations, shuffle_records, write_jsonl


def _dataset_args(args: argparse.Namespace) -> List[Tuple[str, str, str, str]]:
    datasets = []
    if args.r2r_annotations or args.r2r_image_root:
        if not args.r2r_annotations or not args.r2r_image_root:
            raise ValueError("Both --r2r-annotations and --r2r-image-root are required for R2R")
        datasets.append(("r2r", "R2R", args.r2r_annotations, args.r2r_image_root))
    if args.rxr_annotations or args.rxr_image_root:
        if not args.rxr_annotations or not args.rxr_image_root:
            raise ValueError("Both --rxr-annotations and --rxr-image-root are required for RxR")
        datasets.append(("rxr", "RxR", args.rxr_annotations, args.rxr_image_root))
    if not datasets:
        raise ValueError("Provide at least one dataset: R2R and/or RxR annotation + image root")
    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2r-annotations", type=str, default=None, help="Path to R2R annotations.json/jsonl")
    parser.add_argument("--r2r-image-root", type=str, default=None, help="Root directory for R2R frame paths")
    parser.add_argument("--rxr-annotations", type=str, default=None, help="Path to RxR annotations.json/jsonl")
    parser.add_argument("--rxr-image-root", type=str, default=None, help="Root directory for RxR frame paths")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument(
        "--output-format",
        choices=("openai", "qwen"),
        default="openai",
        help="Output schema. `openai` writes messages/content image_url records.",
    )
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames/images per sample")
    parser.add_argument(
        "--pad-mode",
        choices=("black", "repeat_first", "repeat_last", "none", "drop"),
        default="repeat_first",
        help="How to handle samples with fewer frames than --num-frames",
    )
    parser.add_argument(
        "--path-mode",
        choices=("dataset", "absolute", "relative", "keep"),
        default="dataset",
        help="How frame paths should be written into the JSONL",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip samples if any sampled frame path does not exist",
    )
    parser.add_argument("--limit-per-dataset", type=int, default=None, help="Optional debug limit per source dataset")
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include id/source/video_id/original_index beside messages. Default keeps only OpenAI messages.",
    )
    parser.add_argument(
        "--image-detail",
        choices=("low", "high", "auto"),
        default=None,
        help="Optional OpenAI image_url.detail value for openai output.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle merged records before writing")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--strict", action="store_true", help="Raise on the first malformed sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_dir = output_path.parent

    all_records = []
    summary: Dict[str, Dict[str, int]] = {}
    for source, dataset_name, annotation_path, image_root in _dataset_args(args):
        records, stats = convert_navila_annotations(
            annotation_path,
            source=source,
            dataset_name=dataset_name,
            image_root=image_root,
            output_dir=output_dir,
            num_frames=args.num_frames,
            pad_mode=args.pad_mode,
            path_mode=args.path_mode,
            skip_missing_images=args.skip_missing_images,
            limit=args.limit_per_dataset,
            strict=args.strict,
            output_format=args.output_format,
            include_metadata=args.include_metadata,
            image_detail=args.image_detail,
        )
        all_records.extend(records)
        summary[source] = stats

    if args.shuffle:
        all_records = shuffle_records(all_records, seed=args.seed)

    written = write_jsonl(all_records, output_path)
    summary["total"] = {"written": written}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
