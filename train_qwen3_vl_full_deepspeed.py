#!/usr/bin/env python
"""DeepSpeed full fine-tuning script for Qwen3-VL with Unsloth.

Launch example:

deepspeed --num_gpus=4 qwenTrain/train_qwen3_vl_full_deepspeed.py \
  --dataset-path qwenTrain/r2r_rxr_openai_2w.jsonl \
  --image-root /home/rem/图片/naviladata/dataset \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --output-dir qwenTrain/outputs/qwen3_vl_8b_full_ds \
  --log-dir qwenTrain/logs/qwen3_vl_8b_full_ds \
  --deepspeed-config qwenTrain/deepspeed_zero3_bf16.json

This script uses Hugging Face Trainer's DeepSpeed integration. ZeRO-3 shards
parameters, gradients, and optimizer states across GPUs.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Tuple


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


try:
    from train_qwen3_vl_full_single_gpu import (
        DEFAULT_NAVILA_PROMPT_TEMPLATE,
        DEFAULT_QWEN_INSTRUCTION_PART,
        DEFAULT_QWEN_RESPONSE_PART,
        OpenAIJsonlVisionDataset,
        resolve_resume_arg,
        set_full_trainable,
    )
except ImportError:
    from qwenTrain.train_qwen3_vl_full_single_gpu import (
        DEFAULT_NAVILA_PROMPT_TEMPLATE,
        DEFAULT_QWEN_INSTRUCTION_PART,
        DEFAULT_QWEN_RESPONSE_PART,
        OpenAIJsonlVisionDataset,
        resolve_resume_arg,
        set_full_trainable,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, type=Path, help="OpenAI-style JSONL training file")
    parser.add_argument(
        "--image-root",
        required=True,
        type=Path,
        help="Directory containing R2R/ and RxR/. Example: /home/rem/图片/naviladata/dataset",
    )
    parser.add_argument("--model-path", default="/home/rem/work/allModel/Qwen/Qwen3-VL-2B-Instruct", type=str, help="Qwen3-VL-8B model path or HF repo id")
    parser.add_argument("--output-dir", default="/outputs", type=Path, help="Directory to save checkpoints/final model")
    parser.add_argument("--log-dir", default="/outputs/logs", type=Path, help="Directory for rank logs and TensorBoard logs")
    parser.add_argument(
        "--deepspeed-config",
        type=Path,
        default=Path("qwenTrain/deepspeed_zero3_bf16.json"),
        help="DeepSpeed JSON config path",
    )
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Use >0 for debug runs; -1 means full epochs")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional debug limit")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset order before distributed sampling")
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=DEFAULT_NAVILA_PROMPT_TEMPLATE,
        help=(
            "Template applied to the instruction text. Available placeholders: "
            "{instruction}, {num_images}, {num_history_images}."
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path, or 'true' to let Trainer auto-detect the latest checkpoint",
    )
    parser.add_argument("--bf16", dest="bf16", action="store_true", default=True, help="Use bf16 training")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false", help="Disable bf16 and use fp16")
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to reduce VRAM",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing",
    )
    return parser.parse_args()


def get_rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))


def is_rank0() -> bool:
    return get_rank() == 0


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    rank = get_rank()
    handlers = [logging.FileHandler(log_dir / f"train_rank{rank}.log", encoding="utf-8")]
    if rank == 0:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | rank={rank} | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def setup_torch_device() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())


def load_model(args: argparse.Namespace) -> Tuple[Any, Any]:
    from unsloth import FastVisionModel

    kwargs = dict(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
    )
    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            **kwargs,
            full_finetuning=True,
        )
    except TypeError:
        logging.warning("FastVisionModel.from_pretrained did not accept full_finetuning=True; loading 16-bit model.")
        model, tokenizer = FastVisionModel.from_pretrained(**kwargs)

    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    set_full_trainable(model)
    FastVisionModel.for_training(model)
    return model, tokenizer


def main() -> None:
    args = parse_args()
    if args.local_rank is not None:
        os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_dir)

    if not args.deepspeed_config.exists():
        raise FileNotFoundError(args.deepspeed_config)

    setup_torch_device()

    import torch

    torch.manual_seed(args.seed + get_rank())
    random.seed(args.seed + get_rank())

    logging.info("Arguments: %s", vars(args))
    logging.info(
        "Distributed env: rank=%d local_rank=%d world_size=%d",
        get_rank(),
        get_local_rank(),
        get_world_size(),
    )
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(get_local_rank())
        logging.info("GPU: %s | %.1f GB", props.name, props.total_memory / 1024**3)

    dataset = OpenAIJsonlVisionDataset(
        dataset_path=args.dataset_path,
        image_root=args.image_root,
        prompt_template=args.prompt_template,
        limit_samples=args.limit_samples,
        shuffle=args.shuffle,
        seed=args.seed,
        show_progress=is_rank0(),
    )
    logging.info("Loaded %d training samples", len(dataset))
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    model, tokenizer = load_model(args)

    from trl import SFTConfig, SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(
            model,
            tokenizer,
            max_seq_length=args.max_seq_length,
            train_on_responses_only=True,
            instruction_part=DEFAULT_QWEN_INSTRUCTION_PART,
            response_part=DEFAULT_QWEN_RESPONSE_PART,
            completion_only_loss=True,
        ),
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(args.output_dir),
            logging_dir=str(args.log_dir / "tensorboard"),
            deepspeed=str(args.deepspeed_config),
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            optim=args.optim,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            bf16=args.bf16,
            fp16=not args.bf16,
            dataloader_num_workers=args.num_workers,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_seq_length,
            report_to=["tensorboard"] if is_rank0() else [],
            disable_tqdm=not is_rank0(),
        ),
    )

    if is_rank0():
        logging.info("Starting DeepSpeed full fine-tuning with %s", args.deepspeed_config)
    train_result = trainer.train(resume_from_checkpoint=resolve_resume_arg(args.resume_from_checkpoint))
    logging.info("Training finished: %s", train_result)

    if is_rank0():
        logging.info("Saving model and tokenizer to %s", args.output_dir)
        trainer.save_model(str(args.output_dir))
        tokenizer.save_pretrained(str(args.output_dir))
    logging.info("Done")


if __name__ == "__main__":
    main()
