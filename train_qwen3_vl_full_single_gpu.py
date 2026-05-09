#!/usr/bin/env python
"""Single-GPU full fine-tuning script for Qwen3-VL with Unsloth.

Expected dataset format is the OpenAI-style JSONL produced by
`qwenTrain/prepare_r2r_rxr_qwen.py`:

{"messages": [{"role": "user", "content": [{"type": "text", "text": "..."},
{"type": "image_url", "image_url": {"url": "R2R/train/...jpg"}}]},
{"role": "assistant", "content": "..."}]}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from tqdm.auto import tqdm

try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:  # type: ignore[no-redef]
        pass


DEFAULT_NAVILA_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical "
    "observations in the first {num_history_images} image(s), and the current observation in the last image. "
    'Your assigned task is: "{instruction}" Analyze this series of images to decide your next action, '
    "which could be turning left or right by a specific degree, moving forward a certain distance, "
    "or stop if the task is completed."
)

DEFAULT_QWEN_INSTRUCTION_PART = "<|im_start|>user\n"
DEFAULT_QWEN_RESPONSE_PART = "<|im_start|>assistant\n"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="", type=Path, help="OpenAI-style JSONL training file")
    parser.add_argument(
        "--image-root",
        default="",
        type=Path,
        help="Directory containing R2R/ and RxR/. Example: /home/rem/图片/naviladata/dataset",
    )
    parser.add_argument("--model-path", default="", type=str, help="Qwen3-VL-8B model path or HF repo id")
    parser.add_argument("--output-dir", default="", type=Path, help="Directory to save checkpoints/final model")
    parser.add_argument("--log-dir", default="", type=Path, help="Directory for trainer logs and train.log")

    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Use >0 for debug runs; -1 means full epochs")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--optim", type=str, default="adamw_8bit")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional debug limit")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset order before training")
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
    parser.add_argument(
        "--save-merged-16bit",
        action="store_true",
        help="Also call save_pretrained_merged after training if supported by Unsloth.",
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


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def resolve_image_path(url: str, image_root: Path) -> Path:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return Path(parsed.path)
    if parsed.scheme in {"http", "https", "data"}:
        raise ValueError(f"Remote/data images are not supported by this local training script: {url[:80]}")
    path = Path(url)
    if path.is_absolute():
        return path
    return image_root / path


def normalize_assistant_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [item.get("text", "") for item in content if item.get("type") == "text"]
        return "\n".join(texts).strip()
    return str(content)


class OpenAIJsonlVisionDataset(Dataset):
    """Lazy JSONL dataset that opens images in __getitem__."""

    def __init__(
        self,
        dataset_path: Path,
        image_root: Path,
        prompt_template: str = "{instruction}",
        limit_samples: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 3407,
        show_progress: bool = True,
    ) -> None:
        self.dataset_path = dataset_path
        self.image_root = image_root
        self.prompt_template = prompt_template
        self.records: List[Dict[str, Any]] = []

        total_lines = sum(1 for _ in dataset_path.open("rb"))
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in tqdm(f, total=total_lines, desc="Indexing JSONL", unit="rows", disable=not show_progress):
                if not line.strip():
                    continue
                self.records.append(json.loads(line))
                if limit_samples is not None and len(self.records) >= limit_samples:
                    break

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        messages = record["messages"]
        user = messages[0]
        assistant = messages[1]

        user_texts = []
        image_paths = []
        for item in user["content"]:
            if item.get("type") == "text":
                user_texts.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                image_paths.append(resolve_image_path(item["image_url"]["url"], self.image_root))

        instruction = "\n".join(text for text in user_texts if text).strip()
        prompt_text = self.prompt_template.format(
            instruction=instruction,
            num_images=len(image_paths),
            num_history_images=max(len(image_paths) - 1, 0),
        )

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        from PIL import Image

        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            content.append({"type": "image", "image": image})

        answer = normalize_assistant_content(assistant["content"])
        return {
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
        }


def set_full_trainable(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info("Trainable parameters: %s / %s (%.2f%%)", trainable, total, 100.0 * trainable / max(total, 1))


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


def resolve_resume_arg(value: Optional[str]) -> Any:
    if value is None:
        return None
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return None
    return value


def main() -> None:
    args = parse_args()
    setup_logging(args.log_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logging.info("Arguments: %s", vars(args))
    logging.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logging.info("GPU: %s | %.1f GB", props.name, props.total_memory / 1024**3)

    dataset = OpenAIJsonlVisionDataset(
        dataset_path=args.dataset_path,
        image_root=args.image_root,
        prompt_template=args.prompt_template,
        limit_samples=args.limit_samples,
        shuffle=args.shuffle,
        seed=args.seed,
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
            logging_dir=str(args.log_dir),
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
            report_to=["tensorboard"],
            disable_tqdm=False,
        ),
    )

    logging.info("Starting full fine-tuning")
    train_result = trainer.train(resume_from_checkpoint=resolve_resume_arg(args.resume_from_checkpoint))
    logging.info("Training finished: %s", train_result)

    logging.info("Saving model and tokenizer to %s", args.output_dir)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    if args.save_merged_16bit and hasattr(model, "save_pretrained_merged"):
        merged_dir = args.output_dir / "merged_16bit"
        logging.info("Saving merged 16-bit model to %s", merged_dir)
        model.save_pretrained_merged(str(merged_dir), tokenizer)

    logging.info("Done")


if __name__ == "__main__":
    main()
