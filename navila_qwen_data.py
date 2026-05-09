"""Data conversion and loading helpers for training Qwen3-VL on NaVILA VLN data.

This module intentionally does not import or modify the original NaVILA/LLaVA
training code.  It reads NaVILA-style annotations and emits OpenAI-style or
Qwen-VL compatible JSONL. It also provides a small Dataset/Collator for custom
SFT loops.
"""

from __future__ import annotations

import json
import os
import random
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse


IMAGE_PLACEHOLDER = "<image>"
NAVIGATION_PROMPT = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a video "
    'of historical observations {history_images}, and current observation {current_image}. '
    'Your assigned task is: "{instruction}" Analyze this series of images to decide your next action, '
    "which could be turning left or right by a specific degree, moving forward a certain distance, "
    "or stop if the task is completed."
)


def load_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSON list or JSONL annotation file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        first_char = ""
        while True:
            char = f.read(1)
            if not char:
                return []
            if not char.isspace():
                first_char = char
                break
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list in {path}")
            return data
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: str | Path) -> int:
    """Write records as UTF-8 JSONL and return the number of written rows."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into memory."""
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_instruction(instruction: Any) -> str:
    """Clean NaVILA instruction text following the original LazyVLNCEDataset logic."""
    text = str(instruction).replace("\r\n", " ").replace("\n", " ").strip()
    text = text.capitalize()
    text = re.sub(r"(?<=\.\s)([a-z])", lambda match: match.group().upper(), text)
    text = re.sub(r"\s+\.", ".", text)
    return text


def normalize_answer(answer: Any) -> str:
    """Convert answer fields to a deterministic text target."""
    if isinstance(answer, list):
        if not answer:
            return ""
        answer = answer[0]
    return str(answer).replace("\r\n", " ").replace("\n", " ").strip()


def _coerce_frame_path(frame: Any) -> str:
    if isinstance(frame, dict):
        for key in ("path", "frame", "image", "file"):
            if key in frame:
                return str(frame[key])
        raise ValueError(f"Frame dictionary has no path-like key: {frame}")
    return str(frame)


def _linspace_endpoint_false_indices(length: int, count: int) -> List[int]:
    if count <= 0:
        return []
    if length <= 1:
        return [0] * count
    # Equivalent to np.linspace(0, length - 1, count, endpoint=False, dtype=int).
    return [int((length - 1) * i / count) for i in range(count)]


def write_black_png(path: str | Path, size: int = 448) -> Path:
    """Write a square black RGB PNG using only the standard library."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    scanline = b"\x00" + b"\x00\x00\x00" * size
    raw = scanline * size
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return path


def ensure_black_padding_image(output_dir: str | Path, size: int = 448) -> Path:
    """Create the black padding image used for sequences shorter than num_frames."""
    return write_black_png(Path(output_dir) / "assets" / f"black_{size}.png", size=size)


def sample_frame_paths(
    frames: Sequence[Any],
    num_frames: int = 8,
    pad_mode: str = "black",
    pad_frame_path: Optional[str] = None,
) -> List[str]:
    """Sample frames using the same historical/current layout as NaVILA.

    The final returned frame is always the latest/current frame.  Earlier frames
    are uniformly sampled from the available history.  If too few frames exist,
    path-based conversion cannot create NaVILA's black PIL padding, so the
    default is to repeat the first frame at the front.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    paths = [_coerce_frame_path(frame) for frame in frames]
    if not paths:
        return []

    if len(paths) < num_frames:
        deficit = num_frames - len(paths)
        if pad_mode == "black":
            if pad_frame_path is None:
                raise ValueError("pad_frame_path is required when pad_mode='black'")
            paths = [pad_frame_path] * deficit + paths
        elif pad_mode == "repeat_first":
            paths = [paths[0]] * deficit + paths
        elif pad_mode == "repeat_last":
            paths = paths + [paths[-1]] * deficit
        elif pad_mode == "none":
            pass
        elif pad_mode == "drop":
            return []
        else:
            raise ValueError(f"Unsupported pad_mode: {pad_mode}")

    if len(paths) <= num_frames:
        return paths

    latest_frame = paths[-1]
    history_indices = _linspace_endpoint_false_indices(len(paths), num_frames - 1)
    return [paths[i] for i in history_indices] + [latest_frame]


def resolve_frame_path(
    frame_path: str,
    image_root: str | Path,
    output_dir: str | Path,
    path_mode: str = "absolute",
    dataset_name: Optional[str] = None,
) -> str:
    """Resolve a frame path for JSONL output."""
    path = Path(frame_path)
    if image_root and not path.is_absolute():
        path = Path(image_root) / path

    if path_mode == "absolute":
        return str(path.resolve(strict=False))
    if path_mode == "relative":
        return os.path.relpath(path.resolve(strict=False), Path(output_dir).resolve(strict=False))
    if path_mode == "dataset":
        if dataset_name is None:
            raise ValueError("dataset_name is required when path_mode='dataset'")
        image_root_path = Path(image_root)
        raw_frame_path = Path(frame_path)
        if not raw_frame_path.is_absolute():
            return str(Path(dataset_name) / image_root_path.name / raw_frame_path)
        try:
            return str(raw_frame_path.resolve(strict=False).relative_to(image_root_path.parent.resolve(strict=False)))
        except ValueError:
            return os.path.relpath(raw_frame_path.resolve(strict=False), Path(output_dir).resolve(strict=False))
    if path_mode == "keep":
        return str(path)
    raise ValueError(f"Unsupported path_mode: {path_mode}")


def build_navigation_prompt(instruction: str, num_images: int) -> str:
    """Build the NaVILA-style navigation prompt with Qwen image placeholders."""
    if num_images <= 0:
        raise ValueError("num_images must be positive")
    history_images = (IMAGE_PLACEHOLDER + "\n") * max(num_images - 1, 0)
    current_image = IMAGE_PLACEHOLDER + "\n"
    return NAVIGATION_PROMPT.format(
        history_images=history_images,
        current_image=current_image,
        instruction=instruction,
    )


def extract_navila_vln_fields(sample: Dict[str, Any]) -> Tuple[List[Any], str, str]:
    """Extract frames, instruction, and target action from one R2R/RxR sample."""
    if "frames" not in sample:
        raise KeyError("sample does not contain `frames`; expected NaVILA VLN annotation format")
    instruction = sample.get("q")
    answer = sample.get("a")
    if instruction is None:
        raise KeyError("sample does not contain `q` instruction")
    if answer is None:
        raise KeyError("sample does not contain `a` answer")
    return sample["frames"], normalize_instruction(instruction), normalize_answer(answer)


def make_qwen_record(
    sample: Dict[str, Any],
    *,
    source: str,
    index: int,
    image_root: str | Path,
    output_dir: str | Path,
    dataset_name: Optional[str] = None,
    num_frames: int = 8,
    pad_mode: str = "black",
    padding_image_path: Optional[str | Path] = None,
    path_mode: str = "absolute",
    skip_missing_images: bool = False,
) -> Optional[Dict[str, Any]]:
    """Convert one NaVILA R2R/RxR item into Qwen-VL finetuning JSON format."""
    frames, instruction, answer = extract_navila_vln_fields(sample)
    sampled_frames = sample_frame_paths(
        frames,
        num_frames=num_frames,
        pad_mode=pad_mode,
        pad_frame_path=str(padding_image_path) if padding_image_path is not None else None,
    )
    if not sampled_frames:
        return None

    image_paths = [
        resolve_frame_path(
            frame,
            image_root=image_root,
            output_dir=output_dir,
            path_mode=path_mode,
            dataset_name=dataset_name or source,
        )
        for frame in sampled_frames
    ]
    if skip_missing_images:
        for original_frame in sampled_frames:
            original_path = Path(original_frame)
            check_path = original_path if original_path.is_absolute() else Path(image_root) / original_path
            if not check_path.exists():
                return None

    prompt = build_navigation_prompt(instruction, len(image_paths))
    sample_id = sample.get("id") or sample.get("video_id") or f"{source}_{index:09d}"
    return {
        "id": f"{source}:{sample_id}",
        "source": source,
        "video_id": sample.get("video_id"),
        "original_index": index,
        "image": image_paths,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
        "task": instruction,
        "answer": answer,
    }


def make_openai_record(
    sample: Dict[str, Any],
    *,
    source: str,
    index: int,
    image_root: str | Path,
    output_dir: str | Path,
    dataset_name: Optional[str] = None,
    num_frames: int = 8,
    pad_mode: str = "repeat_first",
    padding_image_path: Optional[str | Path] = None,
    path_mode: str = "dataset",
    skip_missing_images: bool = False,
    include_metadata: bool = False,
    image_detail: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Convert one NaVILA R2R/RxR item into OpenAI-style chat JSONL format.

    The user text is only the navigation instruction, not the full NaVILA prompt.
    """
    frames, instruction, answer = extract_navila_vln_fields(sample)
    sampled_frames = sample_frame_paths(
        frames,
        num_frames=num_frames,
        pad_mode=pad_mode,
        pad_frame_path=str(padding_image_path) if padding_image_path is not None else None,
    )
    if not sampled_frames:
        return None

    dataset_name = dataset_name or source
    image_paths = [
        resolve_frame_path(
            frame,
            image_root=image_root,
            output_dir=output_dir,
            path_mode=path_mode,
            dataset_name=dataset_name,
        )
        for frame in sampled_frames
    ]

    if skip_missing_images:
        missing = []
        for original_frame, resolved_path in zip(sampled_frames, image_paths):
            original_path = Path(original_frame)
            if original_path.is_absolute():
                check_path = original_path
            else:
                check_path = Path(image_root) / original_path
            if not check_path.exists():
                missing.append(resolved_path)
        if missing:
            return None

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": instruction}]
    for image_path in image_paths:
        image_url: Dict[str, Any] = {"url": image_path}
        if image_detail is not None:
            image_url["detail"] = image_detail
        user_content.append({"type": "image_url", "image_url": image_url})

    record: Dict[str, Any] = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]
    }
    if include_metadata:
        sample_id = sample.get("id") or sample.get("video_id") or f"{source}_{index:09d}"
        record.update(
            {
                "id": f"{source}:{sample_id}",
                "source": source,
                "video_id": sample.get("video_id"),
                "original_index": index,
            }
        )
    return record


def convert_navila_annotations(
    annotation_path: str | Path,
    *,
    source: str,
    image_root: str | Path,
    output_dir: str | Path,
    dataset_name: Optional[str] = None,
    num_frames: int = 8,
    pad_mode: str = "black",
    path_mode: str = "absolute",
    skip_missing_images: bool = False,
    limit: Optional[int] = None,
    strict: bool = False,
    output_format: str = "openai",
    include_metadata: bool = False,
    image_detail: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Convert one NaVILA annotation file to OpenAI or Qwen records."""
    raw_samples = load_json_or_jsonl(annotation_path)
    records: List[Dict[str, Any]] = []
    stats = {"loaded": len(raw_samples), "written": 0, "skipped": 0, "errors": 0}
    padding_image_path = ensure_black_padding_image(output_dir) if pad_mode == "black" else None
    if output_format not in {"openai", "qwen"}:
        raise ValueError(f"Unsupported output_format: {output_format}")

    for index, sample in enumerate(raw_samples):
        if limit is not None and len(records) >= limit:
            break
        try:
            maker = make_openai_record if output_format == "openai" else make_qwen_record
            common_kwargs = dict(
                sample=sample,
                source=source,
                dataset_name=dataset_name,
                index=index,
                image_root=image_root,
                output_dir=output_dir,
                num_frames=num_frames,
                pad_mode=pad_mode,
                padding_image_path=padding_image_path,
                path_mode=path_mode,
                skip_missing_images=skip_missing_images,
            )
            if output_format == "openai":
                record = maker(
                    **common_kwargs,
                    include_metadata=include_metadata,
                    image_detail=image_detail,
                )
            else:
                record = maker(**common_kwargs)
        except Exception:
            stats["errors"] += 1
            if strict:
                raise
            continue
        if record is None:
            stats["skipped"] += 1
            continue
        records.append(record)

    stats["written"] = len(records)
    return records, stats


def split_text_and_images(text: str, image_paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Turn a placeholder string into Qwen chat content items."""
    chunks = text.split(IMAGE_PLACEHOLDER)
    placeholder_count = len(chunks) - 1
    if placeholder_count != len(image_paths):
        raise ValueError(
            f"Prompt has {placeholder_count} image placeholders but record has {len(image_paths)} images"
        )

    content: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        if chunk:
            content.append({"type": "text", "text": chunk})
        if idx < len(image_paths):
            content.append({"type": "image", "image": to_qwen_image_ref(image_paths[idx])})
    return content


def to_qwen_image_ref(path_or_url: str) -> str:
    """Convert a local absolute path to the file URI style used by Qwen examples."""
    parsed = urlparse(path_or_url)
    if parsed.scheme in {"file", "http", "https", "data"}:
        return path_or_url
    path = Path(path_or_url)
    if path.is_absolute():
        return path.as_uri()
    return path_or_url


def local_path_from_image_ref(path_or_url: str) -> str:
    """Return a PIL-loadable local path from a Qwen image reference."""
    parsed = urlparse(path_or_url)
    if parsed.scheme == "file":
        return parsed.path
    return path_or_url


def qwen_messages_from_record(record: Dict[str, Any], include_answer: bool = True) -> List[Dict[str, Any]]:
    """Convert a JSONL record to Qwen3-VL chat messages."""
    if "messages" in record:
        messages: List[Dict[str, Any]] = []
        for message in record["messages"]:
            role = message["role"]
            if role == "assistant":
                if include_answer:
                    content = message["content"]
                    if isinstance(content, str):
                        content = [{"type": "text", "text": content}]
                    messages.append({"role": "assistant", "content": content})
                continue
            content = message["content"]
            if isinstance(content, str):
                qwen_content = [{"type": "text", "text": content}]
            else:
                qwen_content = []
                for item in content:
                    if item.get("type") == "text":
                        qwen_content.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "image_url":
                        qwen_content.append({"type": "image", "image": item["image_url"]["url"]})
                    else:
                        qwen_content.append(item)
            messages.append({"role": role, "content": qwen_content})
        return messages

    conversations = record["conversations"]
    if not conversations or conversations[0]["from"] != "human":
        raise ValueError("Expected first conversation turn to be from `human`")

    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": split_text_and_images(conversations[0]["value"], record.get("image", [])),
        }
    ]
    if include_answer:
        if len(conversations) < 2 or conversations[1]["from"] != "gpt":
            raise ValueError("Expected second conversation turn to be from `gpt`")
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": conversations[1]["value"]}],
            }
        )
    return messages


class QwenNavJsonlDataset:
    """Small JSONL dataset wrapper.

    It avoids importing torch at module import time so conversion-only workflows
    can run in lightweight environments.  torch Dataset behavior is duck-typed.
    """

    def __init__(self, jsonl_path: str | Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.records = read_jsonl(self.jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        return {
            "record": record,
            "messages": qwen_messages_from_record(record, include_answer=True),
            "prompt_messages": qwen_messages_from_record(record, include_answer=False),
        }


def _build_processor_inputs(
    processor: Any,
    messages_batch: Sequence[List[Dict[str, Any]]],
    *,
    add_generation_prompt: bool,
    padding: bool,
    max_length: Optional[int],
) -> Dict[str, Any]:
    """Apply Qwen chat template and process visual inputs."""
    texts = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        for messages in messages_batch
    ]

    processor_kwargs: Dict[str, Any] = {
        "text": texts,
        "padding": padding,
        "return_tensors": "pt",
    }
    if max_length is not None:
        processor_kwargs["max_length"] = max_length
        processor_kwargs["truncation"] = True

    try:
        from qwen_vl_utils import process_vision_info

        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages_batch,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            processor_kwargs.update(video_kwargs)
            processor_kwargs["do_resize"] = False
        except TypeError:
            image_inputs, video_inputs = process_vision_info(messages_batch)
        if image_inputs is not None:
            processor_kwargs["images"] = image_inputs
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs
    except ImportError:
        from PIL import Image

        image_inputs = []
        for messages in messages_batch:
            for message in messages:
                content = message.get("content", [])
                if isinstance(content, str):
                    continue
                for item in content:
                    if item.get("type") == "image":
                        image_inputs.append(Image.open(local_path_from_image_ref(item["image"])).convert("RGB"))
        if image_inputs:
            processor_kwargs["images"] = image_inputs

    return processor(**processor_kwargs)


@dataclass
class Qwen3VLDataCollator:
    """Collator for Qwen3-VL SFT.

    The collator masks all user/image prompt tokens with ``ignore_index`` and
    leaves only assistant answer tokens as language-model labels.
    """

    processor: Any
    ignore_index: int = -100
    max_length: Optional[int] = None

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        messages_batch = [feature["messages"] for feature in features]
        prompt_messages_batch = [feature["prompt_messages"] for feature in features]

        batch = _build_processor_inputs(
            self.processor,
            messages_batch,
            add_generation_prompt=False,
            padding=True,
            max_length=self.max_length,
        )
        labels = batch["input_ids"].clone()

        for row, prompt_messages in enumerate(prompt_messages_batch):
            prompt_inputs = _build_processor_inputs(
                self.processor,
                [prompt_messages],
                add_generation_prompt=True,
                padding=False,
                max_length=self.max_length,
            )
            prompt_len = min(prompt_inputs["input_ids"].shape[-1], labels.shape[-1])
            labels[row, :prompt_len] = self.ignore_index

        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = self.ignore_index
        batch["labels"] = labels
        return batch


def shuffle_records(records: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    return shuffled
