# Data Processing for NaVILA R2R/RxR

This folder is standalone. It reads NaVILA-style R2R/RxR annotations and writes OpenAI-style JSONL by default without changing the original project code.

## Convert R2R + RxR

```bash
python qwenTrain/prepare_r2r_rxr_qwen.py \
  --r2r-annotations /PATH_TO_DATA/NaVILA-Dataset/R2R/annotations.json \
  --r2r-image-root /PATH_TO_DATA/NaVILA-Dataset/R2R/train \
  --rxr-annotations /PATH_TO_DATA/NaVILA-Dataset/RxR/annotations.json \
  --rxr-image-root /PATH_TO_DATA/NaVILA-Dataset/RxR/train \
  --output qwenTrain/r2r_rxr_qwen.jsonl \
  --num-frames 8
```

By default, image paths are relative from the dataset name, for example `R2R/train/914/frame_0.jpg` and `RxR/train/23289/frame_0.jpg`. Samples shorter than 8 frames repeat the first frame at the beginning so all image paths still point inside `R2R/` or `RxR/`.

The output rows look like:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Walk out of the bathroom and into the theater."},
        {"type": "image_url", "image_url": {"url": "R2R/train/7816/frame_0.jpg"}}
      ]
    },
    {"role": "assistant", "content": "The next action is move forward 75 cm."}
  ]
}
```

The text field is only the original navigation instruction. The longer training prompt can be added later in your training collator or prompt template.

To write the older Qwen-style `image` + `conversations` format:

```bash
python qwenTrain/prepare_r2r_rxr_qwen.py ... --output-format qwen
```

## Use With a Custom Transformers Trainer

```python
from transformers import AutoProcessor
from qwenTrain.navila_qwen_data import QwenNavJsonlDataset, Qwen3VLDataCollator

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
dataset = QwenNavJsonlDataset("qwenTrain/r2r_rxr_qwen.jsonl")
collator = Qwen3VLDataCollator(processor=processor, max_length=4096)
```

The collator converts each row into Qwen chat messages, runs the processor, and masks user/image prompt tokens with `-100` so only the assistant action text is trained.

For best compatibility with Qwen3-VL visual preprocessing, install `qwen-vl-utils` in the Qwen training environment.

## Single-GPU Full Fine-tuning

```bash
python qwenTrain/train_qwen3_vl_full_single_gpu.py \
  --dataset-path qwenTrain/r2r_rxr_openai_20k.jsonl \
  --image-root /PATH_TO_DATA/NaVILA-Dataset \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --output-dir qwenTrain/outputs/qwen3_vl_8b_full \
  --log-dir qwenTrain/logs/qwen3_vl_8b_full \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --max-seq-length 4096 \
  --num-train-epochs 1
```

`--image-root` must be the directory that contains `R2R/` and `RxR/`, because the JSONL stores paths such as `R2R/train/914/frame_0.jpg`.

## Multi-GPU Full Fine-tuning

```bash
torchrun --nproc_per_node=4 qwenTrain/train_qwen3_vl_full_multi_gpu.py \
  --dataset-path qwenTrain/r2r_rxr_openai_20k.jsonl \
  --image-root /PATH_TO_DATA/NaVILA-Dataset \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --output-dir qwenTrain/outputs/qwen3_vl_8b_full_ddp \
  --log-dir qwenTrain/logs/qwen3_vl_8b_full_ddp \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --max-seq-length 4096 \
  --num-train-epochs 1
```

This script uses DDP. Each GPU holds a full model replica, so it improves throughput but does not shard model or optimizer memory.

## DeepSpeed Full Fine-tuning

Use ZeRO-3 to shard parameters, gradients, and optimizer states across GPUs:

```bash
deepspeed --num_gpus=4 qwenTrain/train_qwen3_vl_full_deepspeed.py \
  --dataset-path qwenTrain/r2r_rxr_openai_20k.jsonl \
  --image-root /PATH_TO_DATA/NaVILA-Dataset \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --output-dir qwenTrain/outputs/qwen3_vl_8b_full_ds \
  --log-dir qwenTrain/logs/qwen3_vl_8b_full_ds \
  --deepspeed-config qwenTrain/deepspeed_zero3_bf16.json \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --max-seq-length 4096 \
  --num-train-epochs 1
```

If ZeRO-3 without offload still runs out of memory, switch to:

```bash
--deepspeed-config qwenTrain/deepspeed_zero3_offload_bf16.json
```

CPU offload reduces GPU memory pressure but is much slower and needs enough host RAM.
