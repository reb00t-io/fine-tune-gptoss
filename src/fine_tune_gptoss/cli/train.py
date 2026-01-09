from __future__ import annotations

import argparse
from pathlib import Path

import unsloth  # noqa: F401  # Must be imported before transformers/trl/peft for patching.

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from unsloth import FastLanguageModel

from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from fine_tune_gptoss.paths import default_outputs_dir, default_processed_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune GPT-OSS with Unsloth (LoRA via TRL SFTTrainer).")

    p.add_argument(
        "--model",
        default="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        help="Base model (default: unsloth/gpt-oss-20b-unsloth-bnb-4bit).",
    )
    p.add_argument(
        "--dataset-dir",
        default=None,
        help="Path created by prepare-dataset. Defaults to data/processed/HuggingFaceH4__Multilingual-Thinking__train.",
    )
    p.add_argument("--output-dir", default=None, help="Output dir for LoRA adapters (default: outputs/gpt-oss-20b-lora).")

    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 4-bit loading (default: true). Pass --no-load-in-4bit to disable.",
    )

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)

    # Training
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=3407)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Unsloth fine-tuning requires an NVIDIA GPU (e.g. A100). "
            "Run this on a CUDA machine/container."
        )

    dataset_dir = (
        Path(args.dataset_dir)
        if args.dataset_dir
        else (default_processed_dir() / "HuggingFaceH4__Multilingual-Thinking__train")
    )
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset dir not found: {dataset_dir}\n"
            "Run: prepare-dataset (or pass --dataset-dir)."
        )

    output_dir = Path(args.output_dir) if args.output_dir else (default_outputs_dir() / "gpt-oss-20b-bnb4bit-lora")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading formatted dataset from: {dataset_dir}")
    train_dataset = load_from_disk(str(dataset_dir))
    if isinstance(train_dataset, DatasetDict):
        # Prefer a conventional split name if present; otherwise take the first.
        split_name = "train" if "train" in train_dataset else next(iter(train_dataset.keys()))
        train_dataset = train_dataset[split_name]
    assert isinstance(train_dataset, Dataset)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Sample training example: {train_dataset[0]}")
    print(f"Loading base model via Unsloth: {args.model}")

    device_map = "cuda:0" if torch.cuda.device_count() == 1 else "auto"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=bool(args.load_in_4bit),
        full_finetuning=False,
        device_map=device_map,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=str(output_dir),
            report_to="none",
        ),
    )

    print("Training...")
    trainer_stats = trainer.train()

    print("Saving LoRA adapters...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Done.")
    if trainer_stats and hasattr(trainer_stats, "metrics"):
        print(trainer_stats.metrics)


if __name__ == "__main__":
    main()
