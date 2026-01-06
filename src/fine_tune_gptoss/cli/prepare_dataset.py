from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from fine_tune_gptoss.dataset_prep import format_sharegpt_messages_as_text
from fine_tune_gptoss.paths import default_processed_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download + format a ShareGPT-style dataset for GPT-OSS fine-tuning.")
    p.add_argument(
        "--dataset",
        default="HuggingFaceH4/Multilingual-Thinking",
        help="Hugging Face dataset name (default: HuggingFaceH4/Multilingual-Thinking).",
    )
    p.add_argument("--split", default="train", help="Dataset split (default: train).")
    p.add_argument(
        "--model",
        default="unsloth/gpt-oss-20b",
        help="Tokenizer source model (default: unsloth/gpt-oss-20b).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output directory for datasets.save_to_disk(). Defaults to data/processed/<dataset>__<split>.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests (e.g. 512).",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    out_dir = Path(args.out) if args.out else (default_processed_dir() / f"{args.dataset.replace('/', '__')}__{args.split}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset} [{args.split}]")
    dataset = load_dataset(args.dataset, split=args.split)

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Loading tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Standardizing ShareGPT schema via Unsloth...")
    from unsloth.chat_templates import standardize_sharegpt

    dataset = standardize_sharegpt(dataset)

    print("Formatting examples into a single 'text' column...")
    dataset = dataset.map(
        lambda batch: format_sharegpt_messages_as_text(batch, tokenizer=tokenizer),
        batched=True,
    )

    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

    print(f"Saving to: {out_dir}")
    dataset.save_to_disk(str(out_dir))

    print("Done.")
    print(f"Examples: {len(dataset)}")
    print("First formatted sample:\n")
    print(dataset[0]["text"][:2000])


if __name__ == "__main__":
    main()
