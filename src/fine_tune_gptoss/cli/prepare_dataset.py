from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import cast

import unsloth  # noqa: F401  # Must be imported before transformers for patching.

from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

from fine_tune_gptoss.dataset_prep import format_sharegpt_messages_as_text
from fine_tune_gptoss.paths import default_processed_dir, default_raw_dir
from fine_tune_gptoss.repo_dataset import build_repo_dataset_samples, clone_github_repo
from unsloth.chat_templates import standardize_sharegpt


def _is_git_dataset_spec(value: str) -> bool:
    v = value.strip()
    if v.startswith("repo:"):
        return True
    # Common git URL / SCP-like forms.
    if v.startswith("git@") or v.startswith("ssh://") or ("://" in v):
        return True
    return False


def _slugify_dataset_tag(value: str) -> str:
    tag = value.strip()
    for ch in (":", "/", "@", "\\", " "):
        tag = tag.replace(ch, "__")
    return tag


def _load_git_repo_dataset(*, repo: str, args) -> Dataset:
    if not repo:
        raise ValueError("Invalid --dataset value (empty repo/git URL)")

    repo_slug = _slugify_dataset_tag(repo).removesuffix(".git")
    repo_dir = default_raw_dir() / "repos" / repo_slug

    print(f"Cloning repo dataset: {repo} -> {repo_dir}")
    clone_github_repo(repo, dest_dir=repo_dir, ref=args.repo_ref, depth=int(args.repo_depth))

    print("Building dataset from repo files (.py/.md)...")
    samples = build_repo_dataset_samples(
        repo,
        repo_dir=repo_dir,
        max_samples=args.max_samples,
        max_file_bytes=int(args.repo_max_file_bytes),
        max_chars_per_sample=int(args.repo_max_chars_per_sample),
    )
    return Dataset.from_list([{"messages": s.messages} for s in samples])

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download + format a ShareGPT-style dataset for GPT-OSS fine-tuning.")
    p.add_argument(
        "--dataset",
        default="HuggingFaceH4/Multilingual-Thinking",
        help=(
            "Dataset name. Either a Hugging Face dataset (default: HuggingFaceH4/Multilingual-Thinking) "
            "or a git URL (e.g. git@github.com:psf/requests.git)."
        ),
    )
    p.add_argument("--split", default="train", help="Dataset split (default: train).")
    p.add_argument(
        "--model",
        default="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        help="Tokenizer source model (default: unsloth/gpt-oss-20b-unsloth-bnb-4bit).",
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

    # repo:* dataset mode
    p.add_argument(
        "--repo-ref",
        default=None,
        help="Optional git ref (branch/tag/commit) to checkout when using --dataset repo:...",
    )
    p.add_argument(
        "--repo-depth",
        type=int,
        default=1,
        help="Shallow clone depth for repo: datasets (default: 1).",
    )
    p.add_argument(
        "--repo-max-file-bytes",
        type=int,
        default=250_000,
        help="Skip repo files larger than this many bytes (default: 250000).",
    )
    p.add_argument(
        "--repo-max-chars-per-sample",
        type=int,
        default=12_000,
        help="Chunk long repo files into samples of at most this many characters (default: 12000).",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    dataset_tag = _slugify_dataset_tag(str(args.dataset))
    out_dir = Path(args.out) if args.out else (default_processed_dir() / f"{dataset_tag}__{args.split}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    if out_dir.exists():
        resp = input(f"Output dir already exists: {out_dir}\nOverwrite? [Y/n] ").strip().lower()
        if resp in {"n", "no"}:
            print("Aborted.")
            raise SystemExit(1)

        shutil.rmtree(out_dir)

    dataset_value = str(args.dataset)
    if _is_git_dataset_spec(dataset_value):
        dataset = _load_git_repo_dataset(repo=dataset_value, args=args)
    else:
        print(f"Loading dataset: {args.dataset} [{args.split}]")
        dataset = cast(Dataset, load_dataset(args.dataset, split=args.split))

        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Loading tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if not _is_git_dataset_spec(dataset_value):
        print("Standardizing ShareGPT schema via Unsloth...")
        dataset = standardize_sharegpt(dataset)

    print("Formatting examples into a single 'text' column...")
    dataset = cast(
        Dataset,
        dataset.map(
        lambda batch: format_sharegpt_messages_as_text(batch, tokenizer=tokenizer),
        batched=True,
        ),
    )

    cols_to_drop = [c for c in (dataset.column_names or []) if c != "text"]
    dataset = dataset.remove_columns(cols_to_drop)

    print(f"Saving to: {out_dir}")
    dataset.save_to_disk(str(out_dir))

    print("Done.")
    print("First formatted sample:\n")
    first_row = cast(dict[str, list[str]], dataset.select([0]).to_dict())
    first_text = first_row["text"][0]
    print(first_text)
    print(f"\nSamples: {len(dataset)}")


if __name__ == "__main__":
    main()
