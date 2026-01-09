from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoTextSample:
    messages: list[dict[str, str]]


def _run_git(args: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=str(cwd) if cwd else None, check=True)


def clone_github_repo(
    repo: str,
    *,
    dest_dir: Path,
    ref: str | None = None,
    depth: int = 1,
) -> Path:
    """Clone a repo into dest_dir.

    `repo` can be either:
      - "owner/name" (GitHub shorthand; cloned via https://github.com/<owner>/<name>.git)
      - a full git URL, including SSH/scp-like forms (e.g. git@github.com:owner/name.git)

    If dest_dir exists, it is deleted and re-cloned.
    """

    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    raw = repo.strip()
    is_full_url = raw.startswith("git@") or raw.startswith("ssh://") or ("://" in raw)
    if is_full_url:
        url = raw
    else:
        # Allow shorthand like "owner/name" and also "owner/name.git".
        if raw.endswith(".git") and raw.count("/") == 1 and (":" not in raw) and ("@" not in raw):
            raw = raw[: -len(".git")]
        url = f"https://github.com/{raw}.git"

    _run_git(["clone", "--depth", str(depth), url, str(dest_dir)])

    if ref:
        _run_git(["checkout", ref], cwd=dest_dir)

    return dest_dir


def iter_repo_source_files(repo_dir: Path) -> list[Path]:
    exts = {".py", ".md", ".markdown"}

    files: list[Path] = []
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        if path.suffix.lower() not in exts:
            continue
        files.append(path)

    files.sort(key=lambda p: str(p))
    return files


def _chunk_text(text: str, *, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            nl = text.rfind("\n", start, end)
            if nl > start + 200:
                end = nl + 1
        chunks.append(text[start:end])
        start = end

    return chunks


def build_repo_dataset_samples(
    repo: str,
    *,
    repo_dir: Path,
    max_samples: int | None,
    max_file_bytes: int = 250_000,
    max_chars_per_sample: int = 12_000,
) -> list[RepoTextSample]:
    """Convert repo files into ShareGPT-style message conversations."""

    samples: list[RepoTextSample] = []

    for file_path in iter_repo_source_files(repo_dir):
        try:
            size = file_path.stat().st_size
        except OSError:
            continue

        if size <= 0 or size > max_file_bytes:
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        rel_path = str(file_path.relative_to(repo_dir))
        chunks = _chunk_text(content, max_chars=max_chars_per_sample)

        for idx, chunk in enumerate(chunks, start=1):
            chunk_suffix = "" if len(chunks) == 1 else f" (chunk {idx}/{len(chunks)})"
            user_content = (
                f"Repository: {repo}\n"
                f"File: {rel_path}{chunk_suffix}\n\n"
                "Write the full contents of this file."
            )

            samples.append(
                RepoTextSample(
                    messages=[
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": chunk},
                    ]
                )
            )

            if max_samples is not None and len(samples) >= max_samples:
                return samples

    return samples
