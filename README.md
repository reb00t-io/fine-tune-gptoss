# fine-tune-gptoss

Plain-Python repo to fine-tune **OpenAI GPT-OSS** models using **Unsloth + TRL**.

- Default model: `unsloth/gpt-oss-20b`

## Quickstart

### 1) Install

This project targets **Linux + NVIDIA CUDA** (A100/H100/etc). macOS CPU-only won’t work for Unsloth training.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Install a CUDA-enabled PyTorch for your machine (example; pick the right one for your CUDA version):

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch
```

### 2) Prepare dataset

Downloads and formats the dataset into a single `text` column using the model’s chat template.

```bash
prepare-dataset
```

Optional smoke test:

```bash
prepare-dataset --max-samples 512
```

### 3) Train (LoRA)

```bash
train --max-steps 30
```

Outputs LoRA adapters to `outputs/gpt-oss-20b-lora/`.

## Repo layout

- `src/fine_tune_gptoss/` – library code + CLIs
- `src/fine_tune_gptoss/cli/prepare_dataset.py` – dataset download/format
- `src/fine_tune_gptoss/cli/train.py` – training entry
- `data/` – local datasets (created at runtime)
- `outputs/` – training outputs (created at runtime)
- `reference/` – upstream notebook export(s) for reference only

## Notes

- Dataset used by default: `HuggingFaceH4/Multilingual-Thinking`.
- To try other base models later, pass `--model <hf_repo>` to both `prepare-dataset` (tokenizer) and `train` (model).
