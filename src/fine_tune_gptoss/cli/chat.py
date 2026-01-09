from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
from typing import cast

# Chat stability: Unsloth's GPT-OSS patches use torch.compile internally.
# On some setups this can crash in interactive generate() loops.
# Respect a user-provided value, but default to disabling compile for chat.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import unsloth  # noqa: F401  # Must be imported before transformers/trl/peft for patching.

import torch
import torch._inductor.config as inductor_config
from peft import PeftModel
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from transformers import AutoTokenizer, TextStreamer
from unsloth import FastLanguageModel

from fine_tune_gptoss.gptoss_stream import GptOssStreamRewriter
from fine_tune_gptoss.paths import default_data_dir, default_outputs_dir


class _RewritingTextStreamer(TextStreamer):
    def __init__(self, tokenizer, *, rewriter: GptOssStreamRewriter) -> None:
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=False)
        self._rewriter = rewriter

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:  # type: ignore[override]
        self._rewriter.push(text)
        if stream_end:
            self._rewriter.flush()


def _normalize_chat_inputs(value) -> dict[str, torch.Tensor]:
    """Normalize outputs of tokenizer.apply_chat_template to a dict with tensors.

    Different tokenizer versions return different shapes:
    - dict / BatchEncoding with "input_ids" (+ optional "attention_mask")
    - a plain torch.Tensor of input_ids
    """

    if isinstance(value, Mapping) and "input_ids" in value:
        input_ids = value["input_ids"]
        attention_mask = value.get("attention_mask") if hasattr(value, "get") else None
        out: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            out["attention_mask"] = attention_mask
        return out

    if isinstance(value, torch.Tensor):
        # No attention mask provided; assume all tokens are real tokens.
        return {"input_ids": value, "attention_mask": torch.ones_like(value)}

    raise RuntimeError(
        "tokenizer.apply_chat_template returned an unsupported type. "
        "Expected dict-like with input_ids or a torch.Tensor. "
        f"Got: {type(value)!r}"
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chat with a pretrained GPT-OSS model (interactive).")
    p.add_argument(
        "--model",
        default="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        help="Base model or local path (default: unsloth/gpt-oss-20b-unsloth-bnb-4bit).",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Max sequence length (default: 4096).",
    )
    p.add_argument(
        "--adapter-dir",
        default=None,
        help=(
            "Optional PEFT LoRA adapter directory. If omitted and outputs/gpt-oss-20b-bnb4bit-lora exists, "
            "it will be loaded automatically. Use --adapter-dir none to disable loading adapters."
        ),
    )
    p.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 4-bit loading (default: true). Pass --no-load-in-4bit to disable.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per answer (default: 512).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p (default: 0.9).",
    )
    p.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sampling for generation (default: true). Pass --no-do-sample for greedy decoding.",
    )
    p.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Optional GPT-OSS reasoning effort (low/medium/high).",
    )
    p.add_argument(
        "--cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable TorchInductor CUDA graphs (default: false). "
            "CUDA graphs can be unstable for repeated interactive generation on some setups."
        ),
    )
    return p


def _nice_prompt() -> FormattedText:
    return FormattedText([
        ("class:prompt", "You"),
        ("", "> "),
    ])


def _developer_prompt() -> FormattedText:
    return FormattedText([
        ("class:prompt", "Developer"),
        ("", "> "),
    ])


def _confirm_prompt(text: str) -> FormattedText:
    return FormattedText([
        ("class:prompt", text),
        ("", " "),
    ])


def _ask_yes_no(session: PromptSession, *, prompt: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = session.prompt(_confirm_prompt(f"{prompt} {suffix}"))
        val = (raw or "").strip().lower()
        if not val:
            return default
        if val in {"y", "yes"}:
            return True
        if val in {"n", "no"}:
            return False


def _preview_saved_message(text: str, *, max_chars: int = 2000) -> str:
    t = (text or "").strip("\n")
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "\n\n... (truncated)"


def main() -> None:
    args = build_argparser().parse_args()

    # Avoid TorchInductor CUDA graphs by default for interactive chat.
    # They can crash on repeated generate() calls with checkpoint pool state errors.
    try:
        inductor_config.triton.cudagraphs = bool(args.cuda_graphs)
        # cudagraph_trees is a separate mechanism and is the one visible in your stack traces.
        inductor_config.triton.cudagraph_trees = bool(args.cuda_graphs)
    except Exception:
        pass

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. GPT-OSS inference is expected to run on an NVIDIA GPU. "
            "Run this on a CUDA machine/container."
        )

    device_map = "cuda:0" if torch.cuda.device_count() == 1 else "auto"

    adapter_dir: Path | None
    adapter_arg = (args.adapter_dir or "").strip() if isinstance(args.adapter_dir, str) else args.adapter_dir
    if isinstance(adapter_arg, str) and adapter_arg.lower() in {"none", "null", "no", "false"}:
        adapter_dir = None
    elif adapter_arg:
        adapter_dir = Path(str(adapter_arg))
    else:
        candidate = default_outputs_dir() / "gpt-oss-20b-bnb4bit-lora"
        adapter_dir = candidate if candidate.exists() else None

    print(f"Loading model: {args.model} with adapter: {adapter_dir or 'none'}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=bool(args.load_in_4bit),
        full_finetuning=False,
        device_map=device_map,
    )

    if adapter_dir is not None:
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

        # Accept either the output root dir or a checkpoint-* dir.
        adapter_root = adapter_dir
        if adapter_root.is_dir() and not (adapter_root / "adapter_model.safetensors").exists():
            raise FileNotFoundError(
                "Adapter dir does not look like a PEFT adapter (missing adapter_model.safetensors): "
                f"{adapter_root}"
            )

        print(f"Loading LoRA adapter: {adapter_root}")
        model = PeftModel.from_pretrained(model, str(adapter_root), is_trainable=False)
        model.eval()

        # Enforce tokenizer/chat template from the adapter folder.
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_root), trust_remote_code=True)

    # Optional Unsloth inference patch (if available in this version).
    for_inference = getattr(FastLanguageModel, "for_inference", None)
    if callable(for_inference):
        for_inference(model)

    print("\nInteractive chat. Type your question and press Enter.")
    print("- Exit with Ctrl-D or Ctrl-C.\n")

    style = Style.from_dict(
        {
            "prompt": "ansicyan bold",
        }
    )
    user_session = PromptSession(history=InMemoryHistory(), style=style)
    dev_session = PromptSession(history=InMemoryHistory(), style=style)

    dev_file = default_data_dir() / "chat" / "developer_message.txt"
    saved_dev: str | None = None
    if dev_file.exists():
        try:
            saved_dev = dev_file.read_text(encoding="utf-8").strip() or None
        except OSError:
            saved_dev = None

    developer_message: str | None = None
    if saved_dev is not None:
        print("Saved developer message found:\n")
        print("----- BEGIN SAVED DEVELOPER MESSAGE -----")
        print(_preview_saved_message(saved_dev))
        print("----- END SAVED DEVELOPER MESSAGE -----\n")
        use_saved = _ask_yes_no(user_session, prompt="Use saved developer message?", default=True)
        if use_saved:
            developer_message = saved_dev

    if developer_message is None:
        print("Optional developer message (role=developer).")
        print("- Paste/type it now.")
        print("- Finish with Esc+Enter.")
        print("- Leave empty for none.\n")

        dev_text = dev_session.prompt(
            _developer_prompt(),
            multiline=True,
            prompt_continuation=lambda width, line_number, is_soft_wrap: "> ",
        )

        developer_message = dev_text.strip() or None

        # Persist non-empty developer message for next time.
        if developer_message is not None:
            try:
                dev_file.parent.mkdir(parents=True, exist_ok=True)
                dev_file.write_text(developer_message, encoding="utf-8")
            except OSError:
                pass

    while True:
        try:
            user_text = user_session.prompt(_nice_prompt(), multiline=False)
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        user_text = (user_text or "").strip()
        if not user_text:
            continue

        messages: list[dict[str, str]] = []
        if developer_message is not None:
            messages.append({"role": "developer", "content": developer_message})
        messages.append({"role": "user", "content": user_text})

        template_kwargs: dict[str, object] = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        if args.reasoning_effort is not None:
            template_kwargs["reasoning_effort"] = args.reasoning_effort

        try:
            inputs = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            # Some tokenizers may not support reasoning_effort.
            template_kwargs.pop("reasoning_effort", None)
            inputs = tokenizer.apply_chat_template(messages, **template_kwargs)

        inputs = _normalize_chat_inputs(inputs)
        inputs = {k: cast(torch.Tensor, v).to(model.device) for k, v in inputs.items()}

        gen_kwargs: dict[str, object] = {
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": bool(args.do_sample),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if bool(args.do_sample):
            gen_kwargs["temperature"] = float(args.temperature)
            gen_kwargs["top_p"] = float(args.top_p)

        print("\n---\n")
        rewriter = GptOssStreamRewriter()
        streamer = _RewritingTextStreamer(tokenizer, rewriter=rewriter)

        with torch.inference_mode():
            _ = model.generate(**inputs, streamer=streamer, **gen_kwargs)

        # Some transformer versions may not call stream_end reliably.
        rewriter.flush()
        print("\n")


if __name__ == "__main__":
    main()
