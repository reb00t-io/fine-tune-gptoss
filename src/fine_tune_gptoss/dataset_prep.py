from __future__ import annotations

from typing import Any


def format_sharegpt_messages_as_text(
    examples: dict[str, Any],
    *,
    tokenizer,
) -> dict[str, list[str]]:
    """Formats ShareGPT-style `messages` into model chat-template text."""

    convos = examples.get("messages")
    if convos is None:
        raise KeyError("Expected dataset examples to contain a 'messages' column.")

    texts: list[str] = []
    for convo in convos:
        texts.append(
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

    return {"text": texts}
