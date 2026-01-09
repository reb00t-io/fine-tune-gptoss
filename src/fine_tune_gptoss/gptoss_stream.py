from __future__ import annotations

import re
import sys


class GptOssStreamRewriter:
    """Rewrite GPT-OSS channel markers while streaming.

    Expected model output format (when special tokens are not skipped) commonly includes:

      Analysis:
        <|channel|>analysis<|message|>...<|end|>

      Tool calls (commentary channel):
        <|channel|>commentary to=functions>search({...})<|call|>commentary<|end|>

      Final:
        <|start|>assistant<|channel|>final<|message|>...<|return|>

    Rendered output is optimized for terminal display:
      - Analysis is displayed as: Thinking: ...
      - Tool calls are displayed as: Tool: ...
      - Final is displayed after a blank line.

    Notes:
      - This is a best-effort streaming parser: markers may be split across chunks.
      - Any unknown / stray <|...|> markers are stripped.
    """

    _ANALYSIS_START = "<|channel|>analysis<|message|>"
    _ANALYSIS_END = "<|end|>"

    _FINAL_START = "<|start|>assistant<|channel|>final<|message|>"
    _FINAL_END = "<|return|>"

    _COMMENTARY_MSG_START_1 = "<|channel|>commentary<|message|>"
    _COMMENTARY_MSG_START_2 = "<|start|>assistant<|channel|>commentary<|message|>"

    _COMMENTARY_TOOL_START_1 = "<|channel|>commentary to=functions>"
    _COMMENTARY_TOOL_START_2 = "<|start|>assistant<|channel|>commentary to=functions>"
    _COMMENTARY_TOOL_END = "<|call|>commentary<|end|>"

    _STATE_RAW = "raw"
    _STATE_ANALYSIS = "analysis"
    _STATE_FINAL = "final"
    _STATE_COMMENTARY_MSG = "commentary_msg"
    _STATE_COMMENTARY_TOOL = "commentary_tool"

    def __init__(self) -> None:
        self._buf = ""
        self._state = self._STATE_RAW

        self._printed_thinking_prefix = False
        self._thinking_color_on = False

        self._printed_commentary_prefix = False
        self._printed_tool_prefix = False

        self._BLUE = "\033[34m"
        self._RESET = "\033[0m"

        self._max_marker_len = max(
            len(self._ANALYSIS_START),
            len(self._ANALYSIS_END),
            len(self._FINAL_START),
            len(self._FINAL_END),
            len(self._COMMENTARY_MSG_START_1),
            len(self._COMMENTARY_MSG_START_2),
            len(self._COMMENTARY_TOOL_START_1),
            len(self._COMMENTARY_TOOL_START_2),
            len(self._COMMENTARY_TOOL_END),
        )
        # Keep enough tail to not split markers across chunks.
        self._tail_keep = self._max_marker_len + 32

    def _write(self, text: str) -> None:
        if not text:
            return
        sys.stdout.write(text)
        sys.stdout.flush()

    @staticmethod
    def _strip_markers(text: str) -> str:
        # Some streams include role starters like '<|start|>assistant'.
        # If we only strip '<|start|>' we can accidentally leak 'assistant'.
        text = text.replace("<|start|>assistant", "")
        text = text.replace("<|start|>system", "")
        text = text.replace("<|start|>user", "")
        return re.sub(r"<\|[^>]+\|>", "", text)

    def push(self, chunk: str) -> None:
        self._buf += chunk
        self._drain(allow_partial=False)

    def flush(self) -> None:
        self._drain(allow_partial=True)
        if self._buf:
            self._write(self._strip_markers(self._buf))
            self._buf = ""

        # Ensure we never leave the terminal in colored mode.
        if self._thinking_color_on:
            self._write(self._RESET)
            self._thinking_color_on = False

    def _drain(self, *, allow_partial: bool) -> None:
        while True:
            if self._state == self._STATE_RAW:
                idx_analysis = self._buf.find(self._ANALYSIS_START)
                idx_final = self._buf.find(self._FINAL_START)

                idx_cmsg1 = self._buf.find(self._COMMENTARY_MSG_START_1)
                idx_cmsg2 = self._buf.find(self._COMMENTARY_MSG_START_2)
                idx_ctool1 = self._buf.find(self._COMMENTARY_TOOL_START_1)
                idx_ctool2 = self._buf.find(self._COMMENTARY_TOOL_START_2)

                candidates = [
                    i
                    for i in (idx_analysis, idx_final, idx_cmsg1, idx_cmsg2, idx_ctool1, idx_ctool2)
                    if i != -1
                ]

                # No markers found: flush safe prefix (keeping overlap).
                if not candidates:
                    if allow_partial:
                        return
                    if len(self._buf) <= self._tail_keep:
                        return

                    safe = self._buf[: -self._tail_keep]
                    self._buf = self._buf[-self._tail_keep :]
                    self._write(self._strip_markers(safe))
                    continue

                idx = min(candidates)
                prefix = self._buf[:idx]
                self._buf = self._buf[idx:]
                self._write(self._strip_markers(prefix))

                if self._buf.startswith(self._ANALYSIS_START):
                    self._buf = self._buf[len(self._ANALYSIS_START) :]
                    if not self._printed_thinking_prefix:
                        if not self._thinking_color_on:
                            self._write(self._BLUE)
                            self._thinking_color_on = True
                        self._write("Thinking: ")
                        self._printed_thinking_prefix = True
                    self._state = self._STATE_ANALYSIS
                    continue

                if self._buf.startswith(self._FINAL_START):
                    self._buf = self._buf[len(self._FINAL_START) :]
                    # Ensure a blank line between analysis/commentary and final.
                    self._write("\n\n")
                    self._state = self._STATE_FINAL
                    continue

                if self._buf.startswith(self._COMMENTARY_MSG_START_2):
                    self._buf = self._buf[len(self._COMMENTARY_MSG_START_2) :]
                    if not self._printed_commentary_prefix:
                        self._write("Commentary: ")
                        self._printed_commentary_prefix = True
                    self._state = self._STATE_COMMENTARY_MSG
                    continue

                if self._buf.startswith(self._COMMENTARY_MSG_START_1):
                    self._buf = self._buf[len(self._COMMENTARY_MSG_START_1) :]
                    if not self._printed_commentary_prefix:
                        self._write("Commentary: ")
                        self._printed_commentary_prefix = True
                    self._state = self._STATE_COMMENTARY_MSG
                    continue

                if self._buf.startswith(self._COMMENTARY_TOOL_START_2):
                    self._buf = self._buf[len(self._COMMENTARY_TOOL_START_2) :]
                    if not self._printed_tool_prefix:
                        self._write("Tool: ")
                        self._printed_tool_prefix = True
                    self._state = self._STATE_COMMENTARY_TOOL
                    continue

                if self._buf.startswith(self._COMMENTARY_TOOL_START_1):
                    self._buf = self._buf[len(self._COMMENTARY_TOOL_START_1) :]
                    if not self._printed_tool_prefix:
                        self._write("Tool: ")
                        self._printed_tool_prefix = True
                    self._state = self._STATE_COMMENTARY_TOOL
                    continue

                # Fallback: drop one char and keep going.
                self._buf = self._buf[1:]
                continue

            if self._state == self._STATE_ANALYSIS:
                end = self._buf.find(self._ANALYSIS_END)
                if end == -1:
                    if allow_partial:
                        return
                    if len(self._buf) <= self._tail_keep:
                        return
                    safe = self._buf[: -self._tail_keep]
                    self._buf = self._buf[-self._tail_keep :]
                    self._write(safe)
                    continue

                content = self._buf[:end]
                self._buf = self._buf[end + len(self._ANALYSIS_END) :]
                self._write(content)

                if self._thinking_color_on:
                    self._write(self._RESET)
                    self._thinking_color_on = False
                self._state = self._STATE_RAW
                continue

            if self._state == self._STATE_COMMENTARY_MSG:
                end = self._buf.find(self._ANALYSIS_END)
                if end == -1:
                    if allow_partial:
                        return
                    if len(self._buf) <= self._tail_keep:
                        return
                    safe = self._buf[: -self._tail_keep]
                    self._buf = self._buf[-self._tail_keep :]
                    self._write(self._strip_markers(safe))
                    continue

                content = self._buf[:end]
                self._buf = self._buf[end + len(self._ANALYSIS_END) :]
                self._write(self._strip_markers(content))
                self._write("\n")
                self._printed_commentary_prefix = False
                self._state = self._STATE_RAW
                continue

            if self._state == self._STATE_COMMENTARY_TOOL:
                end = self._buf.find(self._COMMENTARY_TOOL_END)
                if end == -1:
                    if allow_partial:
                        return
                    if len(self._buf) <= self._tail_keep:
                        return
                    safe = self._buf[: -self._tail_keep]
                    self._buf = self._buf[-self._tail_keep :]
                    self._write(self._strip_markers(safe))
                    continue

                content = self._buf[:end]
                self._buf = self._buf[end + len(self._COMMENTARY_TOOL_END) :]
                self._write(self._strip_markers(content))
                self._write("\n")
                self._printed_tool_prefix = False
                self._state = self._STATE_RAW
                continue

            if self._state == self._STATE_FINAL:
                end = self._buf.find(self._FINAL_END)
                if end == -1:
                    if allow_partial:
                        return
                    if len(self._buf) <= self._tail_keep:
                        return
                    safe = self._buf[: -self._tail_keep]
                    self._buf = self._buf[-self._tail_keep :]
                    self._write(safe)
                    continue

                content = self._buf[:end]
                self._buf = self._buf[end + len(self._FINAL_END) :]
                self._write(content)
                self._state = self._STATE_RAW
                continue

            return
