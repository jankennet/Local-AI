"""
continuation.py

When a final text reply is cut off because the model hit its per-call
token cap (finish_reason == "length"), we keep generating -- appending the
partial reply as an assistant message and asking the model to continue --
until it finishes naturally ("stop"), the token budget is exhausted, or the
continuation-round cap is hit. This prevents mid-sentence cut-offs like a
fixed max_tokens would otherwise produce.

Two entry points:
  - continue_text: non-streaming, returns the full accumulated reply.
  - stream_text_continuation: streaming, yields content_delta events for each
    continuation chunk plus a final {"type":"done","reply":...} event.

Only the completed reply should be persisted by the caller (never the
intermediate partials).
"""

import logging
from typing import AsyncGenerator, Callable, Optional

logger = logging.getLogger(__name__)

MAX_CONTINUATION_ROUNDS = 4
CONTINUATION_CHUNK_TOKENS = 1024
CUTOFF_MARKER = "\n\n… (answer cut off — context limit reached)"
REPEAT_MARKER = "\n\n… (stopped — it started repeating itself)"

# Minimum cleaned text length below which we treat a chunk as too short to
# judge repetition (pauses, interjections, etc. are not evidence of a loop).
_MIN_REPETITION_SIGNAL = 30
_LEADING_OVERLAP = 80
_NGRAM = 40


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def detect_repetition(text: str, preceding_text: str) -> bool:
    """True when *text* largely re-emits content already in *preceding_text*.

    Catches degenerate loops where a continuation restarts from the beginning
    of the reply (or re-emits whole blocks) instead of continuing.
    """
    norm_text = _normalize(text)
    if len(norm_text) < _MIN_REPETITION_SIGNAL:
        return False
    norm_preceding = _normalize(preceding_text)
    if not norm_preceding:
        return False

    # Restarting from earlier content: the leading block already exists.
    if norm_text[:_LEADING_OVERLAP] in norm_preceding:
        return True

    # n-gram overlap: if most of the new chunk is already present, it's a repeat.
    chunks = [
        norm_text[i : i + _NGRAM]
        for i in range(0, len(norm_text) - _NGRAM + 1, _NGRAM)
    ]
    if len(chunks) < 3:
        return False
    present = sum(1 for c in chunks if c in norm_preceding)
    return present / len(chunks) >= 0.6


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


async def continue_text(
    completion_client,
    messages: list,
    partial_content: str,
    temperature: float,
    remaining_budget: int,
    chunk_tokens: int = CONTINUATION_CHUNK_TOKENS,
    max_rounds: int = MAX_CONTINUATION_ROUNDS,
    cutoff_marker: str = CUTOFF_MARKER,
    repeat_marker: str = REPEAT_MARKER,
    count_tokens: Optional[Callable[[str], int]] = None,
) -> tuple[str, bool, int]:
    """Continue a truncated reply until the model stops naturally.

    Returns (full_content, completed_naturally, continuation_tokens).
    completed_naturally is False when the reply was stopped by the budget or
    round cap -- in that case the cutoff marker is appended. If a continuation
    chunk just re-emits already-written text (a degeneration loop), that chunk
    is discarded and the repeat marker is appended instead.
    """
    token_counter = count_tokens or _estimate_tokens
    content = partial_content
    used_tokens = 0
    completed = False
    stopped_repeat = False

    for _ in range(max_rounds):
        chunk = min(chunk_tokens, max(0, remaining_budget - used_tokens))
        if chunk <= 0:
            break

        continuation_messages = messages + [{"role": "assistant", "content": content}]
        message = await completion_client.complete_with_tools(
            continuation_messages, [], chunk, temperature
        )
        piece = message.get("content") or ""
        if not piece:
            completed = True
            break
        if detect_repetition(piece, content):
            stopped_repeat = True
            break
        content += piece
        used_tokens += token_counter(piece)

        reason = message.get("finish_reason")
        if reason == "length":
            continue
        # "stop", "tool_calls" (unexpected with empty tools), or a backend that
        # omitted finish_reason after a finish event -- assume we're done.
        completed = True
        break

    if stopped_repeat and repeat_marker:
        content = content.rstrip() + repeat_marker
    elif not completed and cutoff_marker:
        content = content.rstrip() + cutoff_marker

    return content, completed, used_tokens


async def stream_text_continuation(
    completion_client,
    messages: list,
    partial_content: str,
    temperature: float,
    remaining_budget: int,
    chunk_tokens: int = CONTINUATION_CHUNK_TOKENS,
    max_rounds: int = MAX_CONTINUATION_ROUNDS,
    cutoff_marker: str = CUTOFF_MARKER,
    repeat_marker: str = REPEAT_MARKER,
    count_tokens: Optional[Callable[[str], int]] = None,
) -> AsyncGenerator[dict, None]:
    """Stream a continuation of a truncated reply.

    Yields {"type":"content_delta","content":...} for each continuation token
    and finally {"type":"done","reply":<full reply>}. The full reply includes
    the original partial_content. If a continuation chunk starts re-emitting
    already-written text, streaming stops early, the duplicated text is dropped
    from the reply, and the repeat marker is emitted instead.
    """
    token_counter = count_tokens or _estimate_tokens
    content = partial_content
    used_tokens = 0
    completed = False
    stopped_repeat = False

    for _ in range(max_rounds):
        chunk = min(chunk_tokens, max(0, remaining_budget - used_tokens))
        if chunk <= 0:
            break

        continuation_messages = messages + [{"role": "assistant", "content": content}]
        round_base = content
        piece = ""
        stopped_repeat = False
        async for event in completion_client.complete_with_tools_stream(
            continuation_messages, [], chunk, temperature
        ):
            if event["type"] == "content":
                piece += event["content"]
                if detect_repetition(piece, round_base):
                    stopped_repeat = True
                    break
                content += event["content"]
                yield {"type": "content_delta", "content": event["content"]}
            elif event["type"] == "finish":
                if event.get("finish_reason") != "length":
                    completed = True
                break

        if stopped_repeat:
            content = round_base
            break
        if piece:
            used_tokens += token_counter(piece)
        if completed:
            break

    if stopped_repeat and repeat_marker:
        content = content.rstrip() + repeat_marker
        yield {"type": "content_delta", "content": repeat_marker}
    elif not completed and cutoff_marker:
        content = content.rstrip() + cutoff_marker
        yield {"type": "content_delta", "content": cutoff_marker}

    yield {"type": "done", "reply": content}