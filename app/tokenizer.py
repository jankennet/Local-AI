"""
tokenizer.py

Token counting, isolated behind a small interface (ISP: consumers only
need `.count(text)`, nothing else). The concrete implementation uses the
model's OWN vocab (vocab_only=True — no weights loaded, just the tokenizer)
so counts are exact for whichever GGUF model is actually running, instead
of a generic approximation like tiktoken would give.

Swappable: if you ever want a different counting strategy (e.g. a cached/
batched version, or a remote tokenizer service), implement TokenCounter
and nothing else in the app needs to change (DIP).
"""

from functools import lru_cache
from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class LlamaVocabTokenCounter:
    """Exact token counts via the model's own tokenizer, weights not loaded."""

    def __init__(self, model_path: str):
        from llama_cpp import Llama
        self._vocab = Llama(model_path=model_path, vocab_only=True, verbose=False)
        self._count_cache = lru_cache(maxsize=1024)(self._count_uncached)

    def _count_uncached(self, text: str) -> int:
        if not text:
            return 0
        return len(self._vocab.tokenize(text.encode("utf-8")))

    def count(self, text: str) -> int:
        return self._count_cache(text)

    def clear_cache(self) -> None:
        """Clear the token count cache."""
        self._count_cache.cache_clear()

    def cache_info(self) -> str:
        """Return cache statistics."""
        return str(self._count_cache.cache_info())
