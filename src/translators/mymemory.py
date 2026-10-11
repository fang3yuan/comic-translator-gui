"""MyMemory translation provider (free public API, no key; ~5000 chars/day anonymous)."""

from __future__ import annotations

import re
from typing import Any, List

from src.translators.base import BaseTranslator, ConfigurationError

_URL = "https://api.mymemory.translated.net/get"
_MAX_CHARS = 450  # the API rejects queries longer than 500 characters


def split_chunks(text: str, limit: int = _MAX_CHARS) -> List[str]:
    """Split text at sentence/word boundaries into pieces of at most ``limit`` chars.

    Args:
        text: Text to split.
        limit: Maximum piece length.

    Returns:
        Non-empty pieces whose concatenation (with spaces) equals the input words.
    """
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    chunks: List[str] = []
    cur = ""
    for sent in sentences:
        for word_group in ([sent] if len(sent) <= limit else sent.split(" ")):
            if len(cur) + len(word_group) + 1 <= limit:
                cur = f"{cur} {word_group}".strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = word_group
    if cur:
        chunks.append(cur)
    return chunks


class MyMemoryProvider(BaseTranslator):
    """MyMemory public translation memory + MT. Good free fail-over for Google."""

    name = "MyMemory (free)"

    def __init__(self, **kwargs: Any) -> None:
        """Create the provider; keyword arguments go to ``BaseTranslator``."""
        super().__init__(**kwargs)

    def _translate_chunk(self, chunk: str) -> str:
        """Translate one chunk (<= 450 chars)."""
        try:
            import requests
        except ImportError as exc:
            raise ConfigurationError("The 'requests' package is missing - open Setup and press Install.") from exc
        resp = requests.get(
            _URL,
            params={"q": chunk, "langpair": f"{self.source}|{self.target}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        status = int(data.get("responseStatus", 200))
        text = str(data.get("responseData", {}).get("translatedText", "")).strip()
        if status == 429 or "MYMEMORY WARNING" in text.upper():
            raise RuntimeError("MyMemory daily limit reached (429 / all available free translations)")
        if status != 200 or not text:
            raise RuntimeError(f"MyMemory error {status}: {data.get('responseDetails', '')}")
        return text

    def _translate_one(self, text: str) -> str:
        """Translate ``text`` (split into chunks when long)."""
        return " ".join(self._translate_chunk(c) for c in split_chunks(text))
