"""Google Translate provider: public web endpoint, no key, one request per page."""

from __future__ import annotations

from typing import Any, List, Sequence

from src.translators.base import BaseTranslator, BatchMismatch, ConfigurationError

_URL = "https://translate.googleapis.com/translate_a/single"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
_MAX_CHARS = 3500  # per request; a comic page is far below this


class GoogleTranslateProvider(BaseTranslator):
    """Free Google Translate web endpoint (the one browser extensions use).

    All blocks of a page are sent in a single request, one per line, and split back by
    line. This keeps the request count tiny (no rate-limit bans) and gives the engine the
    context of the whole page. If Google merges or splits lines, the texts are translated
    one by one instead.
    """

    name = "Google Translate"
    supports_batch = True

    @staticmethod
    def parse_response(data: Any) -> str:
        """Extract the translated text from Google's nested-list response.

        Args:
            data: Decoded JSON (``[[[translated, source, ...], ...], ...]``).

        Returns:
            Concatenated translation (line breaks preserved).

        Raises:
            ValueError: If the structure is unexpected.
        """
        if not (isinstance(data, list) and data and isinstance(data[0], list)):
            raise ValueError(f"unexpected Google response: {str(data)[:120]}")
        try:
            return "".join(seg[0] for seg in data[0] if isinstance(seg, list) and seg and seg[0])
        except (TypeError, IndexError, KeyError) as exc:
            raise ValueError(f"unexpected Google response: {str(data)[:120]}") from exc

    def _request(self, text: str) -> str:
        """Send one request and return the raw translation."""
        try:
            import requests
        except ImportError as exc:
            raise ConfigurationError("The 'requests' package is missing - open Setup and press Install.") from exc
        resp = requests.post(
            _URL,
            params={"client": "gtx", "sl": self.source, "tl": self.target, "dt": "t"},
            data={"q": text},
            headers=_HEADERS,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        result = self.parse_response(resp.json())
        if not result.strip():
            raise RuntimeError("empty response from Google Translate")
        return result

    def _translate_one(self, text: str) -> str:
        """Translate a single string."""
        return self._request(text)

    def _translate_many(self, texts: Sequence[str]) -> List[str]:
        """Translate ``texts`` joined by newlines (chunked), then split the answer back.

        Raises:
            BatchMismatch: If the number of answer lines differs from the number of texts.
        """
        flat = [" ".join(t.split()) for t in texts]  # a newline inside a text would break the mapping
        out: List[str] = []
        chunk: List[str] = []
        size = 0
        for t in flat + [None]:  # type: ignore[list-item]
            if t is None or (chunk and size + len(t) + 1 > _MAX_CHARS):
                lines = [ln.strip() for ln in self._request("\n".join(chunk)).split("\n")]
                lines = [ln for ln in lines if ln]
                if len(lines) != len(chunk):
                    raise BatchMismatch(f"{len(chunk)} texts but {len(lines)} lines returned")
                out.extend(lines)
                chunk, size = [], 0
            if t is not None:
                chunk.append(t)
                size += len(t) + 1
        return out
