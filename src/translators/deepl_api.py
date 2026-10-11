"""Official DeepL REST API provider (Free or Pro key)."""

from __future__ import annotations

from typing import Any, List, Sequence

from src.translators.base import BaseTranslator, ConfigurationError


class DeepLApiProvider(BaseTranslator):
    """Translate with the official DeepL API.

    Keys ending with ``:fx`` use the Free endpoint, others the Pro endpoint.
    """

    name = "DeepL API Key"
    supports_batch = True

    def __init__(self, api_key: str = "", **kwargs: Any) -> None:
        """Create the provider.

        Args:
            api_key: DeepL authentication key.
            **kwargs: Forwarded to ``BaseTranslator``.
        """
        super().__init__(**kwargs)
        self.api_key = api_key.strip()

    @property
    def endpoint(self) -> str:
        """REST endpoint matching the key type."""
        host = "api-free.deepl.com" if self.api_key.endswith(":fx") else "api.deepl.com"
        return f"https://{host}/v2/translate"

    def _post(self, texts: Sequence[str]) -> List[str]:
        """Send one request with one or more texts."""
        if not self.api_key:
            raise ConfigurationError("DeepL API key is empty - paste it in Translation settings. / مفتاح DeepL فارغ، ألصقه في إعدادات الترجمة.")
        try:
            import requests
        except ImportError as exc:
            raise ConfigurationError("The 'requests' package is missing - open Setup and press Install.") from exc
        resp = requests.post(
            self.endpoint,
            headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"},
            json={
                "text": list(texts),
                "source_lang": self.source.upper(),
                "target_lang": self.target.upper(),
            },
            timeout=self.timeout,
        )
        if resp.status_code in (401, 403):
            raise ConfigurationError(f"DeepL rejected the API key (HTTP {resp.status_code})")
        if resp.status_code == 456:
            raise ConfigurationError("DeepL quota exceeded")
        resp.raise_for_status()
        return [item["text"] for item in resp.json()["translations"]]

    def _translate_one(self, text: str) -> str:
        """Translate a single string."""
        return self._post([text])[0]

    def _translate_many(self, texts: Sequence[str]) -> List[str]:
        """Translate many strings in a single request."""
        return self._post(texts)
