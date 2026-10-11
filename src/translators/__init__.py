"""Translation providers, factory and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.config import PROVIDER_NAMES, Settings
from src.translators.base import (
    BaseTranslator,
    ConfigurationError,
    TranslationError,
    TranslatorChain,
    explain_error,
)
from src.translators.deepl_api import DeepLApiProvider
from src.translators.google import GoogleTranslateProvider
from src.translators.mymemory import MyMemoryProvider

__all__ = [
    "BaseTranslator",
    "ConfigurationError",
    "ProviderCheck",
    "TranslationError",
    "TranslatorChain",
    "create_translator",
    "diagnose",
    "explain_error",
]

TEST_SENTENCES = ["Hello, how are you?", "I will find you."]


def _build_provider(name: str, settings: Settings, **override: Any) -> BaseTranslator:
    """Instantiate a provider from its display name (unknown names fall back to Google).

    Args:
        name: One of ``config.PROVIDER_NAMES``.
        settings: Application settings.
        **override: Options replacing the values taken from ``settings``.

    Returns:
        Configured provider.
    """
    common: Dict[str, Any] = {
        "retries": settings.get("translation.retries", 2),
        "backoff": settings.get("translation.retry_backoff", 1.0),
        "min_interval": settings.get("translation.min_interval", 0.5),
        "timeout": settings.get("translation.timeout", 20),
    }
    common.update(override)
    factories: Dict[str, Callable[[], BaseTranslator]] = {
        "Google Translate": lambda: GoogleTranslateProvider(**common),
        "MyMemory (free)": lambda: MyMemoryProvider(**common),
        "DeepL API Key": lambda: DeepLApiProvider(api_key=settings.get("translation.deepl_api_key", ""), **common),
    }
    return factories.get(name, factories["Google Translate"])()


def create_translator(settings: Settings) -> TranslatorChain:
    """Build the primary provider, followed by the backup provider if enabled.

    Args:
        settings: Application settings.

    Returns:
        A ``TranslatorChain``.
    """
    names: List[str] = [settings.get("translation.provider", "Google Translate")]
    if settings.get("translation.use_fallback", True):
        names += [n for n in settings.get("translation.fallback_providers", []) if n not in names]
    return TranslatorChain([_build_provider(n, settings) for n in names])


@dataclass
class ProviderCheck:
    """Result of testing one provider.

    Attributes:
        name: Provider display name.
        ok: ``True`` if the test sentences were translated.
        message: The translation (on success) or the explained error.
    """

    name: str
    ok: bool
    message: str


def diagnose(settings: Settings, names: Optional[Sequence[str]] = None) -> List[ProviderCheck]:
    """Test providers with short sentences and explain every failure.

    Args:
        settings: Application settings (API key, timeout...).
        names: Providers to test (default: all).

    Returns:
        One ``ProviderCheck`` per provider.
    """
    results: List[ProviderCheck] = []
    for name in names or PROVIDER_NAMES:
        try:
            provider = _build_provider(name, settings, retries=1, timeout=12, min_interval=0)
            out = provider.translate_batch(TEST_SENTENCES)
            results.append(ProviderCheck(name, True, " | ".join(out)))
        except TranslationError as exc:
            results.append(ProviderCheck(name, False, str(exc).split(": ", 1)[-1]))
        except Exception as exc:  # noqa: BLE001
            results.append(ProviderCheck(name, False, explain_error(exc)))
    return results
