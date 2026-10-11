"""Translator base class: retry, throttling, caching, readable errors and a fail-over chain."""

from __future__ import annotations

import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class TranslationError(Exception):
    """Raised when translation fails after all retries / providers."""


class ConfigurationError(TranslationError):
    """Non-retryable error (missing API key, missing dependency, ...)."""


class BatchMismatch(Exception):
    """A batched answer could not be split back into the original texts (retry one by one)."""


def is_rate_limited(exc: BaseException) -> bool:
    """Whether ``exc`` means "too many requests" (retrying immediately only makes it worse)."""
    low = f"{type(exc).__name__}: {exc}".lower()
    return any(k in low for k in ("429", "too many", "rate limit", "quota", "all available free translations"))


def explain_error(exc: BaseException) -> str:
    """Turn a low-level exception into a short, human-readable cause (English + Arabic).

    Args:
        exc: The exception raised by a provider.

    Returns:
        One-line explanation with a suggested fix.
    """
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if isinstance(exc, ConfigurationError):
        return str(exc)
    if is_rate_limited(exc):
        return ("Too many requests (rate limit) - the service asked us to slow down; the backup provider is used. "
                "/ طلبات كثيرة، الخدمة طلبت التمهّل، سيُستخدم المزوّد الاحتياطي.")
    if any(k in low for k in ("name resolution", "getaddrinfo", "connection", "max retries", "timed out",
                              "timeout", "unreachable", "offline", "proxy")):
        return ("Cannot reach the service - check your internet / VPN / firewall. "
                "/ لا يوجد اتصال بالخدمة، تأكد من الإنترنت أو الـVPN أو الجدار الناري.")
    if any(k in low for k in ("403", "forbidden", "blocked", "captcha")):
        return ("The service blocked the request (HTTP 403) - try another provider or a VPN. "
                "/ الخدمة حجبت الطلب، جرّب مزودًا آخر أو VPN.")
    if "ssl" in low or "certificate" in low:
        return ("SSL/certificate problem - check date/time, antivirus HTTPS scanning or proxy. "
                "/ مشكلة شهادة SSL، تحقق من الوقت والتاريخ ومضاد الفيروسات.")
    return text[:300]


class BaseTranslator(ABC):
    """Base class of all providers.

    Requests are sent **sequentially** with a minimum interval (no bursts, so free
    services do not ban us). Providers that can translate a whole page in one request set
    ``supports_batch = True`` and implement ``_translate_many``.
    """

    name: str = "base"
    supports_batch: bool = False

    def __init__(
        self,
        source: str = "en",
        target: str = "ar",
        retries: int = 2,
        backoff: float = 1.0,
        min_interval: float = 0.5,
        timeout: float = 20.0,
    ) -> None:
        """Initialise common options.

        Args:
            source: Source language code.
            target: Target language code.
            retries: Attempts per request (>= 1).
            backoff: Base delay in seconds, doubled after each failed attempt.
            min_interval: Minimum seconds between two outgoing requests.
            timeout: Network timeout in seconds.
        """
        self.source = source
        self.target = target
        self.retries = max(1, int(retries))
        self.backoff = float(backoff)
        self.min_interval = float(min_interval)
        self.timeout = float(timeout)
        self._sleep: Callable[[float], None] = time.sleep
        self._lock = threading.Lock()
        self._last_request = 0.0
        self._cache: Dict[str, str] = {}

    # -- to implement ---------------------------------------------------
    @abstractmethod
    def _translate_one(self, text: str) -> str:
        """Translate a single non-empty string (may raise)."""

    def _translate_many(self, texts: Sequence[str]) -> List[str]:
        """Translate several strings in one request (``supports_batch`` providers only)."""
        raise NotImplementedError

    # -- helpers --------------------------------------------------------
    def _throttle(self) -> None:
        """Block until ``min_interval`` seconds passed since the last request."""
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = time.monotonic()

    def _with_retry(self, fn: Callable[[], object]) -> object:
        """Run ``fn`` with throttling and exponential back-off.

        Configuration problems, batch mismatches and rate limits are not retried: the
        first two cannot succeed, and retrying a rate limit only extends the ban.

        Args:
            fn: Zero-argument callable performing the request.

        Returns:
            Whatever ``fn`` returns.

        Raises:
            TranslationError: After the last failed attempt.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                self._throttle()
                return fn()
            except (ConfigurationError, BatchMismatch):
                raise
            except Exception as exc:  # noqa: BLE001 - network libs raise anything
                last_exc = exc
                logger.warning("%s attempt %d/%d failed: %s", self.name, attempt + 1, self.retries, exc)
                if is_rate_limited(exc):
                    break
                if attempt < self.retries - 1:
                    self._sleep(self.backoff * (2 ** attempt) + random.uniform(0, 0.2))
        raise TranslationError(f"{self.name}: {explain_error(last_exc)}" if last_exc else self.name)

    # -- public API -----------------------------------------------------
    def translate(self, text: str) -> str:
        """Translate one string.

        Args:
            text: Source text.

        Returns:
            Translation (empty string for blank input).
        """
        text = text.strip()
        if not text:
            return ""
        if text not in self._cache:
            result = str(self._with_retry(lambda: self._translate_one(text))).strip()
            if not result:
                raise TranslationError(f"{self.name} returned an empty translation")
            self._cache[text] = result
        return self._cache[text]

    def translate_batch(self, texts: Sequence[str]) -> List[str]:
        """Translate many strings, preserving order.

        Uses one request for the whole page when the provider supports it (fast, and the
        engine sees the context of the page); falls back to one request per text only if
        the batched answer cannot be split back.

        Args:
            texts: Source strings.

        Returns:
            Translations aligned with ``texts``.

        Raises:
            TranslationError: If a text could not be translated.
        """
        pending = list(dict.fromkeys(t.strip() for t in texts if t.strip() and t.strip() not in self._cache))
        if self.supports_batch and len(pending) > 1:
            try:
                out = self._with_retry(lambda: self._translate_many(pending))
                if len(out) == len(pending) and all(str(o).strip() for o in out):  # type: ignore[arg-type]
                    self._cache.update({t: str(o).strip() for t, o in zip(pending, out)})  # type: ignore[arg-type]
            except BatchMismatch:
                logger.info("%s: batched answer did not split back, translating one by one", self.name)
        for t in pending:
            self.translate(t)
        return [self._cache[t.strip()] if t.strip() else "" for t in texts]


class TranslatorChain(BaseTranslator):
    """Try several providers in order until one succeeds.

    A provider that failed is tried last on the following calls, so one broken service does
    not slow down every page.
    """

    name = "chain"

    def __init__(self, providers: Sequence[BaseTranslator]) -> None:
        """Create a chain.

        Args:
            providers: Providers in priority order (must not be empty).
        """
        if not providers:
            raise ValueError("TranslatorChain needs at least one provider")
        super().__init__(retries=1, min_interval=0.0)
        self.providers = list(providers)
        self.last_used: Optional[str] = None
        self.on_fallback: Optional[Callable[[str, str, Exception], None]] = None
        self._dead: set[str] = set()

    def _translate_one(self, text: str) -> str:
        return self.translate_batch([text])[0]

    def translate(self, text: str) -> str:
        """Translate one string using the first working provider."""
        return self.translate_batch([text])[0]

    def translate_batch(self, texts: Sequence[str]) -> List[str]:
        """Translate using the first provider that succeeds.

        Args:
            texts: Source strings.

        Returns:
            Translations aligned with ``texts``.

        Raises:
            TranslationError: If every provider fails (the message lists each cause).
        """
        order = [p for p in self.providers if p.name not in self._dead]
        order += [p for p in self.providers if p.name in self._dead]
        errors: List[str] = []
        for idx, provider in enumerate(order):
            try:
                out = provider.translate_batch(texts)
            except TranslationError as exc:
                self._dead.add(provider.name)
                errors.append(f"- {exc}")
                if self.on_fallback and idx + 1 < len(order):
                    self.on_fallback(provider.name, order[idx + 1].name, exc)
                continue
            self._dead.discard(provider.name)
            self.last_used = provider.name
            return out
        raise TranslationError("All translation providers failed:\n" + "\n".join(errors))
