"""Tests for translators: retry, fallback chain and Arabic post-processing."""

from typing import List

import pytest

from src.config import Settings
from src.translators import create_translator
from src.translators.base import (
    BaseTranslator,
    ConfigurationError,
    TranslationError,
    TranslatorChain,
)
from src.translators.deepl_api import DeepLApiProvider
from src.translators.google import GoogleTranslateProvider
from src.translators.mymemory import MyMemoryProvider, split_chunks
from src.translators import diagnose, explain_error
from src.utils.arabic_utils import contains_arabic, postprocess_translation, shape_arabic


class Flaky(BaseTranslator):
    """Fails ``fail_times`` times, then succeeds."""

    name = "flaky"

    def __init__(self, fail_times: int, **kw):
        super().__init__(min_interval=0, backoff=0, **kw)
        self.fail_times, self.calls = fail_times, 0
        self._sleep = lambda s: None

    def _translate_one(self, text: str) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("boom")
        return f"ar:{text}"


class Broken(BaseTranslator):
    name = "broken"

    def __init__(self):
        super().__init__(retries=2, min_interval=0, backoff=0)
        self._sleep = lambda s: None

    def _translate_one(self, text: str) -> str:
        raise RuntimeError("down")


def test_retry_succeeds_after_failures():
    t = Flaky(2, retries=3)
    assert t.translate("hello") == "ar:hello"
    assert t.calls == 3


def test_retry_gives_up():
    with pytest.raises(TranslationError):
        Flaky(5, retries=2).translate("hello")


def test_cache_avoids_second_call():
    t = Flaky(0)
    t.translate("a")
    t.translate("a")
    assert t.calls == 1


def test_chain_falls_back():
    events: List[str] = []
    chain = TranslatorChain([Broken(), Flaky(0)])
    chain.on_fallback = lambda a, b, e: events.append(f"{a}->{b}")
    assert chain.translate_batch(["x", "y"]) == ["ar:x", "ar:y"]
    assert chain.last_used == "flaky"
    assert events == ["broken->flaky"]


def test_chain_raises_when_all_fail():
    with pytest.raises(TranslationError) as exc:
        TranslatorChain([Broken(), Broken()]).translate("x")
    assert "All translation providers failed" in str(exc.value)


def test_missing_api_key_is_not_retried():
    p = DeepLApiProvider(api_key="", retries=5, min_interval=0)
    with pytest.raises(ConfigurationError):
        p.translate("hi")


def test_factory_builds_chain_with_fallback():
    s = Settings({"translation": {"provider": "MyMemory (free)", "fallback_providers": ["Google Translate"]}})
    chain = create_translator(s)
    assert [p.name for p in chain.providers] == ["MyMemory (free)", "Google Translate"]
    s = Settings({"translation": {"provider": "Removed Provider", "use_fallback": False}})
    assert [p.name for p in create_translator(s).providers] == ["Google Translate"]  # unknown -> Google


def test_empty_text():
    assert Flaky(0).translate("   ") == ""
    assert Flaky(0).translate_batch(["", "a"]) == ["", "ar:a"]


def test_postprocess_punctuation_and_digits():
    out = postprocess_translation("هل أنت بخير? نعم, أنا بخير...", arabic_digits=False)
    assert "؟" in out and "،" in out and "…" in out
    assert postprocess_translation("عندي 3 تفاحات", arabic_digits=True).count("٣") == 1


def test_shape_arabic_changes_text_and_keeps_arabic():
    shaped = shape_arabic("مرحبا")
    assert shaped != "مرحبا"
    assert contains_arabic(shaped)
    assert shaped.count("\n") == 0


def test_chain_skips_dead_provider_next_time():
    broken, good = Broken(), Flaky(0)
    chain = TranslatorChain([broken, good])
    chain.translate_batch(["a"])
    assert "broken" in chain._dead
    broken_calls = []
    broken._translate_one = lambda t: broken_calls.append(t) or (_ for _ in ()).throw(RuntimeError("x"))
    chain.translate_batch(["b"])
    assert broken_calls == []  # not tried again while a healthy provider exists


def test_batch_dedupes_and_keeps_order():
    t = Flaky(0)
    assert t.translate_batch(["ab", "cd", "ab"]) == ["ar:ab", "ar:cd", "ar:ab"]
    assert t.calls == 2  # duplicate text is translated once


def test_rate_limit_is_not_retried_and_fails_over():
    class Limited(BaseTranslator):
        name = "limited"

        def __init__(self):
            super().__init__(retries=5, min_interval=0, backoff=0)
            self.calls = 0

        def _translate_one(self, text):
            self.calls += 1
            raise RuntimeError("HTTP 429 Too Many Requests")

    lim = Limited()
    chain = TranslatorChain([lim, Flaky(0)])
    assert chain.translate_batch(["x"]) == ["ar:x"]
    assert lim.calls == 1  # no retry storm on a 429


def test_google_sends_whole_page_in_one_request(monkeypatch):
    import requests

    sent = []

    class Resp:
        def __init__(self, q):
            self.q = q

        def raise_for_status(self):
            pass

        def json(self):
            return [[[f"ar:{ln}\n", ln] for ln in self.q.split("\n")]]

    def fake_post(url, params=None, data=None, headers=None, timeout=None):
        sent.append(data["q"])
        return Resp(data["q"])

    monkeypatch.setattr(requests, "post", fake_post)
    p = GoogleTranslateProvider(min_interval=0)
    assert p.translate_batch(["one", "two", "three"]) == ["ar:one", "ar:two", "ar:three"]
    assert len(sent) == 1 and sent[0] == "one\ntwo\nthree"


def test_google_falls_back_to_single_requests_on_line_mismatch(monkeypatch):
    import requests

    class Resp:
        def __init__(self, q):
            self.q = q

        def raise_for_status(self):
            pass

        def json(self):
            return [[[self.q.replace("\n", " ").upper(), self.q]]]  # merges the lines

    monkeypatch.setattr(requests, "post", lambda url, params=None, data=None, headers=None, timeout=None: Resp(data["q"]))
    out = GoogleTranslateProvider(min_interval=0).translate_batch(["a b", "c d"])
    assert out == ["A B", "C D"]


def test_google_parse_response():
    data = [[["مرحبا ", "Hello ", None, None, 1], ["كيف حالك؟", "how are you?", None, None, 1]], None, "en"]
    assert GoogleTranslateProvider.parse_response(data) == "مرحبا كيف حالك؟"
    with pytest.raises(ValueError):
        GoogleTranslateProvider.parse_response("garbage")


def test_google_uses_http_and_reports_errors(monkeypatch):
    import requests

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return [[["مرحبا", "Hello"]]]

    seen = {}

    def fake_post(url, params=None, data=None, headers=None, timeout=None):
        seen.update(url=url, params=params, data=data)
        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    p = GoogleTranslateProvider(min_interval=0)
    assert p.translate("Hello") == "مرحبا"
    assert seen["params"]["tl"] == "ar" and seen["data"] == {"q": "Hello"}

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("Name resolution failed")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(TranslationError) as exc:
        GoogleTranslateProvider(min_interval=0, retries=1).translate("Other")
    assert "internet" in str(exc.value).lower()


def test_mymemory_chunking_and_limit(monkeypatch):
    text = ("This is a sentence. " * 60).strip()
    chunks = split_chunks(text)
    assert all(len(c) <= 450 for c in chunks) and " ".join(chunks).split() == text.split()

    import requests

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"responseStatus": 200, "responseData": {"translatedText": "MYMEMORY WARNING: YOU USED ALL"}}

    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
    with pytest.raises(TranslationError) as exc:
        MyMemoryProvider(min_interval=0, retries=1).translate("hi")
    assert "limit" in str(exc.value).lower()


def test_explain_error_variants():
    assert "internet" in explain_error(ConnectionError("getaddrinfo failed")).lower()
    assert "rate limit" in explain_error(RuntimeError("HTTP 429 Too Many Requests")).lower()
    assert "blocked" in explain_error(RuntimeError("403 Forbidden")).lower()


def test_diagnose_reports_every_provider_without_network(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(requests, "get", boom)
    results = {r.name: r for r in diagnose(Settings())}
    assert set(results) >= {"Google Translate", "MyMemory (free)", "DeepL API Key"} and len(results) == 3
    assert not results["Google Translate"].ok and "internet" in results["Google Translate"].message.lower()
    assert "key" in results["DeepL API Key"].message.lower()
