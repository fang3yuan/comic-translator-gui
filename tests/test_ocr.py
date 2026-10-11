"""Tests for OCR helpers, line grouping and style estimation (no model download needed)."""

import cv2
import numpy as np
import pytest

from src.core.detector import Detector
from src.core.models import FontRole, OCRLine
from src.core.ocr import OCREngine, clean_ocr_text, estimate_style
from src.utils.font_matcher import classify_roles


def _line(x1, y1, x2, y2, text="word", conf=0.9):
    return OCRLine([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], text, conf)


def test_clean_text_joins_hyphen_and_sentence_cases():
    assert clean_ocr_text("WHAT ARE YOU\nDOING? I DON'T KNOW.") == "What are you doing? I don't know."
    assert clean_ocr_text("incred-\nible") == "incredible"
    assert clean_ocr_text("It is fine.") == "It is fine."


def test_group_lines_merges_same_bubble_only():
    det = Detector()
    lines = [_line(100, 100, 200, 120), _line(105, 124, 195, 144), _line(500, 400, 600, 420)]
    groups = det.group_lines(lines)
    assert sorted(len(g) for g in groups) == [1, 2]


def test_reading_order_rtl_and_ltr():
    det = Detector()
    img = np.full((600, 800, 3), 255, np.uint8)
    lines = [_line(100, 50, 200, 80, "left"), _line(600, 55, 700, 85, "right"), _line(300, 400, 400, 430, "low")]
    rtl = det.build_regions(img, lines, "rtl")
    ltr = det.build_regions(img, lines, "ltr")
    assert [r.text for r in rtl] == ["right", "left", "low"]
    assert [r.text for r in ltr] == ["left", "right", "low"]
    assert [r.id for r in rtl] == [0, 1, 2]


def test_estimate_style_detects_colours_and_size():
    img = np.full((200, 300, 3), 255, np.uint8)
    cv2.putText(img, "TEXT", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # red (BGR)
    region = Detector().build_regions(img, [_line(35, 70, 200, 125, "TEXT")], "ltr")[0]
    style = estimate_style(img, region)
    assert style.color[0] > 180 and style.color[1] < 80  # red text
    assert style.bg_color == (255, 255, 255)
    assert 40 <= style.font_size <= 60


def test_classify_roles():
    img = np.full((600, 800, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 100), (400, 160), (230, 230, 230), -1)
    regions = Detector().build_regions(
        img, [_line(120, 110, 380, 140, "Meanwhile, back at the base"), _line(500, 400, 700, 460, "BOOM")], "ltr"
    )
    for r in regions:
        r.source_style.font_size = 24
    classify_roles(regions)
    roles = {r.text: r.role for r in regions}
    assert roles["BOOM"] == FontRole.SFX


def test_engine_filters_noise():
    eng = OCREngine(min_confidence=0.3)
    kept = eng._filter([_line(0, 0, 50, 20, "ok", 0.9), _line(0, 0, 50, 20, "low", 0.1),
                        _line(0, 0, 50, 20, "~~", 0.9), _line(0, 0, 3, 3, "x", 0.9)])
    assert [l.text for l in kept] == ["ok"]


def test_rapidocr_reads_english_with_spaces():
    pytest.importorskip("rapidocr")
    from PIL import Image, ImageDraw, ImageFont

    from src.config import resolve_path

    im = Image.new("RGB", (700, 200), "white")
    font = ImageFont.truetype(str(resolve_path("fonts/default_arabic.ttf")), 36)
    ImageDraw.Draw(im).text((30, 60), "WHAT ARE YOU DOING?", font=font, fill="black")
    lines = OCREngine().read_lines(np.array(im)[:, :, ::-1].copy())
    text = " ".join(l.text for l in lines).upper()
    assert "WHAT ARE YOU" in text


def test_missing_engine_gives_actionable_error(monkeypatch):
    from src.core.ocr import OCRUnavailableError

    import sys

    eng = OCREngine()
    monkeypatch.setitem(sys.modules, "rapidocr", None)  # makes "from rapidocr import ..." fail
    with pytest.raises(OCRUnavailableError) as exc:
        eng.read_lines(np.zeros((50, 50, 3), np.uint8))
    assert "Setup" in str(exc.value)
