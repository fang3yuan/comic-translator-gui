"""Long text must stay inside elliptical bubbles."""

import math

from PIL import Image

from src.config import Settings, resolve_path
from src.core.models import TextRegion
from src.core.renderer import Typesetter, fit_text, load_font, text_width

FONT = str(resolve_path("fonts/default_arabic.ttf"))
LONG = "لقد انتظرت هذه اللحظة طويلا ولن أسمح لأحد بأن يقف في طريقي مهما كان الثمن الذي سأدفعه"


def test_ellipse_fit_keeps_every_line_inside_the_ellipse():
    bw, bh = 200, 150
    fit = fit_text(LONG, FONT, bw, bh, ellipse=(bw * 1.1, bh * 1.1))
    font = load_font(FONT, fit.font_size)
    top = -fit.height / 2
    for i, line in enumerate(fit.lines):
        far = max(abs(top + i * fit.line_height), abs(top + (i + 1) * fit.line_height))
        t = min(1.0, far / (bh * 1.1 / 2))
        assert text_width(font, line) <= bw * 1.1 * math.sqrt(1 - t * t) + 1


def test_render_long_text_does_not_touch_bubble_corners():
    page = Image.new("RGBA", (400, 300), (255, 255, 255, 255))
    region = TextRegion(id=0, bbox=(100, 90, 300, 210), bubble_bbox=(60, 50, 340, 250), bubble_shape="ellipse",
                        target_bbox=(75, 65, 325, 235), translation=LONG)
    Typesetter(Settings()).render_region(page, region)
    px = page.convert("L").load()
    cx, cy, rx, ry = 200, 150, 140, 100
    for y in range(300):
        for x in range(400):
            if px[x, y] < 200:
                assert ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0, (x, y)
