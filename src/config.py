"""Application settings: defaults, persistence and path constants."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT: Path = Path(
    os.environ.get("COMIC_TRANSLATOR_HOME", Path(__file__).resolve().parents[1])
)
FONTS_DIR: Path = PROJECT_ROOT / "fonts"
MODELS_DIR: Path = PROJECT_ROOT / "models"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "settings.json"

PROVIDER_NAMES = ["Google Translate", "MyMemory (free)", "DeepL API Key"]
ROLES = ["dialogue", "narration", "sfx", "whisper"]

_ROLE_STYLE_BASE: Dict[str, Any] = {
    "color": "#000000",
    "match_source_color": True,
    "stroke_enabled": False,
    "stroke_color": "#FFFFFF",
    "stroke_auto": True,
    "stroke_width": 2,
    "shadow_enabled": False,
    "shadow_color": "#000000",
    "shadow_offset": 2,
    "opacity": 1.0,
    "max_font_ratio": 1.4,
}


def _role_style(**overrides: Any) -> Dict[str, Any]:
    style = copy.deepcopy(_ROLE_STYLE_BASE)
    style.update(overrides)
    return style


SETTINGS_VERSION = 4

DEFAULT_SETTINGS: Dict[str, Any] = {
    "version": SETTINGS_VERSION,
    "general": {
        "theme": "dark",
        "reading_order": "rtl",
        "output_format": "png",
        "jpg_quality": 95,
        "save_project_json": True,
        "first_run_done": False,
    },
    "detector": {
        "mask_dilate": 3,
    },
    "ocr": {
        "min_confidence": 0.25,
        "tile_height": 1500,
        "max_side": 1280,
    },
    "inpainting": {
        "backend": "auto",
        "lama_model": "models/lama_fp32.onnx",
        "fallback": "telea",
        "radius": 5,
        "flat_fill": True,
    },
    "translation": {
        "provider": "Google Translate",
        "use_fallback": True,
        "fallback_providers": ["MyMemory (free)"],
        "deepl_api_key": "",
        "retries": 2,
        "retry_backoff": 1.0,
        "min_interval": 0.5,
        "timeout": 20,
        "concise": True,
        "arabic_digits": False,
        "erase_credits": True,
        "on_failure": "raise",
    },
    "fonts": {
        "default_font": "fonts/default_arabic.ttf",
        "mappings": {"dialogue": "", "narration": "", "sfx": "", "whisper": ""},
    },
    "styles": {
        "dialogue": _role_style(),
        "narration": _role_style(max_font_ratio=1.2),
        "sfx": _role_style(
            stroke_enabled=True,
            stroke_width=3,
            max_font_ratio=3.0,
        ),
        "whisper": _role_style(max_font_ratio=1.2),
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Args:
        base: Dictionary providing default values.
        override: Dictionary whose values take precedence.

    Returns:
        A new merged dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a possibly project-relative path to an absolute one.

    Args:
        path: Absolute path or path relative to ``PROJECT_ROOT``.

    Returns:
        Absolute ``Path``.
    """
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


class Settings:
    """Nested settings store with dotted-key access and JSON persistence."""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        """Create settings, merging ``data`` over the defaults.

        Args:
            data: Optional partial settings dictionary.
        """
        data = dict(data or {})
        if data and int(data.get("version", 1)) < SETTINGS_VERSION:
            if int(data.get("version", 1)) < 2:  # v2: translated text takes the original colour
                for style in (data.get("styles") or {}).values():
                    if isinstance(style, dict):
                        style["match_source_color"] = True
            if int(data.get("version", 1)) < 4 and isinstance(data.get("ocr"), dict):  # v4: faster OCR defaults
                data["ocr"].pop("tile_height", None)
            tr = data.get("translation")  # v3: only Google / MyMemory / DeepL API remain
            if isinstance(tr, dict):
                if tr.get("provider") not in PROVIDER_NAMES:
                    tr["provider"] = "Google Translate"
                tr["fallback_providers"] = [p for p in tr.get("fallback_providers", []) if p in PROVIDER_NAMES] \
                    or ["MyMemory (free)"]
                tr["min_interval"] = max(float(tr.get("min_interval", 0.5)), 0.5)
            data["version"] = SETTINGS_VERSION
        self.data: Dict[str, Any] = _deep_merge(DEFAULT_SETTINGS, data)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Settings":
        """Load settings from disk, falling back to defaults on any error.

        Args:
            path: JSON file path. Defaults to ``config/settings.json``.

        Returns:
            Loaded ``Settings`` instance.
        """
        path = path or CONFIG_PATH
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return cls(json.load(fh))
        except (OSError, json.JSONDecodeError):
            return cls()

    def save(self, path: Optional[Path] = None) -> None:
        """Persist settings as pretty-printed UTF-8 JSON.

        Args:
            path: Destination file. Defaults to ``config/settings.json``.
        """
        path = path or CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Read a value using ``"section.key"`` notation.

        Args:
            dotted: Dotted key path.
            default: Value returned if the key is missing.

        Returns:
            The stored value or ``default``.
        """
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        """Write a value using ``"section.key"`` notation, creating sections.

        Args:
            dotted: Dotted key path.
            value: Value to store.
        """
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def role_style(self, role: str) -> Dict[str, Any]:
        """Return the merged style dictionary for a comic role.

        Args:
            role: One of ``ROLES``.

        Returns:
            Style dictionary (copy).
        """
        return _deep_merge(_ROLE_STYLE_BASE, self.data.get("styles", {}).get(role, {}))
