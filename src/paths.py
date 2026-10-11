"""Filesystem locations used by the application.

The application "home" is the repository root when running from a checkout
(or an editable install). For regular installs the home falls back to
``~/.comic-translator``. Set ``COMIC_TRANSLATOR_HOME`` to override both.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_home() -> Path:
    """Work out the application home directory.

    Returns:
        The directory that holds ``fonts/``, ``models/``, ``config/`` and ``assets/``.
    """
    env = os.environ.get("COMIC_TRANSLATOR_HOME")
    if env:
        return Path(env).expanduser().resolve()
    root = Path(__file__).resolve().parents[1]
    if (root / "config").exists() or (root / "fonts").exists():
        return root
    return Path.home() / ".comic-translator"


APP_HOME: Path = _resolve_home()
FONTS_DIR: Path = APP_HOME / "fonts"
MODELS_DIR: Path = APP_HOME / "models"
ASSETS_DIR: Path = APP_HOME / "assets"
CONFIG_DIR: Path = APP_HOME / "config"
SETTINGS_FILE: Path = CONFIG_DIR / "settings.json"
DEFAULT_FONT_NAME = "default_arabic.ttf"


def ensure_dirs() -> None:
    """Create all application directories if they do not exist yet."""
    for directory in (FONTS_DIR, MODELS_DIR, ASSETS_DIR, CONFIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_model_path(value: str) -> Path:
    """Resolve a (possibly relative) model path against the app home.

    Args:
        value: Absolute path, or a path relative to ``APP_HOME``.

    Returns:
        An absolute :class:`~pathlib.Path` (the file may not exist).
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else APP_HOME / path
