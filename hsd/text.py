"""Text helpers: language detection, cleaning and word splitting."""

import re
import unicodedata

from . import config

_BENGALI = re.compile(r"[ঀ-৿]")
_SPACES = re.compile(r"\s+")


def is_bangla(text: str) -> bool:
    """True when a meaningful share of the letters are Bengali script."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    bn = sum(1 for c in letters if _BENGALI.match(c))
    return bn / len(letters) >= 0.3


def clean(text: str) -> str:
    """Normalise unicode, collapse whitespace and enforce the length limit."""
    text = unicodedata.normalize("NFC", text or "")
    text = _SPACES.sub(" ", text).strip()
    return text[: config.MAX_CHARS]


def was_truncated(raw: str) -> bool:
    return len(_SPACES.sub(" ", (raw or "")).strip()) > config.MAX_CHARS


def words(text: str) -> list[str]:
    return text.split(" ") if text else []
