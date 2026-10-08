"""Modular text cleaning functions for the Wikipedia corpus."""
import html
import re
import unicodedata

from bs4 import BeautifulSoup
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # make language detection deterministic


def strip_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return html.unescape(text)


def normalize_unicode(text: str) -> str:
    """Normalize to NFKC and drop control / zero-width characters."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )


def clean_whitespace(text: str) -> str:
    """Collapse runs of whitespace into a single space and trim."""
    return re.sub(r"\s+", " ", text).strip()


def is_english(text: str, min_chars: int = 20) -> bool:
    """Return True if the text is detected as English."""
    if len(text) < min_chars:
        return False
    try:
        return detect(text[:2000]) == "en"
    except LangDetectException:
        return False


def clean_text(text: str) -> str:
    """Full cleaning pipeline (without language filtering)."""
    if not isinstance(text, str):
        return ""
    text = strip_html(text)
    text = normalize_unicode(text)
    return clean_whitespace(text)
