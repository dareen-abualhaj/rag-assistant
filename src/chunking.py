"""Recursive text chunking: split on the largest natural boundary first."""
import re
from typing import List

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_keep_sep(text: str, sep: str) -> List[str]:
    """Split on `sep` but keep the separator attached to the preceding part."""
    parts = re.split(f"({re.escape(sep)})", text)
    merged = [parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
              for i in range(0, len(parts), 2)]
    return [m for m in merged if m]


def _split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    if not separators:  # no separator left: hard cut
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = separators[0], separators[1:]
    pieces, current = [], ""
    for part in _split_keep_sep(text, sep):
        if len(current) + len(part) <= chunk_size:
            current += part
        else:
            if current:
                pieces.append(current)
                current = ""
            if len(part) > chunk_size:  # part itself too big: recurse deeper
                pieces.extend(_split(part, chunk_size, rest))
            else:
                current = part
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into chunks of at most `chunk_size` chars with `overlap` chars shared."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    text = text.strip()
    if not text:
        return []

    # leave room for the overlap prefix so final chunks stay <= chunk_size
    pieces = _split(text, chunk_size - overlap, SEPARATORS)
    if overlap == 0:
        return [p.strip() for p in pieces if p.strip()]

    chunks = []
    for i, piece in enumerate(pieces):
        prefix = pieces[i - 1][-overlap:] if i > 0 else ""
        chunk = (prefix + " " + piece).strip() if prefix else piece.strip()
        if chunk:
            chunks.append(chunk)
    return chunks