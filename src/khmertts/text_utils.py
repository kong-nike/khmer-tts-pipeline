"""
khmertts.text_utils
-------------------
Small, dependency-free text helpers shared across the pipeline.

Kept minimal on purpose: heavyweight Khmer normalization (number/currency
verbalization, segmentation) is delegated to the dedicated libraries
(`tha`, `khmernumber`, `khmercut`) in the normalization stage. These helpers
cover the universal cleanup that every stage needs.
"""
from __future__ import annotations

import unicodedata

# Khmer Unicode blocks
KHMER_MAIN = range(0x1780, 0x1800)      # Khmer
KHMER_SYMBOLS = range(0x19E0, 0x1A00)   # Khmer Symbols
# Common Khmer punctuation that is legitimate in transcripts
KHMER_PUNCT = set("។៕៖ៗ៘៙៚")


def normalize_unicode(text: str) -> str:
    """NFC-normalize and collapse whitespace.

    NFC collapses byte-different-but-visually-identical Khmer sequences into a
    single canonical form, so the model does not have to learn several
    encodings of the same word.
    """
    text = unicodedata.normalize("NFC", text)
    # remove zero-width space (U+200B), often used inconsistently in Khmer
    text = text.replace("\u200b", " ")
    return " ".join(text.split())


def is_khmer_char(ch: str) -> bool:
    """True for Khmer letters/signs/symbols (not spaces or punctuation)."""
    cp = ord(ch)
    return cp in KHMER_MAIN or cp in KHMER_SYMBOLS


def non_khmer_chars(text: str) -> dict[str, int]:
    """Return a count of characters that are neither Khmer, whitespace, nor
    standard Khmer punctuation. These are normalization targets (digits,
    Latin letters, foreign symbols)."""
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isspace() or ch in KHMER_PUNCT or is_khmer_char(ch):
            continue
        counts[ch] = counts.get(ch, 0) + 1
    return counts
