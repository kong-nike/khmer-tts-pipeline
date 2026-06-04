"""Tests for khmertts.text_utils.

Run:  pytest -q
"""
import sys
from pathlib import Path

# allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from khmertts.text_utils import non_khmer_chars, normalize_unicode  # noqa: E402


def test_normalize_collapses_whitespace():
    assert normalize_unicode("ខ្ញុំ   ទៅ\n\nផ្ទះ") == "ខ្ញុំ ទៅ ផ្ទះ"


def test_normalize_strips_zero_width_space():
    assert "\u200b" not in normalize_unicode("ខ្ញុំ\u200bទៅ")


def test_normalize_is_idempotent():
    s = "ខ្ញុំ ទៅ ផ្ទះ ។"
    assert normalize_unicode(normalize_unicode(s)) == normalize_unicode(s)


def test_non_khmer_chars_flags_digits_and_latin():
    found = non_khmer_chars("តម្លៃ គឺ 5000 រៀល ABC ។")
    assert set("5000ABC") <= set(found)          # digits + latin detected
    assert "។" not in found                       # khmer punctuation ignored
    assert "ត" not in found                        # khmer letters ignored


def test_non_khmer_chars_clean_text_is_empty():
    assert non_khmer_chars("ខ្ញុំ ទៅ ផ្ទះ ។") == {}
