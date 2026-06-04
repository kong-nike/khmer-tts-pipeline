"""Tests for khmertts.quality_gates.

Run:  pytest -q
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from khmertts.quality_gates import GATES, evaluate, render_report  # noqa: E402


def _write_manifest(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="|", lineterminator="\n").writerows(rows)


def _write_audit(path, recs):
    fields = ["file", "duration", "samplerate", "clip_ratio", "audio_sha1",
              "silent", "chars_per_sec"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(recs)


def test_clean_clip_passes(tmp_path):
    man, aud = tmp_path / "m.csv", tmp_path / "a.csv"
    _write_manifest(man, [("a.wav", "good sentence here")])
    _write_audit(aud, [{"file": "a.wav", "duration": "3.5", "samplerate": "16000",
                        "clip_ratio": "0.0", "audio_sha1": "aaa", "silent": "False",
                        "chars_per_sec": "12"}])
    per_clip, summary = evaluate(man, aud, GATES)
    assert per_clip == [("a.wav", [])]
    assert summary["n_clips"] == 1


def test_flags_long_clipped_and_bad_rate(tmp_path):
    man, aud = tmp_path / "m.csv", tmp_path / "a.csv"
    _write_manifest(man, [("long.wav", "x"), ("clip.wav", "y"), ("fast.wav", "z")])
    _write_audit(aud, [
        {"file": "long.wav", "duration": "20", "samplerate": "16000",
         "clip_ratio": "0.0", "audio_sha1": "h1", "silent": "False", "chars_per_sec": "12"},
        {"file": "clip.wav", "duration": "3", "samplerate": "16000",
         "clip_ratio": "0.2", "audio_sha1": "h2", "silent": "False", "chars_per_sec": "12"},
        {"file": "fast.wav", "duration": "3", "samplerate": "16000",
         "clip_ratio": "0.0", "audio_sha1": "h3", "silent": "False", "chars_per_sec": "99"},
    ])
    per_clip, summary = evaluate(man, aud, GATES)
    reasons = {name: r for name, r in per_clip}
    assert "too_long" in reasons["long.wav"]
    assert "clipped" in reasons["clip.wav"]
    assert "odd_char_rate" in reasons["fast.wav"]


def test_duplicate_audio_flags_all_but_first(tmp_path):
    man, aud = tmp_path / "m.csv", tmp_path / "a.csv"
    _write_manifest(man, [("a.wav", "one"), ("b.wav", "two"), ("c.wav", "three")])
    same = {"duration": "3", "samplerate": "16000", "clip_ratio": "0.0",
            "silent": "False", "chars_per_sec": "12"}
    _write_audit(aud, [
        {"file": "a.wav", "audio_sha1": "DUP", **same},
        {"file": "b.wav", "audio_sha1": "DUP", **same},
        {"file": "c.wav", "audio_sha1": "uniq", **same},
    ])
    per_clip, summary = evaluate(man, aud, GATES)
    reasons = {name: r for name, r in per_clip}
    assert reasons["a.wav"] == []                 # first occurrence kept
    assert "duplicate" in reasons["b.wav"]        # second flagged
    assert reasons["c.wav"] == []
    assert summary["n_dup_clips"] == 1


def test_empty_transcript_flagged(tmp_path):
    man, aud = tmp_path / "m.csv", tmp_path / "a.csv"
    _write_manifest(man, [("a.wav", "   ")])
    _write_audit(aud, [{"file": "a.wav", "duration": "3", "samplerate": "16000",
                        "clip_ratio": "0.0", "audio_sha1": "h", "silent": "False",
                        "chars_per_sec": "12"}])
    per_clip, summary = evaluate(man, aud, GATES)
    assert "empty_transcript" in per_clip[0][1]


def test_render_report_is_markdown(tmp_path):
    man, aud = tmp_path / "m.csv", tmp_path / "a.csv"
    _write_manifest(man, [("a.wav", "good one")])
    _write_audit(aud, [{"file": "a.wav", "duration": "3.5", "samplerate": "16000",
                        "clip_ratio": "0.0", "audio_sha1": "h", "silent": "False",
                        "chars_per_sec": "12"}])
    per_clip, summary = evaluate(man, aud, GATES)
    md = render_report("demo", summary, GATES, per_clip)
    assert "# Dataset Audit Report" in md
    assert "## 2. Quality gates" in md
    assert "PASS" in md
