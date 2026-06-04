#!/usr/bin/env bash
# apply_update.sh — writes the updated/new files into the repo.
# Run this from INSIDE your repo:   bash apply_update.sh
set -e
echo ">> writing updated files..."
mkdir -p "src/khmertts"
cat > "src/khmertts/quality_gates.py" << 'KH_UPDATE_EOF_7f3a'
#!/usr/bin/env python3
"""
Apply explicit data-quality gates to an audited corpus and generate a readable
Dataset Audit Report (Markdown) with a PASS/FAIL verdict per gate.

Consumes the manifest + the audit.csv produced by `khmertts.audit_dataset`
(run `make audit` first). Each gate has a stated threshold so every keep/drop
decision is justified by a rule, not a guess.

Outputs:
    quality_gates.csv          per-clip pass/fail with the failing reasons
    dataset_audit_report.md    the report to show your supervisor

Run:
    python -m khmertts.quality_gates \
        --manifest metadata.csv \
        --audit    audit.csv \
        --dataset_name "kh-tts-dataset-master (single speaker)" \
        --report dataset_audit_report.md
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date
from pathlib import Path

# ---- Default gate thresholds (override on the command line) ----
GATES = {
    "sample_rate_allowed": (16000, 22050),
    "min_dur": 1.0,
    "max_dur": 14.0,
    "max_clip_ratio": 0.01,    # < 1% of samples at full scale
    "min_char_rate": 2.0,
    "max_char_rate": 30.0,
    "max_silent": 0,           # no fully-silent clips allowed
    "max_empty_transcript": 0,
}


def read_manifest(path: Path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line:
                name, text = line.split("|", 1)
                pairs.append((name, text))
    return pairs


def read_audit(path: Path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["file"]] = r
    return rows


def _f(row, key, default=0.0):
    try:
        return float(row.get(key) or default)
    except (ValueError, TypeError):
        return default


def evaluate(manifest: Path, audit: Path, gates: dict):
    pairs = read_manifest(manifest)
    audit_rows = read_audit(audit)

    per_clip = []          # (file, [reasons])
    reason_counts = Counter()
    sample_rates = Counter()
    durations = []
    hashes = Counter()
    seen_hashes = set()
    n_empty = 0

    for name, text in pairs:
        a = audit_rows.get(name)
        reasons = []
        if a is None:
            reasons.append("not_in_audit")
        else:
            sr = int(_f(a, "samplerate"))
            dur = _f(a, "duration")
            clip_ratio = _f(a, "clip_ratio")
            crate = _f(a, "chars_per_sec")
            silent = (a.get("silent") == "True")
            sample_rates[sr] += 1
            durations.append(dur)
            h = a.get("audio_sha1") or ""
            if h:
                hashes[h] += 1
                # flag every occurrence after the first as a duplicate
                if h in seen_hashes:
                    reasons.append("duplicate")
                else:
                    seen_hashes.add(h)

            if sr not in gates["sample_rate_allowed"]:
                reasons.append("bad_sample_rate")
            if dur and dur < gates["min_dur"]:
                reasons.append("too_short")
            if dur and dur > gates["max_dur"]:
                reasons.append("too_long")
            if clip_ratio > gates["max_clip_ratio"]:
                reasons.append("clipped")
            if silent:
                reasons.append("silent")
            if dur and not (gates["min_char_rate"] <= crate <= gates["max_char_rate"]):
                reasons.append("odd_char_rate")

        if not text.strip():
            reasons.append("empty_transcript")
            n_empty += 1

        for r in reasons:
            reason_counts[r] += 1
        per_clip.append((name, reasons))

    # duplicate clips = those flagged above (all but the first of each hash group)
    n_dup_clips = reason_counts.get("duplicate", 0)

    summary = {
        "n_clips": len(pairs),
        "sample_rates": dict(sample_rates),
        "durations": durations,
        "n_empty": n_empty,
        "n_dup_clips": n_dup_clips,
        "n_unique_audio": len(hashes),
        "reason_counts": reason_counts,
    }
    return per_clip, summary


def gate_table(summary: dict, gates: dict):
    """Return list of (gate, threshold, result, detail)."""
    rc = summary["reason_counts"]
    srs = summary["sample_rates"]
    durs = summary["durations"]
    n = summary["n_clips"]

    rows = []
    # sample rate
    ok_sr = all(sr in gates["sample_rate_allowed"] for sr in srs)
    rows.append(("Sample rate", f"in {gates['sample_rate_allowed']} Hz",
                 ok_sr, ", ".join(f"{k}Hz×{v}" for k, v in srs.items())))
    # duration band
    bad_dur = rc.get("too_short", 0) + rc.get("too_long", 0)
    rows.append(("Duration", f"{gates['min_dur']}–{gates['max_dur']} s",
                 bad_dur == 0, f"{bad_dur} out of band "
                 f"(short={rc.get('too_short',0)}, long={rc.get('too_long',0)})"))
    # clipping
    rows.append(("Clipping ratio", f"< {gates['max_clip_ratio']*100:.0f}% of samples",
                 rc.get("clipped", 0) == 0, f"{rc.get('clipped',0)} clips exceed"))
    # silence
    rows.append(("No silent clips", f"≤ {gates['max_silent']}",
                 rc.get("silent", 0) <= gates["max_silent"], f"{rc.get('silent',0)} silent"))
    # empty transcripts
    rows.append(("Empty transcripts", f"≤ {gates['max_empty_transcript']}",
                 summary["n_empty"] <= gates["max_empty_transcript"],
                 f"{summary['n_empty']} empty"))
    # char rate
    rows.append(("Character rate", f"{gates['min_char_rate']}–{gates['max_char_rate']} chars/s",
                 rc.get("odd_char_rate", 0) == 0, f"{rc.get('odd_char_rate',0)} out of band"))
    # duplicates
    rows.append(("Duplicate audio", "0 duplicates",
                 summary["n_dup_clips"] == 0, f"{summary['n_dup_clips']} clips share audio"))
    return rows


def render_report(dataset_name: str, summary: dict, gates: dict, per_clip: list) -> str:
    durs = summary["durations"] or [0.0]
    total_h = sum(durs) / 3600
    n = summary["n_clips"]
    n_fail = sum(1 for _, r in per_clip if r)
    n_pass = n - n_fail
    table = gate_table(summary, gates)
    overall = all(ok for _, _, ok, _ in table)

    def mark(ok):
        return "✅ PASS" if ok else "⚠️ FAIL"

    lines = []
    lines.append(f"# Dataset Audit Report")
    lines.append("")
    lines.append(f"**Dataset:** {dataset_name}  ")
    lines.append(f"**Date:** {date.today().isoformat()}  ")
    lines.append(f"**Overall gate verdict:** {mark(overall)}")
    lines.append("")
    lines.append("## 1. Summary")
    lines.append("")
    lines.append(f"- Clips: **{n}**")
    lines.append(f"- Total audio: **{total_h:.2f} hours** ({sum(durs)/60:.1f} min)")
    lines.append(f"- Duration min / mean / max: "
                 f"{min(durs):.2f} / {sum(durs)/len(durs):.2f} / {max(durs):.2f} s")
    lines.append(f"- Sample rates: {', '.join(f'{k} Hz × {v}' for k,v in summary['sample_rates'].items())}")
    lines.append(f"- Unique audio fingerprints: {summary['n_unique_audio']} "
                 f"(duplicates: {summary['n_dup_clips']})")
    lines.append(f"- Clips passing all gates: **{n_pass}** / {n}  "
                 f"(failing: {n_fail})")
    lines.append("")
    lines.append("## 2. Quality gates")
    lines.append("")
    lines.append("| Gate | Threshold | Result | Detail |")
    lines.append("|------|-----------|--------|--------|")
    for gate, thr, ok, detail in table:
        lines.append(f"| {gate} | {thr} | {mark(ok)} | {detail} |")
    lines.append("")
    lines.append("## 3. Failure breakdown")
    lines.append("")
    if summary["reason_counts"]:
        lines.append("| Reason | Clips |")
        lines.append("|--------|-------|")
        for r, c in summary["reason_counts"].most_common():
            lines.append(f"| {r} | {c} |")
    else:
        lines.append("No clips failed any gate.")
    lines.append("")
    lines.append("## 4. Decision")
    lines.append("")
    lines.append(f"- **{n_pass}** clips pass all gates and form the *Gold Standard* "
                 f"training set.")
    lines.append(f"- **{n_fail}** clips are excluded; reasons listed above and "
                 f"per-clip in `quality_gates.csv`.")
    lines.append(f"- The clean set is produced by `make gold` "
                 f"(`metadata_gold.csv`) and built into a training-ready dataset "
                 f"with `make dataset-gold`.")
    lines.append("")
    lines.append("## 5. Reproducibility")
    lines.append("")
    lines.append("```bash")
    lines.append("make manifest   # pair audio + transcripts")
    lines.append("make audit      # compute per-clip metrics -> audit.csv")
    lines.append("make report     # apply gates -> this report + quality_gates.csv")
    lines.append("make gold       # filter to the passing set -> metadata_gold.csv")
    lines.append("```")
    lines.append("")
    lines.append("_Generated by `khmertts.quality_gates`. Every figure above is "
                 "computed from the dataset; none are estimated._")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="metadata.csv")
    ap.add_argument("--audit", default="audit.csv")
    ap.add_argument("--report", default="dataset_audit_report.md")
    ap.add_argument("--gates_out", default="quality_gates.csv")
    ap.add_argument("--dataset_name", default="Khmer TTS dataset")
    ap.add_argument("--min_dur", type=float, default=GATES["min_dur"])
    ap.add_argument("--max_dur", type=float, default=GATES["max_dur"])
    ap.add_argument("--max_clip_ratio", type=float, default=GATES["max_clip_ratio"])
    args = ap.parse_args()

    gates = dict(GATES)
    gates["min_dur"] = args.min_dur
    gates["max_dur"] = args.max_dur
    gates["max_clip_ratio"] = args.max_clip_ratio

    per_clip, summary = evaluate(Path(args.manifest), Path(args.audit), gates)

    with open(args.gates_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["file", "verdict", "reasons"])
        for name, reasons in per_clip:
            w.writerow([name, "FAIL" if reasons else "PASS", ";".join(reasons)])

    report = render_report(args.dataset_name, summary, gates, per_clip)
    Path(args.report).write_text(report, encoding="utf-8")

    n_fail = sum(1 for _, r in per_clip if r)
    print(f"clips        : {summary['n_clips']}")
    print(f"passed gates : {summary['n_clips'] - n_fail}")
    print(f"failed gates : {n_fail}")
    print(f"report       : {args.report}")
    print(f"per-clip      : {args.gates_out}")


if __name__ == "__main__":
    main()
KH_UPDATE_EOF_7f3a
echo "   wrote src/khmertts/quality_gates.py"
mkdir -p "tests"
cat > "tests/test_quality_gates.py" << 'KH_UPDATE_EOF_7f3a'
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
KH_UPDATE_EOF_7f3a
echo "   wrote tests/test_quality_gates.py"
mkdir -p "src/khmertts"
cat > "src/khmertts/audit_dataset.py" << 'KH_UPDATE_EOF_7f3a'
#!/usr/bin/env python3
"""
Quality map of the corpus BEFORE training. Catches the issues that quietly
wreck TTS training: sample-rate/channel mismatch, silence, clipping, bad
durations, and non-Khmer characters that need text normalization.

Outputs:
    audit.csv    one row per clip, all metrics
    flagged.csv  only the clips worth a human look

Run:
    python -m khmertts.audit_dataset \
        --manifest metadata.csv \
        --audio    data/kh-tts-dataset-master/wav16
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from .text_utils import non_khmer_chars


def read_manifest(path: Path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            name, text = line.split("|", 1)
            pairs.append((name, text))
    return pairs


def audit(manifest: Path, audio_dir: Path, audit_out: Path, flag_out: Path,
          min_dur: float, max_dur: float):
    pairs = read_manifest(manifest)
    rows, flagged, durations = [], [], []
    srs, chans, char_counter = Counter(), Counter(), Counter()

    for name, text in pairs:
        path = audio_dir / name
        rec = {"file": name, "text_len": len(text), "n_words": len(text.split())}
        try:
            info = sf.info(path)
            sr, ch, dur = info.samplerate, info.channels, info.duration
            rec.update(samplerate=sr, channels=ch, duration=round(dur, 3))
            srs[sr] += 1
            chans[ch] += 1
            durations.append(dur)

            audio, _ = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            rec["peak"] = round(peak, 4)
            rec["silent"] = peak < 1e-3
            rec["clipped"] = peak >= 0.999
            # fraction of samples at/near full scale (a clip with a few clipped
            # samples is fine; one that is mostly clipped is distorted)
            if audio.size:
                rec["clip_ratio"] = round(float(np.mean(np.abs(audio) >= 0.99)), 5)
            else:
                rec["clip_ratio"] = 0.0
            # content hash for duplicate detection (hash the raw samples, not the
            # file, so two files with identical audio collide regardless of header)
            rec["audio_sha1"] = hashlib.sha1(
                np.ascontiguousarray(audio)).hexdigest()[:16] if audio.size else ""
            rec["chars_per_sec"] = round(len(text) / dur, 1) if dur > 0 else 0

            reasons = []
            if dur < min_dur:
                reasons.append("too_short")
            if dur > max_dur:
                reasons.append("too_long")
            if rec["silent"]:
                reasons.append("silent")
            if rec["clipped"]:
                reasons.append("clipped")
            if dur > 0 and (rec["chars_per_sec"] > 35 or rec["chars_per_sec"] < 3):
                reasons.append("odd_rate")
            rec["flags"] = ";".join(reasons)
            if reasons:
                flagged.append(rec)
        except Exception as e:  # unreadable / corrupt file
            rec["flags"] = f"unreadable:{e}"
            flagged.append(rec)

        char_counter.update(text)
        rows.append(rec)

    fields = ["file", "duration", "samplerate", "channels", "peak", "clip_ratio",
              "audio_sha1", "silent", "clipped", "text_len", "n_words",
              "chars_per_sec", "flags"]
    for out, data in ((audit_out, rows), (flag_out, flagged)):
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(data)

    # aggregate non-Khmer characters across the whole corpus
    suspicious: Counter = Counter()
    for _, text in pairs:
        suspicious.update(non_khmer_chars(text))

    total_sec = sum(durations)
    dn = np.array(durations) if durations else np.array([0.0])
    print("=" * 60)
    print(f"clips                 : {len(rows)}")
    print(f"total audio           : {total_sec/3600:.2f} hours ({total_sec/60:.1f} min)")
    print(f"duration min/mean/max : {dn.min():.2f} / {dn.mean():.2f} / {dn.max():.2f} s")
    print(f"sample rates          : {dict(srs)}")
    print(f"channels              : {dict(chans)}")
    print(f"distinct characters   : {len(char_counter)}")
    print(f"flagged clips         : {len(flagged)}  -> {flag_out}")
    print(f"per-clip audit        : {audit_out}")
    print("=" * 60)
    if suspicious:
        print("Non-Khmer characters to normalize (digits, latin, symbols):")
        for ch, n in sorted(suspicious.items(), key=lambda kv: -kv[1]):
            print(f"  U+{ord(ch):04X} {ch!r}  count={n}")
    else:
        print("No non-Khmer characters found — transcripts look clean.")
    return rows, flagged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="metadata.csv")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--audit_out", default="audit.csv")
    ap.add_argument("--flag_out", default="flagged.csv")
    ap.add_argument("--min_dur", type=float, default=0.5)
    ap.add_argument("--max_dur", type=float, default=15.0)
    args = ap.parse_args()
    audit(Path(args.manifest), Path(args.audio), Path(args.audit_out),
          Path(args.flag_out), args.min_dur, args.max_dur)


if __name__ == "__main__":
    main()
KH_UPDATE_EOF_7f3a
echo "   wrote src/khmertts/audit_dataset.py"
cat > "Makefile" << 'KH_UPDATE_EOF_7f3a'
# Reproducible pipeline. Override paths on the command line, e.g.:
#   make manifest AUDIO=data/kh-tts-dataset-master/wav16 TEXT=data/kh-tts-dataset-master/text

PYTHON ?= python
AUDIO  ?= data/kh-tts-dataset-master/wav16
TEXT   ?= data/kh-tts-dataset-master/text
MANIFEST ?= metadata.csv
GOLD     ?= metadata_gold.csv
HF_DIR ?= hf_dataset
MAX_DUR ?= 14.0
DATASET_NAME ?= kh-tts-dataset-master (single speaker)

.PHONY: help setup manifest audit gold report dataset dataset-gold baseline test pipeline clean

help:
	@echo "Targets:"
	@echo "  setup        - create venv + install deps (no sudo)"
	@echo "  manifest     - pair wavs + transcripts -> $(MANIFEST)"
	@echo "  audit        - quality report (audit.csv, flagged.csv)"
	@echo "  report       - apply quality gates -> dataset_audit_report.md"
	@echo "  gold         - filter to a clean 'Gold Standard' set -> $(GOLD)"
	@echo "  dataset      - HuggingFace dataset from $(MANIFEST) (raw)"
	@echo "  dataset-gold - HuggingFace dataset from $(GOLD) (cleaned)"
	@echo "  baseline     - synthesize with stock mms-tts-khm"
	@echo "  test         - run unit tests"
	@echo "  pipeline     - manifest + audit + report + gold"
	@echo "  clean        - remove generated artifacts"

setup:
	bash scripts/setup_env.sh

manifest:
	$(PYTHON) -m khmertts.build_manifest --audio $(AUDIO) --text $(TEXT) --out $(MANIFEST)

audit:
	$(PYTHON) -m khmertts.audit_dataset --manifest $(MANIFEST) --audio $(AUDIO)

gold:
	$(PYTHON) -m khmertts.clean_manifest --manifest $(MANIFEST) --audit audit.csv --out $(GOLD) --max_dur $(MAX_DUR)

report:
	$(PYTHON) -m khmertts.quality_gates --manifest $(MANIFEST) --audit audit.csv --dataset_name "$(DATASET_NAME)" --max_dur $(MAX_DUR)

dataset:
	$(PYTHON) -m khmertts.make_hf_dataset --manifest $(MANIFEST) --audio $(AUDIO) --out_dir $(HF_DIR)

dataset-gold:
	$(PYTHON) -m khmertts.make_hf_dataset --manifest $(GOLD) --audio $(AUDIO) --out_dir hf_dataset_gold

baseline:
	$(PYTHON) -m khmertts.baseline_synthesize --out_dir baseline_samples

test:
	pytest -q

pipeline: manifest audit report gold

clean:
	rm -rf metadata.csv metadata_gold.csv dropped.csv audit.csv flagged.csv quality_gates.csv dataset_audit_report.md $(HF_DIR) hf_dataset_gold baseline_samples __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
KH_UPDATE_EOF_7f3a
echo "   wrote Makefile"
cat > "README.md" << 'KH_UPDATE_EOF_7f3a'
# Khmer TTS — Text Analysis & Data Pipeline

A data pipeline and fine-tuning workflow for building a **production-quality
Khmer text-to-speech (TTS)** voice. The project's thesis: *cleaner data and
proper Khmer text normalization produce more intelligible, less robotic speech*
— and it sets out to measure that, not just assert it.

> Research internship project · Institute of Digital Research & Innovation (IDRI),
> Cambodia Academy of Digital Technology (CADT).

---

## Why this exists

Neural TTS is only as good as its training data. Khmer corpora carry noise,
audio/transcript misalignments, and inconsistent Unicode — which surface as
glitchy or robotic speech, especially in long-form content like audiobooks.
This repo builds the pipeline to clean and normalize the data, fine-tunes a
Khmer TTS model on it, and benchmarks the result.

## What's here

```
khmer-tts-pipeline/
├── src/khmertts/             # the pipeline package
│   ├── text_utils.py         #   shared Khmer text helpers (NFC, char checks)
│   ├── build_manifest.py     #   pair wavs + transcripts -> metadata.csv
│   ├── audit_dataset.py      #   pre-training quality report
│   ├── quality_gates.py      #   PASS/FAIL gates + Dataset Audit Report (markdown)
│   ├── clean_manifest.py     #   filter to a clean "Gold Standard" set
│   ├── make_hf_dataset.py    #   build a HuggingFace dataset (train/eval)
│   └── baseline_synthesize.py#   zero-shot reference with stock mms-tts-khm
├── scripts/setup_env.sh      # one-time, no-sudo environment setup
├── tests/                    # unit tests (pytest)
├── docs/implementation_plan.md  # full 3-month roadmap, research & decisions
├── data/                     # dataset goes here (not committed)
├── Makefile                  # reproducible commands
├── requirements.txt
└── pyproject.toml
```

## Quickstart

```bash
# 1. environment (creates .venv, installs everything, no sudo)
bash scripts/setup_env.sh
source .venv/bin/activate

# 2. put the dataset under data/ (see data/README.md for the layout)

# 3. run the prep pipeline
make manifest      # data/.../wav16 + data/.../text  ->  metadata.csv
make audit         # quality report: audit.csv + flagged.csv
make report        # PASS/FAIL quality gates -> dataset_audit_report.md
make gold          # filter long/bad clips -> metadata_gold.csv + dropped.csv
make dataset-gold  # HuggingFace dataset from the clean set -> hf_dataset_gold/
make baseline      # zero-shot samples with stock mms-tts-khm

# run tests
make test
```

Each step is also runnable directly, e.g.:

```bash
python -m khmertts.build_manifest \
    --audio data/kh-tts-dataset-master/wav16 \
    --text  data/kh-tts-dataset-master/text \
    --out   metadata.csv
```

## Approach (short version)

1. **Fine-tune, don't train from scratch.** Start from Meta's Khmer VITS
   checkpoint (`facebook/mms-tts-khm`) using the `ylacombe/finetune-hf-vits`
   recipe — far more reliable than training VITS2 from zero on ~4 hours of data.
2. **Prove the thesis with a controlled A/B.** Fine-tune the *same* checkpoint
   twice — on raw data vs. the cleaned "Gold Standard" set — and compare.
3. **Stand on the existing Khmer NLP stack** (`tha`, `khmernumber`, `khmercut`,
   Montreal Forced Aligner) rather than reinventing it.

Evaluation: **CER/WER** (intelligibility, via a Khmer ASR judge), **MCD**
(acoustic distance vs. ground truth), and a small **MOS** listening test.

See [`docs/implementation_plan.md`](docs/implementation_plan.md) for the full
month-by-month plan, the research behind these decisions, and the risk register.

## Hardware notes

- Built to run on a single ~16 GB GPU (e.g. RTX A4000). Prep steps are CPU-light;
  only fine-tuning needs significant VRAM.
- On a **shared** GPU, check free memory before launching a run
  (`nvidia-smi --query-gpu=memory.free --format=csv`) and run training inside
  `tmux` so it survives an SSH drop.

## Licensing

Code: MIT (see [`LICENSE`](LICENSE)). **The dataset and models are not.** The
`mms-tts-khm` checkpoint is **CC-BY-NC** (non-commercial); the training data is
supervisor/IDRI-owned. Confirm terms before any production or public release.

## Status

Data pipeline + baseline: ✅ implemented and tested.
Cleaning pipeline, fine-tuning, and evaluation: see the roadmap.
KH_UPDATE_EOF_7f3a
echo "   wrote README.md"
cat > ".gitignore" << 'KH_UPDATE_EOF_7f3a'
# ---- Python ----
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
build/
dist/
.pytest_cache/
.ipynb_checkpoints/

# ---- Virtual envs ----
.venv/
venv/
env/

# ---- Data & models: NEVER commit (large / licensed / supervisor-owned) ----
data/**
!data/README.md
hf_dataset/
hf_dataset*/
*.arrow
*.wav
*.flac
*.mp3
*.pth
*.safetensors
*.ckpt
checkpoints/
outputs/
runs/
wandb/

# ---- Generated artifacts ----
metadata.csv
metadata_gold.csv
dropped.csv
audit.csv
flagged.csv
quality_gates.csv
dataset_audit_report.md
baseline_samples/

# ---- OS / editor ----
.DS_Store
.idea/
.vscode/
*.swp
KH_UPDATE_EOF_7f3a
echo "   wrote .gitignore"
mkdir -p "data"
cat > "data/README.md" << 'KH_UPDATE_EOF_7f3a'
# Data (not tracked in git)

Audio and transcripts are **not** committed — they are large and the dataset is
supervisor-owned. The `.gitignore` keeps everything here out of git except this
file. Place the dataset here so the default paths work:

```
data/
  kh-tts-dataset-master/
    wav16/   kh_atr_m001_a0001.wav ...   # audio, 16 kHz mono
    text/    kh_atr_m001_a0001.txt ...   # one transcript per clip, same basename
```

Corpus summary (single speaker, from `make audit` on the real data):

| Field            | Value            |
|------------------|------------------|
| Speaker          | kh_atr_m001      |
| Clips            | 4000             |
| Total duration   | 3.98 hours       |
| Sample rate      | 16 kHz mono      |
| Language         | Khmer (khm)      |
| Clips dropped    | 18 (too long)    |

> Confirm usage rights with the supervisor before publishing any model trained
> on this data (see licensing note in the main README).
KH_UPDATE_EOF_7f3a
echo "   wrote data/README.md"
echo ">> done. Now run:  python -m pytest -q"
