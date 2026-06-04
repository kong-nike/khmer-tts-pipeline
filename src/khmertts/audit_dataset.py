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

    fields = ["file", "duration", "samplerate", "channels", "peak",
              "silent", "clipped", "text_len", "n_words", "chars_per_sec", "flags"]
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
