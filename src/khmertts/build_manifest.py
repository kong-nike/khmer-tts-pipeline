#!/usr/bin/env python3
"""
Pair each .wav with its .txt transcript and write an LJSpeech-style manifest:

    filename|transcript

Run:
    python -m khmertts.build_manifest \
        --audio data/kh-tts-dataset-master/wav16 \
        --text  data/kh-tts-dataset-master/text \
        --out   metadata.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .text_utils import normalize_unicode


def build_manifest(audio_dir: Path, text_dir: Path, out_path: Path) -> dict:
    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(
            f"No .wav files in {audio_dir} — check the paths are not swapped "
            f"(audio=wav16, text=text)."
        )

    rows, missing, empty = [], 0, 0
    for wav in wavs:
        txt = text_dir / f"{wav.stem}.txt"
        if not txt.exists():
            print(f"⚠️  no transcript for {wav.name}")
            missing += 1
            continue
        text = normalize_unicode(txt.read_text(encoding="utf-8"))
        if not text:
            print(f"⚠️  empty transcript: {txt.name}")
            empty += 1
            continue
        rows.append((wav.name, text))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", lineterminator="\n")
        w.writerows(rows)

    return {"wavs": len(wavs), "pairs": len(rows), "missing": missing, "empty": empty}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="folder of .wav files (wav16)")
    ap.add_argument("--text", required=True, help="folder of .txt transcripts (text)")
    ap.add_argument("--out", default="metadata.csv")
    args = ap.parse_args()

    stats = build_manifest(Path(args.audio), Path(args.text), Path(args.out))
    print("-" * 50)
    print(f"wav files found     : {stats['wavs']}")
    print(f"pairs written       : {stats['pairs']}  ->  {args.out}")
    print(f"missing transcripts : {stats['missing']}")
    print(f"empty transcripts   : {stats['empty']}")


if __name__ == "__main__":
    main()
