#!/usr/bin/env python3
"""
Turn metadata.csv + wavs into a HuggingFace dataset (audio + text) split into
train/eval and saved to disk, ready for the fine-tuning step.

Run:
    python -m khmertts.make_hf_dataset \
        --manifest metadata.csv \
        --audio    data/kh-tts-dataset-master/wav16 \
        --out_dir  hf_dataset \
        --eval_frac 0.02
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Audio, Dataset


def make_dataset(manifest: Path, audio_dir: Path, out_dir: Path,
                 sr: int, eval_frac: float, seed: int):
    files, texts = [], []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            name, text = line.split("|", 1)
            p = audio_dir / name
            if p.exists():
                files.append(str(p))
                texts.append(text)

    ds = Dataset.from_dict({"audio": files, "text": texts})
    ds = ds.cast_column("audio", Audio(sampling_rate=sr))
    split = ds.train_test_split(test_size=eval_frac, seed=seed)
    split.save_to_disk(str(out_dir))
    return len(split["train"]), len(split["test"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="metadata.csv")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_dir", default="hf_dataset")
    ap.add_argument("--sr", type=int, default=16000,
                    help="target sample rate (mms-tts-khm = 16000)")
    ap.add_argument("--eval_frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n_train, n_eval = make_dataset(Path(args.manifest), Path(args.audio),
                                   Path(args.out_dir), args.sr, args.eval_frac, args.seed)
    print(f"train: {n_train}  eval: {n_eval}")
    print(f"saved -> {args.out_dir}")
    print("load later: from datasets import load_from_disk; load_from_disk('hf_dataset')")


if __name__ == "__main__":
    main()
