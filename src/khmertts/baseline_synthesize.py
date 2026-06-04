#!/usr/bin/env python3
"""
Run the STOCK facebook/mms-tts-khm model on a few sentences. Keep these WAVs as
your 'before any training' reference, to compare against the fine-tuned model.

Runs on CPU if no GPU memory is free (slow but fine for a handful of clips).

Run:
    python -m khmertts.baseline_synthesize --out_dir baseline_samples
    python -m khmertts.baseline_synthesize --from_manifest metadata.csv --n 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SENTENCES = [
    "ប្រាសាទ អង្គរវត្ត របស់ កម្ពុជា គឺជា បេតិកភណ្ឌ ពិភពលោក ។",
    "ខ្ញុំ នឹង ជិះ តាក់ស៊ី ទៅ សណ្ឋាគារ ។",
    "សួស្ដី ! តើ អ្នក សុខសប្បាយ ជា ទេ ?",
]


def load_sentences(from_manifest: str | None, n: int):
    if not from_manifest:
        return DEFAULT_SENTENCES
    sents = []
    with open(from_manifest, encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                sents.append(line.rstrip("\r\n").split("|", 1)[1])
            if len(sents) >= n:
                break
    return sents


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/mms-tts-khm")
    ap.add_argument("--out_dir", default="baseline_samples")
    ap.add_argument("--from_manifest", default=None)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    import torch
    import scipy.io.wavfile
    from transformers import AutoTokenizer, VitsModel

    sentences = load_sentences(args.from_manifest, args.n)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  | model: {args.model}")

    model = VitsModel.from_pretrained(args.model).to(device)
    tok = AutoTokenizer.from_pretrained(args.model)
    sr = model.config.sampling_rate
    print(f"model sampling rate: {sr}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(sentences, 1):
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            wav = model(**inputs).waveform[0].cpu().numpy()
        path = out / f"baseline_{i:02d}.wav"
        scipy.io.wavfile.write(path, rate=sr, data=wav)
        print(f"  wrote {path}  ({len(wav)/sr:.2f}s)  <- {text[:40]}...")

    print("\nDone. Keep these as the zero-shot reference for A/B comparison.")


if __name__ == "__main__":
    main()
