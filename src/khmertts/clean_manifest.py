#!/usr/bin/env python3
"""
Produce a filtered "Gold Standard" manifest from an audited corpus.

Reads the manifest + the audit.csv (from `khmertts.audit_dataset`) and drops
clips that are out of the safe duration band or carry quality flags
(too_long, too_short, silent, clipped, odd_rate, unreadable). VITS/MMS
fine-tuning is sensitive to over-long clips (GPU OOM / instability), so this
step is required before training, not optional.

Outputs:
    metadata_gold.csv  the kept clips (filename|transcript)
    dropped.csv        the removed clips, with the reason

Run:
    python -m khmertts.clean_manifest \
        --manifest metadata.csv \
        --audit    audit.csv \
        --out      metadata_gold.csv \
        --min_dur 1.0 --max_dur 14.0
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

DEFAULT_DROP_FLAGS = {"too_long", "too_short", "silent", "clipped", "odd_rate", "unreadable"}


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


def read_audit(path: Path):
    """file -> {'duration': float, 'flags': set[str]}"""
    info = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            flags = set(filter(None, (row.get("flags") or "").split(";")))
            # 'unreadable:<msg>' becomes the single tag 'unreadable'
            flags = {fl.split(":", 1)[0] for fl in flags}
            try:
                dur = float(row.get("duration") or 0)
            except ValueError:
                dur = 0.0
            info[row["file"]] = {"duration": dur, "flags": flags}
    return info


def clean(manifest: Path, audit: Path, out: Path, dropped_out: Path,
          min_dur: float, max_dur: float, drop_flags: set[str]):
    pairs = read_manifest(manifest)
    audit_info = read_audit(audit)

    kept, dropped = [], []
    reasons = Counter()
    for name, text in pairs:
        meta = audit_info.get(name)
        why = []
        if meta is None:
            why.append("not_in_audit")
        else:
            dur = meta["duration"]
            if dur and dur < min_dur:
                why.append("too_short")
            if dur and dur > max_dur:
                why.append("too_long")
            why.extend(sorted(meta["flags"] & drop_flags))
        why = sorted(set(why))
        if why:
            dropped.append((name, ";".join(why)))
            for r in why:
                reasons[r] += 1
        else:
            kept.append((name, text))

    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", lineterminator="\n")
        w.writerows(kept)
    with open(dropped_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["file", "reason"])
        w.writerows(dropped)

    print("=" * 50)
    print(f"input clips   : {len(pairs)}")
    print(f"kept (gold)   : {len(kept)}  ->  {out}")
    print(f"dropped       : {len(dropped)}  ->  {dropped_out}")
    if reasons:
        print("drop reasons  :")
        for r, n in reasons.most_common():
            print(f"    {r:<14} {n}")
    print("=" * 50)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="metadata.csv")
    ap.add_argument("--audit", default="audit.csv")
    ap.add_argument("--out", default="metadata_gold.csv")
    ap.add_argument("--dropped_out", default="dropped.csv")
    ap.add_argument("--min_dur", type=float, default=1.0)
    ap.add_argument("--max_dur", type=float, default=14.0,
                    help="drop clips longer than this (VITS stability / GPU memory)")
    ap.add_argument("--keep_flags", nargs="*", default=[],
                    help="flag names to TOLERATE (not drop), e.g. odd_rate")
    args = ap.parse_args()

    drop_flags = DEFAULT_DROP_FLAGS - set(args.keep_flags)
    clean(Path(args.manifest), Path(args.audit), Path(args.out), Path(args.dropped_out),
          args.min_dur, args.max_dur, drop_flags)


if __name__ == "__main__":
    main()