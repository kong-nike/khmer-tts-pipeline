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
