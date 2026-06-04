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
make manifest    # data/.../wav16 + data/.../text  ->  metadata.csv
make audit       # quality report: audit.csv + flagged.csv
make dataset     # HuggingFace dataset -> hf_dataset/
make baseline    # zero-shot samples with stock mms-tts-khm

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
