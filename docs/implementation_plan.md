# Implementation Plan — Text Analysis for Khmer TTS

**Project:** High-quality data pipeline + fine-tuned neural TTS for production-ready Khmer speech
**Intern:** Lean Sokkong · IDRI / CADT · Research Intern
**Duration:** 4 May 2026 – 4 August 2026 (3 months)

---

## 0. Executive summary — what changed after research

The proposal is sound, but three strategic decisions will determine whether you ship something working in 3 months or get stuck:

1. **Fine-tune an existing Khmer VITS checkpoint instead of training VITS2 from scratch.** The only free Khmer TTS corpus (OpenSLR SLR42) is ~4 hours, single-gender. That is enough to *fine-tune* a pre-trained model but not to *train* a stable VITS2 from zero. Meta's `facebook/mms-tts-khm` is already a Khmer VITS model; fine-tuning it is the proven, low-risk path.

2. **Reframe the central experiment as a controlled A/B.** Fine-tune the *same* base checkpoint twice — once on the raw corpus, once on your cleaned "Gold Standard" corpus — and compare WER + MCD. This isolates *data quality* as the only variable, which is exactly the thesis you're trying to prove. Keep "VITS2 from scratch" as a stretch goal, not a dependency.

3. **Stand on the existing Khmer NLP stack.** A large fraction of the "Text Analysis" and "forced alignment" work is already open-sourced (the `seanghay` ecosystem). Your contribution is the *pipeline that wires it together*, the cleaning logic, and the rigorous before/after evaluation — not re-implementing a Khmer normalizer or G2P from scratch.

---

## 1. Pre-project research insights

### 1.1 The Khmer language facts that drive every design choice

- **No word boundaries.** Khmer text is written without spaces between words. Almost everything downstream (normalization, G2P, alignment) depends on first running **word/syllable segmentation**. This is step zero of text analysis, not an afterthought.
- **Abugida, not alphabet; not tonal.** 33 consonants, dependent + independent vowels, and **subscript consonants (coeng)** stacked below the base. Khmer uses a two-register (clear vs. breathy) phonation system rather than tone. Prosody modelling differs from Thai/Vietnamese.
- **Unicode is messy.** The "inconsistent Unicode representations" in the proposal is a real, documented problem: non-canonical ordering of coeng + vowel sequences, zero-width spaces (ZWSP, U+200B) used inconsistently as soft word separators, and visually-identical-but-bytewise-different strings. **Canonical reordering + NFC normalization is mandatory** before training, or the model learns to map several byte sequences to the same sound (instability).
- **G2P is hard.** Published rule-based / WFST Khmer G2P sits around 39–51% word error; many Pali/Sanskrit loanwords have unpronounced final consonants with no clean rule. Implication: don't bet the project on perfect phonemes. Use character/MMS-style input as your baseline and treat phoneme input as an experiment.

### 1.2 The data landscape

| Resource | Size | Notes | License |
|---|---|---|---|
| **OpenSLR SLR42** (`km_kh_male`) | ~3.97 hrs, ~2.9k utts, male only | The proposal's primary corpus. Clean but small and single-speaker. | CC-BY-SA 4.0 |
| `seanghay/khmer_kheng_info_speech` | ~3k word-level recordings (kheng.info dictionary) | Word-level, good for G2P/lexicon, not sentences. | check card |
| `seanghay/khmer_mpwt_speech` | sentence-level | Used for an existing MMS fine-tune. | check card |
| **Khmer ASR Cultural Dataset (DDD)** | ~106 hrs, 8 speakers, metadata | Large, multi-speaker, cultural domain. Built for ASR but usable for TTS data study. | CC-BY-SA 4.0 |
| Google FLEURS (km) | small | Useful as held-out eval / for the ASR judge. | CC-BY |

**Insight:** SLR42 alone proves the data-cleaning thesis fine. But if you want a stronger final voice, the DDD cultural set massively expands available hours. Mind that mixing corpora introduces multi-speaker variance — for a single clean voice, fine-tune per-speaker.

### 1.3 The tooling landscape (build on these)

| Need | Existing tool | Use it for |
|---|---|---|
| Word segmentation | `khmercut`, `seanghay/khmersegment` | Tokenizing before normalization/G2P |
| Text normalization & verbalization | **`tha`** (Khmer Text Normalization & Verbalization Toolkit) | Numbers, currency, dates → spoken form (your Objective 2 core) |
| Numeral expansion | `khmernumber` | Khmer & Arabic numerals → words |
| Inverse text normalization | `khmertagger` (XLM-R based) | Punctuation restoration / number recognition |
| G2P / phonemizer | `khmerphonemizer`, `native-khmer-g2p` | Optional phoneme front-end |
| Forced alignment | **`seanghay/khmer-acoustic-model-mfa`** (Montreal Forced Aligner) | Your Objective 1 forced alignment |
| Base TTS model | **`facebook/mms-tts-khm`** (VITS) | The checkpoint you fine-tune |
| Fine-tuning recipe | **`ylacombe/finetune-hf-vits`** | The training harness for MMS/VITS |
| ASR judge (for WER loop) | `seanghay/whisper-small-khmer-v2` | Transcribe synthesized speech to measure intelligibility |
| Reference / curated list | `seanghay/awesome-khmer-language` | Discover everything else |
| VITS2 (stretch) | `p0p4k/vits2_pytorch`, `daniilrobnikov/vits2` | From-scratch VITS2 comparison |

---

## 2. Architecture & approach decision

**Primary track (must succeed):** Fine-tune `facebook/mms-tts-khm` (VITS) using `finetune-hf-vits`, on raw vs. cleaned data, char-based MMS tokenizer.

**Secondary track (if time allows):** Add a phoneme front-end (G2P) and compare against char input.

**Stretch track (nice-to-have):** Train `vits2_pytorch` from scratch on the cleaned + expanded (DDD) corpus, to demonstrate the VITS2 architecture named in the proposal.

> ⚠️ **License flag for "production-ready":** the MMS checkpoint is **CC-BY-NC** (non-commercial). If IDRI intends commercial deployment, the deliverable model must come from the **from-scratch VITS2 track on permissively-licensed data (SLR42 is CC-BY-SA)**, not the MMS fine-tune. Confirm the intended use with your supervisor (HIM Soklong) in week 1.

---

## 3. The experiment that proves the thesis

This is the spine of your final report. Hold everything constant except the training data.

```
Base checkpoint: facebook/mms-tts-khm  (frozen starting point)
        │
        ├── Fine-tune A  ← RAW SLR42 (no cleaning)
        └── Fine-tune B  ← CLEANED "Gold Standard" SLR42 (your pipeline)

Same held-out test set (sentences NOT in training), same hyper-params, same steps.

Metrics on the test set:
  • WER / CER  (intelligibility)  — synth → seanghay/whisper-small-khmer-v2 → compare to reference text
  • MCD        (acoustic distance) — synth vs. ground-truth recording, after DTW alignment
  • MOS / CMOS (naturalness)      — small human listening test (even 1–5 native listeners is publishable for LRLs)

Hypothesis: B < A on WER/CER and MCD, B > A on MOS.
```

**Why CER matters more than WER for Khmer:** because word segmentation of the ASR output is fuzzy, character error rate is a more stable intelligibility metric. Report both, lead with CER.

---

## 4. Week 0 — environment & baseline (before the official clock matters)

- [ ] GPU access confirmed (a single 16–24 GB GPU, e.g., T4/A10/3090, is enough for MMS fine-tuning).
- [ ] Repo + experiment tracking: GitHub + Weights & Biases (or TensorBoard). Every run = config + data version + metrics.
- [ ] **Data versioning:** adopt DVC or at least a strict folder/manifest convention (`raw/`, `interim/`, `gold/`) so "which data made this model" is never ambiguous.
- [ ] Reproduce two baselines *day one*:
  - [ ] Run `facebook/mms-tts-khm` zero-shot on 10 sentences → this is your "before any work" reference.
  - [ ] Run `seanghay/whisper-small-khmer-v2` on real audio → confirm your ASR judge works and note its own error floor (you can't measure synthesis intelligibility below the ASR's own error rate).
- [ ] Download SLR42; load `tha`, `khmercut`, `khmernumber`, MFA + Khmer acoustic model; smoke-test each.

---

## 5. Month 1 — Data audit & text normalization

**Goal:** A reproducible text pipeline + a full quality map of the corpus.

### 5.1 Data audit (week 1–2)
- Inventory every utterance: duration, sample rate, speaker, transcript length, characters used.
- **Audio quality metrics per clip:** estimate SNR (e.g., WADA-SNR or Brouhaha), clipping, leading/trailing silence, peak/RMS levels, sample-rate consistency. Produce a histogram + a flag column.
- **Transcript audit:** detect non-Khmer characters, digits, foreign words, empty/duplicate transcripts, ZWSP usage, mismatched encodings.
- Output: a `corpus_report.html`/notebook with distributions and a per-clip quality table. This *is* a deliverable.

### 5.2 Text normalization pipeline (week 2–4)
Build `normalize(text) -> spoken_text` as an ordered, testable pipeline:
1. **Unicode canonicalization** — NFC + Khmer coeng/vowel reordering (handle U+17B4/B5 invisible vowels, ZWSP). This directly attacks "inconsistent Unicode representations."
2. **Cleaning** — strip/normalize punctuation, normalize whitespace, remove control chars.
3. **Word segmentation** — `khmercut`.
4. **Verbalization** — numbers, currency (៛/$), dates, percentages, abbreviations → spoken Khmer via `tha` + `khmernumber`. Patch gaps with your own rules; this is where you add value.
5. **(Optional) G2P** — `khmerphonemizer` to emit a phoneme stream for the phoneme-input experiment.

- **Test-driven:** keep a `tests/normalization_cases.tsv` of tricky inputs → expected outputs (e.g., `1,234.5` → spoken Khmer; `25%`; `៛5000`; mixed Latin). Grow it whenever you find a failure. This becomes a small, citable contribution.

**Month 1 deliverables:** corpus audit report; versioned text-normalization library with unit tests; a documented list of normalization edge cases for Khmer.

---

## 6. Month 2 — Audio cleaning pipeline & fine-tuning

**Goal:** The "Gold Standard" cleaned dataset + first fine-tuned model.

### 6.1 Audio cleaning pipeline (week 5–6)
Ordered, each step logged so you can ablate:
1. **Resample / mono / loudness-normalize** to the model's expected rate (MMS-khm uses 16 kHz; confirm in config). Normalize loudness (EBU R128 / -23 LUFS or peak norm).
2. **Silence trimming** (VAD or energy-based) — trim leading/trailing and over-long internal silences.
3. **Denoising** (only where SNR is low — don't over-process clean clips): DeepFilterNet / resemble-enhance / demucs. **A/B this**; aggressive denoising can hurt TTS by removing natural detail.
4. **Forced alignment** with MFA + `seanghay/khmer-acoustic-model-mfa` → phone/word timestamps. Use alignment to (a) **detect transcript↔audio mismatches** (drop or fix), (b) re-segment long clips, (c) flag clips with abnormal phone durations.
5. **SNR/quality filtering** — drop clips below an SNR threshold or with alignment failures. Record how many you drop and why.

Output two frozen, versioned datasets: `raw_manifest` and `gold_manifest`.

### 6.2 Fine-tuning (week 6–8)
- Use `finetune-hf-vits` with `facebook/mms-tts-khm`.
- Train **Model A** (raw) and **Model B** (gold) with identical configs/steps/seed.
- Track loss curves; listen to checkpoints every N steps (TTS loss does not correlate perfectly with quality — your ears + WER matter more).
- Watch for the classic VITS failure modes: metallic/robotic artefacts (often data or duration-predictor issues), mispronunciations (G2P/normalization), and instability on long inputs (chunking — see §7.3).

**Month 2 deliverables:** versioned audio-cleaning pipeline; `gold` vs `raw` datasets with drop logs; two trained checkpoints A and B; intermediate WER/CER numbers.

---

## 7. Month 3 — Evaluation, audiobook & documentation

### 7.1 Rigorous evaluation (week 9–10)
- Compute **CER/WER** (via Whisper-Khmer), **MCD** (DTW-aligned vs. ground truth), on the held-out set, for A vs B vs zero-shot baseline.
- Run a **small MOS test**: 15–20 sentences, 3–5 native listeners, 1–5 scale, A vs B blind. Even a one-listener pilot is methodologically defensible for low-resource languages — report it honestly.
- Produce a results table + plots. This is the core evidence for the proposal's Objective 4.

### 7.2 The 5-minute audiobook (week 10–11)
- Pick a public-domain / permissively-licensed Khmer passage.
- **Long-form synthesis ≠ one forward pass.** Pipeline: sentence-split → normalize each → synthesize per sentence → insert natural inter-sentence pauses → concatenate (with short cross-fade) → loudness-normalize the full track.
- This directly addresses the proposal's "unnatural robotic tones in long-form content": much of the long-form roughness is a *chunking/prosody/pause* problem, not just a data problem — document that finding.
- Measure WER on the audiobook too (run the ASR judge over it).

### 7.3 Documentation & handover (week 11–12)
- Final report: problem → data audit findings → pipeline → A/B results → audiobook demo → limitations → future work.
- Reproducible repo: README, environment file, `make`-style commands to rebuild gold data and retrain.
- A short "Khmer TTS data-cleaning checklist" others at IDRI can reuse — a durable institutional artefact.

**Month 3 deliverables:** evaluation report with A/B/baseline metrics + MOS; 5-minute audiobook WAV + the demo script; final written report; clean public-ready repo.

---

## 8. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| VITS2-from-scratch fails on ~4 hrs | High | Make MMS fine-tune the primary; VITS2 is stretch only. |
| MMS license blocks production use | Med | Confirm intended use week 1; keep a CC-BY-SA from-scratch path for the deliverable model if commercial. |
| Denoising hurts naturalness | Med | A/B denoised vs not; only denoise low-SNR clips. |
| G2P errors degrade pronunciation | Med | Char-input baseline first; phoneme input as a measured experiment, not a dependency. |
| ASR judge error floor masks gains | Med | Report ASR's own error rate; rely on CER + MOS, not WER alone. |
| Single-speaker, single-gender voice | Low/known | Acknowledge as scope; note DDD multi-speaker set as future expansion. |
| Scope creep (cleaning is a rabbit hole) | High | Time-box cleaning to Month 2; "good enough gold set" beats perfect. |

---

## 9. Stretch goals (only after the A/B is done)
- Train `vits2_pytorch` from scratch on cleaned SLR42 + DDD for a true VITS2 result.
- Multi-speaker fine-tune using the DDD cultural corpus.
- Phoneme-input vs char-input ablation as a second paper-worthy result.
- Publish the cleaning pipeline + normalization test set as an open contribution to `awesome-khmer-language`.

---

## 10. Key references
- Meta MMS Khmer VITS checkpoint — `huggingface.co/facebook/mms-tts-khm`
- VITS/MMS fine-tuning recipe — `github.com/ylacombe/finetune-hf-vits`
- Existing Khmer MMS fine-tune — `huggingface.co/KrorngAI/mms-tts-khm-finetuned`
- OpenSLR SLR42 (Khmer TTS data) — `openslr.org/42`
- Khmer NLP ecosystem index — `github.com/seanghay/awesome-khmer-language`
- Khmer normalization (`tha`), numbers (`khmernumber`), segmentation (`khmercut`), MFA model (`seanghay/khmer-acoustic-model-mfa`)
- Khmer ASR judge — `huggingface.co/seanghay/whisper-small-khmer-v2`
- VITS2 implementations — `github.com/p0p4k/vits2_pytorch`, `github.com/daniilrobnikov/vits2`
- Khmer ASR Cultural Dataset (DDD, ~106 hrs) — Mozilla Data Collective

*Note: verify each dataset/model license before any production or published use.*
