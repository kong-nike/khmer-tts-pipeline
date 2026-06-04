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

Corpus summary (single speaker, fill in after running the audit):

| Field            | Value            |
|------------------|------------------|
| Speaker          | kh_atr_m001      |
| Clips            | ~4000            |
| Total duration   | _run `make audit`_ |
| Sample rate      | 16 kHz mono      |
| Language         | Khmer (khm)      |

> Confirm usage rights with the supervisor before publishing any model trained
> on this data (see licensing note in the main README).
