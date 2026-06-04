# Reproducible pipeline. Override paths on the command line, e.g.:
#   make manifest AUDIO=data/kh-tts-dataset-master/wav16 TEXT=data/kh-tts-dataset-master/text

PYTHON ?= python
AUDIO  ?= data/kh-tts-dataset-master/wav16
TEXT   ?= data/kh-tts-dataset-master/text
MANIFEST ?= metadata.csv
GOLD     ?= metadata_gold.csv
HF_DIR ?= hf_dataset
MAX_DUR ?= 14.0

.PHONY: help setup manifest audit gold dataset dataset-gold baseline test pipeline clean

help:
	@echo "Targets:"
	@echo "  setup        - create venv + install deps (no sudo)"
	@echo "  manifest     - pair wavs + transcripts -> $(MANIFEST)"
	@echo "  audit        - quality report (audit.csv, flagged.csv)"
	@echo "  gold         - filter to a clean 'Gold Standard' set -> $(GOLD)"
	@echo "  dataset      - HuggingFace dataset from $(MANIFEST) (raw)"
	@echo "  dataset-gold - HuggingFace dataset from $(GOLD) (cleaned)"
	@echo "  baseline     - synthesize with stock mms-tts-khm"
	@echo "  test         - run unit tests"
	@echo "  pipeline     - manifest + audit + gold"
	@echo "  clean        - remove generated artifacts"

setup:
	bash scripts/setup_env.sh

manifest:
	$(PYTHON) -m khmertts.build_manifest --audio $(AUDIO) --text $(TEXT) --out $(MANIFEST)

audit:
	$(PYTHON) -m khmertts.audit_dataset --manifest $(MANIFEST) --audio $(AUDIO)

gold:
	$(PYTHON) -m khmertts.clean_manifest --manifest $(MANIFEST) --audit audit.csv --out $(GOLD) --max_dur $(MAX_DUR)

dataset:
	$(PYTHON) -m khmertts.make_hf_dataset --manifest $(MANIFEST) --audio $(AUDIO) --out_dir $(HF_DIR)

dataset-gold:
	$(PYTHON) -m khmertts.make_hf_dataset --manifest $(GOLD) --audio $(AUDIO) --out_dir hf_dataset_gold

baseline:
	$(PYTHON) -m khmertts.baseline_synthesize --out_dir baseline_samples

test:
	pytest -q

pipeline: manifest audit gold

clean:
	rm -rf metadata.csv metadata_gold.csv dropped.csv audit.csv flagged.csv $(HF_DIR) hf_dataset_gold baseline_samples __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +