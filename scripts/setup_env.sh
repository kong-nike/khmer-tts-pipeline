#!/usr/bin/env bash
# ============================================================
# setup_env.sh  —  one-time environment setup (no sudo needed)
# Run from the repo root.
# ============================================================
set -e

echo ">> Creating / activating venv"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

echo ">> Installing PyTorch"
# Use the default PyPI wheel — it ships a recent torch (>=2.7) with a bundled
# CUDA runtime that works on this server's CUDA 13 driver (backward compatible).
# Do NOT pin an old CUDA index (e.g. cu124) — that can downgrade torch below
# what current transformers requires and break model loading.
pip install "torch>=2.7"

echo ">> Installing project (editable) + dependencies"
pip install -e .
pip install -r requirements.txt

# Audio decoding note:
#   `datasets` decodes audio via torchcodec, which needs ffmpeg present.
#   - If `ffmpeg -version` works, you're fine.
#   - No ffmpeg and no sudo? either:  pip install imageio-ffmpeg
#                                  or: pip install "datasets<3.0"

echo ">> Sanity check"
python - <<'PY'
import torch, numpy, transformers, datasets, khmertts
print("khmertts   :", khmertts.__version__)
print("torch      :", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu        :", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
print("datasets   :", datasets.__version__)
PY

echo ">> Done. Reactivate later with:  source .venv/bin/activate"