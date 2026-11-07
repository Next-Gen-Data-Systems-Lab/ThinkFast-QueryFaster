#!/bin/bash
# Executable for the Kev job. Kev needs Python 3.12+, a newer torch and a newer
# transformers than the vLLM image, so it gets its own environment on scratch.
set -euo pipefail

# The commit whose code was read. Change it only after reading the new one.
KEV_COMMIT=2074cc44256200bb37540103fa4572901faa10f3
FLA_VERSION=0.5.2

echo "=== host $(hostname)  $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
export HOME="$SCRATCH/home"
export HF_HOME="$SCRATCH/hf"
export SCIKIT_LEARN_DATA="$SCRATCH/sklearn_data"
export UV_CACHE_DIR="$SCRATCH/uv"
export TOKENIZERS_PARALLELISM=false
# The job's uid has no passwd entry in the image, and torch needs a user name for
# its compile cache.
export USER=kev LOGNAME=kev
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor" TRITON_CACHE_DIR="$SCRATCH/triton"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME" "$HF_HOME" "$SCIKIT_LEARN_DATA" results

# ---- arguments from the submit file -------------------------------------
RUN="${1:-jaredpalmer/kev-9b@v1.0}"
DATASET="${2:-20newsgroups}"
SIZES="${3:-1000}"
MODES="${4:-separate,packed}"
TAG="${5:-kev9b}"
MAX_QUERIES="${6:-0}"
echo "=== run=$RUN dataset=$DATASET sizes=$SIZES modes=$MODES tag=$TAG ==="

python3 -m pip install -q --user uv
git clone -q https://github.com/jaredpalmer/kev.git "$SCRATCH/kev"
git -C "$SCRATCH/kev" checkout -q "$KEV_COMMIT"
(cd "$SCRATCH/kev" && uv sync -q --frozen --extra serve --python 3.12)
PY="$SCRATCH/kev/.venv/bin/python"
# The fused kernels are Kev's default on NVIDIA, and it refuses any other version.
uv pip install -q --python "$PY" "flash-linear-attention==$FLA_VERSION"
"$PY" -c "import torch, transformers, kev; print('torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.get_device_name(0))"

PYTHONPATH="$PWD" "$PY" -m src.kev_scan \
    --run "$RUN" --dataset "$DATASET" --corpus-sizes "$SIZES" \
    --modes "$MODES" --tag "$TAG" --max-queries "$MAX_QUERIES"

echo "=== done $(date -Is) ==="
ls -la results
