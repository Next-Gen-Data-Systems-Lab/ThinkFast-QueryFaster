#!/bin/bash
# Executable for scan jobs that need a newer torch and transformers than the vLLM
# image: src.jev_scan and src.rerank_scan. Qwen3.5 needs a newer torch and transformers than
# the vLLM image, so the job builds its own environment on scratch. The versions
# are the ones the Kev job ran with; JEV-9B was trained on transformers 5.16.
set -euo pipefail

echo "=== host $(hostname)  $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
export HOME="$SCRATCH/home"
export HF_HOME="$SCRATCH/hf"
export SCIKIT_LEARN_DATA="$SCRATCH/sklearn_data"
export TOKENIZERS_PARALLELISM=false
# The job's uid has no passwd entry in the image, and torch needs a user name.
export USER=jev LOGNAME=jev
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor" TRITON_CACHE_DIR="$SCRATCH/triton"
mkdir -p "$HOME" "$HF_HOME" "$SCIKIT_LEARN_DATA" results

# ---- arguments from the submit file -------------------------------------
MODULE="${1:-jev_scan}"
DATASET="${2:-20newsgroups}"
SIZES="${3:-1000}"
TAG="${4:-jev9b}"
MAX_QUERIES="${5:-0}"
MODEL="${6:--}"
echo "=== module=$MODULE dataset=$DATASET sizes=$SIZES tag=$TAG max_queries=$MAX_QUERIES model=$MODEL ==="
case "$MODULE" in jev_scan|rerank_scan) ;; *) echo "unknown module $MODULE"; exit 2;; esac
EXTRA=()
if [ "$MODEL" != "-" ]; then EXTRA=(--model "$MODEL"); fi
export SYNERGY_DIR="$PWD/data/synergy"

python3 -m venv "$SCRATCH/venv"
PY="$SCRATCH/venv/bin/python"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q "torch==2.8.0" "transformers==5.17.0" "peft==0.21.0" \
    "accelerate==1.15.0" "safetensors==0.8.0" "huggingface-hub==1.32.0" \
    "flash-linear-attention==0.5.2" numpy scikit-learn pandas datasets
# Without causal-conv1d, transformers runs Qwen3.5's short convolutions in plain
# PyTorch, which cut the pilot to 74 decisions per second. This is the prebuilt
# wheel for torch 2.8, CUDA 12 and Python 3.12, so nothing is compiled.
CONV_WHEEL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
"$PY" -m pip install -q "$CONV_WHEEL" || echo "WARNING: causal-conv1d not installed, convolutions will be slow"
"$PY" -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, torch.cuda.get_device_name(0))"

PYTHONPATH="$PWD" "$PY" -m "src.$MODULE" "${EXTRA[@]}" \
    --dataset "$DATASET" --corpus-sizes "$SIZES" --tag "$TAG" --max-queries "$MAX_QUERIES"

echo "=== done $(date -Is) ==="
ls -la results
