#!/bin/bash
# Executable for the CHTC GPU job. Runs inside the container named in the
# submit file, in the job's scratch directory, with code/ already transferred.
set -euo pipefail

echo "=== host $(hostname)  $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python3 -c "import torch;print('torch', torch.__version__, torch.cuda.get_device_name(0))" || true

# Keep model weights and datasets on scratch, not in $HOME (CHTC quota).
export HF_HOME="${_CONDOR_SCRATCH_DIR:-$PWD}/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export SCIKIT_LEARN_DATA="${_CONDOR_SCRATCH_DIR:-$PWD}/sklearn_data"
mkdir -p "$HF_HOME" "$SCIKIT_LEARN_DATA" results figs

# ---- arguments from the submit file -------------------------------------
DATASET="${1:-20newsgroups}"
MODEL="${2:-Qwen/Qwen2.5-7B-Instruct}"
SIZES="${3:-1000,5000,10000}"
BATCHES="${4:-32,64,128,256}"
K="${5:-100}"
TAG="${6:-run}"

OUT="results/${DATASET}_${TAG}.jsonl"
echo "=== dataset=$DATASET model=$MODEL sizes=$SIZES batches=$BATCHES k=$K ==="

# Smoke test first: catches a typo in 30 seconds instead of after the model
# has spent ten minutes loading onto the GPU.
echo "=== smoke test (mock model, no GPU) ==="
python3 -m src.runner --dataset synthetic --arms classify,ann,hybrid \
    --model mock --embed-backend tfidf \
    --corpus-sizes 300 --batch-sizes 32 --k 50 \
    --out results/_smoke.jsonl
echo "=== smoke test passed ==="

python3 -m src.runner \
    --dataset "$DATASET" \
    --arms classify,ann,hybrid \
    --model "$MODEL" \
    --corpus-sizes "$SIZES" \
    --batch-sizes "$BATCHES" \
    --k "$K" \
    --out "$OUT"

python3 -m src.analyze --results "$OUT" --outdir "figs/${DATASET}_${TAG}"

echo "=== done $(date -Is) ==="
ls -la results figs
