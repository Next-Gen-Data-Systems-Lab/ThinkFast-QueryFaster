#!/bin/bash
# Executable for the CHTC GPU job. Runs inside the container named in the
# submit file, in the job's scratch directory, with src/ already transferred.
set -euo pipefail

echo "=== host $(hostname)  $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python3 -c "import torch;print('torch', torch.__version__, torch.cuda.get_device_name(0))" || true

# Keep everything on scratch: the container's filesystem is read-only, and
# model weights and datasets would not fit in the CHTC home quota.
SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
export HOME="$SCRATCH/home"
export HF_HOME="$SCRATCH/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export SCIKIT_LEARN_DATA="$SCRATCH/sklearn_data"
mkdir -p "$HOME" "$HF_HOME" "$SCIKIT_LEARN_DATA" results figs

# HTCondor sets CUDA_VISIBLE_DEVICES to a GPU UUID, but vLLM 0.6.3 parses it as
# an integer and crashes. Swap the UUID for the matching device index.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
    GPU_UUID="$CUDA_VISIBLE_DEVICES"
    GPU_INDEX=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
        | awk -F', ' -v u="$GPU_UUID" '$2 == u { print $1 }')
    if [[ -z "$GPU_INDEX" ]]; then
        echo "ERROR: assigned GPU $GPU_UUID not found by nvidia-smi" >&2
        exit 1
    fi
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
    echo "=== GPU $GPU_UUID is device $GPU_INDEX ==="
fi

# The vLLM image does not ship everything src/ needs. Install the rest from
# requirements.txt, leaving out vllm itself; the constraints stop pip changing
# any package version the image already has.
if ! python3 -c "import sklearn, pandas, matplotlib, sentence_transformers, datasets" 2>/dev/null; then
    echo "=== installing missing python packages ==="
    python3 -m pip list --format=freeze > "$SCRATCH/constraints.txt"
    grep -v '^vllm' requirements.txt > "$SCRATCH/requirements-job.txt"
    python3 -m pip install -q --user \
        -c "$SCRATCH/constraints.txt" -r "$SCRATCH/requirements-job.txt"
fi

# ---- arguments from the submit file -------------------------------------
DATASET="${1:-20newsgroups}"
MODEL="${2:-Qwen/Qwen2.5-7B-Instruct}"
SIZES="${3:-1000,5000,10000}"
BATCHES="${4:-32,64,128,256}"
K="${5:-100}"
TAG="${6:-run}"
ARMS="${7:-classify,ann,hybrid}"

OUT="results/${DATASET}_${TAG}.jsonl"
echo "=== dataset=$DATASET model=$MODEL sizes=$SIZES batches=$BATCHES k=$K arms=$ARMS ==="

# Smoke test first: catches a typo in 30 seconds instead of after the model
# has spent ten minutes loading onto the GPU.
echo "=== smoke test (mock model, no GPU) ==="
python3 -m src.runner --dataset synthetic --arms classify,ann,hybrid,agent \
    --model mock --embed-backend tfidf \
    --corpus-sizes 300 --batch-sizes 32 --k 50 \
    --out results/_smoke.jsonl
echo "=== smoke test passed ==="

# One process per batch size: deleting a vLLM engine does not reliably release
# its GPU memory, so a second engine in the same process can run out. Every
# process appends to the same file.
#
# Per-document scores are saved for the last batch size only: batch size does
# not change the scores, so one copy is enough.
IFS=, read -ra BATCH_LIST <<< "$BATCHES"
LAST_BS="${BATCH_LIST[${#BATCH_LIST[@]}-1]}"
for BS in "${BATCH_LIST[@]}"; do
    SCORES=""
    if [[ "$BS" == "$LAST_BS" ]]; then
        SCORES="results/${DATASET}_${TAG}.scores.jsonl"
    fi
    python3 -m src.runner \
        --dataset "$DATASET" \
        --arms "$ARMS" \
        --model "$MODEL" \
        --corpus-sizes "$SIZES" \
        --batch-sizes "$BS" \
        --k "$K" \
        --out "$OUT" \
        --scores-out "$SCORES"
done

python3 -m src.analyze --results "$OUT" --outdir "figs/${DATASET}_${TAG}"

echo "=== done $(date -Is) ==="
ls -la results figs
