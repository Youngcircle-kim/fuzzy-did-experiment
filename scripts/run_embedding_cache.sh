#!/usr/bin/env bash

set -euo pipefail

CONFIG="configs/experiments/embedding_facenet512.yaml"

NUM_GPUS=4
MAX_IDENTITIES=""
RESUME="--resume"

mkdir -p outputs/logs

for GPU in $(seq 0 $((NUM_GPUS-1)))
do
    echo "Launching GPU ${GPU}..."

    CUDA_VISIBLE_DEVICES=${GPU} \
    TF_USE_LEGACY_KERAS=1 \
    TF_GPU_ALLOCATOR=cuda_malloc_async \
    PYTHONPATH="$PWD/src:${PYTHONPATH:-}" \
    python scripts/extract_embedding_cache.py \
        --config "${CONFIG}" \
        --shard-index ${GPU} \
        ${RESUME} \
        ${MAX_IDENTITIES:+--max-identities ${MAX_IDENTITIES}} \
        > "outputs/logs/cache_shard_${GPU}.log" 2>&1 &

done

wait

echo
echo "======================================="
echo "Embedding cache extraction completed."
echo "======================================="