#!/bin/bash
set -euo pipefail

LOG=/workspace/logs/verify_correctness/4_gpu.log
CMD=(python train.py /data/dfrc_n5000.binpack /data/fishpack32.binpack \
        --threads 4 \
        --num-workers 16 \
        --batch-size 65536 \
        --l1=128 --features='HalfKAv2_hm^' \
        --max_epochs=5 \
        --default_root_dir /workspace/logs/verify_correctness/ \
        --gpus="0,1,2,3"
)


{
    echo "===== COMMAND ====="
    printf '%q ' "${CMD[@]}"
    echo
    echo "===== START ====="
    echo
    time "${CMD[@]}"
} 2>&1 | tee "$LOG"
