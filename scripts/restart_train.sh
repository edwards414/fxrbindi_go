#!/bin/bash
# Resume training from the latest checkpoint (e.g. after a crash or GPU change).
# Usage: scripts/restart_train.sh [GPUS] [RUN_DIR]
set -e
GPUS="${1:-0,3,4}"
RUN="${2:-runs/v1}"
cd /home/go_ai
if pgrep -f "gozero.train --run-dir $RUN" > /dev/null; then
    echo "training for $RUN already running:"; pgrep -af "gozero.train --run-dir $RUN"; exit 1
fi
RESUME=""
[ -f "$RUN/latest.pkl" ] && RESUME="--resume $RUN/latest.pkl"
nohup env CUDA_VISIBLE_DEVICES="$GPUS" XLA_PYTHON_CLIENT_MEM_FRACTION=0.72 \
    python -m gozero.train --run-dir "$RUN" --selfplay-batch 1024 --sims 32 --train-batch 2048 \
    $RESUME >> "$RUN/train.log" 2>&1 &
echo "restarted (PID $!) on GPUs $GPUS $RESUME"
