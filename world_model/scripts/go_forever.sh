#!/usr/bin/env bash
# Auto-restart trainer on crash. Resumes from latest checkpoint if present.
# Run from world_model/.
#
# Usage:
#   ./scripts/go_forever.sh

set -u
RUNS_DIR="runs/phase1a_wm"
mkdir -p "$RUNS_DIR"

while true; do
    LATEST="$RUNS_DIR/latest.pt"

    if [[ -f "$LATEST" ]]; then
        echo "[go_forever] Resuming from $LATEST"
        python -u scripts/train.py --resume "$LATEST" 2>&1 | tee -a "$RUNS_DIR/train.log"
    else
        echo "[go_forever] Fresh start"
        python -u scripts/train.py 2>&1 | tee -a "$RUNS_DIR/train.log"
    fi

    EXIT_CODE=${PIPESTATUS[0]}
    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[go_forever] Training completed cleanly. Exiting."
        break
    fi

    echo "[go_forever] Trainer exited with code $EXIT_CODE. Restarting in 10s..."
    sleep 10
done
