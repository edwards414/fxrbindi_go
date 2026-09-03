#!/usr/bin/env bash
# Add newly idle H100s to an in-progress GoZero run without losing a completed
# iteration. A candidate must have no compute process and remain idle for
# several samples before the trainer is restarted from its atomic checkpoint.
set -euo pipefail

repo_dir="${GOZERO_REPO_DIR:-/home/gozero19}"
run_dir="${GOZERO_RUN_DIR:-$repo_dir/runs/v5_19x19}"
python_bin="${GOZERO_PYTHON:-/home/go_ai/.venv-gozero/bin/python}"
all_gpus="${GOZERO_ALL_GPUS:-0 1 2 3 4 5 6 7}"
sample_seconds="${GOZERO_GPU_SAMPLE_SECONDS:-30}"
stable_samples="${GOZERO_GPU_STABLE_SAMPLES:-4}"
max_idle_memory_mib="${GOZERO_GPU_MAX_IDLE_MEMORY_MIB:-100}"
max_idle_utilization="${GOZERO_GPU_MAX_IDLE_UTILIZATION:-5}"
log_file="$run_dir/elastic-gpu.log"

mkdir -p "$run_dir"
exec 9>"$run_dir/elastic-gpu.lock"
if ! flock -n 9; then
  echo "another elastic H100 watcher is already running" >&2
  exit 1
fi

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$log_file"
}

find_train_pid() {
  pgrep -fo 'gozero[.]train.*v5_19x19' || true
}

find_finalizer_pid() {
  pgrep -fo 'scripts/finalize_h100_training[.]sh [0-9]+' || true
}

train_gpus() {
  local train_pid="$1"
  tr '\0' '\n' <"/proc/$train_pid/environ" \
    | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' \
    | tail -n 1
}

gpu_is_idle() {
  local gpu="$1" processes stats memory utilization
  processes="$(nvidia-smi --id="$gpu" \
    --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true)"
  [[ -z "$processes" ]] || return 1
  stats="$(nvidia-smi --id="$gpu" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null)" || return 1
  memory="${stats%%,*}"
  utilization="${stats##*,}"
  memory="${memory//[[:space:]]/}"
  utilization="${utilization//[[:space:]]/}"
  [[ "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || return 1
  (( memory <= max_idle_memory_mib && utilization <= max_idle_utilization ))
}

checkpoint_matches_metrics() {
  "$python_bin" - "$run_dir/metrics.jsonl" "$run_dir/latest.pkl" <<'PY'
import json
import pathlib
import pickle
import sys

metrics_path = pathlib.Path(sys.argv[1])
checkpoint_path = pathlib.Path(sys.argv[2])
try:
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if rows and checkpoint.get("iteration") == rows[-1].get("iter"):
        print(rows[-1]["iter"])
        raise SystemExit(0)
except (OSError, ValueError, EOFError, pickle.UnpicklingError):
    pass
raise SystemExit(1)
PY
}

stop_process() {
  local pid="$1" label="$2" attempts=0
  [[ -n "$pid" ]] || return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  kill "$pid"
  while kill -0 "$pid" 2>/dev/null; do
    attempts=$((attempts + 1))
    if (( attempts >= 60 )); then
      log "$label PID $pid did not stop after 60 seconds; sending KILL"
      kill -KILL "$pid"
      break
    fi
    sleep 1
  done
}

start_training() {
  local gpu_csv="$1"
  cd "$repo_dir"
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu_csv" \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.88 \
    JAX_COMPILATION_CACHE_DIR="$repo_dir/.jax_cache" \
    "$python_bin" -m gozero.train \
    --env-id go_19x19 --run-dir "$run_dir" \
    --channels 192 --blocks 12 --compute-dtype bfloat16 \
    --resume "$run_dir/latest.pkl" \
    --selfplay-batch 64 --sims 32 --max-considered 16 \
    --max-steps 722 --pass-guard-ply 100 --train-batch 2048 \
    --lr 0.001 --weight-decay 0.0001 --warmup-iters 20 \
    --decay-iters 1000 --iters 1000 --eval-every 25 \
    --eval-batch 32 --save-every 25 --seed 42 \
    >>"$run_dir/train.log" 2>&1 &
  printf '%s\n' "$!"
}

start_finalizer() {
  local train_pid="$1" gpu_csv="$2"
  cd "$repo_dir"
  nohup env \
    GOZERO_PYTHON="$python_bin" \
    GOZERO_TRAIN_GPUS="$gpu_csv" \
    bash scripts/finalize_h100_training.sh "$train_pid" \
    >>"$run_dir/finalize.log" 2>&1 &
  printf '%s\n' "$!"
}

declare -A stable=()
for gpu in $all_gpus; do
  stable["$gpu"]=0
done

log "elastic watcher started; stable=${stable_samples}x${sample_seconds}s, idle memory<=${max_idle_memory_mib}MiB, utilization<=${max_idle_utilization}%"

while true; do
  train_pid="$(find_train_pid)"
  if [[ -z "$train_pid" ]]; then
    if [[ -f "$run_dir/release-ready.txt" ]]; then
      log "release is ready; watcher exiting"
      exit 0
    fi
    log "trainer not found; finalizer owns recovery, retrying"
    sleep "$sample_seconds"
    continue
  fi

  current_csv="$(train_gpus "$train_pid")"
  if [[ -z "$current_csv" ]]; then
    log "could not read CUDA_VISIBLE_DEVICES for trainer PID $train_pid"
    sleep "$sample_seconds"
    continue
  fi

  current_words="${current_csv//,/ }"
  candidates=()
  for gpu in $all_gpus; do
    if [[ " $current_words " == *" $gpu "* ]]; then
      stable["$gpu"]=0
      continue
    fi
    if gpu_is_idle "$gpu"; then
      stable["$gpu"]=$((stable["$gpu"] + 1))
      log "GPU $gpu idle sample ${stable[$gpu]}/$stable_samples"
      if (( stable["$gpu"] >= stable_samples )); then
        candidates+=("$gpu")
      fi
    else
      stable["$gpu"]=0
    fi
  done

  if (( ${#candidates[@]} == 0 )); then
    sleep "$sample_seconds"
    continue
  fi

  # Recheck immediately before migration so a newly claimed GPU is never used.
  ready=()
  for gpu in "${candidates[@]}"; do
    if gpu_is_idle "$gpu"; then
      ready+=("$gpu")
    else
      stable["$gpu"]=0
    fi
  done
  if (( ${#ready[@]} == 0 )); then
    sleep "$sample_seconds"
    continue
  fi

  target_csv="$(printf '%s\n' $current_words "${ready[@]}" | sort -n -u | paste -sd, -)"
  completed_iter="$(checkpoint_matches_metrics)" || {
    log "checkpoint and metrics are between atomic updates; retrying"
    sleep 2
    continue
  }
  log "checkpoint $completed_iter is complete; migrating $current_csv -> $target_csv"

  finalizer_pid="$(find_finalizer_pid)"
  stop_process "$finalizer_pid" "finalizer"
  stop_process "$train_pid" "trainer"

  new_train_pid="$(start_training "$target_csv")"
  sleep 10
  if ! kill -0 "$new_train_pid" 2>/dev/null; then
    log "expanded trainer failed to start; restoring $current_csv"
    new_train_pid="$(start_training "$current_csv")"
    target_csv="$current_csv"
    sleep 10
    kill -0 "$new_train_pid"
  fi
  new_finalizer_pid="$(start_finalizer "$new_train_pid" "$target_csv")"
  log "trainer PID $new_train_pid and finalizer PID $new_finalizer_pid running on $target_csv"

  for gpu in $all_gpus; do
    stable["$gpu"]=0
  done
  if (( $(tr ',' '\n' <<<"$target_csv" | wc -l) >= 8 )); then
    log "all eight H100s are active; watcher exiting"
    exit 0
  fi
  sleep "$sample_seconds"
done
