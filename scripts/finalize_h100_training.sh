#!/usr/bin/env bash
# Run inside the H100 clone. The script prepares only a complete, evaluated
# iteration-1000 release bundle; any failed gate stops before marking it ready.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 TRAIN_PID" >&2
  exit 2
fi
train_pid="$1"
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
run_dir="$repo_dir/runs/v5_19x19"
python_bin="${GOZERO_PYTHON:-/home/go_ai/.venv-gozero/bin/python}"
gpu="${GOZERO_FINALIZE_GPU:-0}"
train_gpus="${GOZERO_TRAIN_GPUS:-0,1,2,3,4,6,7}"

cd "$repo_dir"
last_iteration() {
  "$python_bin" -c '
import json, pathlib
p = pathlib.Path("runs/v5_19x19/metrics.jsonl")
rows = [json.loads(line) for line in p.read_text().splitlines() if line]
print(rows[-1]["iter"] if rows else 0)
'
}

restart_count=0
while true; do
  while kill -0 "$train_pid" 2>/dev/null; do
    sleep 60
  done
  last_iter="$(last_iteration)"
  if (( last_iter >= 1000 )); then
    break
  fi
  if (( restart_count >= 3 )); then
    echo "training stopped at iteration $last_iter after $restart_count restarts" >&2
    exit 1
  fi
  restart_count=$((restart_count + 1))
  echo "training stopped at iteration $last_iter; restart $restart_count/3" >&2
  env \
    CUDA_VISIBLE_DEVICES="$train_gpus" \
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
    >> "$run_dir/train.log" 2>&1 &
  train_pid="$!"
done

env CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  "$python_bin" -m gozero.evaluate \
  --ckpt "$run_dir/latest.pkl" --vs-random --games 256 --sims 32 \
  > "$run_dir/eval-random.txt" 2>&1

# GNU Go mode is sequential within one process. Split the 20 games into
# black/white pairs across every H100 that completed training, preserving an
# exactly balanced colour assignment while cutting the wall-clock gate time.
IFS=',' read -r -a eval_gpus <<< "$train_gpus"
if (( ${#eval_gpus[@]} == 0 )); then
  echo "no GPUs available for GNU Go evaluation" >&2
  exit 1
fi
pair_base=$((10 / ${#eval_gpus[@]}))
pair_extra=$((10 % ${#eval_gpus[@]}))
eval_pids=()
eval_files=()
for index in "${!eval_gpus[@]}"; do
  pairs="$pair_base"
  if (( index < pair_extra )); then
    pairs=$((pairs + 1))
  fi
  (( pairs > 0 )) || continue
  games=$((pairs * 2))
  eval_gpu="${eval_gpus[$index]}"
  shard="$run_dir/eval-gnugo-gpu${eval_gpu}.txt"
  eval_files+=("$shard")
  env CUDA_VISIBLE_DEVICES="$eval_gpu" XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    "$python_bin" -m gozero.evaluate \
    --ckpt "$run_dir/latest.pkl" --games "$games" --sims 128 \
    --max-plies 722 --seed "$((4200 + index))" \
    --vs-gtp "/usr/games/gnugo --mode gtp --boardsize 19 --komi 7.5 --chinese-rules --level 10 --play-out-aftermath --capture-all-dead" \
    > "$shard" 2>&1 &
  eval_pids+=("$!")
done
eval_failed=0
for eval_pid in "${eval_pids[@]}"; do
  if ! wait "$eval_pid"; then
    eval_failed=1
  fi
done
if (( eval_failed )); then
  echo "one or more parallel GNU Go evaluation shards failed" >&2
  for shard in "${eval_files[@]}"; do
    echo "===== $shard =====" >&2
    tail -n 40 "$shard" >&2 || true
  done
  exit 1
fi
"$python_bin" scripts/aggregate_eval_results.py \
  --output "$run_dir/eval-gnugo.txt" --expected-games 20 \
  "${eval_files[@]}"

env CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  "$python_bin" -m scripts.benchmark_checkpoint \
  --ckpt "$run_dir/latest.pkl" \
  > "$run_dir/benchmark-latency.json"

random_win="$(sed -n 's/.*(\([0-9.]*\)% wins).*/\1/p' "$run_dir/eval-random.txt" | tail -n 1)"
gnugo_win="$(sed -n 's/.*(\([0-9.]*\)% wins).*/\1/p' "$run_dir/eval-gnugo.txt" | tail -n 1)"
latency="$(tail -n 1 "$run_dir/benchmark-latency.json")"
if [[ -z "$random_win" || -z "$gnugo_win" || "$latency" != \{*\} ]]; then
  echo "could not parse evaluation or latency output" >&2
  exit 1
fi
if ! awk -v win="$random_win" 'BEGIN { exit !(win >= 95.0) }'; then
  echo "refusing to publish: random win rate $random_win% is below 95%" >&2
  exit 1
fi
if ! awk -v win="$gnugo_win" 'BEGIN { exit !(win >= 50.0) }'; then
  echo "refusing to publish: GNU Go level 10 win rate $gnugo_win% is below 50%" >&2
  exit 1
fi

"$python_bin" scripts/gen_app_stats.py \
  --vs-random "$random_win" --vs-gnugo "$gnugo_win" \
  --latency "$latency"
"$python_bin" -m py_compile \
  gozero/net.py gozero/train.py gozero/server.py \
  scripts/gen_app_stats.py scripts/benchmark_checkpoint.py \
  scripts/aggregate_eval_results.py

(cd "$run_dir" && sha256sum latest.pkl > latest.pkl.sha256)
printf '%s\n' \
  "release ready at $(date -u +%FT%TZ)" \
  "iteration=1000" \
  "random_win=$random_win" \
  "gnugo_win=$gnugo_win" \
  "latency=$latency" \
  > "$run_dir/release-ready.txt"
