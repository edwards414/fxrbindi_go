#!/usr/bin/env bash
# Run inside the H100 clone. The script publishes only a complete, evaluated
# iteration-1000 run; any failed gate stops before git push.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: GITHUB_TOKEN_FILE=/path/to/token $0 TRAIN_PID" >&2
  exit 2
fi
if [[ -z "${GITHUB_TOKEN_FILE:-}" || ! -r "$GITHUB_TOKEN_FILE" ]]; then
  echo "GITHUB_TOKEN_FILE must name a readable token file" >&2
  exit 2
fi

train_pid="$1"
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
run_dir="$repo_dir/runs/v5_19x19"
run_rel="runs/v5_19x19"
python_bin="${GOZERO_PYTHON:-/home/go_ai/.venv-gozero/bin/python}"
gpu="${GOZERO_FINALIZE_GPU:-0}"

while kill -0 "$train_pid" 2>/dev/null; do
  sleep 60
done

cd "$repo_dir"
last_iter="$($python_bin -c '
import json, pathlib
p = pathlib.Path("runs/v5_19x19/metrics.jsonl")
rows = [json.loads(line) for line in p.read_text().splitlines() if line]
print(rows[-1]["iter"] if rows else 0)
')"
if [[ "$last_iter" != "1000" ]]; then
  echo "refusing to publish incomplete training: iteration $last_iter" >&2
  exit 1
fi

env CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  "$python_bin" -m gozero.evaluate \
  --ckpt "$run_dir/latest.pkl" --vs-random --games 256 --sims 32 \
  > "$run_dir/eval-random.txt" 2>&1

env CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  "$python_bin" -m gozero.evaluate \
  --ckpt "$run_dir/latest.pkl" --games 20 --sims 128 --max-plies 722 \
  --vs-gtp "/usr/games/gnugo --mode gtp --boardsize 19 --komi 7.5 --chinese-rules --level 10 --play-out-aftermath --capture-all-dead" \
  > "$run_dir/eval-gnugo.txt" 2>&1

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

"$python_bin" scripts/gen_app_stats.py \
  --vs-random "$random_win" --vs-gnugo "$gnugo_win" \
  --latency "$latency"
"$python_bin" -m py_compile \
  gozero/net.py gozero/train.py gozero/server.py \
  scripts/gen_app_stats.py scripts/benchmark_checkpoint.py

git fetch origin main h100-19x19-training
git merge --ff-only origin/h100-19x19-training
git merge-base --is-ancestor origin/main HEAD
git lfs install --local
git add -f "$run_rel/latest.pkl"
git add \
  "$run_rel/config.json" \
  "$run_rel/metrics.jsonl" \
  "$run_rel/eval-random.txt" \
  "$run_rel/eval-gnugo.txt" \
  "$run_rel/benchmark-latency.json" \
  app/assets/model_stats.json
git -c user.name="GoZero H100 Trainer" \
  -c user.email="actions@users.noreply.github.com" \
  commit -m "feat: deploy trained 19x19 model"

askpass="$(mktemp)"
trap 'rm -f "$askpass"' EXIT
chmod 700 "$askpass"
printf '%s\n' \
  '#!/bin/sh' \
  'case "$1" in' \
  '  *Username*) printf "%s\\n" "x-access-token" ;;' \
  '  *Password*) exec cat "$GITHUB_TOKEN_FILE" ;;' \
  'esac' > "$askpass"
export GITHUB_TOKEN_FILE
GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
  git push https://github.com/edwards414/fxrbindi_go.git HEAD:main

printf '%s\n' "published iteration 1000 at $(date -u +%FT%TZ)" \
  > "$run_dir/finalize-complete.txt"
