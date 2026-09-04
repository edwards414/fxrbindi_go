#!/usr/bin/env bash
# Run on the inference host after the repository has been fast-forwarded.
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
model_paths=("runs/v5_19x19/latest.pkl" "runs/v4/latest.pkl")
checksum_paths=("runs/v5_19x19/latest.pkl.sha256" "runs/v4/latest.pkl.sha256")
model_labels=("19x19" "9x9")
# 兩顆模型都會在啟動時預先 JIT 三種棋力；健康檢查一成功仍會立即結束。
deploy_timeout="${GOZERO_DEPLOY_TIMEOUT:-1200}"
compose=(docker compose --env-file /dev/null)

cd "$repo_dir"
command -v docker >/dev/null
command -v curl >/dev/null
command -v git >/dev/null
command -v python3 >/dev/null
command -v sha256sum >/dev/null

# A normal Git checkout may leave an LFS pointer in place of the model. CI
# uploads the verified binary before calling this script; interactive deploys
# may still resolve the pointer locally when git-lfs is installed.
for index in "${!model_paths[@]}"; do
  model_path="${model_paths[$index]}"
  checksum_path="${checksum_paths[$index]}"
  label="${model_labels[$index]}"
  if grep -q '^version https://git-lfs.github.com/spec/v1$' "$model_path"; then
    if ! git lfs version >/dev/null 2>&1; then
      echo "$label checkpoint is an LFS pointer and git-lfs is unavailable: $model_path" >&2
      exit 1
    fi
    git lfs install --local >/dev/null
    git lfs pull --include="$model_path" --exclude=""
  fi
  if [[ ! -s "$model_path" ]] || grep -q '^version https://git-lfs.github.com/spec/v1$' "$model_path"; then
    echo "$label checkpoint is missing or is still a Git LFS pointer: $model_path" >&2
    exit 1
  fi
  if [[ ! -f "$checksum_path" ]]; then
    echo "checkpoint checksum is missing: $checksum_path" >&2
    exit 1
  fi
  (cd "$(dirname "$model_path")" && sha256sum -c "$(basename "$checksum_path")")
done

# The developer .env starts with a non KEY=VALUE SSH shortcut, so Compose must
# not auto-parse it. All production settings have safe defaults in compose.yaml.
"${compose[@]}" config --quiet
"${compose[@]}" up -d --build --remove-orphans

deadline=$((SECONDS + deploy_timeout))
until curl --fail --silent --show-error --max-time 3 \
    http://127.0.0.1:8765/health >/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "engine did not become healthy within ${deploy_timeout}s" >&2
    "${compose[@]}" ps >&2
    "${compose[@]}" logs --tail=120 gozero-server >&2
    exit 1
  fi
  sleep 5
done

"${compose[@]}" ps
health_json="$(curl --fail --silent --show-error http://127.0.0.1:8765/health)"
python3 -c '
import json
import sys

health = json.loads(sys.argv[1])
expected = {
    "ok": True,
    "model": "gozero go_19x19 192ch x 12blk",
    "iteration": 1000,
    "board_size": 19,
}
wrong = {key: (value, health.get(key)) for key, value in expected.items()
         if health.get(key) != value}
if wrong:
    raise SystemExit(f"deployed model identity mismatch: {wrong}")
expected_models = {
    9: ("gozero go_9x9 192ch x 12blk", 4628),
    19: ("gozero go_19x19 192ch x 12blk", 1000),
}
models = {item.get("board_size"): item for item in health.get("models", [])}
model_wrong = {
    size: {"expected": identity, "actual": models.get(size)}
    for size, identity in expected_models.items()
    if models.get(size, {}).get("model") != identity[0]
    or models.get(size, {}).get("iteration") != identity[1]
}
if health.get("board_sizes") != [9, 19] or model_wrong:
    raise SystemExit(
        f"deployed dual-model identity mismatch: board_sizes={health.get('board_sizes')}, "
        f"models={model_wrong}"
    )
print(json.dumps(health, separators=(",", ":")))
' "$health_json"

# Prove that the public queue contract and one real 19x19 AI move work with
# the deployed checkpoint. The smoke-test game is resigned after validation.
python3 scripts/smoke_test_server.py \
  --base-url http://127.0.0.1:8765 \
  --expected-iteration 1000 \
  --expected-board-size 19 \
  --timeout 300

# The same public endpoint must route 9x9 to the v4 checkpoint while sharing
# the queue with 19x19.
python3 scripts/smoke_test_server.py \
  --base-url http://127.0.0.1:8765 \
  --expected-model "gozero go_9x9 192ch x 12blk" \
  --expected-iteration 4628 \
  --expected-board-size 9 \
  --timeout 300
