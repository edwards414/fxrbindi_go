#!/usr/bin/env bash
# Run on the inference host after the repository has been fast-forwarded.
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
model_path="runs/v5_19x19/latest.pkl"
checksum_path="runs/v5_19x19/latest.pkl.sha256"
# 19x19 的 128-simulation 搜尋在 CPU 上首次 JIT 會比 9x9 久；只影響
# 新 image 的第一次啟動，健康檢查一成功仍會立即結束。
deploy_timeout="${GOZERO_DEPLOY_TIMEOUT:-900}"
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
if grep -q '^version https://git-lfs.github.com/spec/v1$' "$model_path"; then
  if ! git lfs version >/dev/null 2>&1; then
    echo "19x19 checkpoint is an LFS pointer and git-lfs is unavailable: $model_path" >&2
    exit 1
  fi
  git lfs install --local >/dev/null
  git lfs pull --include="$model_path" --exclude=""
fi
if [[ ! -s "$model_path" ]] || grep -q '^version https://git-lfs.github.com/spec/v1$' "$model_path"; then
  echo "19x19 checkpoint is missing or is still a Git LFS pointer: $model_path" >&2
  exit 1
fi
if [[ ! -f "$checksum_path" ]]; then
  echo "checkpoint checksum is missing: $checksum_path" >&2
  exit 1
fi
(cd "$(dirname "$model_path")" && sha256sum -c "$(basename "$checksum_path")")

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
print(json.dumps(health, separators=(",", ":")))
' "$health_json"

# Prove that the public queue contract and one real 19x19 AI move work with
# the deployed checkpoint. The smoke-test game is resigned after validation.
python3 scripts/smoke_test_server.py \
  --base-url http://127.0.0.1:8765 \
  --expected-iteration 1000 \
  --expected-board-size 19 \
  --timeout 300
