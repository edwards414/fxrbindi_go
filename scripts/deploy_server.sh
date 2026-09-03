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
command -v sha256sum >/dev/null

# A normal Git checkout may leave an LFS pointer in place of the 106 MB model.
# Pull it explicitly and fail before replacing the healthy container if either
# the binary or its release checksum is incomplete.
git lfs version >/dev/null
git lfs install --local >/dev/null
git lfs pull --include="$model_path" --exclude=""
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
curl --fail --silent --show-error http://127.0.0.1:8765/health
echo
