#!/usr/bin/env bash
# Run on the inference host after the repository has been fast-forwarded.
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
deploy_timeout="${GOZERO_DEPLOY_TIMEOUT:-300}"
compose=(docker compose --env-file /dev/null)

cd "$repo_dir"
command -v docker >/dev/null
command -v curl >/dev/null

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
