#!/bin/bash
# App Store 截圖產生器。
#
# flutter drive + binding.takeScreenshot() 在 iOS 模擬器上會卡住不回，
# 所以改成：用 debug-only 的 autodemo 鉤子把 app 直接開到指定頁面
# （app/lib/home_page.dart 啟動時讀 Documents/autodemo.txt），
# 再用 simctl 截原生解析度的圖。
#
# 用法：scripts/gen_screenshots.sh [模擬器名稱] [輸出目錄]
#   預設 iPhone 17 Pro Max（6.9 吋 / 1320x2868，App Store Connect 的必要尺寸）
#
# 前置：app 必須是 debug build 且已安裝（autodemo 鉤子有 kDebugMode 保護，
#       release 版不會有這個行為）。
set -euo pipefail

DEVICE="${1:-iPhone 17 Pro Max}"
OUT="${2:-$(cd "$(dirname "$0")/.." && pwd)/app/screenshots/ios-6.9}"
BUNDLE=dev.gozero.gozeroGo

UDID=$(xcrun simctl list devices available \
  | grep -F "$DEVICE (" | head -1 | sed -E 's/.*\(([-0-9A-F]{36})\).*/\1/')
[ -n "$UDID" ] || { echo "找不到模擬器：$DEVICE" >&2; exit 1; }
echo "模擬器 $DEVICE ($UDID)"

xcrun simctl bootstatus "$UDID" -b >/dev/null 2>&1 || xcrun simctl boot "$UDID" || true
mkdir -p "$OUT"

DOCS=$(xcrun simctl get_app_container "$UDID" "$BUNDLE" data)/Documents
mkdir -p "$DOCS"

# name:mode:等待秒數（等待要蓋過引擎往返；深思一手約 1.6 秒）
SHOTS=(
  "01_home::4"
  "02_setup:setup:4"
  "03_game:game:12"
  "04_stats:stats:5"
  "05_history:history:4"
)

for entry in "${SHOTS[@]}"; do
  IFS=: read -r name mode wait <<<"$entry"
  xcrun simctl terminate "$UDID" "$BUNDLE" >/dev/null 2>&1 || true
  rm -f "$DOCS/autodemo.txt"
  [ -n "$mode" ] && printf '%s' "$mode" > "$DOCS/autodemo.txt"
  xcrun simctl launch "$UDID" "$BUNDLE" >/dev/null
  sleep "$wait"
  xcrun simctl io "$UDID" screenshot --type=png "$OUT/$name.png" >/dev/null 2>&1
  size=$(sips -g pixelWidth -g pixelHeight "$OUT/$name.png" \
         | awk '/pixel/{printf "%s ", $2}')
  echo "  ✓ $name.png  (${size%% })"
done

rm -f "$DOCS/autodemo.txt"
echo "完成，輸出於 $OUT"
