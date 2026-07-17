#!/bin/bash
# 玄石 iOS：build IPA 並上傳 TestFlight。
# 認證走這台 Mac 上 Xcode 已登入的 Apple ID（ios/ExportOptions.plist 的 destination=upload），
# 不需要 App Store Connect API 金鑰。build number 用 git commit 數，保證單調遞增。
# 手動執行：scripts/release_ios.sh
# 自動觸發：.git/hooks/post-commit（main 分支、commit 動到 app/ 時，背景執行）
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="/tmp/xuanshi-release.lock"
LOG="$HOME/Library/Logs/xuanshi-release.log"

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "$(date '+%F %T') 已有上傳進行中，跳過（lock: $LOCK）" >> "$LOG"
  exit 0
fi
trap 'rmdir "$LOCK"' EXIT

notify() {
  osascript -e "display notification \"$1\" with title \"玄石 TestFlight\"" >/dev/null 2>&1 || true
}

cd "$REPO/app"
BUILD_NUM=$(git rev-list --count HEAD)

if {
  echo "=== $(date '+%F %T') build $BUILD_NUM 開始 ==="
  flutter build ipa --build-number="$BUILD_NUM"
  xcodebuild -exportArchive \
    -archivePath build/ios/archive/Runner.xcarchive \
    -exportOptionsPlist ios/ExportOptions.plist \
    -exportPath build/ios/upload \
    -allowProvisioningUpdates
  echo "=== $(date '+%F %T') build $BUILD_NUM 上傳完成 ==="
} >> "$LOG" 2>&1; then
  notify "build $BUILD_NUM 已上傳，等 Apple 處理後即可測試"
else
  notify "build $BUILD_NUM 失敗，詳見 ~/Library/Logs/xuanshi-release.log"
  exit 1
fi
