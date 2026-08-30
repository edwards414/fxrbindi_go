#!/bin/bash
# 玄石 iOS：build IPA 並上傳 App Store Connect。
#
# 認證走 App Store Connect API 金鑰（.env 的 ASC_KEY_ID / ASC_ISSUER_ID / ASC_KEY_PATH，
# 金鑰本體在 ~/.appstoreconnect/private_keys/，不進版控）。不依賴 Xcode 有沒有登入帳號——
# 2026-08-12 就是因為鑰匙圈裡沒有 Apple Distribution 憑證、Xcode 也沒有帳號，
# 舊流程整個掛掉。Admin 權限的金鑰能讓 xcodebuild 自動建立所需的憑證與描述檔。
#
# 匯出與上傳刻意拆成兩步：ExportOptions 用 destination=export 先產出本機 IPA，
# 再用 altool 上傳。destination=upload 會把簽章和上傳綁在一起，
# 失敗時只吐一句 "The given data was not valid JSON"，完全無從查起。
#
# build number 用 git commit 數，保證單調遞增。
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

# .env 第一行是 SSH 主機字串不是賦值，直接 source 會噴錯；只取 KEY=VALUE 的行。
eval "$(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$REPO/.env")"
: "${ASC_KEY_ID:?.env 缺 ASC_KEY_ID}"
: "${ASC_ISSUER_ID:?.env 缺 ASC_ISSUER_ID}"
: "${ASC_KEY_PATH:?.env 缺 ASC_KEY_PATH}"
[ -f "$ASC_KEY_PATH" ] || { echo "找不到金鑰 $ASC_KEY_PATH" >&2; exit 1; }

BUILD_NUM=$(cd "$REPO" && git rev-list --count HEAD)

# 每一步都各自判斷成敗。舊版把整段包在 `if { ... }` 裡，而 `set -e` 在 if 的
# 條件式內不生效，xcodebuild 失敗後還是會跑到最後那句 echo，於是 log 上寫著
# 「上傳完成」、通知也顯示成功——實際上什麼都沒送出去。
fail() {
  echo "=== $(date '+%F %T') build $BUILD_NUM 失敗於：$1 ===" >> "$LOG"
  notify "build $BUILD_NUM 失敗（$1），詳見 ~/Library/Logs/xuanshi-release.log"
  exit 1
}

echo "=== $(date '+%F %T') build $BUILD_NUM 開始 ===" >> "$LOG"

# 一定要在 iCloud 同步範圍外建置。這個 repo 在 ~/Desktop 底下，而 macOS 的
# 「桌面與文件」同步是開的，file provider 會不斷幫檔案蓋上 com.apple.FinderInfo，
# codesign 於是拒簽：
#   objective_c.framework: resource fork, Finder information, or similar detritus not allowed
# 而且清不掉——framework 在建置中重新產生後又馬上被蓋回去，xattr -cr 只是徒勞
# （2026-08-12 實測：先清再 build 一樣失敗）。把 app/ 複製到 /private/tmp 建置則 100% 成功。
# build/ 不複製，讓 WORK 保留自己的產物以支援增量建置。
WORK=/private/tmp/xuanshi-build
mkdir -p "$WORK"
rsync -a --delete --exclude build --exclude .dart_tool --exclude ephemeral \
  --exclude previews --exclude screenshots \
  --exclude macos --exclude windows --exclude linux --exclude android --exclude web \
  "$REPO/app/" "$WORK/app/" >> "$LOG" 2>&1 || fail "rsync 到建置目錄"

cd "$WORK/app"
EXPORT_DIR=build/ios/upload
EXPORT_OPTS=$(mktemp -t ExportOptionsLocal).plist
sed 's|<string>upload</string>|<string>export</string>|' ios/ExportOptions.plist > "$EXPORT_OPTS"

# flutter build ipa 最後會自己試著匯出一次，用的是 ios/ExportOptions.plist
# （destination=upload、沒帶 API 金鑰），必定噴 "No Accounts"——但它仍回 0，
# 而我們只要它產出的 archive。所以不看退出碼，直接檢查 archive 在不在。
flutter build ipa --build-number="$BUILD_NUM" >> "$LOG" 2>&1 || true
ARCHIVE=build/ios/archive/Runner.xcarchive
[ -d "$ARCHIVE" ] || fail "flutter build ipa（沒產出 archive）"

rm -rf "$EXPORT_DIR"
xcodebuild -exportArchive \
  -archivePath "$ARCHIVE" \
  -exportOptionsPlist "$EXPORT_OPTS" \
  -exportPath "$EXPORT_DIR" \
  -allowProvisioningUpdates \
  -authenticationKeyPath "$ASC_KEY_PATH" \
  -authenticationKeyID "$ASC_KEY_ID" \
  -authenticationKeyIssuerID "$ASC_ISSUER_ID" >> "$LOG" 2>&1 || fail "exportArchive（簽章）"

IPA=$(find "$EXPORT_DIR" -maxdepth 1 -name '*.ipa' | head -1)
[ -n "$IPA" ] || fail "找不到匯出的 IPA"

# altool 成功時 stdout 有 "UPLOAD SUCCEEDED"；它的退出碼在部分版本不可靠，兩者都查。
UPLOAD_OUT=$(xcrun altool --upload-app -f "$IPA" -t ios \
  --apiKey "$ASC_KEY_ID" --apiIssuer "$ASC_ISSUER_ID" 2>&1) || true
echo "$UPLOAD_OUT" >> "$LOG"
grep -q "UPLOAD SUCCEEDED" <<<"$UPLOAD_OUT" || fail "altool 上傳"

echo "=== $(date '+%F %T') build $BUILD_NUM 上傳完成 ===" >> "$LOG"
notify "build $BUILD_NUM 已上傳，等 Apple 處理後即可測試"
