# App Store 上架文案（玄石 GoZero）

App ID `6791030701` · Bundle `dev.gozero.gozeroGo` · 主要語言 zh-Hant

字數已對照 App Store Connect 上限驗過（見每節標題後的數字）。
ASC 超過上限會直接擋著不讓存，所以改字後請重新數。

---

## 繁體中文（zh-Hant）

### App 名稱 — 12/30

```
玄石 — 十九路圍棋對弈
```

> ASC 現在填的是 `GO AI APP`，送審前要改掉。名稱一旦上架就牽動搜尋排名，
> 不要頻繁更動。

### 副標題 — 14/30

```
與自我對弈學成的 AI 手談
```

### 關鍵字 — 60/100

```
棋類,棋譜,復盤,勝率,人工智慧,益智,棋盤遊戲,單機,策略,黑白棋,深度學習,弈棋,定石,死活,打劫
```

> 逗號分隔、**不要加空格**（空格會被算進 100 字上限）。
> 名稱與副標題裡出現過的詞（圍棋／十九路／對弈／手談／AI）Apple 已自動索引，
> 這裡刻意不重複，把額度留給沒用過的詞。

### 推廣文字 — 48/170

```
9／19 路棋盤自由切換，三段棋力、即時勝率曲線、可自訂貼目與讓子。AI 完全從零自我對弈學成。
```

> 推廣文字不必送審就能改，之後想換活動訊息隨時可換。

### 描述 — 596/4000

```
9 路快速對弈與完整 19 路棋盤自由切換。對手是一個靠自我對弈學會下棋的 AI——它沒有讀過任何一局人類棋譜，所有棋感都是自己下出來的。

■ 三段棋力
輕靈、均衡、深思。三檔用的是同一個神經網路，差別在每一手思考的深度：輕靈憑直覺落子，均衡搜索 32 次，深思搜索 128 次。從入門到夠你認真下，挑一個合手的。

■ 看得見 AI 怎麼想
每下一手，都會顯示 AI 對當前局面的勝率判斷，以及整局的勝率曲線。哪一手是勝負手、哪一步走壞了，攤開來一目了然。

■ 規則隨你調
貼目與讓子都能自訂，讓子最多四子。採中國規則（Tromp-Taylor 計點），預設貼目 7.5，禁全同型。

■ 對局留得住
每一局都會存進「對戰紀錄」，可以回頭逐手復盤。棋譜只存在你自己的裝置上，不會上傳、不會同步，刪除 App 就一併消失。

■ 這隻 AI 有多強
App 內附「模型性能」頁，收錄訓練曲線、網路架構與實測勝率——不是行銷數字，是訓練過程中真實跑出來的評測結果，你可以自己看。

■ 乾淨
不必註冊、不必登入、沒有廣告、沒有內購、沒有任何追蹤或分析工具。不蒐集任何個人資料。

■ 關於連線
AI 的運算在伺服器端進行，所以對弈時需要網路連線。沒有網路時，「對戰紀錄」與「模型性能」仍可正常瀏覽。

隱私政策：https://edwards414.github.io/fxrbindi_go/privacy.html
```

> 「關於連線」那段刻意寫明。App 需要連線才能對弈是事實，寫在描述裡比讓
> 使用者裝完才發現好，也能減少一星評價。

---

## 英文（en-US，選配）

主要語言是 zh-Hant，不加英文也送得出去；加了能讓非中文區的人搜得到。

### App 名稱 — 18/30

```
XuanShi — 19×19 Go
```

### 副標題 — 27/30

```
Go against a self-taught AI
```

### 關鍵字 — 79/100

```
go,baduk,weiqi,board game,19x19,strategy,puzzle,ai opponent,game record,winrate
```

### 推廣文字 — 146/170

```
Three strength levels, live win-rate graph, adjustable komi and handicap. The AI learned entirely from self-play — it has never seen a human game.
```

### 描述 — 1244/4000

```
A full 19×19 board. Your opponent is an AI that learned Go through self-play — it has never studied a single human game.

■ Three strength levels
Intuition, Balanced, and Deep. All three share the same neural network; what changes is how far ahead it looks — pure intuition, 32 searches, or 128 searches per move.

■ See what the AI is thinking
After every move you get its evaluation of the position, plus a win-rate curve for the whole game. Spotting the move that decided it becomes easy.

■ Your rules
Adjustable komi and handicap (up to four stones). Chinese rules (Tromp-Taylor area scoring), default komi 7.5, positional superko.

■ Games worth keeping
Every game is saved for move-by-move review. Records stay on your device — never uploaded, never synced.

■ How strong is it?
The built-in performance page shows the training curves, network architecture, and measured results from the actual training run.

■ Clean
No account, no sign-in, no ads, no in-app purchases, no tracking or analytics. No personal data collected.

■ Connectivity
The AI runs on a server, so playing requires an internet connection. Your game records and the performance page work offline.

Privacy policy: https://edwards414.github.io/fxrbindi_go/privacy.html
```

---

## App 審查資訊 → 備註（英文，貼給審查員看）

這一欄空著是白白提高拒件率。這個 App 的架構有兩個審查員一定會注意到的點
（沒有登入機制、完全依賴外部伺服器），先講清楚比事後回信快兩週。

```
No demo account is needed — the app has no accounts, no sign-in, and no
user-generated content.

How it works: the Go AI runs on our own inference server, not on the device.
On launch the app calls GET https://go.fxrbindi.com/health; each move is a
POST to the same host over HTTPS. The server is monitored and expected to be
available throughout review. If you ever see "連不上對弈引擎" (cannot reach
the engine) on the home screen, please tap the message to retry, or tap
"開始對弈" (Start a game) which retries the connection — and feel free to
contact us so we can confirm the service is up.

The app works without a network connection for everything except playing:
"對戰紀錄" (Game records) and "模型性能" (Model performance) read local data
and remain fully usable offline.

Data: only the move sequence, komi, handicap and chosen strength level are
sent to the server, along with a random per-game id. No personal data, no
device identifiers, and no IP logging. Game records are stored only in the
app's Documents directory on device.

The app is Traditional Chinese only. Main flow: home screen → "開始對弈"
(Start a game) → choose colour/strength/komi/handicap → "開始對弈" → tap an
intersection to place a stone. "虛手" = pass, "悔棋" = undo, "認輸" = resign.
```

---

## 其他 ASC 欄位

| 欄位 | 值 |
|---|---|
| 隱私政策 URL | `https://edwards414.github.io/fxrbindi_go/privacy.html` |
| 支援 URL | `https://edwards414.github.io/fxrbindi_go/support.html` |
| 行銷 URL（選填） | `https://edwards414.github.io/fxrbindi_go/` |
| 主分類 | 遊戲 → 棋盤遊戲 |
| 次分類 | 遊戲 → 益智解謎 |
| 年齡分級 | 4+（ASC 已是 FOUR_PLUS） |
| 版權 | `2026 Li Chang Yao` |
| 銷售範圍 | 排除歐盟 27 國（見下） |
| 截圖 6.9" | `app/screenshots/ios-6.9/` 五張，1320×2868 |

### App 隱私問卷的答案

全部可以答「未收集資料（Data Not Collected）」。伺服器每局只存
level／執子顏色／棋步序列／認輸旗標，不記 IP、不寫存取日誌；棋譜只在裝置本機。
沒有任何廣告或分析 SDK，所以追蹤（Tracking）一律答「否」，也不需要
App Tracking Transparency。

### 為什麼排除歐盟

EU DSA 自 2025-02-17 起要求申報交易者身分，個人開發者必須提供
**會公開顯示在 App Store 產品頁上**的地址與電話。先排除歐盟 27 國即可免除，
日後想開放再申請郵政信箱補申報。
