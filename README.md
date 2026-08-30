# gozero — Gumbel-AlphaZero Go on H100

自研圍棋 AI 訓練系統。整條管線(棋盤規則、MCTS、神經網路)都是 jitted JAX 程式碼,
自對弈完全在 GPU 上向量化執行,單卡同時下上千盤棋。

## 相對 AlphaGo Zero (2017) 的架構改進

| 元件 | AlphaGo Zero | 本系統 |
|---|---|---|
| 根節點搜索 | PUCT + 800 次模擬 | **Gumbel 根選擇(DeepMind 2022)+ 32 次模擬** — 無偏策略改進,搜索成本降 25 倍 |
| 網路全局視野 | 純 3x3 卷積堆疊 | **KataGo 式全局池化偏置**(隔層注入 mean+max 全盤特徵)|
| 正規化 | BatchNorm | GroupNorm(推理 batch=1 與訓練完全一致,多卡無需同步統計)|
| 優化器 | SGD + 手調階梯 | AdamW + warmup/cosine |
| 自對弈 | 分散式 CPU actor + TPU 推理 | 全 GPU 向量化(pgx 環境本身是 JAX 程式)|

## 檔案

- `gozero/net.py` — 策略+價值網路(全局池化殘差塊)
- `gozero/train.py` — 訓練主程式(pmap 多卡:自對弈 → 目標計算 → 一個 epoch 訓練)
- `gozero/mcts.py` — 推理用搜索(evaluate/gtp 共用)
- `gozero/evaluate.py` — 對戰評測:vs checkpoint / pgx baseline / random / 外部 GTP 引擎(GNU Go 等)
- `gozero/gtp.py` — GTP 協議接口(接 Sabaki/GoGui 圖形界面,也是之後 app 的引擎後端)
- `gozero/coords.py` — GTP ↔ pgx 座標轉換(pgx action 0 = 左上角,列優先)

## 訓練

```bash
cd /home/go_ai
CUDA_VISIBLE_DEVICES=0,3,4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python -m gozero.train --run-dir runs/v1 | tee -a runs/v1/train.log
```

指標寫入 `runs/v1/metrics.jsonl`(每迭代一行 JSON),checkpoint 存 `runs/v1/latest.pkl`
與每 25 迭代的 `ckpt_XXXXXX.pkl`。斷點續訓:`--resume runs/v1/latest.pkl`。

進度追蹤:每 10 迭代跟「凍結的舊自我(anchor)」打 384 盤,勝率 >85% 就把 anchor
換成當前模型 —— metrics 裡 `anchor_updated` 出現的頻率就是進步速度。

## 評測

```bash
# vs pgx 官方 AlphaZero baseline(強業餘水準)
python -m gozero.evaluate --ckpt runs/v1/latest.pkl --vs-baseline --games 256

# vs GNU Go level 10(--play-out-aftermath/--capture-all-dead 讓 gnugo 把死子
# 提乾淨再 pass,否則 Tromp-Taylor 會把死子算成活的,勝率會被高估)
python -m gozero.evaluate --ckpt runs/v1/latest.pkl --sims 256 --games 20 \
  --vs-gtp "gnugo --mode gtp --boardsize 9 --komi 7.5 --chinese-rules --level 10 --play-out-aftermath --capture-all-dead"

# 新舊 checkpoint 對戰(雙方都帶搜索)
python -m gozero.evaluate --ckpt new.pkl --vs-ckpt old.pkl --sims 32 --opp-sims 32
```

## iPhone App（Flutter，`app/`）

「玄石」— 9 路對弈 app（墨×原木×宣紙風格），內建模型性能頁。引擎跑在 Mac 上，
模擬器透過 localhost 連線:

```bash
# 1. 啟動引擎伺服器（首次啟動會 JIT 編譯三檔強度，約 40 秒）
JAX_PLATFORMS=cpu .venv/bin/python -m gozero.server --ckpt runs/v4/latest.pkl --port 8765

# 2. 跑 app（iPhone 17 模擬器）
cd app && flutter run -d "iPhone 17"
```

強度三檔對應 MCTS 模擬數 0 / 32 / 128。性能頁資料由
`scripts/gen_app_stats.py` 從 `runs/v1/metrics.jsonl` 與評測結果生成。

## Docker 啟動引擎伺服器

Docker 映像只包 HTTP inference server，不把 Flutter app 或 checkpoint 放進映像；
checkpoint 從本機掛載，對戰狀態寫到 Docker volume。

```bash
# 建 image
docker build -t gozero-server .

# 啟動 server
docker run --rm \
  -p 8765:8765 \
  -v "$PWD/runs/v4/latest.pkl:/models/latest.pkl:ro" \
  -v gozero-server-data:/data \
  gozero-server

# 或用 compose
docker compose up --build
```

健康檢查:

```bash
curl http://127.0.0.1:8765/health
```

### 對弈推理佇列

`/new`、`/move`、`/undo` 不會佔住 HTTP 連線等待推理，而是回傳 `202` 與
`job_id`。App 透過 `GET /jobs/<job_id>` 顯示排隊順位、預估時間並取得結果；
`GET /queue` 可查看 worker、執行中與排隊數量。

- `GOZERO_SEARCH_SLOTS`：同時執行推理的 worker 數，預設 4。
- `GOZERO_MAX_QUEUE`：尚未開始的任務上限，預設 64。
- `GOZERO_JOB_TTL`：完成結果保留秒數，預設 900。

新版 App 會為每個操作傳送 idempotency key，並以 `expected_moves` 保護落子與
悔棋，網路重送不會讓同一盤棋被重複修改。伺服器內部已保留 premium 加權通道，
但公開 API 不接受客戶端自報付費身分；必須接上可信的購買憑證驗證後才能啟用。
沒有 idempotency key 的舊版 App 仍會收到原本的同步 GameState 回應，升級伺服器
不會直接破壞已安裝的客戶端。

### Server CI/CD

推送影響引擎或排隊 App 協定的 commit 到 `main` 時，
`.github/workflows/server-ci-deploy.yml` 會執行 Python 測試、Flutter 分析與測試、
Docker image 建置；全部成功後才以 SSH fast-forward 正式主機並執行
`scripts/deploy_server.sh`。正式環境需建立以下 GitHub `production` environment secrets：

- `DEPLOY_HOST`
- `DEPLOY_USER`
- `DEPLOY_PATH`
- `DEPLOY_SSH_KEY`
- `DEPLOY_KNOWN_HOSTS`

部署腳本會忽略格式不屬於 Compose 的本機 `.env`、重建服務，並等待 `/health`
成功；逾時會輸出容器狀態與引擎 log，讓 workflow 明確失敗。

### iOS TestFlight CI/CD

推送 `app/**` 到 `main` 時，`.github/workflows/ios-testflight.yml` 會先執行
Flutter analyze/test，再於 GitHub macOS runner 匯入專用 App Store 簽章憑證、
下載 `IOS_APP_STORE` provisioning profile、archive、上傳 TestFlight，並在 Apple
處理完成後加入 `Internal Testers`。每次 workflow run/attempt 都有唯一的
`CFBundleVersion`，失敗重跑不會撞到已上傳的 build number。外部測試仍由
App Store Connect 人工送 Beta Review，不會因每次 commit 自動公開。

GitHub `testflight` environment 需要兩個 variables：

- `APPSTORE_ISSUER_ID`
- `APPSTORE_API_KEY_ID`

以及三個 secrets：

- `APPSTORE_API_PRIVATE_KEY`
- `APPSTORE_CERTIFICATES_FILE_BASE64`
- `APPSTORE_CERTIFICATES_PASSWORD`

## 通知與運維

- 里程碑監控:`nohup python scripts/milestone_watch.py --run-dir runs/v1 >> runs/v1/watch.log 2>&1 &`
  —— 每 30 分鐘檢查健康 + 爬評測階梯(random 80% → pgx baseline 50% → GNU Go lv10 50%),
  達標/當機/停滯都會寄 email(憑證在 `.secrets/gmail.env`,chmod 600)
- 斷點續訓:`scripts/restart_train.sh [GPUS] [RUN_DIR]`
- 手動寄信:`python scripts/send_mail.py --subject "..." < body.txt`

## 之後擴到 19x19

`--env-id go_19x19 --channels 192 --blocks 12 --max-steps 512` 即可;
其餘程式碼不變(座標/網路/搜索都以環境尺寸參數化)。

## 規則

Tromp-Taylor(中國規則計分),貼目 7.5,禁全同型(positional superko)—— 由 pgx 實作。
