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
