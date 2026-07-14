"""Generate app/assets/model_stats.json (the app's 模型性能 page data).

The live app engine is **v4** (192ch×12blk, 8.72M, iter 4628) since 2026-07-14.
Architecture / evals / latency / progress below describe v4 (solid measured data).

The two training charts (loss curve, baseline-climb) need the model's own
metrics.jsonl. v4's training log was NOT bundled with its checkpoint, so if
`runs/v4/metrics.jsonl` is absent this script falls back to the v1 small-net
lineage for those charts and stamps a note. Drop `runs/v4/metrics.jsonl` in and
rerun to upgrade the charts to v4.

Rerun after new evals / training:
    .venv/bin/python scripts/gen_app_stats.py \
        --v4-vs-v1 71.9 --v4-vs-baseline 98.4 \
        --latency '{"0": 44, "32": 387, "128": 1470}'
"""
from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "assets" / "model_stats.json"

# vs pgx baseline 里程碑爬升 — v1 小網路世系（來源: runs/v1/watch.log）
V1_BASELINE_CLIMB = [
    (137, 2.3), (217, 0.8), (296, 7.8), (377, 10.9),
    (457, 27.3), (536, 49.2), (616, 64.1), (4681, 93.8),
]


def main():
    p = argparse.ArgumentParser()
    # --- live model (v4) identity ---
    p.add_argument("--channels", type=int, default=192)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--params", default="8.72M")
    p.add_argument("--iter", type=int, default=4628)
    # --- fresh eval numbers (本機 32 sims) ---
    p.add_argument("--v4-vs-v1", type=float, default=71.9,
                   help="v4 vs v1 winrate %% (both 32 sims)")
    p.add_argument("--v4-vs-baseline", type=float, default=98.4,
                   help="v4 vs pgx baseline winrate %% (v4 32 sims)")
    # --- per-move latency (ms), measured against the live server ---
    p.add_argument("--latency", default='{"0": 44, "32": 387, "128": 1470}',
                   help='JSON like {"0": 44, "32": 387, "128": 1470}')
    # --- training charts source; v4 preferred, v1 fallback ---
    p.add_argument("--metrics-dir", default="runs/v4")
    p.add_argument("--fallback-metrics-dir", default="runs/v1")
    args = p.parse_args()

    metrics_path = ROOT / args.metrics_dir / "metrics.jsonl"
    charts_are_v4 = metrics_path.exists()
    if not charts_are_v4:
        metrics_path = ROOT / args.fallback_metrics_dir / "metrics.jsonl"
        print(f"WARNING: {args.metrics_dir}/metrics.jsonl 不存在，"
              f"訓練圖表改用 {args.fallback_metrics_dir}（v1 小網路世系）。"
              f"補上 v4 的 metrics.jsonl 後重跑即可升級。", flush=True)

    rows = [json.loads(l) for l in open(metrics_path)]
    step = max(1, len(rows) // 120)
    loss_curve = [
        {"iter": r["iter"], "policy": r["policy_loss"], "value": r["value_loss"]}
        for r in rows[::step]
    ]
    frames = rows[-1]["frames"]
    iter_s = sum(r["time"] for r in rows[-100:]) / 100

    if charts_are_v4:
        baseline_curve = [{"iter": args.iter, "win": args.v4_vs_baseline}]
        chart_note = ""
    else:
        baseline_curve = [{"iter": i, "win": w} for i, w in V1_BASELINE_CLIMB]
        chart_note = ("　下方兩張訓練圖為小網路 v1 世系的訓練歷程（v4 大網路的訓練 log "
                      "未隨 checkpoint 附上）；架構、評測、延遲則為 v4 實測。")

    lat = json.loads(args.latency)

    stats = {
        "architecture": [
            ["網路", "AZNet 雙頭（策略 82 維 / 價值 tanh 純量）"],
            ["主幹", f"{args.channels} 通道 × {args.blocks} 殘差塊，隔層 KataGo 式全局池化偏置"],
            ["正規化", "GroupNorm（batch=1 推理與訓練一致）"],
            ["參數量", f"{args.params}（v4 大網路；小網路 v1 為 2.63M）"],
            ["輸入", "9×9×17（pgx 圍棋觀測: 8 步歷史 + 手番）"],
        ],
        "training": [
            ["演算法", "Gumbel-AlphaZero（DeepMind 2022）— 根節點 Gumbel 選擇，32 次模擬、至多 16 候選"],
            ["硬體", "3 × NVIDIA H100（JAX pmap，自對弈全程 GPU 向量化）"],
            ["自對弈", f"每迭代 1024 局並行 ≈ {frames:,} 局面/迭代"],
            ["優化器", "AdamW + warmup / cosine（lr 1e-3, wd 1e-4）"],
            ["進度", f"v4 已練 {args.iter:,} 迭代（大網路 anchor 世系最強權重）"],
            ["規則", "Tromp-Taylor（中國規則計分）、貼目 7.5、禁全同型"],
        ],
        "evals": [
            {"opponent": "v1 小網路 (iter 4681)",
             "detail": "128 局 · 雙方 32 sims（本機頭對頭）",
             "winrate": args.v4_vs_v1},
            {"opponent": "pgx AlphaZero baseline",
             "detail": f"iter {args.iter} · 128 局 · 32 sims（本機重測）",
             "winrate": args.v4_vs_baseline},
            {"opponent": "隨機落子", "detail": "純策略 · 空盤即壓制", "winrate": 100},
        ],
        "evals_note": ("v4（192ch×12blk, 8.72M）為 2026-07 起的線上引擎，"
                       "對前代小網路 v1 勝率 " + f"{args.v4_vs_v1:.0f}%、"
                       "對 pgx 官方 AlphaZero baseline（約強業餘）" + f"{args.v4_vs_baseline:.0f}%。"
                       + chart_note),
        "latency": [
            ["直覺（0 sims）", f"~{lat['0']} ms — 純策略網路一次前向"],
            ["均衡（32 sims）", f"~{lat['32']} ms — 與訓練同規格的 Gumbel 搜索"],
            ["深思（128 sims）", f"~{lat['128']} ms — 4 倍搜索深度"],
        ],
        "iters_logged": rows[-1]["iter"],
        "loss_curve": loss_curve,
        "baseline_curve": baseline_curve,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}  (charts source: {'v4' if charts_are_v4 else 'v1 fallback'})")


if __name__ == "__main__":
    main()
