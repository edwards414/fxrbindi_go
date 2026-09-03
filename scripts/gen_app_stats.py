"""Generate the in-app model report for the live 19x19 checkpoint.

Evaluation results and latency are required command-line inputs so a release
cannot silently reuse the old 9x9 marketing numbers.

Example:
    .venv/bin/python scripts/gen_app_stats.py \
      --vs-random 100 --vs-gnugo 65 \
      --latency '{"0": 180, "32": 2100, "128": 7600}'
"""
from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "assets" / "model_stats.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channels", type=int, default=192)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--params", default="9.28M")
    p.add_argument("--iter", type=int, default=1000)
    p.add_argument("--metrics-dir", default="runs/v5_19x19")
    p.add_argument("--vs-random", type=float, required=True,
                   help="win rate percentage from the 256-game random evaluation")
    p.add_argument("--vs-gnugo", type=float, required=True,
                   help="win rate percentage from the 20-game GNU Go level 10 evaluation")
    p.add_argument("--latency", required=True,
                   help='measured server milliseconds, e.g. {"0":180,"32":2100,"128":7600}')
    args = p.parse_args()

    metrics_path = ROOT / args.metrics_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    if not rows or rows[-1]["iter"] != args.iter:
        raise SystemExit(
            f"metrics must end at iteration {args.iter}; got "
            f"{rows[-1]['iter'] if rows else 'no rows'}"
        )

    step = max(1, len(rows) // 120)
    loss_curve = [
        {"iter": r["iter"], "policy": r["policy_loss"], "value": r["value_loss"]}
        for r in rows[::step]
    ]
    if loss_curve[-1]["iter"] != rows[-1]["iter"]:
        r = rows[-1]
        loss_curve.append(
            {"iter": r["iter"], "policy": r["policy_loss"], "value": r["value_loss"]}
        )

    anchor_curve = [
        {"iter": r["iter"], "win": round(100 * r["win_vs_anchor"], 1)}
        for r in rows if "win_vs_anchor" in r
    ]
    lat = json.loads(args.latency)
    for key in ("0", "32", "128"):
        if key not in lat:
            raise SystemExit(f"latency is missing sims={key}")

    frames = rows[-1]["frames"]
    tail = rows[-min(100, len(rows)):]
    iter_s = sum(r["time"] for r in tail) / len(tail)
    stats = {
        "architecture": [
            ["網路", "AZNet 雙頭（策略 362 維 / 價值 tanh 純量）"],
            ["主幹", f"{args.channels} 通道 × {args.blocks} 殘差塊，隔層 KataGo 式全局池化偏置"],
            ["正規化", "GroupNorm（batch=1 推理與訓練一致）"],
            ["參數量", f"{args.params}（v5 19 路模型）"],
            ["輸入", "19×19×17（pgx 圍棋觀測：8 步歷史 + 手番）"],
        ],
        "training": [
            ["演算法", "Gumbel-AlphaZero — 根節點 Gumbel 選擇，32 次模擬、至多 16 候選"],
            ["硬體", "7 × NVIDIA H100 80GB（JAX pmap、BF16 Tensor Core）"],
            ["自對弈", f"每迭代 448 局並行，{frames:,} 個局面"],
            ["優化器", "AdamW + warmup / cosine（lr 1e-3, wd 1e-4）"],
            ["進度", f"v5 已練 {args.iter:,} 迭代，近 100 輪平均 {iter_s:.1f} 秒／輪"],
            ["規則", "19 路 Tromp-Taylor（中國規則計分）、貼目 7.5、禁全同型"],
        ],
        "evals": [
            {
                "opponent": "GNU Go level 10",
                "detail": "20 局 · 玄石 128 sims · 黑白各半",
                "winrate": args.vs_gnugo,
            },
            {
                "opponent": "隨機合法落子",
                "detail": "256 局 · 玄石 32 sims · 黑白各半",
                "winrate": args.vs_random,
            },
        ],
        "evals_note": (
            f"v5 是 192ch×12blk、9.28M 參數的完整 19 路模型，訓練 {args.iter:,} 輪。"
            "勝率皆來自本次 checkpoint 的實際對局，不沿用 9 路模型數據。"
        ),
        "latency_title": "推理延遲 · 每手（H100 單卡）",
        "latency": [
            ["直覺（0 sims）", f"~{lat['0']} ms — 純策略網路一次前向"],
            ["均衡（32 sims）", f"~{lat['32']} ms — 與訓練同規格的 Gumbel 搜索"],
            ["深思（128 sims）", f"~{lat['128']} ms — 4 倍搜索深度"],
        ],
        "iters_logged": rows[-1]["iter"],
        "loss_curve": loss_curve,
        "progress_title": "對訓練錨點勝率",
        "progress_curve": anchor_curve,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUT} from {metrics_path}")


if __name__ == "__main__":
    main()
