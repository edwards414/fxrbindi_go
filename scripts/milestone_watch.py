"""Milestone watcher for the gozero training run.

Every CYCLE seconds:
  1. Health check: alert by email if training crashed (Traceback in train.log)
     or stalled (no new metrics line for STALL_H hours).
  2. Evaluation ladder on the latest checkpoint (GPU EVAL_GPU):
        L1  vs random          (raw policy, 256 games)  > 80% wins
        L2  vs pgx baseline    (32 sims,   128 games)   > 50% wins
        L3  vs GNU Go level 10 (128 sims,   12 games)   > 50% wins
     Each level only runs after the previous one has been achieved.
     First time a level is achieved -> milestone email.

State (fired milestones, alerts) persists in <run_dir>/milestones.json.
Run:  nohup python scripts/milestone_watch.py --run-dir runs/v1 >> runs/v1/watch.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/go_ai")
from scripts.send_mail import send  # noqa: E402

CYCLE = 30 * 60
STALL_H = 2.0
EVAL_GPU = "4"
GNUGO = ("gnugo --mode gtp --boardsize 9 --komi 7.5 --chinese-rules --level 10 "
         "--play-out-aftermath --capture-all-dead")

LADDER = [
    ("random", 80.0, ["--sims", "0", "--games", "256", "--vs-random"]),
    ("pgx_baseline", 50.0, ["--sims", "32", "--games", "128", "--vs-baseline"]),
    ("gnugo_lv10", 50.0, ["--sims", "128", "--games", "6", "--vs-gtp", GNUGO]),
]


def mem_headroom_gb() -> float:
    try:
        cur = int(Path("/sys/fs/cgroup/memory.current").read_text())
        limit = int(Path("/sys/fs/cgroup/memory.max").read_text())
        return (limit - cur) / 2**30
    except Exception:
        return 99.0


def run_eval(ckpt: str, extra: list) -> float | None:
    # Evals run on CPU: a second CUDA process next to training OOM-killed the
    # run at iter 940 (back when the mem cgroup cap was 8GB; it is 64GB now).
    # The container's cpu.max quota was raised 4 -> 32 cores, so gnugo (single-
    # threaded per game) no longer starves: pin to the 32-core quota and give
    # our CPU policy 8 BLAS threads, leaving the rest uncontended for gnugo.
    # Thread caps are still needed so JAX doesn't spawn 224 threads (all cores
    # are visible) and thrash the 32-core quota.
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
               XLA_FLAGS="--xla_cpu_multi_thread_eigen=false")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    cmd = ["taskset", "-c", "0-31",
           "python", "-m", "gozero.evaluate", "--ckpt", ckpt] + extra
    try:
        # gnugo lv10 with aftermath cleanup can take 10-20 min per game
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                             cwd="/home/go_ai", env=env).stdout
        m = re.search(r"\(([\d.]+)% wins\)", out)
        return float(m.group(1)) if m else None
    except Exception as e:
        print(f"eval failed: {e}", flush=True)
        return None


def last_metrics(run_dir: Path, n=3):
    f = run_dir / "metrics.jsonl"
    if not f.exists():
        return []
    lines = f.read_text().strip().splitlines()
    return [json.loads(x) for x in lines[-n:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/v1")
    ap.add_argument("--once", action="store_true", help="single cycle (dry-run)")
    args = ap.parse_args()
    run_dir = Path("/home/go_ai") / args.run_dir
    state_f = run_dir / "milestones.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {
        "achieved": [], "alerted_crash": False, "alerted_stall": False}

    while True:
        try:
            # --- health ---
            log = run_dir / "train.log"
            mets = last_metrics(run_dir)
            if log.exists() and not state["alerted_crash"]:
                tail = log.read_text()[-20000:]
                if "Traceback" in tail:
                    send("[gozero] 訓練 CRASH 警報",
                         f"train.log 出現 Traceback,請檢查。\n\n最後段落:\n{tail[-3000:]}")
                    state["alerted_crash"] = True
            if mets:
                age_h = (time.time() - (run_dir / "metrics.jsonl").stat().st_mtime) / 3600
                if age_h > STALL_H and not state["alerted_stall"]:
                    send("[gozero] 訓練停滯警報",
                         f"metrics.jsonl 已 {age_h:.1f} 小時沒有更新(最後 iter="
                         f"{mets[-1]['iter']})。可能 GPU 被佔用或程序卡死。")
                    state["alerted_stall"] = True
                elif age_h <= STALL_H:
                    state["alerted_stall"] = False

            # --- ladder ---
            ckpt = run_dir / "latest.pkl"
            if mem_headroom_gb() < 1.6:
                print("low memory headroom; skipping eval this cycle", flush=True)
            elif ckpt.exists():
                for name, thresh, extra in LADDER:
                    if name in state["achieved"]:
                        continue
                    win = run_eval(str(ckpt), extra)
                    print(f"iter={mets[-1]['iter'] if mets else '?'} eval {name}: {win}%",
                          flush=True)
                    if win is not None and win > thresh:
                        state["achieved"].append(name)
                        recent = "\n".join(json.dumps(m, ensure_ascii=False) for m in mets)
                        send(f"[gozero] 里程碑達成:{name} 勝率 {win:.1f}%",
                             f"模型 {args.run_dir} 首次超過 {name} 門檻 {thresh}%。\n\n"
                             f"最近指標:\n{recent}\n\n"
                             f"checkpoint: {ckpt}")
                    break  # one ladder level per cycle; climb next cycle
            state_f.write_text(json.dumps(state))
        except Exception as e:
            print(f"watch cycle error: {e}", flush=True)
        if args.once:
            break
        time.sleep(CYCLE)


if __name__ == "__main__":
    main()
