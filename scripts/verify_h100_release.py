"""Verify that a GoZero 19x19 release bundle is complete and self-consistent."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import pickle
import re
from typing import Any


EXPECTED_CONFIG = {
    "env_id": "go_19x19",
    "channels": 192,
    "blocks": 12,
    "compute_dtype": "bfloat16",
    "selfplay_batch": 64,
    "sims": 32,
    "max_considered": 16,
    "max_steps": 722,
    "pass_guard_ply": 100,
    "train_batch": 2048,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "warmup_iters": 20,
    "decay_iters": 1000,
    "eval_every": 25,
    "eval_batch": 32,
    "save_every": 25,
    "seed": 42,
}
RESULT_RE = re.compile(
    r"result:\s*(\d+)W\s+(\d+)L\s+(\d+)D\s*/\s*(\d+)\s+"
    r"\(([0-9]+(?:\.[0-9]+)?)% wins\)"
)


def _read_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


def _read_metrics(path: pathlib.Path, expected_iteration: int) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read metrics {path}: {exc}") from exc
    actual = [row.get("iter") for row in rows]
    expected = list(range(1, expected_iteration + 1))
    if actual != expected:
        raise ValueError(
            f"metrics iterations are not contiguous 1..{expected_iteration}; "
            f"got {len(rows)} rows ending at {actual[-1] if actual else 'none'}"
        )
    return rows


def _parse_result(path: pathlib.Path, expected_games: int) -> float:
    try:
        text = path.read_text()
    except OSError as exc:
        raise ValueError(f"cannot read evaluation {path}: {exc}") from exc
    matches = RESULT_RE.findall(text)
    if not matches:
        raise ValueError(f"evaluation result is missing from {path}")
    wins, losses, draws, games, reported = matches[-1]
    counts = tuple(map(int, (wins, losses, draws, games)))
    if counts[0] + counts[1] + counts[2] != counts[3]:
        raise ValueError(f"evaluation counts do not add up in {path}")
    if counts[3] != expected_games:
        raise ValueError(f"expected {expected_games} games in {path}, got {counts[3]}")
    calculated = round(100 * counts[0] / max(counts[3], 1), 1)
    winrate = float(reported)
    if abs(calculated - winrate) > 0.05:
        raise ValueError(
            f"reported win rate {winrate}% does not match {calculated}% in {path}"
        )
    return winrate


def _verify_sha256(model_path: pathlib.Path, checksum_path: pathlib.Path) -> str:
    try:
        fields = checksum_path.read_text().strip().split()
    except OSError as exc:
        raise ValueError(f"cannot read checksum {checksum_path}: {exc}") from exc
    if len(fields) != 2 or fields[1].lstrip("*") != model_path.name:
        raise ValueError(f"invalid checksum record in {checksum_path}")
    digest = hashlib.sha256()
    try:
        with model_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot read checkpoint {model_path}: {exc}") from exc
    actual = digest.hexdigest()
    if actual != fields[0].lower():
        raise ValueError(f"checkpoint SHA-256 mismatch: expected {fields[0]}, got {actual}")
    return actual


def _require_config(config: dict[str, Any], expected_iteration: int, source: str) -> None:
    expected = {**EXPECTED_CONFIG, "iters": expected_iteration}
    wrong = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if wrong:
        raise ValueError(f"unexpected training config in {source}: {wrong}")


def _verify_stats(
    stats: dict[str, Any],
    expected_iteration: int,
    random_win: float,
    gnugo_win: float,
    latency: dict[str, int],
) -> None:
    if stats.get("iters_logged") != expected_iteration:
        raise ValueError("App stats iteration does not match the release")
    loss_curve = stats.get("loss_curve") or []
    if not loss_curve or loss_curve[-1].get("iter") != expected_iteration:
        raise ValueError("App loss curve does not end at the release iteration")
    evals = {
        entry.get("opponent"): float(entry.get("winrate"))
        for entry in stats.get("evals", [])
    }
    expected_evals = {
        "隨機合法落子": random_win,
        "GNU Go level 10": gnugo_win,
    }
    if evals != expected_evals:
        raise ValueError(f"App evaluation stats do not match: {evals}")
    latency_rows = stats.get("latency") or []
    if len(latency_rows) != 3:
        raise ValueError("App latency table must contain 0, 32 and 128 simulations")
    for row, sims in zip(latency_rows, ("0", "32", "128"), strict=True):
        if f"~{latency[sims]} ms" not in row[1]:
            raise ValueError(f"App latency for {sims} simulations does not match")


def verify_release_bundle(
    run_dir: pathlib.Path,
    stats_path: pathlib.Path,
    *,
    expected_iteration: int = 1000,
) -> dict[str, Any]:
    model_path = run_dir / "latest.pkl"
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    checksum_path = run_dir / "latest.pkl.sha256"
    marker_path = run_dir / "release-ready.txt"

    metrics = _read_metrics(metrics_path, expected_iteration)
    config = _read_json(config_path)
    _require_config(config, expected_iteration, str(config_path))

    try:
        with model_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(f"cannot load checkpoint {model_path}: {exc}") from exc
    if checkpoint.get("iteration") != expected_iteration:
        raise ValueError(
            f"checkpoint iteration is {checkpoint.get('iteration')}, "
            f"expected {expected_iteration}"
        )
    checkpoint_config = checkpoint.get("config") or {}
    _require_config(checkpoint_config, expected_iteration, str(model_path))
    if any(checkpoint_config.get(key) != value for key, value in config.items()):
        raise ValueError("checkpoint config does not match config.json")

    sha256 = _verify_sha256(model_path, checksum_path)
    random_win = _parse_result(run_dir / "eval-random.txt", 256)
    gnugo_win = _parse_result(run_dir / "eval-gnugo.txt", 20)
    if random_win < 95.0:
        raise ValueError(f"random win rate is below release gate: {random_win}%")
    if gnugo_win < 50.0:
        raise ValueError(f"GNU Go win rate is below release gate: {gnugo_win}%")

    latency = _read_json(run_dir / "benchmark-latency.json")
    if set(latency) != {"0", "32", "128"}:
        raise ValueError(f"unexpected benchmark keys: {sorted(latency)}")
    if any(not isinstance(value, int) or value <= 0 for value in latency.values()):
        raise ValueError(f"invalid benchmark latency: {latency}")

    stats = _read_json(stats_path)
    _verify_stats(stats, expected_iteration, random_win, gnugo_win, latency)

    try:
        marker = marker_path.read_text()
    except OSError as exc:
        raise ValueError(f"cannot read release marker {marker_path}: {exc}") from exc
    marker_values = {
        key: value
        for line in marker.splitlines()
        if "=" in line
        for key, value in (line.split("=", 1),)
    }
    try:
        marker_iteration = int(marker_values["iteration"])
        marker_random = float(marker_values["random_win"])
        marker_gnugo = float(marker_values["gnugo_win"])
        marker_latency = json.loads(marker_values["latency"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"release marker is incomplete or invalid: {exc}") from exc
    if marker_iteration != expected_iteration:
        raise ValueError(f"release marker iteration is {marker_iteration}")
    if marker_random != random_win or marker_gnugo != gnugo_win:
        raise ValueError("release marker win rates do not match evaluations")
    if marker_latency != latency:
        raise ValueError("release marker latency does not match benchmark")

    return {
        "iteration": expected_iteration,
        "metrics_rows": len(metrics),
        "checkpoint_bytes": model_path.stat().st_size,
        "sha256": sha256,
        "random_win": random_win,
        "gnugo_win": gnugo_win,
        "latency_ms": latency,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=pathlib.Path, default="runs/v5_19x19")
    parser.add_argument(
        "--stats", type=pathlib.Path, default="app/assets/model_stats.json"
    )
    parser.add_argument("--expected-iteration", type=int, default=1000)
    args = parser.parse_args()
    try:
        result = verify_release_bundle(
            args.run_dir,
            args.stats,
            expected_iteration=args.expected_iteration,
        )
    except ValueError as exc:
        raise SystemExit(f"release verification failed: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
