"""Exercise the deployed 19x19 inference queue and one real AI move."""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any


class SmokeError(ValueError):
    pass


def _request_json(
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    request_id: str | None = None,
    timeout: float = 15,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    body = None if payload is None else json.dumps(payload).encode()
    headers = {
        "Accept": "application/json",
        "User-Agent": "GoZeroDeploySmoke/1.0",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    if request_id is not None:
        headers["Idempotency-Key"] = request_id
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SmokeError(f"{request.method} {path} returned {exc.code}: {detail}") from exc
    except OSError as exc:
        raise SmokeError(f"{request.method} {path} failed: {exc}") from exc
    with response:
        raw = response.read()
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SmokeError(f"{request.method} {path} returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise SmokeError(f"{request.method} {path} did not return a JSON object")
        return response.status, decoded, {
            name.lower(): value for name, value in response.headers.items()
        }


def _validate_queue(queue: Any) -> None:
    if not isinstance(queue, dict):
        raise SmokeError("queue status is missing")
    required = (
        "workers",
        "active",
        "queued",
        "queued_standard",
        "queued_premium",
        "queue_capacity",
        "estimated_tail_wait_seconds",
    )
    wrong = {
        key: queue.get(key)
        for key in required
        if not isinstance(queue.get(key), int) or queue[key] < 0
    }
    if wrong:
        raise SmokeError(f"invalid queue counters: {wrong}")
    if queue["workers"] < 1 or queue["queue_capacity"] < 1:
        raise SmokeError(f"queue is not configured for work: {queue}")
    if queue["queued"] != queue["queued_standard"] + queue["queued_premium"]:
        raise SmokeError(f"queue lane counts do not add up: {queue}")


def _validate_health(
    health: dict[str, Any],
    *,
    expected_model: str,
    expected_iteration: int,
    expected_board_size: int,
) -> None:
    expected = {
        "ok": True,
        "model": expected_model,
        "iteration": expected_iteration,
        "board_size": expected_board_size,
    }
    wrong = {
        key: {"expected": value, "actual": health.get(key)}
        for key, value in expected.items()
        if health.get(key) != value
    }
    if wrong:
        raise SmokeError(f"deployed model identity mismatch: {wrong}")
    _validate_queue(health.get("queue"))


def _validate_game(
    game: dict[str, Any],
    *,
    expected_board_size: int,
    expected_human_color: str,
    require_ai_move: bool,
) -> None:
    area = expected_board_size * expected_board_size
    if game.get("size") != expected_board_size:
        raise SmokeError(f"game board size is {game.get('size')}")
    board = game.get("board")
    if not isinstance(board, list) or len(board) != area or not set(board) <= {0, 1, 2}:
        raise SmokeError("game board payload is invalid")
    legal = game.get("legal")
    if not isinstance(legal, list) or len(legal) != area + 1 or not set(legal) <= {0, 1}:
        raise SmokeError("game legal-action mask is invalid")
    game_id = game.get("game_id")
    if not isinstance(game_id, str) or not re.fullmatch(r"[0-9a-f]{12}", game_id):
        raise SmokeError(f"invalid game id: {game_id}")
    if game.get("human_color") != expected_human_color:
        raise SmokeError(f"wrong human color: {game.get('human_color')}")
    moves = game.get("moves")
    history = game.get("history")
    if not isinstance(moves, int) or moves < 0 or not isinstance(history, list):
        raise SmokeError("game history is invalid")
    if len(history) != moves or any(
        not isinstance(action, int) or not 0 <= action <= area for action in history
    ):
        raise SmokeError("game move count and history do not match")
    if require_ai_move:
        ai_move = game.get("ai_move")
        if not isinstance(ai_move, int) or not 0 <= ai_move <= area:
            raise SmokeError(f"AI did not produce a valid move: {ai_move}")
        if moves < 1 or history[-1] != ai_move:
            raise SmokeError("AI move is missing from game history")


def _wait_for_job(
    base_url: str,
    initial: dict[str, Any],
    *,
    deadline: float,
) -> dict[str, Any]:
    job_id = initial.get("job_id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise SmokeError(f"invalid job id: {job_id}")
    view = initial
    while True:
        status = view.get("status")
        if status == "completed":
            result = view.get("result")
            if not isinstance(result, dict):
                raise SmokeError("completed job has no result")
            return result
        if status == "failed":
            raise SmokeError(f"inference job failed: {view.get('error')}")
        if status not in {"queued", "running"}:
            raise SmokeError(f"invalid inference job status: {status}")
        if time.monotonic() >= deadline:
            raise SmokeError(f"inference job {job_id} did not complete before timeout")
        time.sleep(0.5)
        _, view, _ = _request_json(base_url, f"/jobs/{job_id}")


def smoke_test(
    base_url: str,
    *,
    expected_model: str,
    expected_iteration: int,
    expected_board_size: int,
    timeout: float,
) -> dict[str, Any]:
    status, health, _ = _request_json(base_url, "/health")
    if status != 200:
        raise SmokeError(f"health returned HTTP {status}")
    _validate_health(
        health,
        expected_model=expected_model,
        expected_iteration=expected_iteration,
        expected_board_size=expected_board_size,
    )

    request_id = f"deploy-smoke-{uuid.uuid4().hex}"
    status, initial, headers = _request_json(
        base_url,
        "/new",
        payload={
            "level": "easy",
            "human_color": "white",
            "komi": 7.5,
            "handicap": 0,
            "request_id": request_id,
        },
        request_id=request_id,
    )
    if status != 202:
        raise SmokeError(f"queued /new returned HTTP {status}")
    job_id = initial.get("job_id")
    if headers.get("location") != f"/jobs/{job_id}":
        raise SmokeError(f"queued /new returned wrong Location: {headers.get('location')}")
    game = _wait_for_job(base_url, initial, deadline=time.monotonic() + timeout)
    _validate_game(
        game,
        expected_board_size=expected_board_size,
        expected_human_color="white",
        require_ai_move=True,
    )

    game_id = game["game_id"]
    state_path = f"/state?game_id={urllib.parse.quote(game_id)}"
    status, state, _ = _request_json(base_url, state_path)
    if status != 200 or state.get("history") != game.get("history"):
        raise SmokeError("state resynchronization does not match the completed job")
    _validate_game(
        state,
        expected_board_size=expected_board_size,
        expected_human_color="white",
        require_ai_move=False,
    )

    status, queue, _ = _request_json(base_url, "/queue")
    if status != 200:
        raise SmokeError(f"queue returned HTTP {status}")
    _validate_queue(queue)

    status, resigned, _ = _request_json(
        base_url,
        "/resign",
        payload={"game_id": game_id},
    )
    if (
        status != 200
        or not resigned.get("game_over")
        or (resigned.get("result") or {}).get("reason") != "resign"
    ):
        raise SmokeError("could not finish the deployment smoke-test game")

    return {
        "model": health["model"],
        "iteration": health["iteration"],
        "board_size": health["board_size"],
        "game_id": game_id,
        "ai_move": game["ai_move"],
        "moves": game["moves"],
        "queue": queue,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--expected-model", default="gozero go_19x19 192ch x 12blk"
    )
    parser.add_argument("--expected-iteration", type=int, default=1000)
    parser.add_argument("--expected-board-size", type=int, default=19)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    try:
        result = smoke_test(
            args.base_url,
            expected_model=args.expected_model,
            expected_iteration=args.expected_iteration,
            expected_board_size=args.expected_board_size,
            timeout=args.timeout,
        )
    except SmokeError as exc:
        raise SystemExit(f"server smoke test failed: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
