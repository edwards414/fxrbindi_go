"""Combine independent GoZero evaluation shards into one verified result log."""
from __future__ import annotations

import argparse
import os
import pathlib
import re


RESULT_RE = re.compile(
    r"result:\s*(\d+)W\s+(\d+)L\s+(\d+)D\s*/\s*(\d+)\s+"
    r"\(([0-9]+(?:\.[0-9]+)?)% wins\)"
)


def aggregate_results(
    inputs: list[pathlib.Path], output: pathlib.Path, *, expected_games: int
) -> tuple[int, int, int]:
    if not inputs:
        raise ValueError("no evaluation shards were provided")
    totals = [0, 0, 0, 0]
    sections = []
    for path in inputs:
        try:
            content = path.read_text()
        except OSError as exc:
            raise ValueError(f"cannot read evaluation shard {path}: {exc}") from exc
        matches = RESULT_RE.findall(content)
        if not matches:
            raise ValueError(f"evaluation result is missing from shard {path}")
        wins, losses, draws, games, reported = matches[-1]
        counts = tuple(map(int, (wins, losses, draws, games)))
        if counts[0] + counts[1] + counts[2] != counts[3]:
            raise ValueError(f"evaluation counts do not add up in shard {path}")
        calculated = round(100 * counts[0] / max(counts[3], 1), 1)
        if abs(calculated - float(reported)) > 0.05:
            raise ValueError(f"reported win rate does not match counts in shard {path}")
        totals = [left + right for left, right in zip(totals, counts, strict=True)]
        sections.append(f"===== {path.name} =====\n{content.rstrip()}\n")
    if totals[3] != expected_games:
        raise ValueError(
            f"evaluation shards contain {totals[3]} games, expected {expected_games}"
        )
    wins, losses, draws, games = totals
    result = f"result: {wins}W {losses}L {draws}D / {games}  ({100*wins/games:.1f}% wins)"
    temporary = output.with_name(f".{output.name}.aggregate")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("\n".join(sections) + "\n===== aggregate =====\n" + result + "\n")
    os.replace(temporary, output)
    return wins, losses, draws


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--expected-games", type=int, required=True)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    args = parser.parse_args()
    try:
        wins, losses, draws = aggregate_results(
            args.inputs, args.output, expected_games=args.expected_games
        )
    except ValueError as exc:
        raise SystemExit(f"evaluation aggregation failed: {exc}") from exc
    games = wins + losses + draws
    print(f"result: {wins}W {losses}L {draws}D / {games}  ({100*wins/games:.1f}% wins)")


if __name__ == "__main__":
    main()
