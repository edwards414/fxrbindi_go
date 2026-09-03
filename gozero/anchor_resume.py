"""Recover the frozen self-play evaluation anchor after a training restart."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def resolve_anchor_iteration(metrics_path: str | Path, start_iteration: int) -> int:
    """Return the anchor that was active after ``start_iteration``.

    Evaluation rows record the anchor used for that match.  A successful
    promotion takes effect after the match, so that row's own iteration becomes
    the new anchor.  Non-evaluation rows do not change the anchor.
    """

    anchor_iteration = start_iteration
    path = Path(metrics_path)
    if not path.exists():
        return anchor_iteration

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        iteration = int(row["iter"])
        if iteration > start_iteration:
            continue
        if "anchor_iter" not in row:
            continue
        if row.get("anchor_updated") is True:
            anchor_iteration = iteration
        else:
            anchor_iteration = int(row["anchor_iter"])
    return anchor_iteration


def load_resume_anchor(
    checkpoint: dict[str, Any],
    run_dir: str | Path,
    start_iteration: int,
) -> tuple[Any, int]:
    """Load the frozen anchor parameters for a resumed training run.

    New checkpoints persist ``anchor_iteration``.  Legacy checkpoints are
    reconstructed from metrics so an elastic GPU migration can safely resume
    runs that started before that metadata existed.
    """

    run_path = Path(run_dir)
    if "anchor_iteration" in checkpoint:
        anchor_iteration = int(checkpoint["anchor_iteration"])
    else:
        anchor_iteration = resolve_anchor_iteration(
            run_path / "metrics.jsonl", start_iteration
        )
    if anchor_iteration == start_iteration:
        return checkpoint["params"], anchor_iteration

    anchor_path = run_path / f"ckpt_{anchor_iteration:06d}.pkl"
    with anchor_path.open("rb") as handle:
        anchor_checkpoint = pickle.load(handle)
    actual_iteration = int(anchor_checkpoint.get("iteration", -1))
    if actual_iteration != anchor_iteration:
        raise ValueError(
            f"anchor checkpoint {anchor_path} contains iteration {actual_iteration}, "
            f"expected {anchor_iteration}"
        )
    return anchor_checkpoint["params"], anchor_iteration
