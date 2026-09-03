"""Verify and atomically integrate a downloaded GoZero H100 release bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tarfile
import tempfile
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.verify_h100_release import verify_release_bundle


RUN_FILES = (
    "latest.pkl",
    "metrics.jsonl",
    "config.json",
    "eval-random.txt",
    "eval-gnugo.txt",
    "benchmark-latency.json",
    "latest.pkl.sha256",
    "release-ready.txt",
    "train.log",
    "finalize.log",
)
STATS_FILE = "app/assets/model_stats.json"
EXPECTED_MEMBERS = frozenset(
    {f"runs/v5_19x19/{name}" for name in RUN_FILES} | {STATS_FILE}
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_outer_checksum(
    bundle_path: pathlib.Path, checksum_path: pathlib.Path
) -> str:
    try:
        fields = checksum_path.read_text().strip().split()
    except OSError as exc:
        raise ValueError(f"cannot read bundle checksum {checksum_path}: {exc}") from exc
    if len(fields) != 2 or fields[1].lstrip("*") != bundle_path.name:
        raise ValueError(f"invalid bundle checksum record in {checksum_path}")
    actual = _sha256(bundle_path)
    if fields[0].lower() != actual:
        raise ValueError(
            f"bundle SHA-256 mismatch: expected {fields[0].lower()}, got {actual}"
        )
    return actual


def _extract_exact_bundle(bundle_path: pathlib.Path, destination: pathlib.Path) -> None:
    try:
        archive = tarfile.open(bundle_path, "r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"cannot open release bundle {bundle_path}: {exc}") from exc
    with archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("release bundle contains duplicate members")
        actual = set(names)
        if actual != EXPECTED_MEMBERS:
            missing = sorted(EXPECTED_MEMBERS - actual)
            extra = sorted(actual - EXPECTED_MEMBERS)
            raise ValueError(
                f"unexpected release bundle members; missing={missing}, extra={extra}"
            )
        for member in members:
            if not member.isfile():
                raise ValueError(f"release bundle member is not a regular file: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read release bundle member: {member.name}")
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _atomic_copy(source: pathlib.Path, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.h100-download")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def integrate_release(
    bundle_path: pathlib.Path,
    checksum_path: pathlib.Path,
    repo_root: pathlib.Path,
    *,
    expected_iteration: int = 1000,
) -> dict[str, Any]:
    bundle_path = bundle_path.resolve()
    checksum_path = checksum_path.resolve()
    repo_root = repo_root.resolve()
    outer_sha256 = _verify_outer_checksum(bundle_path, checksum_path)

    with tempfile.TemporaryDirectory(prefix="gozero19-release-") as tmp:
        extracted_root = pathlib.Path(tmp)
        _extract_exact_bundle(bundle_path, extracted_root)
        verified = verify_release_bundle(
            extracted_root / "runs" / "v5_19x19",
            extracted_root / STATS_FILE,
            expected_iteration=expected_iteration,
        )
        for name in RUN_FILES:
            relative = pathlib.Path("runs") / "v5_19x19" / name
            _atomic_copy(extracted_root / relative, repo_root / relative)
        _atomic_copy(extracted_root / STATS_FILE, repo_root / STATS_FILE)

    installed = verify_release_bundle(
        repo_root / "runs" / "v5_19x19",
        repo_root / STATS_FILE,
        expected_iteration=expected_iteration,
    )
    if installed != verified:
        raise ValueError("installed release differs from the verified download")
    return {"bundle_sha256": outer_sha256, **installed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=pathlib.Path)
    parser.add_argument("checksum", type=pathlib.Path)
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--expected-iteration", type=int, default=1000)
    args = parser.parse_args()
    try:
        result = integrate_release(
            args.bundle,
            args.checksum,
            args.repo_root,
            expected_iteration=args.expected_iteration,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"release integration failed: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
