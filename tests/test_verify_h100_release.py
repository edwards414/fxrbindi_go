import hashlib
import json
import pathlib
import pickle
import subprocess
import sys
import tarfile
import tempfile
import unittest

from scripts.integrate_h100_release import EXPECTED_MEMBERS, integrate_release
from scripts.verify_h100_release import EXPECTED_CONFIG, verify_release_bundle


class VerifyH100ReleaseTest(unittest.TestCase):
    def make_bundle(self, root: pathlib.Path, iterations: list[int] | None = None):
        run_dir = root / "runs" / "v5_19x19"
        run_dir.mkdir(parents=True)
        expected_iteration = 3
        config = {**EXPECTED_CONFIG, "iters": expected_iteration}
        config["run_dir"] = "/home/gozero19/runs/v5_19x19"
        config["resume"] = str(run_dir / "latest.pkl")
        config["init_from"] = None
        (run_dir / "config.json").write_text(json.dumps(config))

        rows = []
        for iteration in iterations or [1, 2, 3]:
            rows.append(
                {
                    "iter": iteration,
                    "time": 58.0,
                    "loss": 3.0,
                    "policy_loss": 2.4,
                    "value_loss": 0.6,
                    "frames": 323456,
                }
            )
        (run_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

        model_path = run_dir / "latest.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(
                {
                    "iteration": expected_iteration,
                    "config": config,
                    "params": {},
                    "opt_state": {},
                },
                handle,
            )
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        (run_dir / "latest.pkl.sha256").write_text(f"{digest}  latest.pkl\n")
        (run_dir / "eval-random.txt").write_text(
            "result: 256W 0L 0D / 256  (100.0% wins)\n"
        )
        (run_dir / "eval-gnugo.txt").write_text(
            "result: 13W 7L 0D / 20  (65.0% wins)\n"
        )
        latency = {"0": 10, "32": 250, "128": 950}
        (run_dir / "benchmark-latency.json").write_text(json.dumps(latency) + "\n")
        (run_dir / "release-ready.txt").write_text(
            "release ready at 2026-09-04T04:00:00Z\n"
            "iteration=3\n"
            "random_win=100.0\n"
            "gnugo_win=65.0\n"
            'latency={"0":10,"32":250,"128":950}\n'
        )
        (run_dir / "train.log").write_text("training complete\n")
        (run_dir / "finalize.log").write_text("evaluation complete\n")
        stats_path = root / "app" / "assets" / "model_stats.json"
        stats_path.parent.mkdir(parents=True)
        stats_path.write_text(
            json.dumps(
                {
                    "iters_logged": 3,
                    "loss_curve": [{"iter": 1}, {"iter": 3}],
                    "evals": [
                        {"opponent": "GNU Go level 10", "winrate": 65.0},
                        {"opponent": "隨機合法落子", "winrate": 100.0},
                    ],
                    "latency": [
                        ["直覺（0 sims）", "~10 ms"],
                        ["均衡（32 sims）", "~250 ms"],
                        ["深思（128 sims）", "~950 ms"],
                    ],
                }
            )
        )
        return run_dir, stats_path

    def make_archive(
        self,
        source_root: pathlib.Path,
        output_root: pathlib.Path,
        extra: pathlib.Path | None = None,
    ):
        bundle_path = output_root / "gozero19-v5-release.tar.gz"
        with tarfile.open(bundle_path, "w:gz") as archive:
            for name in sorted(EXPECTED_MEMBERS):
                archive.add(source_root / name, arcname=name)
            if extra is not None:
                archive.add(extra, arcname=extra.name)
        digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        checksum_path = output_root / "gozero19-v5-release.tar.gz.sha256"
        checksum_path.write_text(
            f"{digest}  runs/v5_19x19/{bundle_path.name}\n"
        )
        return bundle_path, checksum_path

    def test_accepts_complete_consistent_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, stats_path = self.make_bundle(pathlib.Path(tmp))
            result = verify_release_bundle(
                run_dir, stats_path, expected_iteration=3
            )
        self.assertEqual(result["iteration"], 3)
        self.assertEqual(result["metrics_rows"], 3)
        self.assertEqual(result["gnugo_win"], 65.0)

    def test_rejects_non_contiguous_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, stats_path = self.make_bundle(
                pathlib.Path(tmp), iterations=[1, 3]
            )
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                verify_release_bundle(
                    run_dir, stats_path, expected_iteration=3
                )

    def test_rejects_corrupt_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, stats_path = self.make_bundle(pathlib.Path(tmp))
            with (run_dir / "latest.pkl").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_release_bundle(
                    run_dir, stats_path, expected_iteration=3
                )

    def test_rejects_strength_below_release_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, stats_path = self.make_bundle(pathlib.Path(tmp))
            (run_dir / "eval-gnugo.txt").write_text(
                "result: 9W 11L 0D / 20  (45.0% wins)\n"
            )
            with self.assertRaisesRegex(ValueError, "below release gate"):
                verify_release_bundle(
                    run_dir, stats_path, expected_iteration=3
                )

    def test_integrates_exact_verified_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "source"
            destination = root / "destination"
            self.make_bundle(source)
            bundle_path, checksum_path = self.make_archive(source, root)
            result = integrate_release(
                bundle_path,
                checksum_path,
                destination,
                expected_iteration=3,
            )
            installed = destination / "runs" / "v5_19x19"
            self.assertEqual(result["iteration"], 3)
            self.assertTrue((installed / "latest.pkl").is_file())
            self.assertTrue((destination / "app/assets/model_stats.json").is_file())

    def test_rejects_archive_with_extra_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "source"
            self.make_bundle(source)
            extra = root / "unexpected.txt"
            extra.write_text("not part of the release\n")
            bundle_path, checksum_path = self.make_archive(source, root, extra)
            with self.assertRaisesRegex(ValueError, "unexpected release bundle members"):
                integrate_release(
                    bundle_path,
                    checksum_path,
                    root / "destination",
                    expected_iteration=3,
                )

    def test_rejects_checksum_record_for_different_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "source"
            self.make_bundle(source)
            bundle_path, checksum_path = self.make_archive(source, root)
            digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
            checksum_path.write_text(f"{digest}  runs/v5_19x19/wrong.tar.gz\n")
            with self.assertRaisesRegex(ValueError, "invalid bundle checksum"):
                integrate_release(
                    bundle_path,
                    checksum_path,
                    root / "destination",
                    expected_iteration=3,
                )

    def test_integration_cli_help_runs_from_repo_root(self):
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "scripts/integrate_h100_release.py", "--help"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--expected-iteration", completed.stdout)


if __name__ == "__main__":
    unittest.main()
