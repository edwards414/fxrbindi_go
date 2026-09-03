import hashlib
import json
import pathlib
import pickle
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
