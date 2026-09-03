import json
import pickle
import tempfile
import unittest
from pathlib import Path

from gozero.anchor_resume import load_resume_anchor, resolve_anchor_iteration


class AnchorResumeTest(unittest.TestCase):
    def test_resolves_last_promoted_anchor(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            metrics_path = Path(temporary_dir) / "metrics.jsonl"
            rows = [
                {"iter": 350, "anchor_iter": 275, "win_vs_anchor": 0.7396},
                {
                    "iter": 375,
                    "anchor_iter": 275,
                    "win_vs_anchor": 0.8594,
                    "anchor_updated": True,
                },
                {"iter": 400, "anchor_iter": 375, "win_vs_anchor": 0.599},
                {"iter": 401, "loss": 2.7},
            ]
            metrics_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            self.assertEqual(resolve_anchor_iteration(metrics_path, 406), 375)

    def test_ignores_metrics_newer_than_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            metrics_path = Path(temporary_dir) / "metrics.jsonl"
            rows = [
                {
                    "iter": 375,
                    "anchor_iter": 275,
                    "anchor_updated": True,
                },
                {
                    "iter": 425,
                    "anchor_iter": 375,
                    "anchor_updated": True,
                },
            ]
            metrics_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            self.assertEqual(resolve_anchor_iteration(metrics_path, 400), 375)

    def test_loads_anchor_checkpoint_for_legacy_resume(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            (run_dir / "metrics.jsonl").write_text(
                json.dumps(
                    {
                        "iter": 375,
                        "anchor_iter": 275,
                        "anchor_updated": True,
                    }
                )
                + "\n"
            )
            with (run_dir / "ckpt_000375.pkl").open("wb") as handle:
                pickle.dump({"iteration": 375, "params": "anchor-params"}, handle)

            params, iteration = load_resume_anchor(
                {"iteration": 406, "params": "current-params"}, run_dir, 406
            )

            self.assertEqual((params, iteration), ("anchor-params", 375))

    def test_prefers_persisted_anchor_iteration(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            (run_dir / "metrics.jsonl").write_text("not valid json\n")
            with (run_dir / "ckpt_000375.pkl").open("wb") as handle:
                pickle.dump({"iteration": 375, "params": "saved-anchor"}, handle)

            params, iteration = load_resume_anchor(
                {
                    "iteration": 430,
                    "params": "current-params",
                    "anchor_iteration": 375,
                },
                run_dir,
                430,
            )

            self.assertEqual((params, iteration), ("saved-anchor", 375))


if __name__ == "__main__":
    unittest.main()
