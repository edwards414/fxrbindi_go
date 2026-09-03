import pathlib
import tempfile
import unittest

from scripts.aggregate_eval_results import aggregate_results
from scripts.verify_h100_release import _parse_result


class AggregateEvaluationResultsTest(unittest.TestCase):
    def test_combines_shards_into_twenty_game_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            first = root / "gpu0.txt"
            second = root / "gpu1.txt"
            output = root / "combined.txt"
            first.write_text(
                "game 1/4 as black: win\nresult: 3W 1L 0D / 4  (75.0% wins)\n"
            )
            second.write_text(
                "game 1/16 as black: win\nresult: 10W 5L 1D / 16  (62.5% wins)\n"
            )
            result = aggregate_results([first, second], output, expected_games=20)
            self.assertEqual(result, (13, 6, 1))
            self.assertEqual(_parse_result(output, 20), 65.0)
            text = output.read_text()
            self.assertIn("===== gpu0.txt =====", text)
            self.assertIn("===== aggregate =====", text)

    def test_rejects_wrong_total_game_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            shard = root / "gpu0.txt"
            shard.write_text("result: 3W 1L 0D / 4  (75.0% wins)\n")
            with self.assertRaisesRegex(ValueError, "expected 20"):
                aggregate_results([shard], root / "combined.txt", expected_games=20)

    def test_rejects_inconsistent_shard_percentage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            shard = root / "gpu0.txt"
            shard.write_text("result: 3W 1L 0D / 4  (50.0% wins)\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                aggregate_results([shard], root / "combined.txt", expected_games=4)


if __name__ == "__main__":
    unittest.main()
