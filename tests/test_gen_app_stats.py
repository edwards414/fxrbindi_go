import unittest

from scripts.gen_app_stats import device_summary


class ModelStatsDeviceSummaryTest(unittest.TestCase):
    def test_reports_dynamic_six_to_seven_gpu_run(self):
        summary = device_summary(
            [
                {"iter": 1, "frames": 323456},
                {"iter": 2, "frames": 277248},
            ],
            {"max_steps": 722, "selfplay_batch": 64},
        )
        self.assertEqual(summary["hardware"], "6–7")
        self.assertEqual(summary["latest_devices"], 6)
        self.assertEqual(summary["latest_games"], 384)
        self.assertEqual(summary["latest_frames"], 277248)

    def test_rejects_frames_that_do_not_match_training_config(self):
        with self.assertRaisesRegex(ValueError, "cannot be produced"):
            device_summary(
                [{"iter": 1, "frames": 277249}],
                {"max_steps": 722, "selfplay_batch": 64},
            )


if __name__ == "__main__":
    unittest.main()
