import unittest

from scripts.smoke_test_server import (
    SmokeError,
    _validate_game,
    _validate_health,
    _validate_queue,
)


def queue_payload():
    return {
        "workers": 4,
        "active": 0,
        "queued": 0,
        "queued_standard": 0,
        "queued_premium": 0,
        "queue_capacity": 64,
        "average_job_seconds": 1.25,
        "estimated_tail_wait_seconds": 0,
    }


def game_payload():
    return {
        "game_id": "012345abcdef",
        "board": [0] * 360 + [1],
        "size": 19,
        "to_move": "white",
        "human_color": "white",
        "moves": 1,
        "history": [360],
        "last_move": 360,
        "ai_move": 360,
        "legal": [1] * 362,
        "black_winrate": 0.51,
        "winrates": [0.5, 0.51],
        "captures": {"black": 0, "white": 0},
        "game_over": False,
        "result": None,
        "komi": 7.5,
        "handicap": 0,
        "setup_plies": 0,
    }


class ServerSmokeValidationTest(unittest.TestCase):
    def test_accepts_expected_health_queue_and_game(self):
        health = {
            "ok": True,
            "model": "gozero go_19x19 192ch x 12blk",
            "iteration": 1000,
            "board_size": 19,
            "queue": queue_payload(),
        }
        _validate_health(
            health,
            expected_model="gozero go_19x19 192ch x 12blk",
            expected_iteration=1000,
            expected_board_size=19,
        )
        _validate_game(
            game_payload(),
            expected_board_size=19,
            expected_human_color="white",
            require_ai_move=True,
        )

    def test_rejects_wrong_board_size(self):
        game = game_payload()
        game["size"] = 9
        with self.assertRaisesRegex(SmokeError, "board size"):
            _validate_game(
                game,
                expected_board_size=19,
                expected_human_color="white",
                require_ai_move=True,
            )

    def test_rejects_missing_ai_move(self):
        game = game_payload()
        game["ai_move"] = None
        with self.assertRaisesRegex(SmokeError, "AI did not produce"):
            _validate_game(
                game,
                expected_board_size=19,
                expected_human_color="white",
                require_ai_move=True,
            )

    def test_rejects_inconsistent_queue_lanes(self):
        queue = queue_payload()
        queue["queued"] = 2
        with self.assertRaisesRegex(SmokeError, "lane counts"):
            _validate_queue(queue)


if __name__ == "__main__":
    unittest.main()
