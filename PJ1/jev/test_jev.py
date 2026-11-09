"""Tests for the optional Jev search integration."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from jev import q9 as experiment
from jev.q9 import Q9JevEvaluator, _answer_score

from jev.q6 import (
    _TransientJevError,
    Q2JevDirectValueHeuristic,
    Q2JevNodeHeuristic,
    Q2JevPreScoreTieBreaker,
    Q2JevTieBreaker,
    Q6JevNodeHeuristic,
    Q6JevScoreHeuristic,
)


class _Walls:
    width = 5
    height = 5

    def __getitem__(self, x):
        return [
            True,
            x in (0, 4),
            x in (0, 4),
            x in (0, 4),
            True,
        ]


class _Problem:
    walls = _Walls()
    goal = (3, 3)

    def getStartState(self):
        return (1, 1)


class _CoinGrid:
    def __init__(self, coins):
        self._coins = tuple(coins)

    def count(self):
        return len(self._coins)

    def asList(self):
        return list(self._coins)

    def __hash__(self):
        return hash(self._coins)

    def __eq__(self, other):
        return isinstance(other, _CoinGrid) and self._coins == other._coins


class _CoinProblem:
    walls = _Walls()

    def getStartState(self):
        return ((1, 1), _CoinGrid(((2, 1), (3, 3))))


class _FakeClient:
    def __init__(self):
        self.calls = []

    def decide(self, state, questions):
        self.calls.append((state, questions))
        if next(iter(questions.values()))["type"] == "choice":
            return {
                question_id: {
                    "choice": next(iter(question["criteria"])),
                    "confidence": 0.8,
                }
                for question_id, question in questions.items()
            }
        return {
            question_id: {
                "score": float(index),
                "confidence": 0.9,
            }
            for index, question_id in enumerate(questions)
        }

    def close(self):
        pass


class _FailingClient:
    def __init__(self):
        self.calls = 0

    def decide(self, state, questions):
        self.calls += 1
        raise RuntimeError("temporary service failure")

    def close(self):
        pass


class _TransientThenSuccessfulClient(_FakeClient):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    def decide(self, state, questions):
        self.attempts += 1
        if self.attempts == 1:
            raise _TransientJevError("temporary timeout")
        return super().decide(state, questions)


class Q2JevTests(unittest.TestCase):
    def test_plateau_method_batches_and_caches_scores(self):
        client = _FakeClient()
        selector = Q2JevTieBreaker(
            client_factory=lambda: client,
            batch_size=8,
        )
        problem = _Problem()
        selector.begin_search(problem)
        candidates = [
            {"state": (1, 1), "g": 1, "h": 4, "f": 5, "order": 1},
            {"state": (1, 2), "g": 2, "h": 3, "f": 5, "order": 2},
        ]

        selected = selector.choose_from_plateau(candidates, problem)
        selected_again = selector.choose_from_plateau(candidates, problem)

        self.assertEqual(selected["state"], (1, 2))
        self.assertEqual(selected_again["state"], (1, 2))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(selector.statistics()["cache_hits"], 2)

    def test_node_method_calls_once_per_state(self):
        client = _FakeClient()
        heuristic = Q2JevNodeHeuristic(
            client_factory=lambda: client,
            batch_size=8,
        )
        problem = _Problem()
        heuristic.begin_search(problem)

        first = heuristic((1, 1), problem)
        repeated = heuristic((1, 1), problem)
        goal = heuristic((3, 3), problem)

        self.assertEqual(first, 4.0)
        self.assertEqual(repeated, 4.0)
        self.assertEqual(goal, 0.0)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(heuristic.statistics()["cache_hits"], 1)

    def test_direct_value_method_uses_integer_choice(self):
        client = _FakeClient()
        heuristic = Q2JevDirectValueHeuristic(
            client_factory=lambda: client,
        )
        problem = _Problem()
        heuristic.begin_search(problem)

        prediction = heuristic((1, 1), problem)

        self.assertEqual(prediction, 0.0)
        question = client.calls[0][1]["candidate"]
        self.assertEqual(question["type"], "choice")
        self.assertIn("0", question["criteria"])
        self.assertIn("254", question["criteria"])
        features = question["instructions"]["candidate"]
        self.assertNotIn("manhattan_h", features)
        self.assertNotIn("horizontal_goal_delta", features)

    def test_prescore_method_scores_every_reachable_position(self):
        client = _FakeClient()
        selector = Q2JevPreScoreTieBreaker(
            client_factory=lambda: client,
            batch_size=4,
        )
        problem = _Problem()

        selector.begin_search(problem)

        stats = selector.statistics()
        self.assertEqual(stats["scored_states"], 9)
        self.assertEqual(stats["api_requests"], 3)
        self.assertEqual(stats["cached_scores"], 9)

    def test_api_failure_disables_requests_for_current_search(self):
        client = _FailingClient()
        selector = Q2JevTieBreaker(
            client_factory=lambda: client,
            batch_size=8,
        )
        problem = _Problem()
        selector.begin_search(problem)
        candidates = [
            {"state": (1, 1), "g": 1, "h": 4, "f": 5, "order": 1},
            {"state": (1, 2), "g": 2, "h": 3, "f": 5, "order": 2},
        ]

        selector.choose_from_plateau(candidates, problem)
        selector.choose_from_plateau(candidates, problem)

        stats = selector.statistics()
        self.assertEqual(client.calls, 1)
        self.assertEqual(stats["api_requests"], 1)
        self.assertEqual(stats["failures"], 1)
        self.assertGreaterEqual(stats["api_seconds"], 0.0)

    def test_q6_node_method_sends_remaining_coins(self):
        client = _FakeClient()
        heuristic = Q6JevNodeHeuristic(
            client_factory=lambda: client,
        )
        problem = _CoinProblem()
        state = problem.getStartState()
        heuristic.begin_search(problem)

        prediction = heuristic(state, problem)

        self.assertEqual(prediction, 0.0)
        question = client.calls[0][1]["candidate"]
        features = question["instructions"]["candidate"]
        self.assertEqual(features["hero_position"], [1, 1])
        self.assertEqual(features["remaining_coin_count"], 2)
        self.assertEqual(features["remaining_coins"], [[2, 1], [3, 3]])
        self.assertNotIn("manhattan_h", features)

    def test_q6_request_limit_uses_fallback(self):
        client = _FakeClient()
        heuristic = Q6JevNodeHeuristic(
            client_factory=lambda: client,
        )
        heuristic.max_requests = 1
        problem = _CoinProblem()
        first = problem.getStartState()
        second = ((1, 2), _CoinGrid(((2, 1), (3, 3))))
        heuristic.begin_search(problem)

        heuristic(first, problem)
        fallback = heuristic(second, problem)

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(fallback, 3.0)
        self.assertEqual(heuristic.statistics()["api_requests"], 1)

    def test_q6_retries_transient_failure(self):
        client = _TransientThenSuccessfulClient()
        heuristic = Q6JevNodeHeuristic(
            client_factory=lambda: client,
        )
        heuristic.retry_backoff = 0
        problem = _CoinProblem()
        heuristic.begin_search(problem)

        prediction = heuristic(problem.getStartState(), problem)

        self.assertEqual(prediction, 0.0)
        self.assertEqual(client.attempts, 2)
        stats = heuristic.statistics()
        self.assertEqual(stats["api_requests"], 2)
        self.assertEqual(stats["retries"], 1)
        self.assertEqual(stats["failures"], 1)

    def test_q6_score_maps_continuous_value(self):
        client = _FakeClient()
        heuristic = Q6JevScoreHeuristic(
            client_factory=lambda: client,
        )
        problem = _CoinProblem()
        heuristic.begin_search(problem)

        prediction = heuristic(problem.getStartState(), problem)

        self.assertEqual(prediction, 0.0)
        question = client.calls[0][1]["candidate_0"]
        self.assertEqual(question["type"], "score")
        self.assertEqual(len(question["criteria"]), 10)
        features = question["instructions"]["candidate"]
        self.assertNotIn("manhattan_h", features)
        self.assertNotIn("mst", features)


class _Q9Walls:
    width = 5
    height = 5

    def __getitem__(self, x):
        return [True, x in (0, 4), x in (0, 4), x in (0, 4), True]


class _Q9Grid:
    def __init__(self, positions):
        self._positions = tuple(positions)

    def count(self):
        return len(self._positions)

    def asList(self):
        return list(self._positions)


class _Q9Dragon:
    def __init__(self, position, weakened_timer=0):
        self._position = position
        self.weakenedTimer = weakened_timer

    def getPosition(self):
        return self._position


class _Q9State:
    def __init__(self):
        self._walls = _Q9Walls()
        self._coins = _Q9Grid(((2, 1), (3, 3)))
        self._hero = (1, 1)
        self._swords = [(3, 1)]
        self._dragons = [_Q9Dragon((3, 2))]
        self._score = 20.0

    def getWalls(self):
        return self._walls

    def getHeroPosition(self):
        return self._hero

    def getCoins(self):
        return self._coins

    def getNumCoins(self):
        return self._coins.count()

    def getSwords(self):
        return list(self._swords)

    def getDragonStates(self):
        return list(self._dragons)

    def getLegalHeroActions(self):
        return ["North", "East"]

    def getScore(self):
        return self._score


class _Q9FailingClient:
    def decide(self, state, questions):
        raise RuntimeError("offline")

    def close(self):
        pass


class _Q9OverallClient:
    def __init__(self):
        self.calls = []

    def decide(self, state, questions):
        self.calls.append((state, questions))
        return {"overall": {"score": 7.0}}

    def close(self):
        pass


class Q9JevTests(unittest.TestCase):
    def test_explicit_unlimited_budget_overrides_import_defaults(self):
        client = _Q9OverallClient()
        with patch("jev.q9.Q9_JEV_ENABLED", False), \
             patch("jev.q9.Q9_JEV_MAX_REQUESTS", 1):
            evaluator = Q9JevEvaluator(client_factory=lambda: client,
                                       enabled=True, max_requests=0)
            state = _Q9State()
            evaluator.score(state)
            state._hero = (2, 1)
            evaluator.score(state)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(evaluator.statistics()["fallback_states"], 0)

    def test_api_failure_uses_local_overall_fallback(self):
        evaluator = Q9JevEvaluator(client_factory=_Q9FailingClient)
        with patch.dict("os.environ", {"JEV_Q9_STRICT": "0"}):
            values = evaluator.evaluate(_Q9State())
        self.assertEqual(set(values), {"overall"})
        self.assertTrue(0 <= values["overall"] <= 100)
        self.assertEqual(evaluator.statistics()["fallback_states"], 1)

    def test_strict_experiment_rejects_fallback(self):
        evaluator = Q9JevEvaluator(client_factory=_Q9FailingClient)
        with patch.dict("os.environ", {"JEV_Q9_STRICT": "1"}):
            with self.assertRaises(RuntimeError):
                evaluator.evaluate(_Q9State())
        self.assertEqual(evaluator.statistics()["fallback_states"], 0)

    def test_score_validation(self):
        self.assertEqual(_answer_score({"score": 9}), 100)
        for value in (float("nan"), float("inf"), -1, 10):
            with self.assertRaises(ValueError):
                _answer_score({"score": value})
        with self.assertRaises(KeyError):
            _answer_score({})

    def test_game_score_is_never_read_and_does_not_change_cache(self):
        client = _Q9OverallClient()
        evaluator = Q9JevEvaluator(client_factory=lambda: client)
        state = _Q9State()
        with patch.object(state, "getScore", side_effect=AssertionError("score read")):
            first = evaluator.score(state)
            state._score = 999
            self.assertEqual(first, evaluator.score(state))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(evaluator.statistics()["cache_hits"], 1)

    def test_overall_mode_sends_one_question_and_returns_one_score(self):
        client = _Q9OverallClient()
        evaluator = Q9JevEvaluator(client_factory=lambda: client, mode="overall")
        state = _Q9State()

        values = evaluator.evaluate(state)
        self.assertAlmostEqual(values["overall"], 700.0 / 9.0)
        self.assertAlmostEqual(evaluator.score(state), 700.0 / 9.0)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(list(client.calls[0][1]), ["overall"])
        self.assertNotIn("score", client.calls[0][1]["overall"]["instructions"]["state"])
        self.assertEqual(evaluator.statistics()["mode"], "overall")


class _Q9ExperimentState:
    def isWin(self):
        return False

    def isLose(self):
        return False

    def getScore(self):
        return 36


class _Q9ExperimentGame:
    state = _Q9ExperimentState()
    moveHistory = [(0, "East"), (1, "West")]

    def __init__(self, failure):
        self.failure = failure

    def run(self):
        raise self.failure


class _Q9ExperimentEvaluator:
    def __init__(self, disabled=False):
        self.disabled = disabled
        self.closed = False

    def statistics(self):
        return {"client_disabled": self.disabled, "requests": 1}

    def close(self):
        self.closed = True


class _Q9ExperimentClient:
    def __init__(self, **kwargs):
        self.last_usage = None

    def decide(self, state, questions):
        return {"overall": {"score": 5}}

    def close(self):
        pass


class Q9ExperimentTests(unittest.TestCase):
    def test_module_entry_point_help_does_not_start_experiment(self):
        result = subprocess.run(
            [sys.executable, "-m", "jev.q9", "--help"], cwd=experiment.ROOT,
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-requests", result.stdout)
        self.assertNotIn("RESULT_DIRECTORY", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_import_is_passive_and_package_export_is_compatible(self):
        code = '''
from unittest.mock import patch
with patch("urllib.request.urlopen", side_effect=AssertionError("network on import")), \\
     patch("pathlib.Path.read_text", side_effect=AssertionError("file read on import")), \\
     patch("os.chdir", side_effect=AssertionError("directory change on import")):
    from jev import Q9JevEvaluator
    from jev.q9 import Q9JevEvaluator as DirectEvaluator
    assert Q9JevEvaluator is DirectEvaluator
    evaluator = Q9JevEvaluator()
    assert evaluator.statistics()["requests"] == 0
'''
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=experiment.ROOT,
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_interruptions_are_not_game_losses(self):
        for error, status, disabled in (
            (experiment.BudgetReached(), "budget_interrupted", False),
            (KeyboardInterrupt(), "user_interrupted", False),
            (RuntimeError(), "api_failed", True),
            (ValueError(), "error", False),
        ):
            evaluator = _Q9ExperimentEvaluator(disabled)
            result = experiment.run_game(_Q9ExperimentGame(error), evaluator)
            self.assertEqual(result["status"], status)
            self.assertIsNone(result["win"])
            self.assertFalse(result["score_is_final"])
            self.assertEqual(result["score"], 36)
            self.assertTrue(evaluator.closed)

    def test_missing_cost_and_incomplete_score_are_not_zero_or_final(self):
        rows = [{"status": "budget_interrupted", "completed": False,
                 "score": 36, "wall_seconds": 1}]
        result = experiment.summarize(rows, [{"usage": {}}], {}, {})
        self.assertEqual(result["completed_games"], 0)
        self.assertIsNone(result["average_completed_score"])
        self.assertIsNone(result["usage"]["cost"]["total"])

    def test_one_pair_saves_results_when_request_budget_stops_real_game(self):
        import multiAgents
        old_cwd = Path.cwd()
        old_evaluator = multiAgents._Q9_JEV_EVALUATOR
        old_enabled, old_weight = multiAgents.USE_JEV, multiAgents.JEV_WEIGHT
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "layouts").mkdir()
                shutil.copyfile(experiment.ROOT / "layouts" / "smallClassic.lay",
                                root / "layouts" / "smallClassic.lay")
                with patch.object(experiment, "ROOT", root), \
                     patch.object(experiment, "billing", return_value={}), \
                     patch.object(experiment, "OpenRouterJevClient", _Q9ExperimentClient), \
                     patch.object(experiment, "Q9_JEV_ENABLED", False), \
                     patch.object(experiment, "Q9_JEV_MAX_REQUESTS", 1), \
                     patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-test"}), \
                     patch("sys.argv", ["q9.py", "--max-requests", "2"]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    experiment.main()
                os.chdir(old_cwd)
                output = next((root / "jev" / "results").iterdir())
                config = json.loads((output / "config.json").read_text())
                rows = json.loads((output / "games.json").read_text())
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(config["games"], 1)
                self.assertEqual(len(rows), 1)
                self.assertTrue(rows[0]["baseline"]["completed"])
                self.assertEqual(rows[0]["status"], "budget_interrupted")
                self.assertEqual(rows[0]["jev"]["requests"], 2)
                self.assertEqual(rows[0]["jev"]["fallback_states"], 0)
                self.assertEqual(summary["completed_games"], 0)
                self.assertTrue((output / "requests.jsonl").exists())
        finally:
            os.chdir(old_cwd)
            multiAgents._Q9_JEV_EVALUATOR = old_evaluator
            multiAgents.USE_JEV, multiAgents.JEV_WEIGHT = old_enabled, old_weight


if __name__ == "__main__":
    unittest.main()
