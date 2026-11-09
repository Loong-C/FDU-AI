"""Q6 Jev teaching example with shared compatibility implementations."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
import time
from urllib.parse import urlsplit

from .rate_limit import JEV_REQUEST_LIMITER


# ---------------------------------------------------------------------------
# Q6 SCORE TEACHING EXAMPLE: EDIT THESE RULES
# ---------------------------------------------------------------------------
# Students can change the experiment here without editing the API, cache,
# retry, or search integration below. Environment variables with matching
# names override the operational defaults at runtime.

Q6_SCORE_PROMPT = (
    "Independently inspect the complete maze, current position, and remaining "
    "coins. Estimate the exact number of legal unit-cost moves in a shortest "
    "route that collects all remaining coins. No traditional heuristic "
    "estimate or prescribed collection order is supplied."
)

Q6_SCORE_LEVELS = [
    "The optimal remaining route costs approximately 0 moves",
    "The optimal remaining route costs approximately 28 moves",
    "The optimal remaining route costs approximately 56 moves",
    "The optimal remaining route costs approximately 85 moves",
    "The optimal remaining route costs approximately 113 moves",
    "The optimal remaining route costs approximately 141 moves",
    "The optimal remaining route costs approximately 169 moves",
    "The optimal remaining route costs approximately 198 moves",
    "The optimal remaining route costs approximately 226 moves",
    "The optimal remaining route costs approximately 254 moves",
]

Q6_SCORE_MAX_COST = 254.0

# One cache entry represents (hero position, complete remaining coin set).
Q6_SCORE_CACHE_RULE = "complete_coin_search_state"

# Used only after the request limit or final API failure.
Q6_SCORE_FALLBACK_RULE = "farthest_remaining_coin_manhattan"

JEV_DEFAULT_PROVIDER = "openrouter"
JEV_OPENROUTER_MODEL = "typesafe/jev-1.13"
JEV_TYPESAFE_MODEL = "jev-latest"
JEV_BATCH_SIZE = 16
JEV_MAX_REQUESTS = 30000
JEV_REQUESTS_PER_SECOND = 5.0
JEV_TIMEOUT_SECONDS = 15.0
JEV_MAX_RETRIES = 2
JEV_RETRY_BACKOFF_SECONDS = 0.5

# Score-to-heuristic rule:
# h_jev = score * Q6_SCORE_MAX_COST / (len(Q6_SCORE_LEVELS) - 1)
# ---------------------------------------------------------------------------


_DIRECTIONS = (
    ("North", (0, 1)),
    ("South", (0, -1)),
    ("East", (1, 0)),
    ("West", (-1, 0)),
)

_PROMISE_LEVELS = [
    "Very unlikely to lie on a shortest route; strong detour or dead-end risk",
    "Unlikely to lie on a shortest route",
    "Uncertain; neither clearly good nor clearly bad",
    "Promising for reaching the goal with little unnecessary travel",
    "Very promising; likely to lie on a shortest route",
]

_COST_RATIO_LEVELS = [
    "Remaining cost is approximately 1.00 times the Manhattan lower bound",
    "Remaining cost is approximately 1.25 times the Manhattan lower bound",
    "Remaining cost is approximately 1.50 times the Manhattan lower bound",
    "Remaining cost is approximately 1.75 times the Manhattan lower bound",
    "Remaining cost is approximately 2.00 times the Manhattan lower bound",
    "Remaining cost is approximately 2.25 times the Manhattan lower bound",
    "Remaining cost is approximately 2.50 times the Manhattan lower bound",
    "Remaining cost is approximately 2.75 times the Manhattan lower bound",
    "Remaining cost is approximately 3.00 times the Manhattan lower bound",
    "Remaining cost is at least 3.25 times the Manhattan lower bound",
]

class _TransientJevError(RuntimeError):
    pass


class OpenRouterJevClient:
    """Minimal client for OpenRouter's Jev Decisions endpoint."""

    def __init__(self, api_key=None, model=None, timeout=None):
        self.api_key = (
            api_key or os.environ.get("OPENROUTER_API_KEY", "")
        ).strip()
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.model = model or os.environ.get(
            "JEV_MODEL", JEV_OPENROUTER_MODEL
        )
        self.timeout = float(
            timeout
            or os.environ.get(
                "JEV_TIMEOUT_SECONDS", str(JEV_TIMEOUT_SECONDS)
            )
        )
        self.url = os.environ.get(
            "JEV_OPENROUTER_URL",
            "https://openrouter.ai/api/alpha/decisions",
        )
        parsed_url = urlsplit(self.url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.hostname:
            raise ValueError("JEV_OPENROUTER_URL must be an HTTP(S) URL")
        self._host = parsed_url.hostname
        self._port = parsed_url.port
        self._target = parsed_url.path or "/"
        if parsed_url.query:
            self._target += "?" + parsed_url.query
        self._https = parsed_url.scheme == "https"
        self._connection = None

    def _get_connection(self):
        if self._connection is None:
            connection_type = (
                http.client.HTTPSConnection
                if self._https
                else http.client.HTTPConnection
            )
            self._connection = connection_type(
                self._host,
                self._port,
                timeout=self.timeout,
            )
        return self._connection

    def _close_connection(self):
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def decide(self, state, questions):
        JEV_REQUEST_LIMITER.acquire()
        body = json.dumps(
            {
                "model": self.model,
                "state": state,
                "questions": questions,
            }
        ).encode("utf-8")
        try:
            connection = self._get_connection()
            connection.request(
                "POST",
                self._target,
                body=body,
                headers={
                    "Authorization": "Bearer " + self.api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Connection": "keep-alive",
                },
            )
            response = connection.getresponse()
            response_body = response.read()
        except (OSError, http.client.HTTPException) as exc:
            self._close_connection()
            raise _TransientJevError(
                "OpenRouter request failed: %s" % exc
            ) from exc

        if response.status < 200 or response.status >= 300:
            message = response_body.decode("utf-8", errors="replace")
            transient_statuses = (408, 429, 500, 502, 503, 504, 524, 529)
            exception_type = (
                _TransientJevError
                if response.status in transient_statuses
                else RuntimeError
            )
            raise exception_type(
                "OpenRouter returned HTTP %d: %s"
                % (response.status, message)
            )

        payload = json.loads(response_body)
        answers = payload.get("answers")
        self.last_usage = payload.get("usage")
        if not isinstance(answers, dict):
            raise RuntimeError("OpenRouter response does not contain answers")
        return answers

    def close(self):
        self._close_connection()


class TypeSafeJevClient:
    """Adapter for the official TypeSafe SDK."""

    def __init__(self, model=None, timeout=None):
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        resolved_timeout = float(
            timeout
            or os.environ.get(
                "JEV_TIMEOUT_SECONDS", str(JEV_TIMEOUT_SECONDS)
            )
        )
        self._client = TypeSafeClient(
            model=model or os.environ.get("JEV_MODEL", JEV_TYPESAFE_MODEL),
            timeout=resolved_timeout,
            retry=RetryPolicy(
                max_retries=2,
                backoff_initial=0.5,
                backoff_max=2.0,
                timeout=max(30.0, resolved_timeout * 3.0),
            ),
        )
        self.url = self._client._config.base_url.rstrip("/") + "/v1/systemone"
        self.last_usage = None

    def decide(self, state, questions):
        JEV_REQUEST_LIMITER.acquire()
        result = self._client.system_one(state=state, questions=questions)
        self.last_usage = getattr(result, "usage", None)
        return result.answers

    def close(self):
        self._client.close()


class _Q2JevBase:
    def __init__(self, client_factory=None, batch_size=None):
        self._client_factory = client_factory
        self.batch_size = batch_size or int(
            os.environ.get("JEV_BATCH_SIZE", str(JEV_BATCH_SIZE))
        )
        if self.batch_size < 1:
            raise ValueError("JEV_BATCH_SIZE must be at least 1")
        self.max_requests = int(
            os.environ.get("JEV_MAX_REQUESTS", str(JEV_MAX_REQUESTS))
        )
        if self.max_requests < 1:
            raise ValueError("JEV_MAX_REQUESTS must be at least 1")
        default_rate = (
            "0"
            if client_factory is not None
            else str(JEV_REQUESTS_PER_SECOND)
        )
        self.requests_per_second = float(
            os.environ.get("JEV_REQUESTS_PER_SECOND", default_rate)
        )
        if self.requests_per_second < 0:
            raise ValueError("JEV_REQUESTS_PER_SECOND cannot be negative")
        self.max_retries = int(
            os.environ.get("JEV_MAX_RETRIES", str(JEV_MAX_RETRIES))
        )
        if self.max_retries < 0:
            raise ValueError("JEV_MAX_RETRIES cannot be negative")
        self.retry_backoff = float(
            os.environ.get(
                "JEV_RETRY_BACKOFF_SECONDS",
                str(JEV_RETRY_BACKOFF_SECONDS),
            )
        )
        if self.retry_backoff < 0:
            raise ValueError("JEV_RETRY_BACKOFF_SECONDS cannot be negative")
        self._client = None
        self._problem = None
        self._shared_state = None
        self._scores = {}
        self._confidences = {}
        self._warning_reported = False
        self._disabled_reason = None
        self._api_disabled = True
        self._stats = {}
        self._status_window = None
        self._status_window_unavailable = False
        self._request_executor = None
        self._reset_stats()

    def _reset_stats(self):
        self._stats = {
            "decision_rounds": 0,
            "api_requests": 0,
            "scored_states": 0,
            "cache_hits": 0,
            "failures": 0,
            "retries": 0,
            "api_seconds": 0.0,
            "throttle_seconds": 0.0,
        }
        self._last_request_started = None

    def begin_search(self, problem):
        self.close()
        self._problem = problem
        self._scores = {}
        self._confidences = {}
        self._warning_reported = False
        self._disabled_reason = None
        self._reset_stats()
        self._shared_state = {
            "task": (
                "Reach the fixed goal using a shortest legal path through "
                "the maze. Walls cannot be crossed."
            ),
            "map": {
                "width": problem.walls.width,
                "height": problem.walls.height,
                "rows_top_to_bottom": self._map_rows(problem),
            },
            "goal": list(problem.goal),
        }
        self._client = self._create_client()
        self._api_disabled = self._client is None

    def _get_status_window(self):
        if (
            os.environ.get("JEV_STATUS_WINDOW") != "1"
            or self._status_window_unavailable
        ):
            return None
        if self._status_window is None:
            try:
                from .status_window import JevStatusWindow
                self._status_window = JevStatusWindow()
            except Exception as exc:
                self._status_window_unavailable = True
                self._warn("status window unavailable (%s)" % exc)
        return self._status_window

    def _decide_with_status(self, questions, question, candidates, started):
        window = self._get_status_window()
        if window is None:
            return self._decide_with_retry(questions)

        if self._request_executor is None:
            self._request_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="jev-api",
            )
        window.begin_request(
            question,
            candidates,
            api_requests=self._stats["api_requests"],
            expected_requests=getattr(self, "expected_request_count", None),
        )
        future = self._request_executor.submit(
            self._decide_with_retry,
            questions,
        )
        while not future.done():
            window.pump(api_requests=self._stats["api_requests"])
            time.sleep(0.05)
        try:
            return future.result()
        except Exception as exc:
            window.set_failure(
                str(exc),
                time.perf_counter() - started,
                api_requests=self._stats["api_requests"],
            )
            raise

    def _create_client(self):
        if self._client_factory is not None:
            return self._client_factory()

        provider = os.environ.get(
            "JEV_PROVIDER", JEV_DEFAULT_PROVIDER
        ).lower()
        try:
            if provider == "openrouter":
                return OpenRouterJevClient()
            if provider == "typesafe":
                return TypeSafeJevClient()
            raise ValueError(
                "JEV_PROVIDER must be 'openrouter' or 'typesafe'"
            )
        except (ImportError, ValueError) as exc:
            self._disabled_reason = (
                "%s; using deterministic fallback" % exc
            )
            self._warn(self._disabled_reason)
            return None

    def _map_rows(self, problem):
        rows = []
        for y in range(problem.walls.height - 1, -1, -1):
            rows.append(
                "".join(
                    "%" if problem.walls[x][y] else " "
                    for x in range(problem.walls.width)
                )
            )
        return rows

    def _candidate_features(self, candidate):
        x, y = candidate["state"]
        goal_x, goal_y = self._problem.goal
        legal_directions = []
        blocked_directions = []
        for name, (dx, dy) in _DIRECTIONS:
            if self._problem.walls[x + dx][y + dy]:
                blocked_directions.append(name)
            else:
                legal_directions.append(name)

        return {
            "position": [x, y],
            "goal": [goal_x, goal_y],
            "path_cost_g": candidate["g"],
            "manhattan_h": candidate["h"],
            "astar_f": candidate["f"],
            "horizontal_goal_delta": goal_x - x,
            "vertical_goal_delta": goal_y - y,
            "legal_directions": legal_directions,
            "blocked_directions": blocked_directions,
            "open_degree": len(legal_directions),
            "is_dead_end": len(legal_directions) <= 1,
        }

    def _request_scores(self, candidates, criteria, question):
        if not candidates:
            return {}
        if self._api_disabled:
            return {}
        if self._stats["api_requests"] >= self.max_requests:
            self._disable_at_request_limit()
            return {}

        questions = {}
        question_states = {}
        for index, candidate in enumerate(candidates):
            question_id = "candidate_%d" % index
            question_states[question_id] = candidate["state"]
            questions[question_id] = {
                "type": "score",
                "instructions": {
                    "candidate": self._candidate_features(candidate),
                    "question": question,
                },
                "criteria": criteria,
            }

        started = time.perf_counter()
        try:
            answers = self._decide_with_status(
                questions, question, candidates, started
            )
            output = {}
            for question_id, state in question_states.items():
                answer = answers[question_id]
                if isinstance(answer, dict):
                    score = answer["score"]
                    confidence = answer.get("confidence", 0.0)
                else:
                    score = answer.score
                    confidence = getattr(answer, "confidence", 0.0)
                output[state] = float(score)
                self._confidences[state] = float(confidence)
            self._stats["scored_states"] += len(output)
            if self._status_window is not None:
                self._status_window.set_values(
                    "Score",
                    output.values(),
                    time.perf_counter() - started,
                    api_requests=self._stats["api_requests"],
                )
            return output
        except Exception as exc:
            if self._status_window is not None:
                self._status_window.set_failure(
                    str(exc),
                    time.perf_counter() - started,
                    api_requests=self._stats["api_requests"],
                )
            self._api_disabled = True
            self._disabled_reason = (
                "Jev API failed during this search; further requests disabled"
            )
            if not isinstance(exc, _TransientJevError):
                self._stats["failures"] += 1
            self._warn(
                "Jev request failed (%s: %s); using deterministic fallback"
                % (type(exc).__name__, exc)
            )
            return {}
        finally:
            self._stats["api_seconds"] += time.perf_counter() - started

    def _request_choice(self, candidate, choices, question):
        if self._api_disabled:
            return None
        if self._stats["api_requests"] >= self.max_requests:
            self._disable_at_request_limit()
            return None

        questions = {
            "candidate": {
                "type": "choice",
                "instructions": {
                    "candidate": self._candidate_features(candidate),
                    "question": question,
                },
                "criteria": {str(choice): None for choice in choices},
            }
        }
        started = time.perf_counter()
        try:
            answers = self._decide_with_status(
                questions, question, [candidate], started
            )
            answer = answers["candidate"]
            if isinstance(answer, dict):
                choice = answer["choice"]
                confidence = answer.get("confidence", 0.0)
            else:
                choice = answer.choice
                confidence = getattr(answer, "confidence", 0.0)
            self._confidences[candidate["state"]] = float(confidence)
            self._stats["scored_states"] += 1
            if self._status_window is not None:
                self._status_window.set_values(
                    "Choice",
                    [choice],
                    time.perf_counter() - started,
                    api_requests=self._stats["api_requests"],
                )
            return int(choice)
        except Exception as exc:
            if self._status_window is not None:
                self._status_window.set_failure(
                    str(exc),
                    time.perf_counter() - started,
                    api_requests=self._stats["api_requests"],
                )
            self._api_disabled = True
            self._disabled_reason = (
                "Jev API failed during this search; further requests disabled"
            )
            if not isinstance(exc, _TransientJevError):
                self._stats["failures"] += 1
            self._warn(
                "Jev request failed (%s: %s); using deterministic fallback"
                % (type(exc).__name__, exc)
            )
            return None
        finally:
            self._stats["api_seconds"] += time.perf_counter() - started

    def _decide_with_retry(self, questions):
        for attempt in range(self.max_retries + 1):
            if self._stats["api_requests"] >= self.max_requests:
                self._disable_at_request_limit()
                raise RuntimeError(self._disabled_reason)

            self._pace_request()
            self._stats["api_requests"] += 1
            try:
                return self._client.decide(self._shared_state, questions)
            except _TransientJevError:
                self._stats["failures"] += 1
                if attempt >= self.max_retries:
                    raise
                self._stats["retries"] += 1
                time.sleep(self.retry_backoff * (2 ** attempt))

        raise RuntimeError("Jev retry loop exited unexpectedly")

    def _pace_request(self):
        if self.requests_per_second == 0:
            self._last_request_started = time.perf_counter()
            return

        minimum_interval = 1.0 / self.requests_per_second
        now = time.perf_counter()
        if self._last_request_started is not None:
            delay = minimum_interval - (now - self._last_request_started)
            if delay > 0:
                time.sleep(delay)
                self._stats["throttle_seconds"] += delay
        self._last_request_started = time.perf_counter()

    def _disable_at_request_limit(self):
        self._api_disabled = True
        self._disabled_reason = (
            "JEV_MAX_REQUESTS=%d reached; further requests disabled"
            % self.max_requests
        )
        self._warn(self._disabled_reason)

    def _candidate(self, state, g=None):
        h = self._manhattan(state)
        return {
            "state": state,
            "g": g,
            "h": h,
            "f": None if g is None else g + h,
        }

    def _manhattan(self, state):
        return (
            abs(state[0] - self._problem.goal[0])
            + abs(state[1] - self._problem.goal[1])
        )

    def _warn(self, message):
        if not self._warning_reported:
            print("[Jev] Warning: " + message)
            self._warning_reported = True

    def statistics(self):
        stats = dict(self._stats)
        stats["cached_scores"] = len(self._scores)
        stats["disabled_reason"] = self._disabled_reason
        return stats

    def close(self):
        if self._request_executor is not None:
            self._request_executor.shutdown(wait=True)
            self._request_executor = None
        if self._client is not None:
            self._client.close()
        self._client = None
        if self._status_window is not None:
            window = self._status_window
            self._status_window = None
            window.finish_search(self._stats)


class Q2JevNodeHeuristic(_Q2JevBase):
    """Call Jev once for each newly encountered state."""

    def __call__(self, state, problem):
        if problem is not self._problem:
            self.begin_search(problem)
        if state == problem.goal:
            return 0.0
        if state in self._scores:
            self._stats["cache_hits"] += 1
            return self._scores[state]

        candidate = self._candidate(state)
        scores = self._request_scores(
            [candidate],
            _COST_RATIO_LEVELS,
            (
                "Estimate the remaining shortest-path cost as a multiple of "
                "the Manhattan lower bound."
            ),
        )
        ratio_score = scores.get(state)
        if ratio_score is None:
            heuristic = float(candidate["h"])
        else:
            heuristic = candidate["h"] * (1.0 + 0.25 * ratio_score)
        self._scores[state] = heuristic
        return heuristic

    def choose_from_plateau(self, candidates, problem):
        self._stats["decision_rounds"] += 1
        return min(candidates, key=lambda candidate: candidate["order"])


class Q2JevDirectValueHeuristic(Q2JevNodeHeuristic):
    """Ask Jev to choose an exact integer remaining cost."""

    def _candidate_features(self, candidate):
        x, y = candidate["state"]
        legal_directions = []
        blocked_directions = []
        for name, (dx, dy) in _DIRECTIONS:
            if self._problem.walls[x + dx][y + dy]:
                blocked_directions.append(name)
            else:
                legal_directions.append(name)

        return {
            "position": [x, y],
            "goal": list(self._problem.goal),
            "legal_directions": legal_directions,
            "blocked_directions": blocked_directions,
            "open_degree": len(legal_directions),
            "is_dead_end": len(legal_directions) <= 1,
        }

    def __call__(self, state, problem):
        if problem is not self._problem:
            self.begin_search(problem)
        if state == problem.goal:
            return 0.0
        if state in self._scores:
            self._stats["cache_hits"] += 1
            return self._scores[state]

        candidate = self._candidate(state)
        prediction = self._request_choice(
            candidate,
            range(255),
            (
                "Independently inspect the complete maze and choose the exact "
                "integer number of legal unit-cost steps in a shortest path "
                "from this candidate to the fixed goal. No heuristic or "
                "lower-bound estimate is supplied."
            ),
        )
        heuristic = float(
            candidate["h"] if prediction is None else prediction
        )
        self._scores[state] = heuristic
        return heuristic


class Q2JevTieBreaker(_Q2JevBase):
    """Score only unseen nodes in the current minimum-f plateau."""

    def __call__(self, state, problem):
        if problem is not self._problem:
            self.begin_search(problem)
        return float(self._manhattan(state))

    def choose_from_plateau(self, candidates, problem):
        if problem is not self._problem:
            self.begin_search(problem)

        self._stats["decision_rounds"] += 1
        missing = [
            candidate for candidate in candidates
            if candidate["state"] not in self._scores
        ]
        self._stats["cache_hits"] += len(candidates) - len(missing)

        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            scores = self._request_scores(
                batch,
                _PROMISE_LEVELS,
                (
                    "How promising is this candidate for reaching the fixed "
                    "goal along a shortest route?"
                ),
            )
            for candidate in batch:
                state = candidate["state"]
                self._scores[state] = scores.get(
                    state, -float(candidate["h"])
                )

        return max(
            candidates,
            key=lambda candidate: (
                self._scores[candidate["state"]],
                -candidate["order"],
            ),
        )


class Q2JevPreScoreTieBreaker(Q2JevTieBreaker):
    """Score every reachable position before A* begins."""

    def begin_search(self, problem):
        super().begin_search(problem)
        candidates = self._reachable_candidates(problem)
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start:start + self.batch_size]
            scores = self._request_scores(
                batch,
                _PROMISE_LEVELS,
                (
                    "How promising is this position for reaching the fixed "
                    "goal along a shortest route?"
                ),
            )
            for candidate in batch:
                state = candidate["state"]
                self._scores[state] = scores.get(
                    state, -float(candidate["h"])
                )
            if self._api_disabled:
                for remaining in candidates[start + self.batch_size:]:
                    self._scores[remaining["state"]] = -float(remaining["h"])
                break

    def _reachable_candidates(self, problem):
        start = problem.getStartState()
        queue = deque([(start, 0)])
        distances = {start: 0}
        candidates = []
        while queue:
            state, distance = queue.popleft()
            candidates.append(self._candidate(state, distance))
            x, y = state
            for _, (dx, dy) in _DIRECTIONS:
                successor = (x + dx, y + dy)
                if problem.walls[successor[0]][successor[1]]:
                    continue
                if successor in distances:
                    continue
                distances[successor] = distance + 1
                queue.append((successor, distance + 1))
        return candidates


class Q6JevNodeHeuristic(_Q2JevBase):
    """Ask Jev for one remaining-cost value per CoinSearchProblem state."""

    expected_request_count = 6000

    def begin_search(self, problem):
        self.close()
        self._problem = problem
        self._scores = {}
        self._confidences = {}
        self._warning_reported = False
        self._disabled_reason = None
        self._reset_stats()
        start_position, start_coins = problem.getStartState()
        self._initial_coin_count = start_coins.count()
        self._shared_state = {
            "task": (
                "Starting from each candidate state, collect every remaining "
                "coin using the fewest legal unit-cost maze moves. A coin is "
                "collected when the hero enters its cell. Walls cannot be "
                "crossed and revisiting cells is allowed."
            ),
            "map": {
                "width": problem.walls.width,
                "height": problem.walls.height,
                "rows_top_to_bottom": self._map_rows(problem),
            },
            "initial_position": list(start_position),
            "initial_coin_count": self._initial_coin_count,
        }
        self._client = self._create_client()
        self._api_disabled = self._client is None

    def __call__(self, state, problem):
        if problem is not self._problem:
            self.begin_search(problem)
        position, coin_grid = state
        if coin_grid.count() == 0:
            return 0.0
        if state in self._scores:
            self._stats["cache_hits"] += 1
            return self._scores[state]

        candidate = {
            "state": state,
            "g": None,
            "h": self._fallback_value(state),
            "f": None,
        }
        prediction = self._request_choice(
            candidate,
            range(255),
            (
                "Independently inspect the complete maze, current position, "
                "and remaining coins. Choose the exact integer number of "
                "legal unit-cost moves in a shortest route that collects all "
                "remaining coins. No traditional heuristic estimate or "
                "prescribed collection order is supplied."
            ),
        )
        heuristic = float(
            candidate["h"] if prediction is None else prediction
        )
        self._scores[state] = heuristic
        return heuristic

    def _candidate_features(self, candidate):
        position, coin_grid = candidate["state"]
        x, y = position
        legal_directions = []
        blocked_directions = []
        for name, (dx, dy) in _DIRECTIONS:
            if self._problem.walls[x + dx][y + dy]:
                blocked_directions.append(name)
            else:
                legal_directions.append(name)

        return {
            "hero_position": list(position),
            "remaining_coins": [
                list(coin) for coin in sorted(coin_grid.asList())
            ],
            "remaining_coin_count": coin_grid.count(),
            "legal_directions": legal_directions,
            "blocked_directions": blocked_directions,
            "open_degree": len(legal_directions),
            "is_dead_end": len(legal_directions) <= 1,
        }

    def _fallback_value(self, state):
        if Q6_SCORE_FALLBACK_RULE != "farthest_remaining_coin_manhattan":
            raise ValueError(
                "Unsupported Q6_SCORE_FALLBACK_RULE: %s"
                % Q6_SCORE_FALLBACK_RULE
            )
        position, coin_grid = state
        coins = coin_grid.asList()
        if not coins:
            return 0
        return max(
            abs(position[0] - coin[0]) + abs(position[1] - coin[1])
            for coin in coins
        )

    def choose_from_plateau(self, candidates, problem):
        self._stats["decision_rounds"] += 1
        return min(candidates, key=lambda candidate: candidate["order"])


class Q6JevScoreHeuristic(Q6JevNodeHeuristic):
    """Map Jev's continuous absolute-cost score to a Q6 heuristic."""

    def __call__(self, state, problem):
        if problem is not self._problem:
            self.begin_search(problem)
        _, coin_grid = state
        if coin_grid.count() == 0:
            return 0.0
        if state in self._scores:
            self._stats["cache_hits"] += 1
            return self._scores[state]

        candidate = {
            "state": state,
            "g": None,
            "h": self._fallback_value(state),
            "f": None,
        }
        scores = self._request_scores(
            [candidate],
            Q6_SCORE_LEVELS,
            Q6_SCORE_PROMPT,
        )
        score = scores.get(state)
        heuristic = float(
            candidate["h"]
            if score is None
            else score * (Q6_SCORE_MAX_COST / (len(Q6_SCORE_LEVELS) - 1))
        )
        self._scores[state] = heuristic
        if self._status_window is not None:
            self._status_window.set_heuristic(heuristic)
        return heuristic
