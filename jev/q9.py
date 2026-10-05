"""Q9 Jev evaluator and optional, ungraded one-game experiment.

Import Q9JevEvaluator for scoring, or run python -m jev.q9 for the experiment.
"""

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import time
import math
from urllib import request

from .core import (
    OpenRouterJevClient,
    TypeSafeJevClient,
    TransientJevError,
)


Q9_JEV_ENABLED = os.environ.get("JEV_Q9_ENABLED", "1").lower() not in {
    "0", "false", "no", "off"
}
Q9_JEV_MODE = os.environ.get("JEV_Q9_MODE", "overall").lower()
Q9_JEV_CACHE_ENABLED = os.environ.get("JEV_Q9_CACHE", "1").lower() not in {
    "0", "false", "no", "off"
}
Q9_JEV_WEIGHT = float(os.environ.get("JEV_Q9_WEIGHT", "0.10"))
Q9_JEV_MAX_REQUESTS = int(os.environ.get("JEV_Q9_MAX_REQUESTS", "128"))
Q9_JEV_MAX_CACHE = int(os.environ.get("JEV_Q9_MAX_CACHE", "4096"))
Q9_JEV_MAX_RETRIES = int(os.environ.get("JEV_Q9_MAX_RETRIES", "2"))
Q9_STATUS_ESTIMATED_REQUESTS = 2000
Q9_JEV_RETRY_BACKOFF = float(
    os.environ.get("JEV_Q9_RETRY_BACKOFF_SECONDS", "0.5")
)

_SCORE_LEVELS = [
    "0: extremely poor",
    "1: very poor",
    "2: poor",
    "3: below average",
    "4: slightly below average",
    "5: average",
    "6: slightly above average",
    "7: good",
    "8: very good",
    "9: nearly ideal",
]


def _answer_score(answer):
    value = _raw_answer_score(answer)
    return value * (100.0 / 9.0)


def _raw_answer_score(answer):
    if isinstance(answer, dict):
        value = answer["score"]
    else:
        value = answer.score
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 9:
        raise ValueError("Jev score must be finite and within 0..9")
    return value


class Q9JevEvaluator:
    """One cached overall judgement per board, with no game score input."""

    def __init__(self, client_factory=None, mode=None, *, enabled=None, max_requests=None):
        self._client_factory = client_factory
        self.enabled = Q9_JEV_ENABLED if enabled is None else enabled
        self.max_requests = Q9_JEV_MAX_REQUESTS if max_requests is None else max_requests
        self.mode = (mode or Q9_JEV_MODE).lower()
        if self.mode != "overall":
            raise ValueError("Q9 supports only JEV_Q9_MODE=overall")
        self._client = None
        self._client_disabled = False
        self._warning_reported = False
        self._cache = OrderedDict()
        self._layout_key = None
        self._context = None
        self._initial_coin_count = None
        self._requests = 0
        self._successful_requests = 0
        self._failures = 0
        self._api_seconds = 0.0
        self._evaluations = 0
        self._cache_hits = 0
        self._fallback_states = 0
        self._status_window = None
        self._status_window_unavailable = False
        self._request_executor = None

    def _warn(self, message):
        if not self._warning_reported:
            print("[Jev Q9] Warning: " + message)
            self._warning_reported = True

    def _make_client(self):
        if self._client_factory is not None:
            return self._client_factory()
        provider = os.environ.get("JEV_PROVIDER", "openrouter").lower()
        if provider == "openrouter":
            return OpenRouterJevClient(
                model=os.environ.get("JEV_MODEL")
            )
        if provider == "typesafe":
            return TypeSafeJevClient(
                model=os.environ.get("JEV_MODEL")
            )
        raise ValueError("JEV_PROVIDER must be 'openrouter' or 'typesafe'")

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
                self._warn("status window unavailable (%s)" % type(exc).__name__)
        return self._status_window

    @staticmethod
    def _status_state_summary(features):
        return (
            "英雄 %s；剩余硬币 %d 枚；剑 %d 把；龙 %d 条"
            % (
                features["hero_position"],
                features["num_coins"],
                len(features["remaining_swords"]),
                len(features["dragons"]),
            )
        )

    def _decide_with_status(self, questions, features, started):
        window = self._get_status_window()
        if window is None:
            return self._client.decide(self._context, questions)

        if self._request_executor is None:
            self._request_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="jev-q9-api",
            )
        question = questions["overall"]["instructions"]["question"]
        window.begin_request(
            question,
            [],
            api_requests=self._requests,
            state_summary=self._status_state_summary(features),
            expected_requests=Q9_STATUS_ESTIMATED_REQUESTS,
        )
        future = self._request_executor.submit(
            self._client.decide,
            self._context,
            questions,
        )
        while not future.done():
            window.pump(api_requests=self._requests)
            time.sleep(0.05)
        try:
            return future.result()
        except Exception as exc:
            window.set_failure(
                type(exc).__name__,
                time.perf_counter() - started,
                api_requests=self._requests,
            )
            raise

    def pump_status_window(self):
        if self._status_window is not None:
            self._status_window.pump(api_requests=self._requests)

    @staticmethod
    def _map_rows(walls):
        return [
            "".join("%" if walls[x][y] else " " for x in range(walls.width))
            for y in range(walls.height - 1, -1, -1)
        ]

    @staticmethod
    def _positions(grid):
        return tuple(sorted(tuple(position) for position in grid.asList()))

    def _state_key(self, state):
        dragons = tuple(
            (
                tuple(dragon.getPosition()),
                int(dragon.weakenedTimer),
                dragon.configuration.direction if hasattr(dragon, "configuration") else None,
            )
            for dragon in state.getDragonStates()
        )
        return (
            tuple(state.getHeroPosition()),
            self._positions(state.getCoins()),
            tuple(sorted(tuple(position) for position in state.getSwords())),
            dragons,
        )

    def _ensure_context(self, state):
        walls = state.getWalls()
        layout_key = (walls.width, walls.height, tuple(self._map_rows(walls)))
        if layout_key == self._layout_key:
            return
        self.close()
        self._layout_key = layout_key
        self._cache.clear()
        self._initial_coin_count = max(1, state.getNumCoins())
        self._context = {
            "task": (
                "Evaluate a Hero game state. Higher values mean a stronger "
                "chance of eventually collecting all coins while surviving."
            ),
            "map": {
                "width": walls.width,
                "height": walls.height,
                "rows_top_to_bottom": list(layout_key[2]),
            },
        }
        if not self.enabled:
            self._client_disabled = True
            return
        try:
            self._client = self._make_client()
            self._client_disabled = False
        except (ImportError, ValueError) as exc:
            self._client_disabled = True
            self._warn("%s; using deterministic fallback" % exc)

    def _features(self, state):
        hero = tuple(state.getHeroPosition())
        dragons = [
            {
                "position": list(dragon.getPosition()),
                "weakened_timer": int(dragon.weakenedTimer),
                "direction": dragon.configuration.direction if hasattr(dragon, "configuration") else None,
            }
            for dragon in state.getDragonStates()
        ]
        legal_actions = list(state.getLegalHeroActions())
        return {
            "hero_position": list(hero),
            "remaining_coins": [list(position) for position in self._positions(state.getCoins())],
            "remaining_swords": [list(position) for position in sorted(state.getSwords())],
            "dragons": dragons,
            "legal_actions": legal_actions,
            "num_coins": state.getNumCoins(),
        }

    def _fallback(self, state):
        hero = tuple(state.getHeroPosition())
        coin_count = state.getNumCoins()
        food = 100.0 * (1.0 - coin_count / float(self._initial_coin_count or 1))

        active_distances = []
        weakened_count = 0
        for dragon in state.getDragonStates():
            if dragon.weakenedTimer > 0:
                weakened_count += 1
            else:
                position = dragon.getPosition()
                active_distances.append(
                    abs(hero[0] - position[0]) + abs(hero[1] - position[1])
                )
        nearest = min(active_distances) if active_distances else 10
        safety = min(100.0, 20.0 + nearest * 12.0 + weakened_count * 15.0)

        power = min(100.0, len(state.getSwords()) * 25.0 + weakened_count * 20.0)
        # Local emergency estimate only; never count this as a Jev response.
        return max(0.0, min(100.0, 0.5 * food + 0.3 * safety + 0.2 * power))

    def _request(self, state):
        if self._client_disabled or self._client is None:
            return None
        if self.max_requests > 0 and self._requests >= self.max_requests:
            self._client_disabled = True
            self._warn("JEV_MAX_REQUESTS reached; using deterministic fallback")
            return None

        features = self._features(state)
        questions = {
            "overall": {
                "type": "score",
                "instructions": {
                    "state": features,
                    "question": (
                        "Give one overall judgement of this Hero state. "
                        "Consider progress toward collecting all coins, "
                        "survival risk, escape options, swords, weakened "
                        "dragons together. Current game score is deliberately "
                        "omitted; judge only the supplied board situation. Return a "
                        "single holistic state-quality score."
                    ),
                },
                "criteria": _SCORE_LEVELS,
            }
        }
        for attempt in range(Q9_JEV_MAX_RETRIES + 1):
            if self.max_requests > 0 and self._requests >= self.max_requests:
                self._client_disabled = True
                return None
            try:
                self._requests += 1
                started = time.perf_counter()
                try:
                    answers = self._decide_with_status(
                        questions, features, started
                    )
                finally:
                    self._api_seconds += time.perf_counter() - started
                raw_score = _raw_answer_score(answers["overall"])
                result = {"overall": raw_score * (100.0 / 9.0)}
                self._successful_requests += 1
                if self._status_window is not None:
                    self._status_window.set_values(
                        "Score（原始 0–9）",
                        [raw_score],
                        time.perf_counter() - started,
                        api_requests=self._requests,
                    )
                    self._status_window.set_derived(
                        "Q9 换算分（0–100）", result["overall"]
                    )
                return result
            except TransientJevError as exc:
                self._failures += 1
                if attempt >= Q9_JEV_MAX_RETRIES:
                    self._client_disabled = True
                    self._warn("JEV request failed; using deterministic fallback")
                    return None
                time.sleep(Q9_JEV_RETRY_BACKOFF * (2 ** attempt))
            except Exception as exc:
                self._failures += 1
                self._client_disabled = True
                self._warn("JEV request failed (%s); using deterministic fallback" % exc)
                return None
        return None

    def evaluate(self, state):
        self._evaluations += 1
        self._ensure_context(state)
        key = self._state_key(state)
        if Q9_JEV_CACHE_ENABLED and key in self._cache:
            self._cache_hits += 1
            result = self._cache.pop(key)
            self._cache[key] = result
            return result

        result = self._request(state)
        if result is None:
            if os.environ.get("JEV_Q9_STRICT", "0") == "1":
                raise RuntimeError("Real Jev evaluation unavailable; strict experiment forbids fallback")
            self._fallback_states += 1
            result = {"overall": self._fallback(state)}
        if Q9_JEV_CACHE_ENABLED:
            self._cache[key] = result
            while len(self._cache) > max(1, Q9_JEV_MAX_CACHE):
                self._cache.popitem(last=False)
        return result

    def score(self, state):
        """Return overall in 0..100, not a calibrated win probability."""
        return self.evaluate(state)["overall"]

    def statistics(self):
        return {
            "requests": self._requests,
            "successful_requests": self._successful_requests,
            "failures": self._failures,
            "api_seconds": self._api_seconds,
            "evaluations": self._evaluations,
            "cache_hits": self._cache_hits,
            "fallback_states": self._fallback_states,
            "cached_states": len(self._cache),
            "client_disabled": self._client_disabled,
            "mode": self.mode,
        }

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
            window.finish_search(self.statistics())


__all__ = [
    "Q9JevEvaluator",
    "Q9_JEV_ENABLED",
    "Q9_JEV_MODE",
    "Q9_JEV_CACHE_ENABLED",
    "Q9_JEV_WEIGHT",
]


# Optional experiment: importing this module does not start a game or make requests.
ROOT = Path(__file__).resolve().parents[1]


class BudgetReached(RuntimeError):
    """End an experimental game without counting it as a loss."""


def billing():
    try:
        req = request.Request('https://openrouter.ai/api/v1/key', headers={
            'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']})
        with request.urlopen(req, timeout=15) as response:
            data = json.load(response)['data']
        return {k: data.get(k) for k in ('usage', 'usage_daily', 'usage_monthly')}
    except Exception as exc:
        return {'unavailable': type(exc).__name__}


def run_game(game, evaluator=None):
    """Preserve the partial game and statistics on failure or Ctrl+C."""
    started = time.perf_counter()
    reason = None
    try:
        game.run()
        status = 'won' if game.state.isWin() else 'lost' if game.state.isLose() else 'incomplete'
    except BudgetReached:
        status, reason = 'budget_interrupted', 'Experiment time or request budget reached'
    except KeyboardInterrupt:
        status, reason = 'user_interrupted', 'KeyboardInterrupt'
    except Exception as exc:
        api_unavailable = evaluator and evaluator.statistics()['client_disabled']
        status = 'api_failed' if api_unavailable else 'error'
        reason = type(exc).__name__  # Do not log exception text containing credentials.
    finally:
        if evaluator is not None:
            evaluator.close()
    completed = status in ('won', 'lost')
    row = {
        'status': status, 'reason': reason, 'completed': completed,
        'win': game.state.isWin() if completed else None,
        'score': game.state.getScore(), 'score_is_final': completed,
        'wall_seconds': time.perf_counter() - started,
        'moves': len(game.moveHistory),
        'hero_moves': sum(index == 0 for index, _ in game.moveHistory),
    }
    if evaluator is not None:
        row['jev'] = evaluator.statistics()
    return row


def summarize(rows, records, before, after):
    completed = [r for r in rows if r.get('completed')]
    usages = [r['usage'] for r in records if isinstance(r.get('usage'), dict)]

    def usage_total(field):
        values = [u[field] for u in usages if isinstance(u.get(field), (int, float))]
        return {'total': sum(values) if values else None, 'records': len(values)}

    return {
        'attempted_games': len(rows), 'completed_games': len(completed),
        'wins': sum(r['win'] for r in completed),
        'average_completed_score': (sum(r['score'] for r in completed) / len(completed))
            if completed else None,
        'statuses': [r['status'] for r in rows],
        'requests': sum(r.get('jev', {}).get('requests', 0) for r in rows),
        'api_seconds': sum(r.get('jev', {}).get('api_seconds', 0) for r in rows),
        'wall_seconds': sum(r['wall_seconds'] for r in rows),
        'usage': {field: usage_total(field) for field in ('cost', 'input_tokens', 'output_tokens')},
        'key_usage_delta_usd': (after['usage'] - before['usage'])
            if isinstance(after.get('usage'), (int, float))
            and isinstance(before.get('usage'), (int, float)) else None,
        'note': 'Ungraded experiment. One game does not estimate win rate reliably. '
                'Interrupted scores are partial and excluded from outcome averages. '
                'Usage totals cover only records providing that field; missing is null. '
                'Key usage delta may include concurrent use or delayed billing.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weight', type=float, default=0.10)
    parser.add_argument('--max-seconds', type=float, default=0,
                        help='time budget per Jev game, checked between evaluations (0: unlimited)')
    parser.add_argument('--max-requests', type=int, default=0,
                        help='HTTP attempt budget per Jev game, including retries (0: unlimited)')
    parser.add_argument('--direct', action='store_true', help='bypass environment HTTP proxy')
    parser.add_argument('--no-status-window', action='store_true',
                        help='do not open the Jev request status window')
    args = parser.parse_args()
    if args.games < 1 or not 0 < args.weight <= 1:
        parser.error('games must be positive and weight must be within (0, 1]')
    if not math.isfinite(args.max_seconds) or args.max_seconds < 0 or args.max_requests < 0:
        parser.error('budgets must be finite and nonnegative')
    if args.direct:
        request.install_opener(request.build_opener(request.ProxyHandler({})))
    if args.no_status_window:
        os.environ['JEV_STATUS_WINDOW'] = '0'
    else:
        os.environ.setdefault('JEV_STATUS_WINDOW', '1')
    env_file = ROOT.parent / '.env'
    if env_file.exists():
        for raw in env_file.read_text(encoding='utf-8-sig').splitlines():
            if not raw.strip() or raw.lstrip().startswith('#') or '=' not in raw:
                continue
            name, value = raw.split('=', 1)
            if name.strip().lower() in ('api_key', 'openrouter_api_key'):
                if not os.environ.get('OPENROUTER_API_KEY'):
                    os.environ['OPENROUTER_API_KEY'] = value.strip().strip('"').strip("'")
    if not os.environ.get('OPENROUTER_API_KEY'):
        parser.error('Set OPENROUTER_API_KEY or api_key in the project .env')
    os.environ.update(JEV_Q9_MODE='overall', JEV_Q9_ENABLED='1',
                      JEV_Q9_STRICT='1', JEV_Q9_WEIGHT=str(args.weight),
                      JEV_Q9_MAX_REQUESTS=str(args.max_requests), JEV_PROVIDER='openrouter')
    import hero
    import layout
    import multiAgents
    from dragonAgents import RandomDragon
    from textDisplay import NullGraphics
    import __main__

    # Layout loading is relative to pj1-search.
    os.chdir(ROOT)
    output = ROOT / 'jev' / 'results' / time.strftime('q9_overall_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        target = output / name
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(target)

    def append(name, value):
        with (output / name).open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(value) + '\n')

    config = {'layout': 'smallClassic', 'depth': 2, 'dragon': 'RandomDragon(1)',
              'games': args.games, 'seeds': list(range(args.seed, args.seed + args.games)),
              'weight': args.weight, 'scale': multiAgents.JEV_SCALE, 'graded': False,
              'mode': 'overall', 'input_game_score': False,
              'request_limit': args.max_requests or None, 'time_budget_seconds': args.max_seconds or None,
              'agent_timeout_seconds': None, 'strict_no_fallback': True,
              'cache': 'fresh per game', 'execution': 'sequential',
              'model': os.environ.get('JEV_MODEL', 'typesafe/jev-1.13'),
              'direct_connection': args.direct}
    save('config.json', config)
    print('RESULT_DIRECTORY', output, flush=True)
    print('Optional ungraded experiment. Network/service latency may make one game take tens of minutes.',
          flush=True)
    before = billing()
    save('billing_before.json', before)
    rows = []

    class ProgressDisplay(NullGraphics):
        def update(self, state):
            if evaluator is not None:
                evaluator.pump_status_window()
            report_progress()

    class BudgetEvaluator(Q9JevEvaluator):
        def evaluate(self, state):
            try:
                if args.max_seconds and time.perf_counter() - started >= args.max_seconds:
                    raise BudgetReached()
                return super().evaluate(state)
            finally:
                report_progress()

        def _request(self, state):
            if args.max_requests and self.statistics()['requests'] >= args.max_requests:
                raise BudgetReached()
            result = super()._request(state)
            if result is None and args.max_requests and self.statistics()['requests'] >= args.max_requests:
                raise BudgetReached()
            return result

    evaluator = None

    class LoggedClient(OpenRouterJevClient):
        def decide(self, state, questions):
            request_started = time.perf_counter()
            record = {'game': game_index + 1}
            try:
                answer = super().decide(state, questions)
                record.update(answers=answer, usage=getattr(self, 'last_usage', None))
                return answer
            except BaseException as exc:
                record['error_type'] = type(exc).__name__
                raise
            finally:
                record['seconds'] = time.perf_counter() - request_started
                append('requests.jsonl', record)

    def report_progress():
        nonlocal last_report
        if time.perf_counter() - last_report < 30:
            return
        last_report = time.perf_counter()
        status = {'game': game_index + 1, 'moves': len(game.moveHistory),
                  'coins': game.state.getNumCoins(), 'score': game.state.getScore(),
                  'elapsed_seconds': time.perf_counter() - started,
                  'jev': evaluator.statistics()}
        save('progress.json', status)
        print('PROGRESS', json.dumps(status), flush=True)

    def make_game(display):
        __main__.__dict__['_display'] = display
        return hero.ClassicGameRules().newGame(
            layout.getLayout('smallClassic'), multiAgents.ExpectimaxAgent(evalFn='better', depth='2'),
            [RandomDragon(1)], display, catchExceptions=False)

    try:
        for game_index, seed in enumerate(config['seeds']):
            random.seed(seed)
            multiAgents.USE_JEV = False
            baseline = run_game(make_game(NullGraphics()))
            row = {'game': game_index + 1, 'seed': seed, 'baseline': baseline,
                   'status': 'not_started', 'completed': False, 'wall_seconds': 0}
            rows.append(row)
            save('games.json', rows)
            if baseline['status'] not in ('won', 'lost'):
                break
            random.seed(seed)
            multiAgents.USE_JEV = True
            multiAgents.JEV_WEIGHT = args.weight
            evaluator = BudgetEvaluator(client_factory=LoggedClient, mode='overall',
                                        enabled=True, max_requests=args.max_requests)
            multiAgents._Q9_JEV_EVALUATOR = evaluator
            started = last_report = time.perf_counter()
            game = make_game(ProgressDisplay())
            print('START_JEV_GAME', game_index + 1, 'SEED', seed, flush=True)
            row.update(run_game(game, evaluator))
            save('games.json', rows)
            save('progress.json', row)
            print('GAME_RESULT', json.dumps(row), flush=True)
            if not row['completed']:
                break
    finally:
        save('games.json', rows)
        after = billing()
        save('billing_after.json', after)
        log = output / 'requests.jsonl'
        records = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()] if log.exists() else []
        summary = summarize(rows, records, before, after)
        save('summary.json', summary)
        print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
