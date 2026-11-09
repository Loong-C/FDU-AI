"""Fixed Q6 feature experiment and its bounded, headless worker process.

python -m Game.Application.feature_evaluation --extractor SimpleExtractor
The grader always selects StudentFeatureExtractor and the protocol below.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from collections.abc import Mapping
import contextlib
import io
import json
import math
from numbers import Real
from pathlib import Path
import random
import subprocess
import sys
import os
import time
import traceback
import threading

from Agents.Runtime.module_loader import load_module_file
from Agents.Runtime.feature_extractors import FeatureExtractor
from Game.Core import utilities as util
from Game.Core.observations import ReadOnlyGameState, ReadOnlyObservations

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAYOUTS = ('mediumGrid', 'smallClassic', 'mediumClassic')
REPEATS = 3
TRAINING_GAMES = 2000
TEST_GAMES = 100
HORIZON = 900
DISCOUNT = 0.99
TOTAL_TEST_GAMES = len(LAYOUTS) * REPEATS * TEST_GAMES
# Integer win boundaries for the 20% / 50% / 80% tiers.
WIN_THRESHOLDS = tuple((TOTAL_TEST_GAMES * percent + 99) // 100 for percent in (20, 50, 80))
MAX_SECONDS = 900
MAX_WORKERS = min(6, os.cpu_count() or 1)
STUDENT_EXTRACTOR = 'Agents.Coursework.feature_extractors.StudentFeatureExtractor'
SIMPLE_EXTRACTOR = 'Agents.Coursework.feature_extractors.SimpleExtractor'


class EvaluationError(ValueError):
    pass


class BoundedLog(io.StringIO):
    """Retain enough student output for diagnosis without unbounded print buffers."""
    def write(self, text):
        remaining = max(0, 16384 - self.tell())
        if remaining:
            super().write(text[:remaining])
        return len(text)


def _finite(value, label):
    if not isinstance(value, Real):
        raise EvaluationError('%s must be a finite real number' % label)
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise EvaluationError('%s must be a finite real number' % label) from error
    if not math.isfinite(number):
        raise EvaluationError('%s must be finite' % label)
    return number


class CheckedExtractor(FeatureExtractor):
    """Validate feature values; read-only observations protect the input state."""
    def __init__(self, extractor):
        if not isinstance(extractor, FeatureExtractor):
            raise EvaluationError('The selected extractor must inherit FeatureExtractor')
        self.extractor = extractor
        # At most the current and previous observations; never retain episodes.
        self._cache = []

    def getFeatures(self, state, action):
        if not isinstance(state, ReadOnlyGameState):
            state = ReadOnlyObservations()(state)
        for previous, features in self._cache:
            if previous is state:
                if action in features:
                    return features[action].copy()
                break
        else:
            features = {}
            self._cache.append((state, features))
            if len(self._cache) > 2:
                self._cache.pop(0)
        result = _feature_values(self.extractor.getFeatures(state, action))
        features[action] = result.copy()
        return result


def _feature_values(values):
    if not isinstance(values, Mapping):
        raise EvaluationError('getFeatures must return a sparse mapping, preferably util.Counter')
    result = util.Counter()
    for key, value in values.items():
        # Counter insertion also checks that the key is hashable.
        result[key] = _finite(value, 'Feature %r' % (key,))
    return result


def check_feature_contract(feature_class, board):
    """Small, separate probes retain strict mutation checks outside training."""
    from Game.Core.observations import ReadOnlyObservations
    from Game.Core.rules import GameState
    extractor = feature_class()
    if not isinstance(extractor, FeatureExtractor):
        raise EvaluationError('The selected extractor must inherit FeatureExtractor')
    state = GameState()
    state.initialize(board, board.getNumDragons())
    state._trackExplored = False
    samples = [state]
    actions = state.getLegalActions()
    if actions:
        samples.append(state.generateHeroSuccessor(actions[0]))
    readonly = ReadOnlyObservations()
    for sample in samples:
        for action in sample.getLegalActions():
            probe = sample.deepCopy()
            before = hash(probe)
            _feature_values(extractor.getFeatures(probe, action))
            if hash(probe) != before:
                raise EvaluationError('getFeatures must not modify GameState')
            observation = readonly(sample)
            first = _feature_values(extractor.getFeatures(observation, action))
            second = _feature_values(extractor.getFeatures(observation, action))
            if first != second:
                raise EvaluationError('getFeatures must return stable features for the same state and action')


def nonzero_weights(agent):
    weights = agent.getWeights()
    if not isinstance(weights, Mapping):
        raise EvaluationError('getWeights must return a mapping')
    result = {}
    for key, value in weights.items():
        number = _finite(value, 'Weight %r' % (key,))
        if number != 0:
            result[key] = number
    return result


def points_for_wins(wins):
    if type(wins) is not int or not 0 <= wins <= TOTAL_TEST_GAMES:
        raise EvaluationError('Invalid test win count')
    return sum(wins >= threshold for threshold in WIN_THRESHOLDS)


def _check_game(game):
    if game.agentCrashed or game.agentTimeout:
        raise EvaluationError('An agent crashed or timed out during feature evaluation')
    _finite(game.state.getScore(), 'Game score')


def evaluate_group(qlearning_path, feature_path, extractor, map_index, repeat):
    """One original map/seed protocol, with fresh modules, agent and weights."""
    from Game.Application import game_setup, layout_loader
    from Game.Presentation.text_display import NullGraphics
    from Agents.Runtime.dragon_agents import RandomDragon
    from Game.Core.observations import ReadOnlyObservations
    if not 0 <= map_index < len(LAYOUTS) or not 0 <= repeat < REPEATS:
        raise EvaluationError('Invalid group index')
    started = time.monotonic()
    qlearning = load_module_file('Agents.Coursework.qlearning_agents', qlearning_path)
    if extractor.startswith('Agents.Coursework.feature_extractors.'):
        if feature_path is None:
            raise EvaluationError('Student feature file is missing')
        load_module_file('Agents.Coursework.feature_extractors', feature_path)
    feature_class = util.lookup(extractor, vars(qlearning))
    name = LAYOUTS[map_index]
    board = layout_loader.getLayout(name)
    initial_coins = board.coins.count()
    check_feature_contract(feature_class, board)
    random.seed(2026 + 1000 * map_index + repeat)
    agent = qlearning.ApproximateQAgent(extractor=extractor, numTraining=TRAINING_GAMES,
                                       alpha=.2, gamma=DISCOUNT, epsilon=.05)
    if (agent.alpha, agent.discount, agent.epsilon, agent.numTraining, agent.episodesSoFar) != (
            .2, DISCOUNT, .05, TRAINING_GAMES, 0):
        raise EvaluationError('The fixed learning parameters must be respected')
    if nonzero_weights(agent):
        raise EvaluationError('Each training group must start with zero weights')
    if not isinstance(agent.featExtractor, feature_class):
        raise EvaluationError('ApproximateQAgent must use the requested extractor')
    agent.featExtractor = CheckedExtractor(agent.featExtractor)
    dragons = [RandomDragon(i + 1) for i in range(board.getNumDragons())]
    observations = ReadOnlyObservations()
    # Keep only small episode batches; no full-game histories in the result.
    training = game_setup.runGames(board, HORIZON, agent, dragons, NullGraphics(),
                                   TRAINING_GAMES, False, numTraining=TRAINING_GAMES,
                                   catchExceptions=False, observationFactory=observations,
                                   trackExplored=False)
    if training:
        raise EvaluationError('Training games must not be returned as test games')
    if (agent.alpha, agent.epsilon, agent.episodesSoFar) != (0, 0, TRAINING_GAMES):
        raise EvaluationError('Training must stop after the fixed number of episodes')
    frozen = nonzero_weights(agent)
    scores, coins, wins, truncated = [], [], 0, 0
    for episode in range(TEST_GAMES):
        random.seed(100000 + 1000 * map_index + 100 * repeat + episode)
        games = game_setup.runGames(board, HORIZON, agent, dragons, NullGraphics(),
                                     1, False, catchExceptions=False,
                                     observationFactory=observations, trackExplored=False)
        if len(games) != 1:
            raise EvaluationError('Exactly one test game must be returned')
        game = games[0]
        _check_game(game)
        if agent.alpha != 0 or agent.epsilon != 0 or nonzero_weights(agent) != frozen:
            raise EvaluationError('Testing must disable exploration and weight updates')
        wins += bool(game.state.isWin())
        truncated += bool(game.truncated)
        scores.append(float(game.state.getScore()))
        coins.append(1 - game.state.getNumCoins() / initial_coins)
    group = dict(layout=name, repeat=repeat, training=TRAINING_GAMES,
                       tests=TEST_GAMES, wins=wins, truncated=truncated,
                       mean_score=sum(scores) / TEST_GAMES,
                       coin_fraction=sum(coins) / TEST_GAMES)
    group['seconds'] = time.monotonic() - started
    return dict(ok=True, group=group,
                qlearning_source=str(Path(qlearning.__file__).resolve()),
                feature_source=str(Path(feature_path).resolve()) if extractor.startswith(
                    'Agents.Coursework.feature_extractors.') else None)


def _aggregate(results, started):
    groups = [item['group'] for item in results]
    wins = sum(group['wins'] for group in groups)
    return dict(ok=True, groups=groups, tests=TOTAL_TEST_GAMES, wins=wins,
                win_rate=wins / TOTAL_TEST_GAMES, seconds=time.monotonic() - started,
                qlearning_source=results[0]['qlearning_source'],
                feature_source=results[0]['feature_source'])


def evaluate(qlearning_path, feature_path=None, extractor=STUDENT_EXTRACTOR):
    """Serial diagnostic API; CLI and grading use bounded parallel workers."""
    started = time.monotonic()
    results = [evaluate_group(qlearning_path, feature_path, extractor, m, r)
               for m in range(len(LAYOUTS)) for r in range(REPEATS)]
    return _aggregate(results, started)


def validate_result(result):
    """Require all map/seed groups so failed games never shrink the denominator."""
    if not isinstance(result, dict) or result.get('ok') is not True:
        raise EvaluationError(str(result.get('error', 'Feature evaluation failed'))
                              if isinstance(result, dict) else 'Invalid worker result')
    groups = result.get('groups')
    expected = [(name, repeat) for name in LAYOUTS for repeat in range(REPEATS)]
    if not isinstance(groups, list) or len(groups) != len(expected):
        raise EvaluationError('Incomplete map/seed groups')
    wins = 0
    for group, pair in zip(groups, expected):
        if (group.get('layout'), group.get('repeat')) != pair:
            raise EvaluationError('Unexpected map/seed order')
        if group.get('training') != TRAINING_GAMES or group.get('tests') != TEST_GAMES:
            raise EvaluationError('Unexpected experiment budget')
        for key in ('wins', 'truncated'):
            if type(group.get(key)) is not int or not 0 <= group[key] <= TEST_GAMES:
                raise EvaluationError('Invalid group ' + key)
        if group['wins'] + group['truncated'] > TEST_GAMES:
            raise EvaluationError('A truncated game cannot also be a win')
        _finite(group.get('mean_score'), 'Mean score')
        fraction = _finite(group.get('coin_fraction'), 'Coin fraction')
        if not 0 <= fraction <= 1:
            raise EvaluationError('Invalid coin fraction')
        wins += group['wins']
    if type(result.get('wins')) is not int or result['wins'] != wins or result.get('tests') != TOTAL_TEST_GAMES:
        raise EvaluationError('Inconsistent experiment totals')
    if _finite(result.get('seconds'), 'Duration') < 0:
        raise EvaluationError('Invalid duration')
    if _finite(result.get('win_rate'), 'Win rate') != wins / TOTAL_TEST_GAMES:
        raise EvaluationError('Inconsistent win rate')
    return result


def run_isolated(qlearning_path, feature_source, timeout=MAX_SECONDS, extractor=STUDENT_EXTRACTOR):
    """Independent processes share one deadline; parent never imports student code."""
    needs_student = extractor.startswith('Agents.Coursework.feature_extractors.')
    if needs_student and (feature_source is None or feature_source.error or not feature_source.path):
        return dict(ok=False, error=getattr(feature_source, 'error', None) or 'Student feature file is missing')
    command = [sys.executable, '-B', '-m', 'Game.Application.feature_evaluation', '--json',
               '--qlearning-code', str(Path(qlearning_path).resolve()), '--extractor', extractor]
    if feature_source is not None and feature_source.path:
        command.extend(['--features-code', str(Path(feature_source.path).resolve())])
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
    started = time.monotonic()
    deadline = started + timeout
    cancelled = threading.Event()
    active = set()
    lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def one(map_index, repeat):
        if cancelled.is_set() or time.monotonic() >= deadline:
            return dict(ok=False, error='Feature evaluation cancelled or timed out')
        process = subprocess.Popen(command + ['--group', str(map_index), str(repeat)],
            cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace',
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        with lock:
            active.add(process)
            if cancelled.is_set():
                process.kill()
        try:
            try:
                stdout, stderr = process.communicate(timeout=max(.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                return dict(ok=False, error='Feature evaluation exceeded the %g-second total limit' % timeout)
            if process.returncode:
                return dict(ok=False, error='Worker exited with code %s: %s' %
                            (process.returncode, stderr[-2000:]))
            result = json.loads(stdout)
            if not isinstance(result, dict) or result.get('ok') is not True:
                return dict(ok=False, error=str(result.get('error', 'Feature evaluation failed')),
                            log=str(result.get('log', ''))[-16384:])
            group = result.get('group', {})
            if (group.get('layout'), group.get('repeat')) != (LAYOUTS[map_index], repeat):
                raise EvaluationError('Unexpected worker group')
            return result
        finally:
            with lock:
                active.discard(process)

    try:
        futures = {pool.submit(one, m, r): (m, r)
                   for m in range(len(LAYOUTS)) for r in range(REPEATS)}
        results = {}
        for future in as_completed(futures, timeout=max(.001, deadline - time.monotonic())):
            result = future.result()
            if not result.get('ok'):
                return result
            results[futures[future]] = result
        ordered = [results[m, r] for m in range(len(LAYOUTS)) for r in range(REPEATS)]
        return validate_result(_aggregate(ordered, started))
    except FuturesTimeoutError:
        return dict(ok=False, error='Feature evaluation exceeded the %g-second total limit' % timeout)
    except (Exception, SystemExit) as error:
        return dict(ok=False, error='%s: %s' % (type(error).__name__, error))
    finally:
        cancelled.set()
        with lock:
            for process in tuple(active):
                if process.poll() is None:
                    process.kill()
        pool.shutdown(wait=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extractor', default=STUDENT_EXTRACTOR)
    parser.add_argument('--qlearning-code', type=Path, default=PROJECT_ROOT / 'Agents/Coursework/qlearning_agents.py')
    parser.add_argument('--features-code', type=Path, default=PROJECT_ROOT / 'Agents/Coursework/feature_extractors.py')
    parser.add_argument('--json', action='store_true', help='Machine-readable result for the grader')
    parser.add_argument('--group', type=int, nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    log = BoundedLog()
    if args.group is not None:
        if not args.json:
            parser.error('--group is an internal worker option')
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = evaluate_group(args.qlearning_code, args.features_code, args.extractor, *args.group)
        except BaseException as error:
            result = dict(ok=False, error='%s: %s' % (type(error).__name__, error))
            traceback.print_exception(type(error), error, error.__traceback__, limit=8, file=log)
        result['log'] = log.getvalue()
    else:
        from Agents.Runtime.module_loader import DeferredStudentModule
        result = run_isolated(args.qlearning_code, DeferredStudentModule(path=str(args.features_code.resolve())),
                              extractor=args.extractor)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    elif result['ok']:
        for name in LAYOUTS:
            groups = [g for g in result['groups'] if g['layout'] == name]
            print('%s: %d/%d wins, mean score %.2f, coin fraction %.3f, %d truncated' % (
                name, sum(g['wins'] for g in groups), REPEATS * TEST_GAMES,
                sum(g['mean_score'] for g in groups) / REPEATS,
                sum(g['coin_fraction'] for g in groups) / REPEATS,
                sum(g['truncated'] for g in groups)))
        print('Total: %d/%d wins (%.2f%%); gamma=%.2f; feature tier %d/3; %.2fs' % (
            result['wins'], result['tests'], 100 * result['win_rate'],
            DISCOUNT, points_for_wins(result['wins']), result['seconds']))
    else:
        print('Feature evaluation failed: ' + result['error'])
    return result


if __name__ == '__main__':
    main()
