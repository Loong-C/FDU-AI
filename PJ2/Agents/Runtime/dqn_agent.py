"""Train and evaluate the CNN Double DQN agent.

Run from the project root: python -m Agents.Runtime.dqn_agent
"""
import argparse
import contextlib
import copy
import io
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn

from Agents.Runtime.learning_agents import ReinforcementAgent
from Agents.Runtime.dragon_agents import RandomDragon
from Game.Application import game_setup, layout_loader
from Game.Core.game import Directions
from Game.Core.rules import ClassicGameRules, GameState, WEAKENED_TIME
from Game.Core.utilities import nearestPoint
from Game.Presentation.text_display import NullGraphics
from Agents.Coursework.dqn_utils import EpsilonScheduler, ReplayBuffer


class HeroCNN(nn.Module):
    """
    Fully convolutional encoder with Hero-conditioned and global features.
    You don't need to modify this class, 
    But you can if you want to experiment with different architectures.    
    """

    def __init__(self, num_actions=5, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        feature_dim = 32 * (1 + 2 * 2)
        self._region_weights = {}
        self.q_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def _pool_weights(self, height, width, device, dtype):
        key = (height, width, device, dtype)
        if key not in self._region_weights:
            weights = torch.zeros((4, height, width), dtype=dtype)
            for row in range(2):
                y0, y1 = row * height // 2, ((row + 1) * height + 1) // 2
                for col in range(2):
                    x0, x1 = col * width // 2, ((col + 1) * width + 1) // 2
                    weights[row * 2 + col, y0:y1, x0:x1] = 1.0 / ((y1 - y0) * (x1 - x0))
            self._region_weights[key] = weights.flatten(1).to(device=device)
        return self._region_weights[key]

    def forward(self, state):
        state = state.float()
        features = self.encoder(state)
        hero_mask = state[:, 0:1]
        hero_mask = hero_mask / hero_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        hero_features = (features * hero_mask).sum(dim=(-2, -1))
        height, width = features.shape[-2:]
        weights = self._pool_weights(height, width, features.device, features.dtype)
        global_features = torch.matmul(features.flatten(2), weights.t()).flatten(1)
        return self.q_head(torch.cat([hero_features, global_features], dim=1))


class DQNAgent(ReinforcementAgent):
    """Double DQN using a PyTorch network and uniform experience replay."""

    ACTIONS = (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP)
    ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}

    def __init__(self, input_shape, hidden_dim=64, capacity=100000,
                 batch_size=32, warmup=5000, train_interval=4,
                 target_interval=2000, device='cpu', epsilon_start=None,
                 epsilon_end=0.005, epsilon_decay_steps=1000000, **kwargs):
        kwargs.setdefault('epsilon', 0.05)
        kwargs.setdefault('gamma', 0.8)
        kwargs.setdefault('alpha', 0.2)
        kwargs.setdefault('numTraining', 0)
        super().__init__(**kwargs)
        self.epsilon_start = None if epsilon_start is None else float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        if self.epsilon_decay_steps <= 0:
            raise ValueError('Epsilon decay steps must be positive')
        if not 0 <= self.epsilon_end <= 1 or (self.epsilon_start is not None and
                not 0 <= self.epsilon_end <= self.epsilon_start <= 1):
            raise ValueError('Epsilon bounds must satisfy 0 <= end <= start <= 1')
        self.epsilon_scheduler = (EpsilonScheduler(self.epsilon_start, self.epsilon_end,
                                                   self.epsilon_decay_steps)
                                  if self.epsilon_start is not None else None)
        self.index = 0
        self.input_shape = tuple(input_shape)
        self.hidden_dim = int(hidden_dim)
        self.batch_size = int(batch_size)
        self.warmup = int(warmup)
        self.train_interval = int(train_interval)
        self.target_interval = int(target_interval)
        capacity = int(capacity)
        if min(capacity, self.batch_size, self.warmup, self.train_interval,
               self.target_interval, self.hidden_dim) <= 0:
            raise ValueError('Network and replay settings must be positive')
        if max(self.batch_size, self.warmup) > capacity:
            raise ValueError('Replay capacity must cover batch size and warmup')
        self.device = torch.device(device)
        self.online = self._make_network().to(self.device)
        self.target = self._make_network().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.alpha)
        self.replay_buffer = ReplayBuffer(capacity)
        self.environment_steps = 0
        self.optimization_steps = 0
        self.step_callback = None

    def training_epsilon(self, step):
        if self.epsilon_scheduler is None:
            return self.epsilon
        return self.epsilon_scheduler.value(step)

    def registerInitialState(self, state):
        if self.isInTraining() and self.epsilon_scheduler is not None:
            self.epsilon = self.training_epsilon(self.environment_steps)
        super().registerInitialState(state)

    def _make_network(self):
        return HeroCNN(num_actions=len(self.ACTIONS), hidden_dim=self.hidden_dim)

    def _encode(self, state):
        walls, coins = state.getWalls(), state.getCoins()
        channels = np.zeros((6, walls.height, walls.width), dtype=np.float32)
        channels[3] = np.asarray(walls.data, dtype=np.uint8).T
        channels[1] = np.asarray(coins.data, dtype=np.uint8).T

        def mark(channel, position, value=1.0):
            if position is None:
                return
            x, y = nearestPoint(position)
            if not (0 <= x < walls.width and 0 <= y < walls.height):
                raise ValueError('Agent position is outside the map')
            channels[channel, y, x] = max(channels[channel, y, x], float(value))

        mark(0, state.getHeroPosition())
        for dragon in state.getDragonStates():
            if dragon.weakenedTimer > 0:
                mark(4, dragon.getPosition(), min(1.0, max(0.0, dragon.weakenedTimer / WEAKENED_TIME)))
            else:
                mark(2, dragon.getPosition())
        for position in state.getSwords():
            mark(5, position)
        if channels.shape != self.input_shape:
            raise ValueError('Game state dimensions do not match the model')
        return channels

    def _tensor(self, states):
        return torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)

    def getLegalActions(self, state):
        return [action for action in super().getLegalActions(state) if action in self.ACTION_INDEX]

    def getAction(self, state):
        legal = self.getLegalActions(state)
        if not legal:
            action = None
        elif random.random() < self.epsilon:
            action = random.choice(legal)
        else:
            action, _ = self.computeActionAndValueFromQValues(state)
        self.doAction(state, action)
        return action

    def getPolicy(self, state):
        return self.computeActionAndValueFromQValues(state)[0]

    def getValue(self, state):
        return self.computeActionAndValueFromQValues(state)[1]

    def getQValues(self, state):
        """Return all action values from a single network evaluation."""
        with torch.no_grad():
            values = self.online(self._tensor(self._encode(state)[None, ...]))
        return values[0].detach().cpu().numpy()

    def getQValue(self, state, action):
        if action not in self.ACTION_INDEX:
            raise ValueError('Q network does not support this action')
        return float(self.getQValues(state)[self.ACTION_INDEX[action]])

    def computeActionAndValueFromQValues(self, state):
        legal_actions = self.getLegalActions(state)
        if not legal_actions:
            return None, 0.0
        values = self.getQValues(state)
        action = max(legal_actions, key=lambda item: values[self.ACTION_INDEX[item]])
        return action, float(values[self.ACTION_INDEX[action]])

    def getWeights(self):
        return {name: value.detach().cpu().numpy().copy()
                for name, value in self.online.state_dict().items()}

    def update(self, state, action, nextState, reward):
        if not self.isInTraining() or action not in self.ACTION_INDEX:
            return
        next_actions = tuple(self.ACTION_INDEX[a] for a in self.getLegalActions(nextState))
        legal_mask = tuple(index in next_actions for index in range(len(self.ACTIONS)))
        self.replay_buffer.add((self._encode(state), self.ACTION_INDEX[action], float(reward),
                            self._encode(nextState) if next_actions else None, legal_mask))
        self.environment_steps += 1
        if (len(self.replay_buffer) >= max(self.warmup, self.batch_size)
                and self.environment_steps % self.train_interval == 0):
            self._train_batch(self.replay_buffer.sample(self.batch_size))
        if self.epsilon_scheduler is not None:
            self.epsilon = self.training_epsilon(self.environment_steps)
        if self.step_callback is not None:
            self.step_callback(self)

    def _train_batch(self, batch):
        states = self._tensor(np.stack([item[0] for item in batch]))
        actions = torch.tensor([item[1] for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=self.device)
        targets = rewards.clone()
        nonterminal = [i for i, item in enumerate(batch) if item[3] is not None]
        if nonterminal:
            next_states = self._tensor(np.stack([batch[i][3] for i in nonterminal]))
            legal_mask = torch.as_tensor(np.asarray([batch[i][4] for i in nonterminal], dtype=np.bool_),
                                         dtype=torch.bool, device=self.device)
            with torch.no_grad():
                next_actions = self.online(next_states).masked_fill(~legal_mask, float('-inf')).argmax(dim=1)
                target_values = self.target(next_states).gather(1, next_actions[:, None]).squeeze(1)
                targets[torch.tensor(nonterminal, dtype=torch.long, device=self.device)] += self.discount * target_values
        predictions = self.online(states).gather(1, actions[:, None]).squeeze(1)
        loss = 0.5 * (predictions - targets).square().mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimization_steps += 1
        if self.optimization_steps % self.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())



class TrainingRunner:
    """Train agents and evaluate independent snapshots."""

    @staticmethod
    def evaluate_snapshot(agent, board, horizon, test_games, seed, include_steps=False):
        """Evaluate an independent copy without changing the training run."""
        replay_buffer = getattr(agent, 'replay_buffer', None)
        # Evaluation disables learning, so its agent does not need replay history.
        memo = {id(replay_buffer): None} if replay_buffer is not None else {}
        evaluation_agent = copy.deepcopy(agent, memo)
        evaluation_agent.numTraining = evaluation_agent.episodesSoFar
        evaluation_agent.alpha = 0.0
        evaluation_agent.epsilon = 0.0
        python_random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        scores = []
        wins = 0
        truncated = 0
        moves = 0
        try:
            for episode in range(test_games):
                random.seed(seed + episode)
                np.random.seed((seed + episode) % (2 ** 32))
                dragons = [RandomDragon(i + 1) for i in range(board.getNumDragons())]
                with contextlib.redirect_stdout(io.StringIO()):
                    games = game_setup.runGames(
                        board, horizon, evaluation_agent, dragons, NullGraphics(),
                        1, False, catchExceptions=False,
                    )
                if len(games) != 1:
                    raise RuntimeError('Expected exactly one evaluation game')
                game = games[0]
                if game.agentCrashed or game.agentTimeout:
                    raise RuntimeError('An agent crashed or timed out during evaluation')
                scores.append(float(game.state.getScore()))
                wins += int(game.state.isWin())
                truncated += int(game.truncated)
                moves += sum(index == 0 for index, _ in game.moveHistory)
        finally:
            random.setstate(python_random_state)
            np.random.set_state(numpy_random_state)
        result = (sum(scores) / len(scores), wins, truncated)
        return (*result, moves / len(scores)) if include_steps else result

    @staticmethod
    def main(argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--maps', nargs='+', default=['mediumGrid', 'smallClassic', 'mediumClassic'])
        parser.add_argument('--training-steps', type=int, default=300000)
        parser.add_argument('--report-interval', type=int, default=15000)
        parser.add_argument('--save-interval', type=int, default=30000)
        parser.add_argument('--epsilon-decay-steps', type=int, default=60000)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--replay-capacity', type=int, default=20000)
        parser.add_argument('--replay-start-size', type=int, default=1000)
        parser.add_argument('--test-games', type=int, default=5)
        parser.add_argument('--horizon', type=int, default=1000, help='Maximum Hero actions per game')
        parser.add_argument('--seed', type=int, default=50020)
        parser.add_argument('--alpha', type=float, default=3e-4)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--epsilon', type=float, default=None, help='Use a fixed epsilon instead of decay')
        parser.add_argument('--epsilon-start', type=float, default=1)
        parser.add_argument('--epsilon-end', type=float, default=0.05)
        parser.add_argument('--hidden-dim', type=int, default=64)
        parser.add_argument('--device', default='cpu')
        parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'checkpoints')
        parser.add_argument('--load', type=Path, help='Load a saved model and evaluate it without training')
        parser.add_argument('--map', dest='evaluation_map', help='Override the checkpoint map during evaluation')
        parser.add_argument('--graphics', action='store_true', help='Show games when evaluating a saved model')
        parser.add_argument('--frame-time', type=float, default=0.1, help='Delay between game rounds in graphical evaluation')
        args = parser.parse_args(argv)
        if min(args.training_steps, args.report_interval, args.save_interval, args.epsilon_decay_steps,
               args.batch_size, args.replay_capacity, args.replay_start_size,
               args.test_games, args.horizon, args.hidden_dim) <= 0:
            parser.error('Step counts, game counts, horizon and hidden size must be positive')
        if max(args.batch_size, args.replay_start_size) > args.replay_capacity:
            parser.error('replay capacity must cover batch size and replay start size')
        if not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
            parser.error('epsilon bounds must satisfy 0 <= end <= start <= 1')
        if args.epsilon is not None and not 0 <= args.epsilon <= 1:
            parser.error('fixed epsilon must be between zero and one')
        if args.evaluation_map and not args.load:
            parser.error('--map requires --load; use --maps for training')
        if args.graphics and not args.load:
            parser.error('--graphics requires --load')
        if not np.isfinite(args.frame_time) or args.frame_time < 0:
            parser.error('frame-time must be finite and nonnegative')

        if args.load:
            checkpoint = torch.load(args.load, map_location=args.device, weights_only=True)
            evaluation_map = args.evaluation_map or checkpoint['map']
            board = layout_loader.getLayout(evaluation_map)
            if board is None:
                raise ValueError('Evaluation map could not be found: %s' % evaluation_map)
            shape = (6, board.height, board.width)
            if checkpoint['model'] != 'cnn':
                raise ValueError('Checkpoint must contain a CNN model')
            if checkpoint['model'] == 'cnn' and checkpoint['input_shape'][0] != shape[0]:
                raise ValueError('Checkpoint input channel count does not match the current encoder')
            agent = DQNAgent(shape, hidden_dim=checkpoint['hidden_dim'],
                                  numTraining=0, device=args.device)
            agent.online.load_state_dict(checkpoint['online'])
            if args.graphics:
                from Game.Presentation.graphics_display import HeroGraphics
                random.seed(args.seed)
                np.random.seed(args.seed % (2 ** 32))
                dragons = [RandomDragon(i + 1) for i in range(board.getNumDragons())]
                print('Loaded %s | map=%s | model=%s | training steps=%s' %
                      (args.load, evaluation_map, checkpoint['model'], checkpoint.get('training_steps', 'unknown')), flush=True)
                game_setup.runGames(board, args.horizon, agent, dragons,
                                    HeroGraphics(frameTime=args.frame_time), args.test_games, False,
                                    tickDuration=args.frame_time)
                return
            mean_score, wins, truncated, mean_steps = TrainingRunner.evaluate_snapshot(
                agent, board, args.horizon, args.test_games, args.seed + 100000, include_steps=True)
            print('Loaded %s | map=%s | model=%s | training steps=%s' %
                  (args.load, evaluation_map, checkpoint['model'], checkpoint.get('training_steps', 'unknown')))
            print('Test mean score: %.2f | Wins: %d/%d | Truncated: %d | Mean steps: %.1f' %
                  (mean_score, wins, args.test_games, truncated, mean_steps))
            return


        args.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        console = sys.stdout
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run_dir = args.output_dir / run_id
        run_dir.mkdir()
        log_path = run_dir / 'training.log'
        with log_path.open('x', encoding='utf-8'):
            pass

        def emit(message):
            with log_path.open('a', encoding='utf-8') as log_file:
                print(message, file=log_file, flush=True)
            print(message, file=console, flush=True)

        emit('Training log: %s' % log_path)
        emit('Training configuration: %s' % vars(args))
        for map_index, name in enumerate(args.maps):
            board = layout_loader.getLayout(name)
            if board is None:
                parser.error('Unknown map: %s' % name)
            seed = args.seed
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            options = {'epsilon_start': args.epsilon_start if args.epsilon is None else None,
                       'epsilon_end': args.epsilon_end}
            agent = DQNAgent((6, board.height, board.width),
                                hidden_dim=args.hidden_dim, numTraining=args.training_steps + 1,
                                alpha=args.alpha, gamma=args.gamma,
                                epsilon=args.epsilon_start if args.epsilon is None else args.epsilon,
                                epsilon_decay_steps=args.epsilon_decay_steps, device=args.device,
                                capacity=args.replay_capacity, batch_size=args.batch_size,
                                warmup=args.replay_start_size, **options)
            dragons = [RandomDragon(i + 1) for i in range(board.getNumDragons())]
            prefix = 'cnn_%s' % name
            emit('\nMap: %s | Model: %s | Device: %s | Seed: %d' % (name, 'cnn', args.device, seed))
            emit(f"{'Steps':>9} | {'Eps':>6} | {'Episodes':>8} | "
                 f"{'TeScore':>8} | {'TeWins':>7} | {'TeSteps':>7} | {'Trunc':>5} | {'TrSec':>8} | {'TeSec':>7} | {'Steps/s':>8}")
            timing = {'start': time.perf_counter(), 'excluded': 0.0, 'last_step': 0}
            best = {'score': None}

            def on_step(current):
                step = current.environment_steps
                final = step == args.training_steps
                if step % args.report_interval and step % args.save_interval and not final:
                    return
                if current.device.type == 'cuda':
                    torch.cuda.synchronize(current.device)
                event_start = time.perf_counter()
                improved = False
                if step % args.report_interval == 0 or final:
                    train_seconds = event_start - timing['start'] - timing['excluded']
                    step_rate = (step - timing['last_step']) / train_seconds if train_seconds > 0 else 0.0
                    test_start = time.perf_counter()
                    score, wins, truncated, test_steps = TrainingRunner.evaluate_snapshot(
                        current, board, args.horizon, args.test_games,
                        args.seed + 100000 + 1000 * map_index, include_steps=True)
                    if current.device.type == 'cuda':
                        torch.cuda.synchronize(current.device)
                    test_seconds = time.perf_counter() - test_start
                    performance = (score, wins)
                    improved = best['score'] is None or performance > best['score']
                    if improved:
                        best['score'] = performance
                    emit(f"{step:>9d} | {current.epsilon:>6.4f} | "
                         f"{current.episodesSoFar:>8d} | "
                         f"{score:>8.2f} | {wins:>2d}/{args.test_games:<4d} | {test_steps:>7.1f} | {truncated:>5d} | "
                         f"{train_seconds:>8.2f} | {test_seconds:>7.2f} | {step_rate:>8.1f}")
                    timing.update(start=event_start, excluded=0.0, last_step=step)
                if step % args.save_interval == 0 or final or improved:
                    checkpoint = {'model': 'cnn', 'map': name, 'input_shape': list(current.input_shape),
                                  'hidden_dim': args.hidden_dim, 'training_steps': step,
                                  'training_games': current.episodesSoFar,
                                  'online': {key: value.detach().cpu() for key, value in current.online.state_dict().items()}}
                    if improved:
                        checkpoint['test_mean_score'] = score
                        checkpoint['test_wins'] = wins
                        checkpoint['test_games'] = args.test_games
                    paths = []
                    if step % args.save_interval == 0 or final:
                        paths.append(run_dir / ('%s_%d.pt' % (prefix, step)))
                    if improved:
                        paths.append(run_dir / ('%s_best.pt' % prefix))
                    for path in paths:
                        fd, temporary_path = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
                        os.close(fd)
                        try:
                            torch.save(checkpoint, temporary_path)
                            os.replace(temporary_path, path)
                        finally:
                            if os.path.exists(temporary_path):
                                os.unlink(temporary_path)
                    if improved:
                        best_path = args.output_dir / ('%s_best_%s.pt' % (prefix, run_id))
                        fd, temporary_path = tempfile.mkstemp(prefix=best_path.name + '.', suffix='.tmp', dir=args.output_dir)
                        os.close(fd)
                        try:
                            shutil.copy2(paths[-1], temporary_path)
                            os.replace(temporary_path, best_path)
                        finally:
                            if os.path.exists(temporary_path):
                                os.unlink(temporary_path)
                timing['excluded'] += time.perf_counter() - event_start

            agent.step_callback = on_step
            rules = ClassicGameRules(30)
            rules.quiet = True
            while agent.environment_steps < args.training_steps:
                remaining = args.training_steps - agent.environment_steps
                before = agent.environment_steps
                GameState.getAndResetExplored()
                game = game_setup.createGame(rules, board, min(args.horizon, remaining), agent,
                                             dragons, NullGraphics(), True, False)
                with contextlib.redirect_stdout(io.StringIO()):
                    game.run()
                if game.agentCrashed or game.agentTimeout:
                    raise RuntimeError('An agent crashed or timed out during training')
                if agent.environment_steps == before:
                    raise RuntimeError('Training game produced no replay transitions')
            agent.step_callback = None
        emit('\nElapsed: %.2f seconds' % (time.monotonic() - started))


if __name__ == '__main__':
    TrainingRunner.main()
