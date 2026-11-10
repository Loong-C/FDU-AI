"""Student implementations for DQN experience replay and exploration.

Run training from the project root with: python -m Agents.Runtime.dqn_agent
"""
import random
from collections import deque

from Game.Core import utilities as util


class ReplayBuffer:
    """Store recent transitions and sample uniformly without replacement.

    Implement the four methods below without changing their signatures.
    You are allowed to use any data structures you like. A simple approach is to use a deque to store the transitions and sample from it.
    But you are free to implement a more sophisticated data structure if you wish. 
    You don't need to care about how this buffer is used by the agent.
    """

    def __init__(self, capacity):
        """Create an empty buffer holding at most capacity transitions."""
        if int(capacity) <= 0:
            raise ValueError('Replay capacity must be positive')
        "*** YOUR CODE HERE ***"
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        """Add a transition to the buffer."""
        "*** YOUR CODE HERE ***"
        self.buffer.append(transition)

    def sample(self, batch_size):
        """Return batch_size distinct stored transitions in a list."""
        "*** YOUR CODE HERE ***"
        return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        """Return the current number of stored transitions, not the capacity."""
        "*** YOUR CODE HERE ***"
        return len(self.buffer)


class EpsilonScheduler:
    """Linearly decay epsilon by environment step and clamp at the end value."""

    def __init__(self, start, end, decay_steps):
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = int(decay_steps)
        if not 0 <= self.end <= self.start <= 1:
            raise ValueError('Epsilon bounds must satisfy 0 <= end <= start <= 1')
        if self.decay_steps <= 0:
            raise ValueError('Epsilon decay steps must be positive')

    def value(self, step):
        """Return the epsilon for the given number of collected transitions."""
        "*** YOUR CODE HERE ***"
        progress = min(1.0, max(0.0, step / self.decay_steps))
        return self.start + (self.end - self.start) * progress


if __name__ == '__main__':
    from Agents.Runtime.dqn_agent import TrainingRunner
    TrainingRunner.main()
