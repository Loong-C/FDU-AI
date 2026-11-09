"""Pace complete game rounds without changing game rules."""

import time


class TickScheduler:
    def __init__(self, duration, wait=time.sleep, clock=time.monotonic):
        self.duration = duration
        self.wait = wait
        self.clock = clock

    def run(self, game):
        try:
            if not game.start():
                return
            tick_start = self.clock()
            for _ in game.runTicks():
                elapsed = self.clock() - tick_start
                remaining = self.duration - elapsed
                if remaining > 0:
                    self.wait(remaining)
                tick_start = self.clock()
        finally:
            game.finish()


class AgentDrivenScheduler:
    """Advance the game only after each agent has returned an action."""

    def run(self, game):
        game.run()
