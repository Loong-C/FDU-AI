"""Process-wide rate limiting shared by the Q6 and Q9 Jev clients."""

from collections import deque
import os
import threading
import time


MAX_REQUESTS_PER_SECOND = 5
MAX_REQUESTS_PER_MINUTE = 300
_SECOND_WINDOW = 1.0
_MINUTE_WINDOW = 60.0


class SlidingWindowRateLimiter:
    """Limit request starts to a rolling one-second window."""

    def __init__(self):
        self._starts = deque()
        self._minute_starts = deque()
        self._last_start = None
        self._lock = threading.Lock()

    def acquire(self):
        """Wait for a request slot and return the time spent waiting."""
        configured_rate = float(
            os.environ.get(
                "JEV_REQUESTS_PER_SECOND", str(MAX_REQUESTS_PER_SECOND)
            )
        )
        if configured_rate < 0:
            raise ValueError("JEV_REQUESTS_PER_SECOND cannot be negative")

        # Zero disables the optional Q6 pacing target, but never the shared
        # safety ceiling. Values above the ceiling are clamped.
        effective_rate = min(
            MAX_REQUESTS_PER_SECOND,
            configured_rate if configured_rate > 0 else MAX_REQUESTS_PER_SECOND,
        )
        minimum_interval = 1.0 / effective_rate
        waited = 0.0

        with self._lock:
            while True:
                now = time.monotonic()
                while self._starts and now - self._starts[0] >= _SECOND_WINDOW:
                    self._starts.popleft()
                while (
                    self._minute_starts
                    and now - self._minute_starts[0] >= _MINUTE_WINDOW
                ):
                    self._minute_starts.popleft()

                window_wait = 0.0
                if len(self._starts) >= MAX_REQUESTS_PER_SECOND:
                    window_wait = self._starts[0] + _SECOND_WINDOW - now

                minute_wait = 0.0
                if len(self._minute_starts) >= MAX_REQUESTS_PER_MINUTE:
                    minute_wait = self._minute_starts[0] + _MINUTE_WINDOW - now

                pace_wait = 0.0
                if self._last_start is not None:
                    pace_wait = self._last_start + minimum_interval - now

                delay = max(window_wait, minute_wait, pace_wait)
                if delay <= 0:
                    started = time.monotonic()
                    self._starts.append(started)
                    self._minute_starts.append(started)
                    self._last_start = started
                    return waited

                time.sleep(delay)
                waited += delay


JEV_REQUEST_LIMITER = SlidingWindowRateLimiter()
