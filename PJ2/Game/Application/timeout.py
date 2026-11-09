# util.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
#
# Attribution Information: This adaptation is based on AI projects developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# util.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: This adaptation is based on AI projects developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import signal, time

class TimeoutFunctionException(Exception):
    """Exception to raise on a timeout"""
    pass


class TimeoutFunction:
    def __init__(self, function, timeout):
        self.timeout = timeout
        self.function = function

    def handle_timeout(self, signum, frame):
        raise TimeoutFunctionException()

    def __call__(self, *args, **keyArgs):
        import math
        import threading
        timeout = float(self.timeout)
        if not math.isfinite(timeout):
            raise ValueError('Timeout must be finite')
        if timeout <= 0:
            raise TimeoutFunctionException()
        # POSIX supports interrupting a running callback, including fractional
        # seconds. Windows/non-main threads can only check after it returns.
        if hasattr(signal, 'setitimer') and threading.current_thread() is threading.main_thread():
            old_handler = signal.getsignal(signal.SIGALRM)
            previous = signal.getitimer(signal.ITIMER_REAL)
            start = time.monotonic()
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, min(timeout, previous[0]) if previous[0] else timeout)
            try:
                return self.function(*args, **keyArgs)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
                if previous[0]:
                    signal.setitimer(signal.ITIMER_REAL, max(1e-6, previous[0] - (time.monotonic() - start)), previous[1])
        start = time.monotonic()
        result = self.function(*args, **keyArgs)
        if time.monotonic() - start >= timeout:
            self.handle_timeout(None, None)
        return result
