"""Tiny in-memory sliding-window rate limiter (per client key).

Good enough for a single-process demo. A multi-worker deployment would need a
shared store (e.g. Redis) -- documented as a known limitation.
"""

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable


class SlidingWindowLimiter:
    def __init__(
        self, max_requests: int, window_seconds: float = 60.0, clock=time.monotonic
    ):
        self.max_requests = max_requests
        self.window = window_seconds
        self._clock: Callable[[], float] = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.max_requests <= 0:
            return True
        now = self._clock()
        cutoff = now - self.window
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= self.max_requests:
                return False
            q.append(now)
            return True
