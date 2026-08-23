"""Causal timestamp-based histories; never select a future observation."""

from collections import deque


class TimeHistory:
    def __init__(self, retention_seconds=1200):
        self.values = deque()
        self.retention = retention_seconds

    def append(self, timestamp, value):
        if self.values and timestamp < self.values[-1][0]:
            raise ValueError("events must be processed in nondecreasing event time")
        self.values.append((timestamp, value))
        cutoff = timestamp - self.retention
        while self.values and self.values[0][0] < cutoff:
            self.values.popleft()

    def at_or_before(self, timestamp):
        for observed_at, value in reversed(self.values):
            if observed_at <= timestamp:
                return observed_at, value
        return None

    def since(self, timestamp):
        return [(observed_at, value) for observed_at, value in self.values
                if observed_at >= timestamp]

    def change(self, now, window):
        if not self.values:
            return None
        current_time, current = self.values[-1]
        prior = self.at_or_before(now - window)
        if prior is None or current is None or prior[1] is None:
            return None
        return current - prior[1]

    def velocity(self, now, window):
        if not self.values:
            return None
        current_time, current = self.values[-1]
        prior = self.at_or_before(now - window)
        if prior is None or current is None or prior[1] is None:
            return None
        elapsed = current_time - prior[0]
        return None if elapsed <= 0 else (current - prior[1]) / elapsed
