"""Causal timestamp-based histories; never select a future observation."""

from bisect import bisect_left, bisect_right


class TimeHistory:
    def __init__(self, retention_seconds=1200):
        self.values = []
        self.timestamps = []
        self.retention = retention_seconds

    def append(self, timestamp, value):
        if self.values and timestamp < self.values[-1][0]:
            raise ValueError("events must be processed in nondecreasing event time")
        self.values.append((timestamp, value))
        self.timestamps.append(timestamp)
        cutoff = timestamp - self.retention
        expired = bisect_left(self.timestamps, cutoff)
        if expired:
            del self.values[:expired]
            del self.timestamps[:expired]

    def at_or_before(self, timestamp):
        index = bisect_right(self.timestamps, timestamp) - 1
        return None if index < 0 else self.values[index]

    def since(self, timestamp):
        index = bisect_left(self.timestamps, timestamp)
        return self.values[index:]

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
