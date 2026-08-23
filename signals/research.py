"""Retrospective descriptive measurements; never called by live features."""

import math

from .mathutil import mean


def lagged_correlations(btc_series, alt_series, lags=(0, 1, 2, 5, 10, 15, 30)):
    """Correlate completed BTC changes with later alt changes at candidate lags."""
    results = {}
    for lag in lags:
        pairs = []
        for btc_time, btc_value in btc_series:
            candidates = [(abs(alt_time - (btc_time + lag)), alt_value)
                          for alt_time, alt_value in alt_series if alt_time >= btc_time]
            if candidates:
                distance, alt_value = min(candidates)
                if distance <= max(0.5, lag / 2 + .5): pairs.append((btc_value, alt_value))
        if len(pairs) < 2:
            results[lag] = None; continue
        xs, ys = zip(*pairs); mx, my = mean(xs), mean(ys)
        numerator = sum((x-mx)*(y-my) for x,y in pairs)
        denominator = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
        results[lag] = None if denominator == 0 else numerator / denominator
    return results
