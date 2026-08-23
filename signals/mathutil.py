"""Small dependency-free robust statistics helpers."""

import math
import statistics


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def stddev(values):
    values = [value for value in values if value is not None]
    return statistics.pstdev(values) if len(values) >= 2 else (0.0 if len(values) == 1 else None)


def mad(values):
    center = median(values)
    return None if center is None else median([abs(value - center) for value in values])


def safe_ratio(numerator, denominator):
    return None if numerator is None or denominator in (None, 0) else numerator / denominator


def log_return(old, new):
    return None if old is None or new is None or old <= 0 or new <= 0 else math.log(new / old)
