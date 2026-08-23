"""UTC timestamp preservation and latency helpers."""

from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc)


def iso_utc(value=None):
    value = value or utc_now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def exchange_timestamp(message):
    """Preserve the most precise exchange timestamp supplied by Kalshi."""
    msg = message.get("msg", message) if isinstance(message, dict) else {}
    if msg.get("time") is not None:
        return str(msg["time"])
    if msg.get("ts_ms") is not None:
        return datetime.fromtimestamp(float(msg["ts_ms"]) / 1000,
                                      timezone.utc).isoformat(timespec="milliseconds")
    if msg.get("ts") is not None:
        value = msg["ts"]
        if isinstance(value, str) and not value.replace(".", "", 1).isdigit():
            return value
        return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
    return None


def parse_timestamp(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def latency_ms(start, end):
    a, b = parse_timestamp(start), parse_timestamp(end)
    return None if a is None or b is None else (b - a).total_seconds() * 1000


def receive_latency_ms(event):
    return latency_ms(event.exchange_timestamp, event.local_receive_timestamp)


def processing_latency_ms(event):
    return latency_ms(event.local_receive_timestamp, event.processing_timestamp)
