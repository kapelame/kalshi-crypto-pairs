"""Connection-generation-aware Kalshi WebSocket sequence validation."""

from dataclasses import dataclass


def channel_family(event_type):
    """Return the independently sequenced Kalshi channel family."""
    if event_type in ("orderbook_snapshot", "orderbook_delta"):
        return "orderbook"
    if event_type in ("market_lifecycle", "market_lifecycle_v2"):
        return "market_lifecycle"
    return event_type or "unknown"


@dataclass(frozen=True)
class SequenceObservation:
    healthy: bool
    generation: int
    family: str
    sid: object
    previous: int | None = None
    current: int | None = None
    generation_changed: bool = False


class SequenceValidator:
    """Validate sequences per (connection generation, channel family, sid).

    Kalshi sequence numbers belong to subscription streams, not to the whole
    account or process.  A connection generation change rotates every stream.
    """

    def __init__(self):
        self.generation = 0
        self.last = {}
        self.gaps = 0

    def begin_generation(self, generation=None):
        next_generation = self.generation + 1 if generation is None else int(generation)
        changed = next_generation != self.generation
        self.generation = next_generation
        if changed:
            self.last.clear()
        return changed

    def observe_result(self, sid, sequence, family="unknown", generation=None,
                       bootstrap=False):
        changed = False
        if generation is not None:
            changed = self.begin_generation(generation)
        family = channel_family(family)
        key = (self.generation, family, sid)
        previous = self.last.get(key)

        # Legacy captures did not store connection generation.  A seq=1 book
        # snapshot is the documented subscription bootstrap and is sufficient
        # evidence of a new generation; trade/ticker restarts alone are not.
        current_generation_has_streams = any(
            key_generation == self.generation
            for key_generation, _, _ in self.last)
        if (generation is None and bootstrap and sequence == 1 and
                current_generation_has_streams):
            changed = self.begin_generation()
            key = (self.generation, family, sid)
            previous = None

        if sid is None or sequence is None:
            return SequenceObservation(True, self.generation, family, sid,
                                       generation_changed=changed)
        healthy = previous is None or sequence == previous + 1
        if not healthy:
            self.gaps += 1
        self.last[key] = sequence
        return SequenceObservation(healthy, self.generation, family, sid,
                                   previous, sequence, changed)

    def observe(self, sid, sequence, family="unknown", generation=None,
                bootstrap=False):
        return self.observe_result(sid, sequence, family, generation,
                                   bootstrap).healthy

    def reset(self):
        self.begin_generation()
