"""Persistence-first causal dispatch to the in-memory research SignalEngine."""

import asyncio
import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass

from signals.engine import SignalEngine
from signals.ordering import event_from_persisted
from signals.replay import RawEventReader
from streaming.timeutil import parse_timestamp


RAW_PERSISTENCE_UNHEALTHY = "RAW_PERSISTENCE_UNHEALTHY"
ENGINE_UNHEALTHY = "SIGNAL_ENGINE_UNHEALTHY"
BACKPRESSURE_UNHEALTHY = "DIRECT_BACKPRESSURE_UNHEALTHY"


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


@dataclass(frozen=True)
class EventEnvelope:
    """The canonical object shared by direct live processing and replay."""
    event: dict

    @property
    def ingest_sequence(self):
        return self.event["_ingest_sequence"]

    @property
    def event_id(self):
        return self.event["event_id"]


class LiveSignalService:
    """One long-lived, event-driven SignalEngine with a coherent cached state."""
    def __init__(self, engine=None, latency_sample_limit=200_000,
                 identity_history_limit=200_000):
        self.engine = engine or SignalEngine()
        self.latest_ingest_sequence = None
        self.raw_watermark = None
        self.state_timestamp = None
        self.state_available_at = None
        self.cached_basket = {}
        self.health = "HEALTHY"
        self.events_processed = 0
        self.ordering_violations = 0
        self.consumed_identity = deque(maxlen=identity_history_limit)
        self.engine_latency_ms = deque(maxlen=latency_sample_limit)
        self.state_latency_ms = deque(maxlen=latency_sample_limit)
        self._digest = hashlib.sha256()

    def consume(self, envelope):
        event = envelope.event if isinstance(envelope, EventEnvelope) else envelope
        available = parse_timestamp(event["local_receive_timestamp"]).timestamp()
        try:
            self.engine.process(event, emit_snapshot=False)
        except ValueError:
            self.ordering_violations += 1
            self.health = ENGINE_UNHEALTHY
            raise
        engine_done = time.time()
        self.engine_latency_ms.append(max(0.0, (engine_done - available) * 1000))
        # Basket computation is part of complete cached research state. Asset
        # state is already held by the same long-lived engine.
        self.cached_basket = self.engine.basket_features(self.engine.last_event_time)
        state_done = time.time()
        self.state_latency_ms.append(max(0.0, (state_done - available) * 1000))
        self.latest_ingest_sequence = event.get("_ingest_sequence")
        self.raw_watermark = event["event_id"]
        self.state_timestamp = event["local_receive_timestamp"]
        self.state_available_at = state_done
        self.events_processed += 1
        identity = (self.latest_ingest_sequence, event["event_id"])
        self.consumed_identity.append(identity)
        self._digest.update(json.dumps(identity, separators=(",", ":")).encode())

    def catch_up(self, db_path, through_sequence=None):
        for event in RawEventReader(db_path).read():
            sequence = event.get("_ingest_sequence")
            if through_sequence is not None and sequence > through_sequence:
                break
            self.consume(event)

    @property
    def digest(self):
        return self._digest.hexdigest()

    def coherent_snapshots(self):
        if self.engine.last_event_time is None:
            return {}, {}
        basket = self.cached_basket
        snapshots = {}
        for asset in self.engine.assets:
            snapshot = self.engine.snapshot(
                asset, self.engine.last_event_time, self.raw_watermark,
                emit_checkpoints=False, basket=basket)
            snapshot["basket"] = basket
            snapshots[asset] = snapshot
        return snapshots, basket

    def latency_summary(self):
        return {
            "engine_p50_ms": _percentile(self.engine_latency_ms, .50),
            "engine_p95_ms": _percentile(self.engine_latency_ms, .95),
            "engine_p99_ms": _percentile(self.engine_latency_ms, .99),
            "state_p50_ms": _percentile(self.state_latency_ms, .50),
            "state_p95_ms": _percentile(self.state_latency_ms, .95),
            "state_p99_ms": _percentile(self.state_latency_ms, .99),
            "engine_max_ms": max(self.engine_latency_ms, default=None),
            "state_max_ms": max(self.state_latency_ms, default=None),
        }


@dataclass
class _Submission:
    event: object
    completed: asyncio.Future


class CausalIngestDispatcher:
    """Single-writer persistence followed immediately by ordered live delivery."""
    def __init__(self, store, service, max_queue=100_000,
                 unhealthy_threshold=50_000):
        self.store = store
        self.service = service
        self.queue = asyncio.Queue(maxsize=max_queue)
        self.unhealthy_threshold = unhealthy_threshold
        self.health = "HEALTHY"
        self.worker = None
        self.accepting = False
        self.max_queue_depth = 0
        self.events_submitted = 0
        self.events_delivered = 0
        self.persistence_failures = 0
        self.dropped_events = 0
        self.sequence_assignment_latency_ms = deque(maxlen=200_000)
        self.persistence_latency_ms = deque(maxlen=200_000)

    async def start(self, catch_up=False):
        if self.worker is not None:
            return
        # Producers cannot publish until catch-up and registration complete.
        # Since this dispatcher is the sole writer, the captured ledger tail is
        # an atomic handoff boundary.
        self.accepting = True
        if catch_up and self.store.ingest_enabled:
            tail = self.store.connection.execute(
                "SELECT COALESCE(MAX(ingest_sequence),0) FROM raw_ingest_log"
            ).fetchone()[0]
            # Publications may queue while this bounded historical catch-up is
            # in progress. The writer starts only after the captured tail is
            # consumed, making the handoff gapless and duplicate-free.
            await asyncio.to_thread(
                self.service.catch_up, self.store.path, tail)
            self.service.engine_latency_ms.clear()
            self.service.state_latency_ms.clear()
        self.worker = asyncio.create_task(self._run(), name="causal-ingest-dispatcher")

    async def publish(self, event):
        if not self.accepting:
            raise RuntimeError("causal ingest dispatcher is not running")
        if self.health == RAW_PERSISTENCE_UNHEALTHY:
            raise RuntimeError(RAW_PERSISTENCE_UNHEALTHY)
        loop = asyncio.get_running_loop()
        completed = loop.create_future()
        await self.queue.put(_Submission(event, completed))
        self.events_submitted += 1
        depth = self.queue.qsize()
        self.max_queue_depth = max(self.max_queue_depth, depth)
        if depth >= self.unhealthy_threshold:
            self.health = BACKPRESSURE_UNHEALTHY
        return await completed

    async def _run(self):
        while True:
            submission = await self.queue.get()
            if submission.event is None:
                submission.completed.set_result(None)
                self.queue.task_done()
                return
            if self.health in {RAW_PERSISTENCE_UNHEALTHY, ENGINE_UNHEALTHY}:
                submission.completed.set_exception(RuntimeError(self.health))
                self.queue.task_done()
                continue
            event = submission.event
            available = parse_timestamp(event.local_receive_timestamp).timestamp()
            persisted = None
            try:
                persisted = self.store.append(event)
                persist_done = time.time()
                latency = max(0.0, (persist_done - available) * 1000)
                self.sequence_assignment_latency_ms.append(latency)
                self.persistence_latency_ms.append(latency)
                envelope = EventEnvelope(event_from_persisted(event, persisted))
                self.service.consume(envelope)
                self.events_delivered += 1
                if (self.health == BACKPRESSURE_UNHEALTHY and
                        self.queue.qsize() < self.unhealthy_threshold // 2):
                    self.health = "HEALTHY"
                submission.completed.set_result(envelope)
            except Exception as exc:
                if persisted is not None:
                    self.health = ENGINE_UNHEALTHY
                else:
                    self.health = RAW_PERSISTENCE_UNHEALTHY
                    self.persistence_failures += 1
                    self.accepting = False
                submission.completed.set_exception(exc)
            finally:
                self.queue.task_done()

    async def stop(self):
        if self.worker is None:
            return
        await self.queue.join()
        loop = asyncio.get_running_loop()
        completed = loop.create_future()
        await self.queue.put(_Submission(None, completed))
        await completed
        await self.worker
        self.worker = None
        self.accepting = False

    def summary(self):
        return {
            "health": self.health,
            "events_submitted": self.events_submitted,
            "events_delivered": self.events_delivered,
            "dropped_events": self.dropped_events,
            "queue_depth": self.queue.qsize(),
            "max_queue_depth": self.max_queue_depth,
            "persistence_failures": self.persistence_failures,
            "ordering_violations": self.service.ordering_violations,
            "live_identity_digest": self.service.digest,
            "latest_ingest_sequence": self.service.latest_ingest_sequence,
            "eligible_count": self.service.cached_basket.get("eligible_count", 0),
            "coherent_window": self.service.cached_basket.get("coherent_window", False),
            **self.service.latency_summary(),
            "persistence_p50_ms": _percentile(self.persistence_latency_ms, .50),
            "persistence_p95_ms": _percentile(self.persistence_latency_ms, .95),
            "persistence_p99_ms": _percentile(self.persistence_latency_ms, .99),
        }
