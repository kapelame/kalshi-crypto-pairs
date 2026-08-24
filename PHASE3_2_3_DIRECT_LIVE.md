# Phase 3.2.3 — Direct live SignalEngine delivery

This phase is a read-only research-state delivery layer. It imports no order API
and does not alter `trader.py`, strategy logic, models, sizing, or execution.

## Architecture

All normalized producers publish `RawEvent` objects to one bounded-FIFO
`CausalIngestDispatcher`. Its single worker:

1. commits the raw row and compact global ingest-ledger row atomically;
2. obtains the authoritative `ingest_sequence`;
3. constructs one canonical `EventEnvelope`;
4. immediately delivers that envelope to one long-lived `SignalEngine`;
5. refreshes cached basket state and completes the producer publication.

The ordering is durability-first. There is no SQLite query, polling interval, or
ledger-tail loop between commit and live mutation. Terminal redraw cadence only
reads cached state. The SQLite-tail `monitor_signals.py` remains an independent
research observer and `RawEventReader` remains the offline replay source.

## Startup handoff

Before its writer starts, the dispatcher captures the committed ledger tail and
replays through exactly that sequence. Publications arriving during catch-up
wait in the bounded FIFO. The writer then starts at the database's next
sequence. Thus the handoff is `persisted <= captured tail`, followed by queued
direct publications `> tail`, with no duplicate or gap.

## Failure and backpressure

A failed raw transaction is rolled back and never reaches the engine. The
dispatcher enters `RAW_PERSISTENCE_UNHEALTHY`, stops accepting observations, and
surfaces the exception. An engine failure enters `SIGNAL_ENGINE_UNHEALTHY`; the
causal-order assertion remains unchanged.

The queue is bounded. Producers await capacity instead of dropping events.
Crossing the safety threshold exposes `DIRECT_BACKPRESSURE_UNHEALTHY` while
causal processing drains the backlog.

## Timing

Instrumentation measures availability to sequence/persistence completion,
engine mutation, and complete cached basket state. Optional terminal state age
is separate. HTTP `request_started_at` remains separate from local availability.

## Commands

```bash
python collect_live_signals.py --db direct_raw.db --env-file .env
python collect_live_signals.py --db direct_raw.db --duration 180 --no-render
python monitor_signals.py --db direct_raw.db --refresh 0.25
python benchmark_direct_live.py --rates 3000 5000 10000 --duration 1
```

## Limitations

SQLite commit latency is intentionally paid before live mutation. Extreme bursts
can create a visible bounded backlog; no event is dropped or reordered. This is
not process-level high availability. The prior aiohttp `Unclosed client session`
warning remains a separate follow-up.
