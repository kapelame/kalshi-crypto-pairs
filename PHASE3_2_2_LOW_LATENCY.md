# Phase 3.2.2 — Low-Latency Causal Ingestion

## Problem

Legacy REST producers sometimes captured `local_receive_timestamp` before the
HTTP request completed. In the validation failure, a Coinbase observation used
`02:54:09.553601` but was not persisted until after an order-book event received
at `02:54:09.639445` had been processed. A timestamp merge therefore needed a
20-second holdback to prove that no earlier receive timestamp remained in flight.

## Timestamp semantics

- `exchange_timestamp`: time supplied by the external exchange, preserved without
  substituting a local clock.
- `request_started_at`: local request initiation time for HTTP observations. It is
  stored in the ingest ledger, not injected into the external raw payload.
- `local_receive_timestamp`: the time a complete WebSocket frame or complete HTTP
  response body is locally available.
- `processing_timestamp`: event normalization/construction time.
- `persistence_timestamp`: raw-store write-boundary time stored in the ingest ledger.
- monitor engine time and render time are measured locally and are not compared to
  an exchange clock except where explicitly labeled.

WebSocket receipt was already correct: it is captured immediately after
`ws.receive()` returns. Coinbase, Kalshi discovery, metadata, settlement lookup,
and order-book recovery now capture response availability after the response body
has been read. Request start remains separately available.

## Global ingest sequence

New databases contain a compact `raw_ingest_log` table:

| Column | Meaning |
| --- | --- |
| `ingest_sequence` | SQLite `INTEGER PRIMARY KEY AUTOINCREMENT` global sequence |
| `event_id` | Unique raw event identity |
| `event_table` | One of the seven raw tables |
| `source_rowid` | Rowid in that raw table |
| `request_started_at` | Optional HTTP request start |
| `persistence_timestamp` | Local write-boundary timestamp |

The raw row and ledger reference are inserted in one `BEGIN IMMEDIATE`
transaction. Both commit or both roll back. Unique constraints prevent duplicate
event references and duplicate `(event_table, source_rowid)` references. Immutable
triggers prohibit ledger updates and deletes. SQLite `AUTOINCREMENT` preserves
continuity across clean shutdown/restart and never reuses a committed sequence.

The ledger duplicates no raw payload. Readers fetch a bounded ledger batch, group
references by table, resolve rowids with bounded `IN` queries, and restore ledger
order.

## Ordering semantics

New capture:

1. The single writer persists an observation.
2. The transaction assigns its global `ingest_sequence`.
3. Live monitor and replay deliver strictly increasing ingest sequences.
4. An external timestamp never retroactively moves information ahead of an event
   that was already available to the live system.

Legacy capture:

- No schema is migrated or backfilled.
- Replay retains receive timestamp, processing timestamp, table priority, rowid
  ordering.
- The monitor retains the bounded Phase 3.2.1 causal-lateness buffer.

`SignalEngine` preserves its causal-order exception. For new captures it validates
strictly increasing ingest sequence; for legacy captures it validates timestamp
order. Feature event time remains the preserved local observation time, while
delivery causality is the ingest sequence.

## Monitor and latency

For new databases the monitor tails `raw_ingest_log` directly. It does not use the
20-second buffer. The display exposes:

- latest ingest sequence;
- source and processing backlog;
- state lag from persistence to current monitor state;
- persistence-to-engine p50/p95/p99;
- engine processing p50/p95/p99;
- engine-to-render lag.

The legacy 20-second setting remains present only for databases without the ledger.
No raw events are dropped when processing falls behind; the ledger is the durable
queue.

## Commands

Fresh capture:

```bash
python collect_stream.py --db phase3_2_2_validation_raw.db
```

Low-latency monitor:

```bash
python monitor_signals.py --db phase3_2_2_validation_raw.db --refresh 0.25
```

Replay uses ingest order automatically when the ledger exists:

```bash
python replay_stream.py --db phase3_2_2_validation_raw.db --speed 0
```

## Limitations

- A database created by legacy code remains a legacy database; it is intentionally
  not mutated or backfilled.
- Persisted time marks the atomic write boundary, not a hardware durable-flush
  acknowledgment.
- Exchange-to-local latency is meaningful only when the external exchange clock is
  comparable and supplied by the event.
- The previously observed `Unclosed client session` shutdown warning is tracked
  separately unless reproduced by code directly changed in this phase.

## Validation

Measured live results are recorded in the Phase 3.2.2 completion report after a
fresh authenticated capture. Tests cover global monotonicity, restart continuity,
transaction rollback, asynchronous REST/WebSocket completion order, exactly-once
monitor delivery, live/replay equivalence, legacy compatibility, burst handling,
and the unchanged causal guard.
