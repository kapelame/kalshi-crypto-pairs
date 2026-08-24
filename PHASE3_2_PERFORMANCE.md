# Phase 3.2 performance architecture

Phase 3.2 separates immutable raw events, per-event in-memory state, optional
diagnostic snapshots, and research checkpoints. It does not contain an order or
trading path.

## Before and after

Previously every raw event constructed, retained, serialized, inserted, and
committed a complete per-asset feature and basket snapshot. Book levels were
sorted repeatedly, trade and price windows were rescanned, and replay loaded all
raw tables into one Python list. The incomplete validation replay produced
466,254 rows and about 2.26 GiB before it was stopped.

Now `SignalEngine.process(..., emit_snapshot=False)` applies every causal state
mutation without materializing unrelated feature families. Full immutable
snapshots are materialized only for the selected persistence policy. The old
per-event behavior remains available as `--snapshot-mode all` for debugging.

## Incremental algorithms

- Order books retain sorted price-key arrays updated with binary insertion and
  deletion. L1/top-3/top-5 depth, imbalance, ratio, and slope are refreshed only
  when a book message changes the book; unrelated events never sort a book.
- Trade windows use per-window recent/earlier deques, running counts and sums,
  directional volumes, and monotonic maximum queues. Expiration is incremental.
- Timestamp histories use binary search for causal at-or-before retrieval.
  Probability velocity and acceleration preserve elapsed-time semantics.
- Price features update on underlying-price observations. Snapshot-time
  evaluation preserves causal rolling cutoffs and target metadata changes.
- Basket state is computed once and shared across same-watermark checkpoint or
  diagnostic snapshot groups. It is not recomputed for raw deltas unless a
  snapshot consumer needs it.

## Streaming replay

`RawEventReader` opens the raw database read-only and maintains one SQLite cursor
per typed raw table. A heap performs a stable k-way merge using:

1. local receive timestamp;
2. processing timestamp;
3. fixed table priority;
4. table rowid.

Python memory is bounded to cursor state and one heap entry per table. Start/stop
queries use append-order rowid bounds with a conservative lookaround and exact
timestamp predicates. SQLite sorts each bounded table stream by the canonical
keys. Full replay streams each table without building a Python event list.

Replay returns counters and a deterministic logical digest by default. Retaining
snapshots in a Python list is explicit test/debug behavior only.

## Persistence modes

- `checkpoints` (default): T+30, T+60, T+120, T+180, T+300, T+600, plus one
  terminal snapshot per outcome.
- `diagnostic`: checkpoints plus a configurable cadence, default 1 Hz per asset.
- `all`: one snapshot for every raw event. This is intentionally non-default.

Every checkpoint stores the scheduled timestamp, first actual causal snapshot
timestamp at or after the schedule, delay in milliseconds, asset eligibility,
exclusion reasons, basket eligible count, coherent-window flag, window identity,
and raw watermark. An imperfect checkpoint is retained rather than discarded.
Outcomes remain physically separate and are joined only after feature freezing.

## SQLite writes and schema

Writes run in configurable batches (default 500) with rollback on failure and a
final flush on normal shutdown or cancellation. Core checkpoint columns are
normalized in `feature_snapshots`. Same-watermark basket state is stored once in
`basket_snapshots` and referenced by digest ID. `contract_outcomes` remains a
separate immutable table. Only the market/checkpoint research index is created.

No migration is attempted for incomplete legacy derived databases, and raw
databases are never altered.

## Monitor and backpressure

The monitor keeps one engine, processes each raw event once with no persistence,
and renders cached state. Catch-up work is moved off the asyncio event-loop
thread and yields every bounded processing chunk. The display reports processed
rate, queued merge rows, maximum backlog, catch-up lag, and processing p50/p95/p99.
Raw events are never dropped or reordered.

## Commands

Checkpoint replay:

```bash
python replay_stream.py --db raw_stream.db --features-db features.db \
  --snapshot-mode checkpoints --speed 0
```

One-hertz diagnostics:

```bash
python replay_stream.py --db raw_stream.db --features-db diagnostics.db \
  --snapshot-mode diagnostic --diagnostic-hz 1
```

Read-only benchmark:

```bash
python benchmark_signals.py --db raw_stream.db --start TIMESTAMP --stop TIMESTAMP \
  --snapshot-mode checkpoints --no-persist
```

## Measured results

On the 19:00:30–19:01:30 ET Phase 3.1 capture, 17,752 events replayed at
7,497–15,735 events/sec across repeated cold/warm runs; a persistence-enabled
two-minute run processed 27,722 events at 12,075 events/sec and wrote 278,528
bytes. The prior full-snapshot engine measured about 647 events/sec on this data.

The 1,636,090-event Phase 3 capture completed checkpoint replay in 105.36 seconds
(15,528 events/sec), generated 90 scheduled checkpoints plus 10 terminal rows,
used about 52.3 MB peak RSS, and wrote 565,248 bytes. Across the exact 35:58.821
capture span this projects to approximately 0.90 MiB/hour, versus the prior
projected 10–14 GB/hour architecture.

## Limitations

- Diagnostic and `all` modes intentionally use more storage than checkpoints.
- Start/stop rowid acceleration relies on the append-only recorder's near-causal
  insertion property and uses a 10,000-row lookaround plus exact timestamp
  filtering. Full replay does not rely on this bound.
- Python dictionaries remain the authoritative full reconstructed books; the
  sorted arrays optimize top-level queries rather than replacing raw fidelity.
