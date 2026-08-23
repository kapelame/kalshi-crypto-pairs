# Phase 3 deterministic research signals

Phase 3 is a read-only, event-time feature and replay layer. It never imports
`trader.py`, never calls an order or portfolio endpoint, and never mutates Phase
2 raw tables. It produces descriptive measurements, not trade recommendations.

## Causal clock and rolling semantics

The engine consumes raw events ordered by `local_receive_timestamp`, then
`processing_timestamp`, stable table priority, and source-table row ID. A
feature snapshot sees only the event currently being processed and earlier
events. All histories are timestamp based, not message-count based.

For a requested window `W`, the historical state is the latest observation at
or before `current_event_time - W`. If none exists, the feature is `null`.
Velocity uses the actual elapsed time between observations rather than assuming
exact sampling:

`velocity_W = (P_now - P_historical) / actual_elapsed_seconds`

Acceleration applies the same causal time-based calculation to the corresponding
velocity history:

`acceleration_W = (velocity_W_now - velocity_W_historical) / elapsed_seconds`

Probability history supports 1, 5, 15, 30, 60, and 120 seconds. Velocity is
reported for 1, 5, 15, 30, and 60 seconds; acceleration for 5, 15, and 30.

## Canonical UP probability

All quote prices use dollar-probability units from 0 to 1. Direct YES quotes and
complementary NO quotes are reconciled into executable YES bounds:

```text
best_yes_bid = max(yes_bid, 1 - no_ask)
best_yes_ask = min(yes_ask, 1 - no_bid)
P_up = (best_yes_bid + best_yes_ask) / 2
spread = best_yes_ask - best_yes_bid
```

All four quote fields are required. Missing, crossed, out-of-range, or older
than `quote_stale_seconds` quotes yield `P_up=null`; no midpoint is fabricated.
Raw yes/no quotes and last trade remain visible separately.

## Underlying and target features

Underlying returns use 1, 5, 15, 30, 60, 120, and 300-second causal histories.
The 300-second realized volatility is `sqrt(sum(log_return^2))`. Also retained:

- most recent jump magnitude (`abs(last log return)`);
- rolling high and low;
- distance from rolling high and low;
- absolute target distance: `reference_price - target`;
- percentage target distance: `(reference_price - target) / target`;
- volatility-adjusted distance:
  `(reference_price - target) / (reference_price * realized_volatility_300s)`.

Probability and underlying histories are separate. Missing or zero volatility
produces a null z-score rather than an infinite value.

## Basket, dispersion, and BTC factor

With configurable `neutral_band=0.02`, a market is UP above 0.52, DOWN below
0.48, and neutral inside the band. Snapshots contain counts, `5/5`/`4/5`/mixed
breadth text, mean/median probability and velocity, synchronized count,
population standard deviations, range, median absolute deviation, and:

`normalized_dispersion = probability_stddev / mean(sqrt(P * (1-P)))`

The four-alt basket is calculated separately. Each alt reports probability,
30-second velocity, and target-distance z-score minus BTC. These are relative
measurements, not causal claims.

`signals.research.lagged_correlations` is retrospective only. It compares
completed BTC/alt measurements at candidate lags 0, 1, 2, 5, 10, 15, and 30
seconds. It is not called by live features and never makes a future value
available to an earlier snapshot.

## Books and trades

WebSocket books are reconstructed from snapshots and signed deltas. Phase 2
explicitly uses `use_yes_price=true`, so NO-side WebSocket levels are treated as
YES ask-scale levels. REST NO bids are complemented with `1 - no_bid`.

Fresh books report level-1, top-3, and top-5 imbalance, five-level bid/ask
depth, depth ratio, spread through canonical quotes, and a simple cumulative
five-level price slope. `book_age_seconds` and `book_fresh` are always present.
If age exceeds `book_stale_seconds=10`, depth features become null.

Trade windows are 5, 15, 30, and 60 seconds. They include count, total
contracts, average/max size, YES/NO aggressive contracts, signed imbalance, and
recent-half minus previous-half volume rate. Direction uses documented
`taker_outcome_side`, falling back to documented legacy `taker_side`. If neither
is `yes` or `no`, directional fields are null; direction is never guessed.

## Contract time, prior windows, and resets

Every snapshot reports seconds since open and seconds until close. Checkpoint
flags fire exactly once, on the first valid event at or after 30, 60, 120, 180,
300, and 600 seconds. Events need not arrive at the exact boundary.

On each immutable `contract_reset`, the old contract is summarized with result,
open/close reference, signed/absolute return, final/extreme/max/min probability,
and realized volatility. Basket prior state includes breadth, 5-of-5 and 4-of-5
direction, mean return, and extremity. New-contract quote, book, and trade state
is cleared; underlying price history remains available because it occurred in
the past and is needed for causal volatility context.

At each new-window checkpoint, current per-asset, basket, BTC-factor, book,
trade, and target-distance features coexist with frozen prior-window context.
Reset labels are descriptive:

- `RESET_UP_CANDIDATE` / `RESET_DOWN_CANDIDATE`: prior and current synchronized direction agree;
- `REVERSAL_UP_CANDIDATE` / `REVERSAL_DOWN_CANDIDATE`: they differ;
- `FRAGMENTED`: insufficient or mixed state.

## Deterministic regime labels

All thresholds live in `signals/config.py`:

| Setting | Default | Purpose |
|---|---:|---|
| `neutral_band` | 0.02 | neutral probability band around 0.50 |
| `synchronized_min_assets` | 4 | synchronized breadth minimum |
| `synchronization_velocity_tolerance` | 0.01 probability/s | effectively flat velocity band |
| `low_dispersion_threshold` | 0.08 | descriptive low-dispersion reference |
| `acceleration_epsilon` | 0.0001 probability/s² | accelerating/decelerating split |
| `quote_stale_seconds` | 10 | quote validity |
| `book_stale_seconds` | 10 | book validity |

Labels are `SYNC_UP_ACCELERATING`, `SYNC_UP_DECELERATING`,
`SYNC_DOWN_ACCELERATING`, `SYNC_DOWN_DECELERATING`, `FRAGMENTED`, and
`NEUTRAL`, plus reset labels above. They segment research data only.

## Derived storage and anti-leakage design

`kalshi_features_v3.db` is separate from raw storage. `feature_snapshots`
contains timestamp, asset, ticker, open/close, target, raw event watermark,
monotonic replay ordinal, feature JSON, and basket JSON. Snapshot IDs are
content-addressed SHA-256 values. `contract_outcomes` is a separate immutable
table. UPDATE and DELETE triggers protect both.

Checkpoint export first freezes the earliest snapshot at or after each
checkpoint, then performs a separate outcome join whose settlement timestamp
must be at or after the frozen snapshot. Settlement never enters feature JSON.
The settlement export row uses the final already-frozen pre-settlement state.

Tests explicitly verify that later trades, prices, books, resets, and settlement
cannot modify earlier Python snapshots or replay output. Stop-at replay prevents
later events from being loaded into the engine at all.

## Replay architecture

`RawEventReader` opens Phase 2 SQLite in read-only mode and performs a stable
merge across typed raw tables. `ReplayEngine` and `monitor_signals.py` both call
the same `SignalEngine.process(event)` method. Replay supports original timing,
speed multipliers, maximum speed (`--speed 0`), inclusive stop timestamp, and
one-market filtering. Existing feature databases are never overwritten.

## Commands

```bash
# Deterministic max-speed replay
.venv/bin/python replay_stream.py --db kalshi_stream_raw.db --speed 0

# 20x replay, one market, bounded in time
.venv/bin/python replay_stream.py --db kalshi_stream_raw.db --speed 20 \
  --market KXBTC15M-... --stop-at 2026-08-23T20:40:00Z

# Read-only terminal monitor over an actively appended raw DB
.venv/bin/python monitor_signals.py --db kalshi_stream_raw.db

# Leakage-safe frozen checkpoint export
.venv/bin/python export_checkpoints.py --features-db kalshi_features_v3.db

# Descriptive completed-window summary
.venv/bin/python research_summary.py --features-db kalshi_features_v3.db
```
