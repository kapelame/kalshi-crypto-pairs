# Phase 2 high-fidelity data layer

This layer is research-only and read-only. It does not import `trader.py`, call
portfolio endpoints, construct orders, or expose order placement. The repaired
REST collector and its historical SQLite/CSV files remain in place.

## Verified instruments and discovery

Active contracts are discovered with:

`GET https://external-api.kalshi.com/trade-api/v2/markets?series_ticker=<series>&status=open&limit=1`

| Asset | Current series |
|---|---|
| BTC | `KXBTC15M` |
| ETH | `KXETH15M` |
| SOL | `KXSOL15M` |
| XRP | `KXXRP15M` |
| DOGE | `KXDOGE15M` |

DOGE was established by querying the current official series list for Crypto,
which returned `KXDOGE15M`, `frequency=fifteen_min`, title `Dogecoin 15 Minute`,
and the CRYPTO15M contract terms. A subsequent current `status=open` query
returned an active generated ticker in the form
`KXDOGE15M-<contract timestamp>-<window suffix>` with fixed-point quotes,
strike, volume, and open interest. The implementation does not synthesize
contract tickers.

## Official interfaces

- REST: <https://external-api.kalshi.com/trade-api/v2>
- WebSocket: <wss://external-api-ws.kalshi.com/trade-api/ws/v2>
- WebSocket connection/authentication:
  <https://docs.kalshi.com/websockets/websocket-connection>
- Ticker schema: <https://docs.kalshi.com/websockets/market-ticker>
- Trade schema: <https://docs.kalshi.com/websockets/public-trades>
- Book schemas: <https://docs.kalshi.com/websockets/orderbook-updates>
- Lifecycle behavior: <https://docs.kalshi.com/getting_started/market_lifecycle>
- Keep-alive: <https://docs.kalshi.com/websockets/connection-keep-alive>

The client subscribes to `ticker`, `trade`, `orderbook_delta`, and
`market_lifecycle_v2`. The `orderbook_delta` subscription produces an initial
`orderbook_snapshot` followed by deltas. It explicitly sends
`use_yes_price=true`, avoiding reliance on Kalshi's documented future default
change for NO-side price scale. REST supplies discovery, initial and sequence-gap
recovery books, rollover validation, prior settlement lookup, and fallback
state.

## Credentials

Kalshi requires the following headers on the WebSocket handshake:

- `KALSHI-ACCESS-KEY`
- `KALSHI-ACCESS-TIMESTAMP`
- `KALSHI-ACCESS-SIGNATURE`

The signature input is exactly
`timestamp_ms + "GET" + "/trade-api/ws/v2"`, signed with RSA-PSS/SHA-256.
Configure only environment variables or an ignored `.env`:

```text
KALSHI_KEY_ID=<placeholder>
KALSHI_KEY_FILE=/absolute/path/outside-repo/private-key.pem
```

`.env` and `*.pem` are ignored. Secret values and private-key contents are never
logged. Without both settings, stream commands exit clearly before creating a
database; unauthenticated REST diagnostics continue to work.

## Append-only database

The default database is `kalshi_stream_raw.db`. It is separate from the legacy
`kalshi_v3.db`. Tables are:

- `market_lifecycle_events`
- `contract_reset_events`
- `ticker_events`
- `trade_events`
- `orderbook_snapshots`
- `orderbook_deltas`
- `underlying_price_events`

Every table uses the same immutable raw envelope:

| Column | Meaning |
|---|---|
| `event_id` | Unique UUID for the received/created event |
| `event_type` | Typed event category |
| `asset` | BTC, ETH, SOL, XRP, or DOGE |
| `market_ticker` | Exact Kalshi market ticker, nullable if unavailable |
| `series_ticker` | Exact Kalshi series |
| `exchange_timestamp` | Exchange-provided time, never replaced by local time |
| `local_receive_timestamp` | UTC time immediately after local receipt |
| `processing_timestamp` | UTC time when the immutable row is constructed |
| `source` | WebSocket, REST discovery/recovery, rollover, or Coinbase REST |
| `raw_payload` | Lossless JSON payload/response |
| `contract_open_time` | Contract open timestamp |
| `contract_close_time` | Contract close timestamp |
| `target` | Floor strike where applicable |
| `sequence` | Exchange WebSocket sequence, nullable when absent |
| `sequence_generation` | Local WebSocket connection/subscription generation, nullable for legacy captures |

SQLite triggers reject UPDATE and DELETE on every raw table. Duplicate
`event_id` inserts are rejected. Missing values remain SQL `NULL`/JSON `null`;
numeric zero remains zero. No derived feature is stored in these tables.

## Time and sequence integrity

All locally generated times are timezone-aware UTC with microsecond formatting.
For exchange time the most precise supplied field is retained in this order:
high-precision `time`, `ts_ms`, then `ts`. The receive-latency helper compares
exchange time with local receipt; processing latency compares local receipt with
processing time.

Sequence identity is `(connection generation, channel family, sid)`. Order-book
snapshot and delta messages share the `orderbook` family; ticker, trade, and
lifecycle channels remain independent. Sequence numbers are not assumed to be
global across channels or connections. Each reconnect/resubscription increments
`sequence_generation`, rotates prior expectations, and permits each new stream
to bootstrap independently. Within one identity, any value other than the prior
sequence plus one is a gap. A gap marks affected state unhealthy, fetches a REST
recovery book, and forces a reconnect/resubscribe. A healthy fresh order-book
snapshot in the new generation restores that market's sequence health.

Legacy raw databases do not contain the generation column. During bounded
replay, a sequence-1 order-book snapshot is treated as trusted evidence of a
new subscription generation; a trade or ticker restart alone is never used to
infer a generation boundary. Persisted lifecycle-v2 rows are not replay-
validated because that subscription is global while the recorder intentionally
retains only lifecycle events belonging to the five tracked markets; the live
transport still validates the complete lifecycle stream before filtering.

## Contract resets

REST discovery is repeated across the five series. When a ticker changes at a
15-minute boundary, an explicit immutable `contract_reset` event records:

- prior and new ticker;
- prior settlement/result if available;
- new target;
- exact new `open_time` (the structural :00/:15/:30/:45 boundary);
- prior contract's final REST market object if available.

Subscriptions then reconnect with the five new active tickers. No directional
relationship between consecutive windows is inferred.

## Commands

REST diagnostic for any asset, including DOGE:

```bash
.venv/bin/python diagnose_kalshi.py --asset DOGE
```

Sixty-second streaming diagnostic:

```bash
.venv/bin/python diagnose_stream.py --duration 60
```

Continuous recording until Ctrl+C:

```bash
.venv/bin/python collect_stream.py
```

The streaming diagnostic reports counts by event type and asset, event rate,
sequence gaps, reconnects, receive-latency distribution, stale markets, all five
active tickers, quote/book validity, and the overall healthy count.

## Known limitations

- A live WebSocket end-to-end run requires user-supplied Kalshi credentials;
  tests use current documented payloads and generated test-only RSA keys.
- Some message classes do not include exchange timestamps. Those rows retain
  `exchange_timestamp=NULL`; latency is not fabricated.
- `market_lifecycle_v2` is subscribed globally and filtered locally to the five
  tracked tickers because lifecycle filtering behavior is less explicit than
  the ticker/trade/book channel documentation.
- Coinbase spot observations are contextual underlying observations, not the CF
  Benchmarks RTI used for official Kalshi settlement. The exact Kalshi strike
  and final result remain sourced from Kalshi REST.
