# Kalshi polling API audit (research-v0.1)

Audited 2026-08-23 against the current official Kalshi API documentation and
read-only production responses. No authenticated or trading endpoint was used.

## Existing calls and failure mode

`collector.py` made two Kalshi calls:

1. `GET /trade-api/v2/markets?series_ticker=<series>&status=open&limit=1`
2. `GET /trade-api/v2/markets/{ticker}/orderbook?depth=10`

It expected integer-cent market fields `yes_bid`, `yes_ask`, `volume`, and
`open_interest`, plus `orderbook.yes` and `orderbook.no` arrays containing
integer-cent price/quantity pairs. It also reads metadata fields that remain
valid: `ticker`, `status`, `floor_strike`, and `close_time`.

The endpoints themselves remain valid. The old production hostname also
remains supported, although `external-api.kalshi.com` is now recommended. The
response schema changed during Kalshi's fixed-point migration:

| Data | Old collector expectation | Current REST field |
|---|---|---|
| YES bid/ask | `yes_bid`, `yes_ask` integer cents | `yes_bid_dollars`, `yes_ask_dollars` fixed-point strings |
| NO bid/ask | not parsed | `no_bid_dollars`, `no_ask_dollars` fixed-point strings |
| Last trade | not parsed | `last_price_dollars` fixed-point string |
| Volume | `volume` integer | `volume_fp` fixed-point contract string |
| Open interest | `open_interest` integer | `open_interest_fp` fixed-point contract string |
| Order book wrapper | `orderbook` | `orderbook_fp` |
| Book sides | `yes`, `no` | `yes_dollars`, `no_dollars` |
| Book levels | integer `[cents, count]` | string `[dollars, count_fp]` |

This explains the split behavior. Discovery, rollover, strike, expiry, status,
and settlement use unchanged metadata fields and valid series filters. Quote
lookups returned `None` because the old keys no longer exist. The collector then
converted missing volume/OI into fake zero with `.get(..., 0)`. Order-book
parsing looked inside a nonexistent wrapper, produced empty sides, and returned
`None`; all non-200 responses and exceptions on that path were silently ignored.

The collected `kalshi_v3.db`/CSV confirms this exactly: 563 rows, zero non-null
BTC mids/bids/order-book imbalances, and volume/OI equal to zero in every row,
while tickers, strikes, expiry time, reference price, and transitions are
populated. The same column pattern is present for ETH, SOL, and XRP.

## Current endpoint/schema source of truth

- Market metadata and all top-of-book/trade/count fields:
  <https://docs.kalshi.com/api-reference/market/get-markets>
- One market (including status, `result`, settlement value/time):
  <https://docs.kalshi.com/api-reference/market/get-market>
- Fixed-point price/count migration:
  <https://docs.kalshi.com/getting_started/fixed_point_migration>
- Order-book semantics and public read-only example:
  <https://docs.kalshi.com/getting_started/orderbook_responses>
- Market lifecycle and settlement/result semantics:
  <https://docs.kalshi.com/getting_started/market_lifecycle>
- Recommended and compatibility hosts:
  <https://docs.kalshi.com/getting_started/api_environments>
- Rate-limit behavior:
  <https://docs.kalshi.com/getting_started/rate_limits>

The current market list response contains `yes_bid_dollars`,
`yes_ask_dollars`, `no_bid_dollars`, `no_ask_dollars`, `last_price_dollars`,
`volume_fp`, `open_interest_fp`, `status`, `result`,
`settlement_value_dollars`, and `settlement_ts`. The one-market endpoint uses
the same market object. `result` becomes `yes` or `no` when determined; a fully
settled REST market has lifecycle status `finalized` (the list filter is still
named `status=settled`).

The order book contains bids only. A YES ask can be inferred from the best NO
bid as `100c - NO bid`; a NO ask can be inferred from the best YES bid. Market
metadata already supplies explicit YES and NO bid/ask fields, so the collector
uses those for top-of-book and uses book levels for imbalance.

## Authentication, naming, and rate limits

No credentials are required for the REST market-list call. Kalshi's official
market-data quick start also performs the order-book request without auth, and
a live unauthenticated read returned HTTP 200 during this audit. The generated
API-reference page currently displays auth headers for that endpoint, which
conflicts with the official quick start and observed behavior. This repair does
not add or request credentials. Auth remains required for portfolio/order APIs
and WebSocket connection setup, neither of which is in scope.

The four configured series names remain valid: `KXBTC15M`, `KXETH15M`,
`KXSOL15M`, and `KXXRP15M`. Live discovery returned the same generated ticker
format already stored by the collector. There is no ticker/series rename behind
this failure.

The old code did detect HTTP 429 for market discovery but did not log the body;
its order-book path silently swallowed all HTTP failures and exceptions. The
repair logs HTTP status/body, timeouts, connection failures, and schema failures,
and retains exponential backoff. Missing fields remain `None`; a real `"0.00"`
is parsed as numeric zero.
