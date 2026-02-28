# Kalshi Crypto Pairs

A framework for trading Kalshi's 15-minute crypto prediction markets (BTC, ETH, SOL, XRP).

Each contract asks: **"Will the price go up in the next 15 minutes?"** — a binary YES/NO settled by CF Benchmarks RTI 60-second averages at open vs close.

## Architecture

```
kalshi_collector_v3.py          Collect market data + real prices (no auth needed)
        │
        ▼
kalshi_ml_v3.py                 Train XGBoost model on settlement outcomes
        │
        ▼
kalshi_live_paper.py            Paper trade with simulated bankroll
        │
        ▼
kalshi_live_trader.py           Live trade with real money (RSA-PSS auth)
        │
        ▼
monitor.py + kalshi_report.py   Dashboard + email alerts
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect data (no API key needed)

```bash
python3 kalshi_collector_v3.py
```

This polls Kalshi's public market data + Coinbase prices every 2 seconds for 4 hours. Outputs `kalshi_v3.db` (SQLite) and `kalshi_v3_features.csv`.

### 3. Check data quality

```bash
python3 kalshi_quality_check_v2.py
```

### 4. Train ML model

```bash
python3 kalshi_ml_v3.py
```

Trains XGBoost to predict settlement outcomes (YES/NO) using features like time-to-expiry, price-vs-strike, orderbook imbalance, and momentum. Reports Brier score, calibration, feature importance, and simulated PnL.

### 5. Paper trade

```bash
python3 kalshi_live_paper.py --duration 7200
```

Runs the strategy with a simulated bankroll. Records all trades to SQLite for analysis.

### 6. Live trade (requires API key)

```bash
cp .env.example .env
# Edit .env with your Kalshi API credentials
python3 kalshi_live_trader.py --demo      # Demo mode first
python3 kalshi_live_trader.py             # Real money
```

## Configuration

Copy `.env.example` to `.env` and fill in your values. See the file for all available environment variables.

### Getting Kalshi API Access

1. Create an account at [kalshi.com](https://kalshi.com)
2. Go to Settings → API Keys
3. Generate an RSA key pair and download the private key (`.pem` file)
4. Set `KALSHI_KEY_ID` and `KALSHI_KEY_FILE` in your `.env`

## Modules

| File | Description |
|------|-------------|
| `kalshi_collector_v3.py` | Data collector — polls markets, orderbooks, and Coinbase prices. Tracks market transitions (ticker changes every 15 min). |
| `kalshi_collector_v2.py` | Legacy collector (v2) — uses cross-asset vw_mid averaging (deprecated, kept for reference). |
| `kalshi_ml_v3.py` | ML training — XGBoost predicting settlement outcomes. Features: TTE, price-vs-strike, OB imbalance, momentum, BTC lead-lag. |
| `kalshi_live_paper.py` | Paper trader — consensus-based strategy (cross-asset z-score agreement), full fee model, Kelly sizing, SQLite logging. |
| `kalshi_live_trader.py` | Live trader — same strategy with real money via Kalshi's RSA-PSS authenticated API. |
| `kalshi_paper_trader.py` | Walk-forward backtester — rule-based vs ML strategy comparison with risk management. |
| `backtest_v4.py` | Backtester — tests strategy variants (stale guard, contract cap, TTE window, conflict guard). |
| `kalshi_quality_check_v2.py` | Data quality check for v3 collector output. |
| `kalshi_quality_check.py` | Legacy quality check for v2 collector data. |
| `monitor.py` | Live dashboard — syncs trader DB from EC2 via SCP, renders Rich terminal UI. |
| `kalshi_report.py` | Email reporter — sends 4-hour trading summaries via AWS SES. |
| `analyze_deviation.py` | Analysis — outlier vs consensus trading strategies. |
| `analyze_factors.py` | Analysis — multi-factor impact on win rate (volatility, spread, OB alignment). |
| `analyze_timing.py` | Analysis — optimal entry timing (TTE window sweep). |

## Strategy Overview

The core strategy uses **cross-asset consensus**:

1. Compute a z-score for each asset using price-vs-strike / expected volatility
2. Convert z-score to model probability P(YES) via normal CDF
3. Check consensus: do 3/4 or 4/4 assets agree on direction?
4. Trade aligned assets when consensus is strong (high avg |z-score|)
5. Position size via fractional Kelly criterion with fee-adjusted edge

Key insight: BTC, ETH, SOL, XRP 15-min crypto markets are correlated. When all four signal the same direction, the signal is stronger.

## API Notes

- **Kalshi API**: `https://api.elections.kalshi.com/trade-api/v2` — no auth needed for market data reads
- **Coinbase API**: `https://api.coinbase.com/v2` — public exchange rates
- Rate limit: 20 reads/sec (basic tier)
- Prices in cents: `yes_bid=24` means 24 cents = 24% implied probability
- `floor_strike` = reference price the real price must beat for YES settlement

## License

MIT
