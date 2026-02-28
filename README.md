# Kalshi Trading Framework

An end-to-end framework for trading on [Kalshi](https://kalshi.com) — the first regulated US event contract exchange. Covers data collection, ML model training, backtesting, paper trading, live execution, and monitoring.

Currently implemented for **crypto 15-minute markets** (BTC, ETH, SOL, XRP), but the architecture generalizes to any Kalshi event series. The collector, ML pipeline, fee model, risk manager, and execution layer are all modular and reusable.

## Architecture

```
Collector               Polls Kalshi API + external data sources (no auth needed)
    │
    ▼
ML Training             XGBoost / GradientBoosting on settlement outcomes
    │
    ▼
Backtester              Walk-forward CV, strategy variant sweeps
    │
    ▼
Paper Trader            Simulated bankroll, full fee model, SQLite logging
    │
    ▼
Live Trader             Real money via Kalshi RSA-PSS authenticated API
    │
    ▼
Monitor + Reports       Rich terminal dashboard, email alerts via AWS SES
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

Polls Kalshi public market data + Coinbase prices every 2s for 4 hours. Outputs SQLite + CSV. To adapt for other markets, change the `SERIES` dict at the top of the file.

### 3. Check data quality

```bash
python3 kalshi_quality_check_v2.py
```

### 4. Train ML model

```bash
python3 kalshi_ml_v3.py
```

Trains a model to predict settlement outcomes. Reports Brier score, calibration, feature importance, and simulated PnL.

### 5. Paper trade

```bash
python3 kalshi_live_paper.py --duration 7200
```

Runs the strategy with a simulated bankroll. All trades logged to SQLite.

### 6. Live trade (requires API key)

```bash
cp .env.example .env
# Edit .env with your Kalshi API credentials
python3 kalshi_live_trader.py --demo      # Demo mode first
python3 kalshi_live_trader.py             # Real money
```

## Configuration

Copy `.env.example` to `.env` and fill in your values.

### Getting Kalshi API Access

1. Create an account at [kalshi.com](https://kalshi.com)
2. Go to Settings → API Keys
3. Generate an RSA key pair and download the private key (`.pem` file)
4. Set `KALSHI_KEY_ID` and `KALSHI_KEY_FILE` in your `.env`

## Adapting to Other Markets

The framework is designed around Kalshi's series/ticker structure. To trade a different market:

1. **Collector**: Change the `SERIES` dict to target your event series (e.g., weather, economics, politics). Add any relevant external data sources alongside the Kalshi orderbook data.
2. **Features**: Modify the feature engineering in `kalshi_ml_v3.py` to match your market's drivers. The current features (time-to-expiry, price-vs-strike, orderbook imbalance, momentum) are general enough for any time-bounded contract.
3. **Strategy**: The consensus logic assumes multiple correlated contracts. For single-contract markets, simplify to model probability vs market price edge.
4. **Risk**: The Kelly sizing, fee model, and exposure limits in the paper/live traders are market-agnostic.

## Modules

| File | Description |
|------|-------------|
| **Data Collection** | |
| `kalshi_collector_v3.py` | Polls markets, orderbooks, and external prices. Tracks market transitions. |
| `kalshi_collector_v2.py` | Legacy collector using cross-asset averaging (kept for reference). |
| `kalshi_quality_check_v2.py` | Data quality report: gaps, completeness, TTE coverage, settlement lookup. |
| **ML & Backtesting** | |
| `kalshi_ml_v3.py` | XGBoost model training on settlement outcomes with calibration analysis. |
| `kalshi_paper_trader.py` | Walk-forward backtester — rule-based vs ML comparison with full risk management. |
| `backtest_v4.py` | Strategy variant sweep (stale guard, contract cap, TTE window, conflict filter). |
| **Trading** | |
| `kalshi_live_paper.py` | Paper trader — consensus strategy, fee model, Kelly sizing, SQLite logging. |
| `kalshi_live_trader.py` | Live trader — RSA-PSS authenticated execution on Kalshi. |
| **Monitoring** | |
| `monitor.py` | Rich terminal dashboard — syncs trader DB from remote server via SCP. |
| `kalshi_report.py` | Email reporter — periodic trading summaries via AWS SES. |
| **Analysis** | |
| `analyze_deviation.py` | Outlier vs consensus strategy comparison. |
| `analyze_factors.py` | Multi-factor win rate analysis (volatility, spread, OB alignment). |
| `analyze_timing.py` | Optimal entry timing via TTE window sweep. |

## Key Concepts

### Kalshi API

- **Base URL**: `https://api.elections.kalshi.com/trade-api/v2`
- **Auth**: Not needed for market data reads. RSA-PSS signing required for orders.
- **Rate limit**: 20 reads/sec (basic tier)
- **Prices**: In cents — `yes_bid=24` means 24% implied probability
- **Structure**: Series (e.g., `KXBTC15M`) → one open market at a time → ticker changes on each period

### Strategy (Crypto Implementation)

The included crypto strategy uses **cross-asset consensus**:

1. Compute z-score per asset: (price - strike) / expected_volatility
2. Convert to model probability P(YES) via normal CDF
3. Check agreement across correlated assets (3/4 or 4/4 consensus)
4. Trade when consensus is strong, size via fractional Kelly with fee-adjusted edge

This approach exploits correlation between related contracts. The same principle applies to any set of correlated Kalshi markets.

### Fee Model

Kalshi charges per-contract fees based on `price * (1 - price)`:
- **Taker**: `ceil(0.07 * contracts * P * (1-P) * 100) / 100`
- **Maker**: `ceil(0.0175 * contracts * P * (1-P) * 100) / 100`

Fees are highest at P=50% and zero at P=0% or P=100%.

## License

MIT
