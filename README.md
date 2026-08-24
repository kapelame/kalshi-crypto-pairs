# Kalshi Trading Framework

An end-to-end framework for trading on [Kalshi](https://kalshi.com) — the first regulated US event contract exchange. Covers data collection, ML training, backtesting, paper trading, live execution, and real-time monitoring.

Currently wired for **crypto 15-minute markets** (BTC, ETH, SOL, XRP, DOGE), but the architecture generalizes to any Kalshi series. **Strategy logic is intentionally left empty** — bring your own signals.

## How It Works

Each contract asks: *"Will the crypto price go up in the next 15 minutes?"*

- Settlement: if CF Benchmarks 60s average at close >= 60s average at open → YES wins
- Prices in cents: `yes_bid=24` means 24% implied probability
- New market opens every 15 minutes with a fresh `floor_strike` reference price

The framework handles everything except the trading strategy: API integration, orderbook data, real-time crypto prices, fee calculation, risk management, order execution, settlement tracking, and monitoring.

## Quick Start

### 1. Install

```bash
git clone https://github.com/kapelame/kalshi-crypto-pairs.git
cd kalshi-crypto-pairs
pip install -r requirements.txt
```

### 2. Collect data (no API key needed)

```bash
python3 collector.py
```

Polls Kalshi markets + Coinbase prices every 2s for 4 hours. Outputs SQLite + CSV. Sample data is included in `data/` so you can skip this step initially.

### Phase 2 read-only stream recording

The high-fidelity recorder keeps the REST collector intact and writes an
independent append-only raw-event database. Kalshi requires authentication for
the WebSocket handshake even though the subscribed market-data channels are
read-only.

```bash
.venv/bin/python diagnose_stream.py --duration 60
.venv/bin/python collect_stream.py
```

See [PHASE2_STREAMING.md](PHASE2_STREAMING.md) for the event schema, credentials,
timestamp guarantees, channels, and recovery behavior. Neither command imports
or invokes the trading client.

### Phase 3 deterministic research features

Phase 3 replays immutable raw events into a separate append-only feature store.
It provides causal probability, acceleration, breadth, dispersion, book/trade,
target-distance, prior-window, and reset-regime measurements without models or
trading decisions.

```bash
.venv/bin/python replay_stream.py --db kalshi_stream_raw.db --speed 0
.venv/bin/python monitor_signals.py --db kalshi_stream_raw.db
.venv/bin/python export_checkpoints.py --features-db kalshi_features_v3.db
.venv/bin/python research_summary.py --features-db kalshi_features_v3.db
```

See [PHASE3_SIGNALS.md](PHASE3_SIGNALS.md) for formulas, thresholds, replay
ordering, derived schema, and structural leakage guards.

### 3. Check data quality

```bash
python3 quality_check.py
```

### 4. Train an ML model

```bash
python3 train.py
```

Trains XGBoost on settlement outcomes. Reports accuracy, Brier score, calibration, and simulated PnL. Uses `data/sample_features.csv` if no collector data is available.

### 5. Paper trade

```bash
python3 paper_trader.py --duration 7200
```

Runs against live markets with simulated bankroll. All trades logged to SQLite.

### 6. Live trade (requires API key)

```bash
python3 trader.py --demo      # Demo mode first
python3 trader.py             # Real money
```

**Important**: The strategy function `your_strategy()` in `trader.py` is empty by default. You must implement your own signal logic before trading. See the docstring and comments in the strategy section for guidance.

### 7. Monitor

```bash
python3 dashboard.py
```

Real-time terminal dashboard: live market data (Kalshi odds, Coinbase prices, orderbook), account balance/positions (if API key set), trade history, and performance stats. Refreshes every 2s.

## API Keys

REST market data requires **no authentication**. Kalshi's WebSocket connection
handshake and all trading/account endpoints require API keys. The Phase 2
recorder uses credentials only to establish a read-only WebSocket connection:

1. Create account at [kalshi.com](https://kalshi.com)
2. Go to **Settings → API Keys**
3. Generate an RSA key pair, download the private key (`.pem` file)
4. Configure:

```bash
cp .env.example .env
# Edit .env:
# KALSHI_KEY_ID=your-key-id-here
# KALSHI_KEY_FILE=/absolute/path/outside-repo/kalshi_private.pem
```

API docs: [Kalshi API Reference](https://trading-api.readme.io/reference/getting-started)

## Deploying on AWS

To run the trader 24/7 on EC2:

```bash
# 1. Launch an EC2 instance (t3.micro is sufficient)
# 2. SSH in and clone the repo
ssh ec2-user@your-ec2-ip
git clone https://github.com/kapelame/kalshi-crypto-pairs.git
cd kalshi-crypto-pairs
pip3 install -r requirements.txt

# 3. Upload your API key
scp kalshi_private.pem ec2-user@your-ec2-ip:~/kalshi-crypto-pairs/

# 4. Set up environment
cp .env.example .env
nano .env  # fill in KALSHI_KEY_ID and KALSHI_KEY_FILE

# 5. Run trader in background
nohup python3 trader.py --duration 86400 > /dev/null 2>&1 &

# 6. Monitor from your local machine
# Set TRADER_DB to sync the DB, or just run:
scp ec2-user@your-ec2-ip:~/kalshi-crypto-pairs/kalshi_live_trader.db .
python3 dashboard.py
```

For persistent deployment, use `systemd`, `screen`, or `tmux`.

## Sample Data

`data/sample_features.csv` contains ~2,500 snapshots (~4 hours) with 67 columns per row:

| Feature | Description |
|---------|-------------|
| `{asset}_mid` | Market mid price (probability) |
| `{asset}_spread` | Bid-ask spread |
| `{asset}_time_to_expiry` | Seconds until settlement |
| `{asset}_floor_strike` | Reference price to beat |
| `{asset}_real_price` | Coinbase spot price |
| `{asset}_price_vs_strike_pct` | % deviation from strike |
| `{asset}_ob_imbalance` | Orderbook YES/NO ratio (-1 to +1) |
| `{asset}_mom_*` | Price momentum (5s, 15s, 30s) |

Use this data with `train.py` to build and evaluate models before collecting your own.

## Modules

| File | Description |
|------|-------------|
| `collector.py` | Polls Kalshi + Coinbase every 2s. Outputs SQLite + CSV. |
| `quality_check.py` | Data quality report: gaps, completeness, settlement coverage. |
| `train.py` | XGBoost model training on settlement outcomes. |
| `backtester.py` | Walk-forward backtester with configurable strategy variants. |
| `paper_trader.py` | Live paper trading — real markets, simulated bankroll. |
| `trader.py` | Live trading — RSA-PSS authenticated orders on Kalshi. |
| `dashboard.py` | Terminal dashboard — live markets, positions, trade history. |
| `collect_stream.py` | Read-only five-asset WebSocket/REST raw-event recorder. |
| `diagnose_stream.py` | Time-limited streaming health and latency diagnostic. |
| `replay_stream.py` | Deterministic raw-event replay into derived features. |
| `benchmark_signals.py` | Read-only bounded/full signal performance benchmark. |
| `monitor_signals.py` | Read-only five-asset research signal monitor. |
| `export_checkpoints.py` | Frozen checkpoint export with post-freeze outcomes. |
| `research_summary.py` | Descriptive completed-window statistics. |

Phase 3.2 replay defaults to research checkpoints rather than per-event feature
rows:

```bash
python replay_stream.py --db raw_stream.db --features-db features.db \
  --snapshot-mode checkpoints --speed 0
```

Use `--snapshot-mode diagnostic --diagnostic-hz 1` for throttled diagnostics.
`--snapshot-mode all` is retained only for deep debugging. See
`PHASE3_2_PERFORMANCE.md` for architecture and measured benchmarks.

## Fee Model

Kalshi charges per-contract fees based on `price * (1 - price)`:

- **Taker**: `ceil(0.07 * contracts * P * (1-P) * 100) / 100`
- **Maker**: `ceil(0.0175 * contracts * P * (1-P) * 100) / 100`

Fees are highest at P=50% ($0.02/contract) and zero at P=0% or P=100%.

## License

MIT
