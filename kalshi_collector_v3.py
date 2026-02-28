#!/usr/bin/env python3
"""
Kalshi Crypto 15-Min Collector v3
=================================
v3 vs v2 核心改进:
  1. 采集 floor_strike (参考价) + time_to_expiry (到期剩余秒数)
  2. 采集 orderbook 深度 → 计算 ob_imbalance 方向信号
  3. 采集 Binance 实时价格 → price_vs_strike (价格vs参考价偏离%)
  4. 废弃跨资产均值回归 (vw_mid/deviation 无经济意义)
  5. 改用 per-asset 动量 + lead-lag 特征
  6. 记录 market ticker 便于后续关联结算结果
  7. 跟踪 15-min 市场切换 (transition)

结算规则:
  每15min一个合约, 问 "价格会涨吗?"
  floor_strike = 开盘前60s CF Benchmarks RTI 均价
  结算价 = close前60s RTI 均价
  结算价 >= floor_strike → YES, 否则 NO

存储:
  SQLite: kalshi_v3.db (实时写入)
  CSV:    kalshi_v3_features.csv (结束时导出)
"""

import asyncio
import aiohttp
import sqlite3
import csv
import os
import sys
import time
import json
import math
from datetime import datetime, timezone
from typing import Optional, Dict, List
from collections import deque


# ==============================================================================
# 配置
# ==============================================================================

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
COINBASE_API = "https://api.coinbase.com/v2"

POLL_SEC = 2.0          # 2s per cycle (more requests per cycle than v2)
MAX_RUNTIME = 14400     # 4 hours

SERIES = {
    "BTC": {"series": "KXBTC15M"},
    "ETH": {"series": "KXETH15M"},
    "SOL": {"series": "KXSOL15M"},
    "XRP": {"series": "KXXRP15M"},
}
ASSETS = list(SERIES.keys())

SESSION_REFRESH_SEC = 600
MAX_CONSECUTIVE_ERRORS = 20
BACKOFF_BASE = 1.0
BACKOFF_MAX = 16.0
PROGRESS_INTERVAL = 300
OB_DEPTH = 10

DB_PATH = "kalshi_v3.db"
CSV_EXPORT = "kalshi_v3_features.csv"
W = 105

# Feature table columns (per asset)
ASSET_COLS = [
    "ticker", "mid", "spread", "yes_bid", "yes_ask",
    "volume", "oi", "floor_strike", "time_to_expiry",
    "ob_imbalance", "real_price", "price_vs_strike_pct",
    "mom_5s", "mom_15s", "mom_30s",
]


# ==============================================================================
# Poller: Kalshi Market + Orderbook + Binance
# ==============================================================================

class Poller:
    def __init__(self):
        self._kal_session: Optional[aiohttp.ClientSession] = None
        self._bin_session: Optional[aiohttp.ClientSession] = None
        self._kal_created: float = 0
        self._consecutive_errors: int = 0
        self._total_errors: int = 0
        self._total_polls: int = 0
        self._last_tickers: Dict[str, str] = {}
        self._transitions: List[dict] = []

    async def _kalshi_sess(self) -> aiohttp.ClientSession:
        now = time.time()
        if (self._kal_session is None or self._kal_session.closed
                or now - self._kal_created > SESSION_REFRESH_SEC):
            if self._kal_session and not self._kal_session.closed:
                try: await self._kal_session.close()
                except: pass
            self._kal_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10, connect=5, sock_read=5),
                headers={"Accept": "application/json"})
            self._kal_created = now
        return self._kal_session

    async def _coinbase_sess(self) -> aiohttp.ClientSession:
        if self._bin_session is None or self._bin_session.closed:
            self._bin_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5))
        return self._bin_session

    # --- Kalshi: market data ---
    async def _poll_market(self, asset: str) -> dict:
        info = SERIES[asset]
        try:
            s = await self._kalshi_sess()
            url = f"{KALSHI_API}/markets?series_ticker={info['series']}&status=open&limit=1"
            async with s.get(url) as r:
                if r.status == 429:
                    return {"asset": asset, "error": "rate_limit"}
                if r.status != 200:
                    return {"asset": asset, "error": f"HTTP {r.status}"}
                data = await r.json()

            mkts = data.get("markets", [])
            if not mkts:
                return {"asset": asset, "error": "no_markets"}

            m = mkts[0]
            yb = m.get("yes_bid")  # cents
            ya = m.get("yes_ask")
            mid, spread = None, None
            if yb is not None and ya is not None and yb > 0 and ya > 0:
                mid = (yb + ya) / 200.0
                spread = (ya - yb) / 100.0
            elif yb and yb > 0:
                mid = yb / 100.0
            elif ya and ya > 0:
                mid = ya / 100.0

            # Parse close_time for time_to_expiry
            tte = None
            ct = m.get("close_time", "")
            if ct:
                try:
                    close_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    tte = (close_dt - datetime.now(timezone.utc)).total_seconds()
                    if tte < 0:
                        tte = 0
                except:
                    pass

            ticker = m.get("ticker", "")
            # Detect market transition
            if asset in self._last_tickers and self._last_tickers[asset] != ticker:
                self._transitions.append({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "asset": asset,
                    "old_ticker": self._last_tickers[asset],
                    "new_ticker": ticker,
                    "new_floor_strike": m.get("floor_strike"),
                })
            self._last_tickers[asset] = ticker

            return {
                "asset": asset, "ticker": ticker,
                "status": m.get("status", ""),
                "yes_bid": yb, "yes_ask": ya,
                "mid": mid, "spread": spread,
                "volume": m.get("volume", 0),
                "oi": m.get("open_interest", 0),
                "floor_strike": m.get("floor_strike"),
                "close_time": ct,
                "time_to_expiry": tte,
                "error": None,
            }
        except asyncio.TimeoutError:
            return {"asset": asset, "error": "timeout"}
        except aiohttp.ClientError as e:
            self._kal_session = None
            return {"asset": asset, "error": f"conn:{type(e).__name__}"}
        except Exception as e:
            return {"asset": asset, "error": str(e)[:100]}

    # --- Kalshi: orderbook ---
    async def _poll_orderbook(self, asset: str, ticker: str) -> dict:
        if not ticker:
            return {"asset": asset, "ob_imbalance": None}
        try:
            s = await self._kalshi_sess()
            url = f"{KALSHI_API}/markets/{ticker}/orderbook?depth={OB_DEPTH}"
            async with s.get(url) as r:
                if r.status != 200:
                    return {"asset": asset, "ob_imbalance": None}
                data = await r.json()

            ob = data.get("orderbook", {})
            yes_bids = ob.get("yes", [])
            no_bids = ob.get("no", [])

            yes_qty = sum(qty for _, qty in yes_bids)
            no_qty = sum(qty for _, qty in no_bids)
            total = yes_qty + no_qty

            imbalance = (yes_qty - no_qty) / total if total > 0 else None

            return {
                "asset": asset,
                "ob_imbalance": round(imbalance, 4) if imbalance is not None else None,
                "yes_depth_qty": yes_qty,
                "no_depth_qty": no_qty,
            }
        except:
            return {"asset": asset, "ob_imbalance": None}

    # --- Coinbase: real prices ---
    async def _poll_prices(self) -> Dict[str, Optional[float]]:
        try:
            s = await self._coinbase_sess()
            url = f"{COINBASE_API}/exchange-rates?currency=USD"
            async with s.get(url) as r:
                if r.status != 200:
                    return {a: None for a in ASSETS}
                data = await r.json()

            rates = data.get("data", {}).get("rates", {})
            prices = {}
            for a in ASSETS:
                rate = rates.get(a)
                if rate:
                    prices[a] = round(1.0 / float(rate), 4)
                else:
                    prices[a] = None
            return prices
        except:
            return {a: None for a in ASSETS}

    # --- Combined poll cycle ---
    async def poll_all(self) -> Dict[str, dict]:
        self._total_polls += 1

        # Step 1: market data (parallel)
        market_tasks = [self._poll_market(a) for a in ASSETS]
        market_results = await asyncio.gather(*market_tasks, return_exceptions=True)

        markets = {}
        for i, r in enumerate(market_results):
            a = ASSETS[i]
            if isinstance(r, Exception):
                markets[a] = {"asset": a, "error": str(r)[:100]}
            else:
                markets[a] = r

        # Step 2: orderbook + Binance (parallel)
        ob_tasks = []
        for a in ASSETS:
            ticker = markets[a].get("ticker", "")
            ob_tasks.append(self._poll_orderbook(a, ticker))
        ob_tasks.append(self._poll_prices())

        ob_results = await asyncio.gather(*ob_tasks, return_exceptions=True)
        ob_data = {}
        for i, r in enumerate(ob_results[:-1]):
            a = ASSETS[i]
            if isinstance(r, dict):
                ob_data[a] = r
            else:
                ob_data[a] = {"asset": a, "ob_imbalance": None}

        binance_prices = ob_results[-1] if isinstance(ob_results[-1], dict) else {a: None for a in ASSETS}

        # Merge everything
        has_error = False
        for a in ASSETS:
            m = markets[a]
            m["ob_imbalance"] = ob_data.get(a, {}).get("ob_imbalance")
            m["real_price"] = binance_prices.get(a)

            # price_vs_strike_pct
            rp = m.get("real_price")
            fs = m.get("floor_strike")
            if rp is not None and fs is not None and fs > 0:
                m["price_vs_strike_pct"] = round((rp - fs) / fs * 100, 4)
            else:
                m["price_vs_strike_pct"] = None

            if m.get("error"):
                has_error = True

        if has_error:
            self._consecutive_errors += 1
            self._total_errors += 1
        else:
            self._consecutive_errors = 0

        return markets

    def get_backoff(self) -> float:
        if self._consecutive_errors <= 1:
            return 0
        return min(BACKOFF_BASE * (2 ** (self._consecutive_errors - 2)), BACKOFF_MAX)

    @property
    def stats_str(self) -> str:
        return (f"polls:{self._total_polls} errs:{self._total_errors} "
                f"consec:{self._consecutive_errors}")

    async def close(self):
        for s in [self._kal_session, self._bin_session]:
            if s and not s.closed:
                try: await s.close()
                except: pass


# ==============================================================================
# Momentum tracker
# ==============================================================================

class MomentumTracker:
    def __init__(self):
        self._history: Dict[str, deque] = {
            a: deque(maxlen=200) for a in ASSETS
        }

    def update(self, asset: str, ts_unix: float, mid: Optional[float]):
        if mid is not None:
            self._history[asset].append((ts_unix, mid))

    def get(self, asset: str, ts_unix: float, window: int) -> Optional[float]:
        hist = self._history[asset]
        if len(hist) < 2:
            return None
        target = ts_unix - window
        best, best_diff = None, float("inf")
        for ts, mid in hist:
            diff = abs(ts - target)
            if diff < best_diff:
                best_diff = diff
                best = mid
        if best is not None and best_diff <= window * 0.5:
            current = hist[-1][1]
            return round(current - best, 6)
        return None


# ==============================================================================
# DataStore
# ==============================================================================

class DataStore:
    def __init__(self, db_path: str = DB_PATH):
        self._path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._columns: List[str] = []

    def init(self):
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        # Build column list
        cols = ["ts TEXT", "ts_unix REAL"]
        for a in ASSETS:
            p = a.lower()
            cols += [
                f"{p}_ticker TEXT",
                f"{p}_mid REAL", f"{p}_spread REAL",
                f"{p}_yes_bid INTEGER", f"{p}_yes_ask INTEGER",
                f"{p}_volume INTEGER", f"{p}_oi INTEGER",
                f"{p}_floor_strike REAL", f"{p}_time_to_expiry REAL",
                f"{p}_ob_imbalance REAL",
                f"{p}_real_price REAL", f"{p}_price_vs_strike_pct REAL",
                f"{p}_mom_5s REAL", f"{p}_mom_15s REAL", f"{p}_mom_30s REAL",
                f"{p}_error TEXT",
            ]
        cols.append("n_valid INTEGER")
        self._columns = [c.split()[0] for c in cols]

        col_defs = ", ".join(cols)
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS features ({col_defs})")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feat_ts ON features(ts_unix)")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS transitions (
                ts TEXT, asset TEXT, old_ticker TEXT, new_ticker TEXT,
                new_floor_strike REAL
            )
        """)
        self._conn.commit()

    def save_row(self, row: dict):
        vals = [row.get(c) for c in self._columns]
        placeholders = ",".join(["?"] * len(vals))
        self._conn.execute(
            f"INSERT INTO features VALUES ({placeholders})", vals)
        self._conn.commit()

    def save_transitions(self, transitions: List[dict]):
        for t in transitions:
            self._conn.execute(
                "INSERT INTO transitions VALUES (?,?,?,?,?)",
                (t["ts"], t["asset"], t["old_ticker"],
                 t["new_ticker"], t.get("new_floor_strike")))
        self._conn.commit()

    def export_csv(self, path: str = CSV_EXPORT) -> int:
        cursor = self._conn.execute(
            "SELECT * FROM features ORDER BY ts_unix")
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        return len(rows)

    def row_count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM features").fetchone()[0]

    def transition_count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM transitions").fetchone()[0]

    def close(self):
        if self._conn:
            self._conn.close()


# ==============================================================================
# Console display
# ==============================================================================

class Console:
    def __init__(self):
        self._n = 0
        self._start = time.time()
        self._last_progress = time.time()
        self._gap_count = 0
        self._gap_sec = 0
        self._last_ts = 0

    def show(self, row: dict, markets: Dict[str, dict]):
        self._n += 1

        # Gap detection
        ts_unix = row.get("ts_unix", 0)
        if self._last_ts > 0:
            gap = ts_unix - self._last_ts
            if gap > 5.0:
                self._gap_count += 1
                self._gap_sec += gap
                print(f"\n  !! GAP #{self._gap_count}: {gap:.0f}s")
        self._last_ts = ts_unix

        elapsed = time.time() - self._start

        if self._n % 40 == 1 or self._n <= 3:
            self._show_full(row, markets, elapsed)
        else:
            # Compact line
            parts = []
            for a in ASSETS:
                p = a.lower()
                mid = row.get(f"{p}_mid")
                parts.append(f"{a}={mid:.2f}" if mid else f"{a}=---")
            tte = row.get("btc_time_to_expiry")
            tte_s = f"tte={tte:.0f}s" if tte is not None else "tte=---"
            print(f"\r  #{self._n:>5} {' '.join(parts)} {tte_s} "
                  f"{elapsed/60:.0f}m/{MAX_RUNTIME/60:.0f}m "
                  f"gaps:{self._gap_count}     ",
                  end="", flush=True)

        # Progress report
        if time.time() - self._last_progress >= PROGRESS_INTERVAL:
            self._show_progress(elapsed)
            self._last_progress = time.time()

    def _show_full(self, row: dict, markets: Dict[str, dict], elapsed: float):
        print(f"\n{'=' * W}")
        print(f"  #{self._n}  {row.get('ts','')}  "
              f"valid:{row.get('n_valid',0)}/4  {elapsed/60:.0f}m")
        print(f"{'=' * W}")
        print(f"  {'ASSET':<5}{'Mid':>6}{'Sprd':>6}{'TTE':>6}"
              f"{'Strike':>10}{'RealP':>10}{'PvS%':>7}"
              f"{'OB_Imb':>7}{'Vol':>8}{'OI':>7}{'Mom5s':>7}")
        print(f"  {'-' * 82}")

        for a in ASSETS:
            p = a.lower()
            m = markets.get(a, {})

            if m.get("error"):
                print(f"  {a:<5}{'---':>6}{'---':>6}{'---':>6}"
                      f"{'---':>10}{'---':>10}{'---':>7}"
                      f"{'---':>7}{'---':>8}{'---':>7}{'---':>7}"
                      f" ERR:{m['error'][:20]}")
                continue

            mid = row.get(f"{p}_mid")
            spread = row.get(f"{p}_spread")
            tte = row.get(f"{p}_time_to_expiry")
            fs = row.get(f"{p}_floor_strike")
            rp = row.get(f"{p}_real_price")
            pvs = row.get(f"{p}_price_vs_strike_pct")
            obi = row.get(f"{p}_ob_imbalance")
            vol = row.get(f"{p}_volume", 0)
            oi = row.get(f"{p}_oi", 0)
            mom = row.get(f"{p}_mom_5s")

            mid_s = f"{mid:.3f}" if mid is not None else "---"
            spr_s = f"{spread:.3f}" if spread is not None else "---"
            tte_s = f"{tte:.0f}s" if tte is not None else "---"
            fs_s = f"${fs:,.0f}" if fs is not None else "---"
            rp_s = f"${rp:,.0f}" if rp is not None else "---"
            pvs_s = f"{pvs:+.3f}" if pvs is not None else "---"
            obi_s = f"{obi:+.3f}" if obi is not None else "---"
            mom_s = f"{mom:+.4f}" if mom is not None else "---"

            print(f"  {a:<5}{mid_s:>6}{spr_s:>6}{tte_s:>6}"
                  f"{fs_s:>10}{rp_s:>10}{pvs_s:>7}"
                  f"{obi_s:>7}{vol:>8}{oi:>7}{mom_s:>7}")

    def _show_progress(self, elapsed: float):
        remaining = MAX_RUNTIME - elapsed
        pct = elapsed / MAX_RUNTIME * 100
        gap_pct = (self._gap_sec / elapsed * 100) if elapsed > 0 else 0

        print(f"\n\n{'#' * 70}")
        print(f"  PROGRESS: {elapsed/60:.0f}m / {MAX_RUNTIME/60:.0f}m ({pct:.0f}%)")
        print(f"  rows={self._n}  gaps={self._gap_count} "
              f"({self._gap_sec:.0f}s = {gap_pct:.1f}%)")
        print(f"  remaining: {remaining/60:.0f} min")

        if self._n >= 5400:
            print(f"  ** READY: {self._n} rows sufficient")
        elif self._n >= 2700:
            print(f"  *  USABLE: {self._n} rows, more is better")
        else:
            need = 2700 - self._n
            eta = need * POLL_SEC
            print(f"  .. NEED MORE: {need} rows to go (~{eta/60:.0f} min)")
        print(f"{'#' * 70}\n")


# ==============================================================================
# Main
# ==============================================================================

async def main():
    print(f"\n{'=' * W}")
    print(f"  Kalshi Collector v3 | Improved Data Pipeline")
    print(f"{'=' * W}")
    print(f"  API:       {KALSHI_API}")
    print(f"  Prices:    {COINBASE_API}")
    asset_str = ", ".join(f"{k}={v['series']}" for k, v in SERIES.items())
    print(f"  assets:    {asset_str}")
    print(f"  poll:      {POLL_SEC}s")
    print(f"  runtime:   {MAX_RUNTIME}s ({MAX_RUNTIME/3600:.1f} hours)")
    print(f"  features:  mid, spread, TTE, floor_strike, OB imbalance, "
          f"Binance price, momentum")
    print(f"  storage:   {DB_PATH} + {CSV_EXPORT}")
    print(f"  Ctrl+C to stop early\n")

    poller = Poller()
    momentum = MomentumTracker()
    store = DataStore()
    store.init()
    console = Console()

    start_time = time.time()

    try:
        while True:
            t0 = asyncio.get_event_loop().time()

            # Backoff
            backoff = poller.get_backoff()
            if backoff > 0:
                print(f"\n  backoff {backoff:.1f}s "
                      f"(consec errors: {poller._consecutive_errors})")
                await asyncio.sleep(backoff)

            # Poll
            markets = await poller.poll_all()

            # Check total failure
            all_error = all(m.get("error") for m in markets.values())
            if all_error:
                if poller._consecutive_errors <= 3:
                    print(f"\n  all-fail, retrying...")
                continue

            # Build feature row
            now_dt = datetime.now(timezone.utc)
            ts_iso = now_dt.isoformat(timespec="milliseconds")
            ts_unix = now_dt.timestamp()

            row = {"ts": ts_iso, "ts_unix": ts_unix}
            n_valid = 0

            for a in ASSETS:
                p = a.lower()
                m = markets[a]

                mid = m.get("mid")
                if mid is not None and not m.get("error"):
                    n_valid += 1

                # Update momentum history
                momentum.update(a, ts_unix, mid)

                row[f"{p}_ticker"] = m.get("ticker", "")
                row[f"{p}_mid"] = mid
                row[f"{p}_spread"] = m.get("spread")
                row[f"{p}_yes_bid"] = m.get("yes_bid")
                row[f"{p}_yes_ask"] = m.get("yes_ask")
                row[f"{p}_volume"] = m.get("volume", 0)
                row[f"{p}_oi"] = m.get("oi", 0)
                row[f"{p}_floor_strike"] = m.get("floor_strike")
                row[f"{p}_time_to_expiry"] = m.get("time_to_expiry")
                row[f"{p}_ob_imbalance"] = m.get("ob_imbalance")
                row[f"{p}_real_price"] = m.get("real_price")
                row[f"{p}_price_vs_strike_pct"] = m.get("price_vs_strike_pct")
                row[f"{p}_mom_5s"] = momentum.get(a, ts_unix, 5)
                row[f"{p}_mom_15s"] = momentum.get(a, ts_unix, 15)
                row[f"{p}_mom_30s"] = momentum.get(a, ts_unix, 30)
                row[f"{p}_error"] = m.get("error")

            row["n_valid"] = n_valid

            # Save transitions
            if poller._transitions:
                store.save_transitions(poller._transitions)
                for t in poller._transitions:
                    print(f"\n  >> TRANSITION {t['asset']}: "
                          f"{t['old_ticker']} -> {t['new_ticker']} "
                          f"(strike: ${t.get('new_floor_strike', '?'):,.2f})")
                poller._transitions.clear()

            # Store
            store.save_row(row)

            # Display
            console.show(row, markets)

            # Time check
            if time.time() - start_time >= MAX_RUNTIME:
                print(f"\n\nTIME: reached {MAX_RUNTIME}s")
                break

            # Sleep
            elapsed_cycle = asyncio.get_event_loop().time() - t0
            await asyncio.sleep(max(0, POLL_SEC - elapsed_cycle))

    except KeyboardInterrupt:
        print(f"\nSTOPPED by user")
    except Exception as e:
        print(f"\nFATAL: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
    finally:
        n_rows = store.export_csv()
        n_trans = store.transition_count()
        elapsed = time.time() - start_time

        await poller.close()
        store.close()

        print(f"\n{'=' * W}")
        print(f"  COLLECTION COMPLETE (v3)")
        print(f"  runtime:      {elapsed:.0f}s ({elapsed/60:.1f}m)")
        print(f"  rows:         {n_rows}")
        print(f"  transitions:  {n_trans} (15-min market changes)")
        print(f"  poller:       {poller.stats_str}")
        print(f"  gaps:         {console._gap_count} "
              f"({console._gap_sec:.0f}s)")
        print(f"  sqlite:       {os.path.abspath(DB_PATH)}")
        print(f"  csv:          {os.path.abspath(CSV_EXPORT)} ({n_rows} rows)")
        print(f"{'=' * W}")

        if n_rows > 0:
            gap_pct = (console._gap_sec / elapsed * 100
                       if elapsed > 0 else 0)
            print(f"\n  QUALITY:")
            if n_rows >= 2700 and gap_pct < 5:
                print(f"  [OK] {n_rows} rows, gap {gap_pct:.1f}%, "
                      f"{n_trans} transitions")
                print(f"  -> Ready: python3 kalshi_quality_check_v2.py")
            elif n_rows >= 1500:
                print(f"  [PARTIAL] {n_rows} rows, gap {gap_pct:.1f}%")
                print(f"  -> Can try, interpret carefully")
            else:
                print(f"  [INSUFFICIENT] {n_rows} rows")
                print(f"  -> Need more data")


# ==============================================================================
# Entry
# ==============================================================================

def _in_nb():
    try:
        return get_ipython().__class__.__name__ in (
            "ZMQInteractiveShell", "Shell", "TerminalInteractiveShell")
    except NameError:
        return False

if _in_nb():
    import nest_asyncio; nest_asyncio.apply()
    asyncio.run(main())
else:
    if __name__ == "__main__":
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            sys.exit(0)
