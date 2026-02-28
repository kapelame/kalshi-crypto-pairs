#!/usr/bin/env python3
"""
Kalshi Live Trader v3
======================
REAL MONEY trading on Kalshi 15-min crypto markets.
Strategy: v3 consensus (cross-asset z-score + trend following)

用法: python3 kalshi_live_trader.py [--duration 7200] [--demo]
"""

import asyncio
import aiohttp
import math
import time
import sys
import os
import json
import sqlite3
import requests
import base64
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from collections import deque, defaultdict
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

# ==============================================================================
# Config
# ==============================================================================

PROD_BASE = "https://api.elections.kalshi.com"
DEMO_BASE = "https://demo-api.kalshi.co"
COINBASE_API = "https://api.coinbase.com/v2"

KEY_ID = os.environ.get("KALSHI_KEY_ID", "")
KEY_FILE = os.environ.get("KALSHI_KEY_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "kalshi_private.pem"))

POLL_SEC = 2.0
DEFAULT_DURATION = 7200

SERIES = {
    "BTC": {"series": "KXBTC15M"},
    "ETH": {"series": "KXETH15M"},
    "SOL": {"series": "KXSOL15M"},
    "XRP": {"series": "KXXRP15M"},
}
ASSETS = list(SERIES.keys())
OB_DEPTH = 10
W = 100

# 各资产15分钟典型波动率
DEFAULT_VOL_15M = {"BTC": 0.15, "ETH": 0.20, "SOL": 0.35, "XRP": 0.30}

LOG_FILE = "kalshi_live_trader.log"
DB_FILE = "kalshi_live_trader.db"

# ==============================================================================
# Kalshi API Client (RSA-PSS Auth)
# ==============================================================================

class KalshiClient:
    def __init__(self, key_id: str, key_file: str, demo: bool = False):
        self.key_id = key_id
        self.base = DEMO_BASE if demo else PROD_BASE
        self.demo = demo
        with open(key_file, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)

    def _sign(self, method: str, path: str) -> dict:
        ts_ms = str(int(time.time() * 1000))
        # Strip query params for signing
        path_clean = path.split("?")[0]
        msg = ts_ms + method + path_clean
        sig = self.private_key.sign(
            msg.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }

    def get(self, path: str, params: dict = None) -> dict:
        r = requests.get(self.base + path, headers=self._sign("GET", path),
                         params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: dict) -> dict:
        r = requests.post(self.base + path, headers=self._sign("POST", path),
                          json=body, timeout=10)
        r.raise_for_status()
        return r.json()

    def delete(self, path: str) -> dict:
        r = requests.delete(self.base + path, headers=self._sign("DELETE", path),
                            timeout=10)
        r.raise_for_status()
        return r.json()

    # --- High-level methods ---

    def get_balance(self) -> float:
        """Returns balance in dollars"""
        data = self.get("/trade-api/v2/portfolio/balance")
        return data.get("balance", 0) / 100.0

    def get_positions(self) -> dict:
        """Returns {ticker: position_count}"""
        data = self.get("/trade-api/v2/portfolio/positions",
                        params={"count_filter": "position", "limit": 100})
        positions = {}
        for mp in data.get("market_positions", []):
            pos = mp.get("position", 0)
            if pos != 0:
                positions[mp["ticker"]] = pos
        return positions

    def place_order(self, ticker: str, side: str, contracts: int,
                    price_cents: int) -> dict:
        """
        Place a limit order.
        side: "yes" or "no"
        price_cents: 1-99 (YES price in cents)
        """
        body = {
            "ticker": ticker,
            "action": "buy",
            "side": side.lower(),
            "count": contracts,
            "type": "limit",
            "yes_price": price_cents,
            "client_order_id": str(uuid.uuid4()),
        }
        log(f"ORDER: {side} x{contracts} {ticker} @{price_cents}c  body={json.dumps(body)}")
        result = self.post("/trade-api/v2/portfolio/orders", body)
        order = result.get("order", {})
        log(f"ORDER RESULT: id={order.get('order_id','')} status={order.get('status','')} "
            f"filled={order.get('fill_count',0)}/{contracts}")
        return result

    def sell_position(self, ticker: str, side: str, contracts: int,
                      price_cents: int) -> dict:
        """Sell existing position. side: 'yes' or 'no'."""
        body = {
            "ticker": ticker,
            "action": "sell",
            "side": side.lower(),
            "count": contracts,
            "type": "limit",
            "yes_price": price_cents,
            "client_order_id": str(uuid.uuid4()),
        }
        log(f"SELL: {side} x{contracts} {ticker} @{price_cents}c  body={json.dumps(body)}")
        result = self.post("/trade-api/v2/portfolio/orders", body)
        order = result.get("order", {})
        log(f"SELL RESULT: id={order.get('order_id','')} status={order.get('status','')} "
            f"filled={order.get('fill_count',0)}/{contracts}")
        return result

    def get_orderbook(self, ticker: str) -> dict:
        """获取orderbook, 返回 best_yes_ask, best_no_ask (cents)"""
        data = self.get(f"/trade-api/v2/markets/{ticker}/orderbook")
        ob = data.get("orderbook", {})
        # yes array: [[price, qty], ...] 买YES的bid
        # no array: [[price, qty], ...]  买NO的bid
        yes_bids = ob.get("yes", [])
        no_bids = ob.get("no", [])
        # best ask for YES = 100 - best_no_bid (对方卖YES = 对方买NO的最高价)
        # best ask for NO = 100 - best_yes_bid
        best_yes_ask = (100 - no_bids[0][0]) if no_bids else 99
        best_no_ask = (100 - yes_bids[0][0]) if yes_bids else 99
        return {"yes_ask": best_yes_ask, "no_ask": best_no_ask,
                "yes_bids": yes_bids, "no_bids": no_bids}

    def cancel_order(self, order_id: str) -> dict:
        return self.delete(f"/trade-api/v2/portfolio/orders/{order_id}")

    def get_market(self, ticker: str) -> dict:
        return self.get(f"/trade-api/v2/markets/{ticker}")


# ==============================================================================
# Fee Model
# ==============================================================================

def taker_fee(contracts: int, price: float) -> float:
    if contracts <= 0 or price <= 0 or price >= 1:
        return 0
    return math.ceil(0.07 * contracts * price * (1 - price) * 100) / 100

def maker_fee(contracts: int, price: float) -> float:
    if contracts <= 0 or price <= 0 or price >= 1:
        return 0
    return math.ceil(0.0175 * contracts * price * (1 - price) * 100) / 100


# ==============================================================================
# Data Recorder (SQLite)
# ==============================================================================

class DataRecorder:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_unix REAL, ts_iso TEXT,
                btc_ticker TEXT, btc_mid REAL, btc_spread REAL, btc_pvs REAL,
                btc_tte REAL, btc_ob REAL, btc_price REAL, btc_vol REAL,
                eth_ticker TEXT, eth_mid REAL, eth_spread REAL, eth_pvs REAL,
                eth_tte REAL, eth_ob REAL, eth_price REAL, eth_vol REAL,
                sol_ticker TEXT, sol_mid REAL, sol_spread REAL, sol_pvs REAL,
                sol_tte REAL, sol_ob REAL, sol_price REAL, sol_vol REAL,
                xrp_ticker TEXT, xrp_mid REAL, xrp_spread REAL, xrp_pvs REAL,
                xrp_tte REAL, xrp_ob REAL, xrp_price REAL, xrp_vol REAL,
                bankroll REAL, session_pnl REAL, n_trades INTEGER,
                positions TEXT, signals TEXT
            );
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_unix REAL, ts_iso TEXT,
                asset TEXT, side TEXT, contracts INTEGER,
                entry_price REAL, fee REAL, ticker TEXT,
                edge REAL, z_score REAL, model_p REAL, confidence REAL,
                order_id TEXT, order_status TEXT,
                settled TEXT, pnl REAL, bankroll_after REAL,
                consensus TEXT, avg_z REAL, agreement REAL,
                tte_at_entry REAL, edge_mode TEXT, bankroll_before REAL
            );
            CREATE TABLE IF NOT EXISTS settlements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_unix REAL, ts_iso TEXT,
                asset TEXT, ticker TEXT, result TEXT
            );
        """)
        self.conn.commit()

    def record_tick(self, ts, markets, risk, signals):
        now_iso = datetime.now(timezone.utc).isoformat()
        vals = [ts, now_iso]
        for a in ASSETS:
            m = markets.get(a, {})
            vals.extend([
                m.get("ticker"), m.get("mid"), m.get("spread"),
                m.get("price_vs_strike_pct"), m.get("time_to_expiry"),
                m.get("ob_imbalance"), m.get("real_price"), m.get("volume"),
            ])
        pos_str = json.dumps({a: p["side"] for a, p in risk.positions.items()})
        sig_str = json.dumps({a: [s[0], round(s[1], 4), s[2]]
                              for a, s in signals.items()}) if signals else "{}"
        vals.extend([risk.bankroll, risk.session_pnl, len(risk.trades), pos_str, sig_str])
        placeholders = ",".join(["?"] * len(vals))
        self.conn.execute(f"INSERT INTO ticks VALUES (NULL, {placeholders})", vals)
        self.conn.commit()

    def record_trade(self, ts, asset, side, contracts, entry, fee, ticker,
                     edge, z_score, model_p, confidence,
                     order_id="", order_status="",
                     consensus="", avg_z=0, agreement=0,
                     tte_at_entry=0, edge_mode="", bankroll_before=0):
        now_iso = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO trades VALUES (NULL, ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [ts, now_iso, asset, side, contracts, entry, fee, ticker,
             edge, z_score, model_p, confidence, order_id, order_status,
             None, None, None,
             consensus, avg_z, agreement, tte_at_entry, edge_mode, bankroll_before])
        self.conn.commit()

    def record_settlement(self, ts, asset, ticker, result_str, pnl=None,
                          bankroll=None):
        now_iso = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO settlements VALUES (NULL, ?,?,?,?,?)",
            [ts, now_iso, asset, ticker, result_str])
        if pnl is not None:
            self.conn.execute(
                "UPDATE trades SET settled=?, pnl=?, bankroll_after=? "
                "WHERE ticker=? AND settled IS NULL",
                [result_str, pnl, bankroll, ticker])
        self.conn.commit()

    def close(self):
        self.conn.close()


# ==============================================================================
# Trend Tracker
# ==============================================================================

class TrendTracker:
    def __init__(self):
        self._prices: Dict[str, list] = {a: [] for a in ASSETS}
        self._settled: list = []

    def update(self, asset: str, ts: float, price: Optional[float]):
        if price is not None:
            self._prices[asset].append((ts, price))
            cutoff = ts - 3600
            self._prices[asset] = [(t, p) for t, p in self._prices[asset] if t > cutoff]

    def record_settlement(self, settled_yes: bool):
        self._settled.append(1 if settled_yes else 0)
        if len(self._settled) > 30:
            self._settled = self._settled[-30:]

    def price_trend(self, asset: str, window_sec: int = 900) -> Optional[float]:
        hist = self._prices[asset]
        if len(hist) < 2:
            return None
        now_ts, now_price = hist[-1]
        target_ts = now_ts - window_sec
        best, best_diff = None, float("inf")
        for ts, price in hist:
            diff = abs(ts - target_ts)
            if diff < best_diff:
                best_diff = diff
                best = price
        if best is not None and best_diff < window_sec * 0.5 and best > 0:
            return (now_price - best) / best * 100
        return None

    def realized_vol(self, asset: str, window_sec: int = 900) -> Optional[float]:
        hist = self._prices[asset]
        if len(hist) < 20:
            return None
        now_ts = hist[-1][0]
        cutoff = now_ts - window_sec
        recent = [(t, p) for t, p in hist if t > cutoff]
        if len(recent) < 15:
            return None
        returns = []
        for i in range(1, len(recent)):
            dt = recent[i][0] - recent[i-1][0]
            if dt > 0 and recent[i-1][1] > 0:
                ret = (recent[i][1] - recent[i-1][1]) / recent[i-1][1] * 100
                returns.append(ret)
        if len(returns) < 10:
            return None
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std_per_sample = math.sqrt(var_r)
        avg_dt = window_sec / len(returns)
        samples_per_15m = 900 / avg_dt if avg_dt > 0 else 450
        return std_per_sample * math.sqrt(samples_per_15m)


def oscillation_score(settled_list: list, n: int = 8) -> float:
    recent = settled_list[-n:] if len(settled_list) >= 3 else settled_list
    if len(recent) < 3:
        return 0.5
    flips = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
    return flips / (len(recent) - 1)


# ==============================================================================
# Risk Manager — v3 (no excessive v4 guards)
# ==============================================================================

class RiskManager:
    def __init__(self, bankroll: float):
        self.bankroll = bankroll
        self.initial_bankroll = bankroll
        self.positions: Dict[str, dict] = {}
        self.trades: list = []
        self.session_pnl = 0.0
        self.peak_bankroll = bankroll
        self.stopped = False
        self.stop_reason = ""
        self.stop_floor = bankroll - 5.00  # 初始止损线 = 本金 - $5
        # v3 config
        self.max_per_trade_pct = 0.06
        self.max_exposure_pct = 0.35
        self.max_same_direction = 4
        self.min_edge = 0.015
        self.kelly_fraction = 0.50
        self.mid_range = (0.20, 0.80)
        self.tte_range = (480, 780)        # 8-13分钟入场
        self.tte_blackout = (600, 660)     # 跳过10-11分钟(历史胜率差)
        self.max_spread = 0.05
        self.use_maker = False  # taker order, 保证成交
        self.net_edge_floor = 0.01
        # 止损止盈 (绝对金额)
        self.stop_loss = 5.00              # 亏 $5 停
        self.take_profit = 5.00            # 赚 $5 停

    @property
    def current_exposure(self) -> float:
        return sum(p["cost"] for p in self.positions.values())

    def direction_count(self, direction: str) -> int:
        return sum(1 for p in self.positions.values() if p["side"] == direction)

    def can_trade(self, asset: str, side: str, mid: float,
                  spread: float, tte: float) -> Tuple[bool, str]:
        if self.stopped:
            return False, "stopped"
        if asset in self.positions:
            return False, "in_pos"
        if mid is None or tte is None:
            return False, "no_data"
        if not (self.mid_range[0] <= mid <= self.mid_range[1]):
            return False, f"mid={mid:.2f}"
        if not (self.tte_range[0] <= tte <= self.tte_range[1]):
            return False, f"tte={tte:.0f}"
        if self.tte_blackout[0] <= tte <= self.tte_blackout[1]:
            return False, f"tte_blackout={tte:.0f}"
        if spread is not None and spread > self.max_spread:
            return False, "spread"
        if self.current_exposure >= self.bankroll * self.max_exposure_pct:
            return False, "exposure"
        if self.direction_count(side) >= self.max_same_direction:
            return False, f"max_{side}"
        return True, "ok"

    def calc_size(self, edge: float, mid: float, side: str) -> Tuple[int, float]:
        price = mid if side == "YES" else (1 - mid)
        if price <= 0:
            return 0, 0
        # 动态合约数: 便宜的多买, 贵的少买
        # 合约价 <= $0.35 → 2份 (赔率好, risk_reward >= 1.86)
        # 合约价 $0.35-$0.65 → 1份 (正常)
        # 合约价 > $0.65 → 1份但要求更高edge
        if price <= 0.35:
            contracts = 2
        else:
            contracts = 1
        if price > 0.65 and abs(edge) < 0.03:
            return 0, 0  # 太贵且edge不够 → 跳过
        cost = contracts * price
        remaining = self.bankroll * self.max_exposure_pct - self.current_exposure
        # 如果2份太多，降到1份
        while contracts > 1 and cost > remaining:
            contracts -= 1
            cost = contracts * price
        if cost > remaining:
            return 0, 0
        fee = maker_fee(contracts, mid) if self.use_maker else taker_fee(contracts, mid)
        net_edge = abs(edge) - fee / contracts if contracts > 0 else 0
        if net_edge < self.net_edge_floor:
            return 0, 0
        return contracts, cost + fee

    def open_position(self, asset: str, side: str, mid: float,
                      contracts: int, cost: float, ticker: str,
                      order_id: str = ""):
        fee = maker_fee(contracts, mid) if self.use_maker else taker_fee(contracts, mid)
        self.positions[asset] = {
            "side": side, "entry_price": mid, "contracts": contracts,
            "cost": cost, "fee": fee, "ticker": ticker,
            "order_id": order_id,
            "entry_time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        }

    def close_position(self, asset: str, settled_yes: bool) -> dict:
        pos = self.positions.pop(asset)
        side, contracts, entry = pos["side"], pos["contracts"], pos["entry_price"]
        fee = pos["fee"]
        if side == "YES":
            pnl = contracts * (1 - entry) - fee if settled_yes else -(contracts * entry + fee)
        else:
            pnl = contracts * entry - fee if not settled_yes else -(contracts * (1 - entry) + fee)
        self.bankroll += pnl
        self.session_pnl += pnl
        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
        # 动态止损: 止损线 = 当前余额 - $5 (trailing stop)
        self.stop_floor = max(self.stop_floor, self.bankroll - self.stop_loss)
        if self.bankroll <= self.stop_floor:
            self.stopped = True
            self.stop_reason = f"STOP LOSS: bank ${self.bankroll:.2f} <= floor ${self.stop_floor:.2f}"
        elif self.session_pnl >= self.take_profit:
            self.stopped = True
            self.stop_reason = f"TAKE PROFIT: ${self.session_pnl:+.2f} >= +${self.take_profit:.2f}"
        trade = {
            "asset": asset, "side": side, "entry": entry,
            "contracts": contracts, "fee": fee,
            "settled": "YES" if settled_yes else "NO",
            "pnl": round(pnl, 4), "bankroll": round(self.bankroll, 2),
            "time": pos["entry_time"],
        }
        self.trades.append(trade)
        return trade


# ==============================================================================
# Strategy — IMPLEMENT YOUR OWN
# ==============================================================================
#
# This is where your trading logic goes. The function below receives live
# market data for all assets and must return a dict of trade signals.
#
# Available data per asset (in `markets[asset]`):
#   mid              - market mid price (probability 0-1)
#   spread           - bid-ask spread
#   time_to_expiry   - seconds until settlement
#   floor_strike     - reference price contract must beat for YES
#   real_price       - live Coinbase spot price
#   price_vs_strike_pct - (real_price - floor_strike) / floor_strike * 100
#   ob_imbalance     - orderbook imbalance (-1 to +1, positive = YES heavy)
#   volume           - total volume traded
#   ticker           - current market ticker
#
# TrendTracker provides:
#   tracker.price_trend(asset, window_sec)  - price change % over window
#   tracker.realized_vol(asset, window_sec) - annualized volatility estimate
#
# Return format: {asset: (side, edge, info_dict)}
#   side = "YES" | "NO" | None (no trade)
#   edge = estimated edge (0-1), must exceed RiskManager.min_edge
#   info_dict = any extra data you want logged (z_score, model_p, etc.)
#
# Example strategies to try:
#   - Z-score: pvs / expected_vol → Φ(z) → compare to market mid
#   - ML model: train on collector data, predict settlement outcome
#   - Cross-asset consensus: if 3/4 assets agree on direction, follow
#   - Mean reversion: fade extreme orderbook imbalances
#   - Momentum: follow price_vs_strike trend in first 5 minutes

def your_strategy(tracker: TrendTracker, markets: dict
                  ) -> Dict[str, Tuple[Optional[str], float, dict]]:
    """
    Implement your strategy here.

    Args:
        tracker: TrendTracker with price history and settlement results
        markets: dict of {asset: market_data} from Poller.poll_all()

    Returns:
        dict of {asset: (side, edge, info)}
        side: "YES", "NO", or None
        edge: float, your estimated edge (will be checked against min_edge)
        info: dict with any signal metadata for logging
    """
    results = {}
    for a in ASSETS:
        # TODO: implement your signal logic here
        # m = markets.get(a, {})
        # mid = m.get("mid")
        # pvs = m.get("price_vs_strike_pct")
        # tte = m.get("time_to_expiry")
        # ob = m.get("ob_imbalance")
        # ... compute signal ...
        results[a] = (None, 0, {})  # no signal by default
    return results


# ==============================================================================
# Poller (market data - no auth needed)
# ==============================================================================

class Poller:
    def __init__(self, api_base: str):
        self.api_base = api_base
        self._kal_session: Optional[aiohttp.ClientSession] = None
        self._cb_session: Optional[aiohttp.ClientSession] = None
        self._kal_created: float = 0

    async def _kal_sess(self) -> aiohttp.ClientSession:
        now = time.time()
        if self._kal_session is None or self._kal_session.closed or now - self._kal_created > 600:
            if self._kal_session and not self._kal_session.closed:
                try: await self._kal_session.close()
                except: pass
            self._kal_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10, connect=5, sock_read=5),
                headers={"Accept": "application/json"})
            self._kal_created = now
        return self._kal_session

    async def _cb_sess(self) -> aiohttp.ClientSession:
        if self._cb_session is None or self._cb_session.closed:
            self._cb_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5))
        return self._cb_session

    async def poll_market(self, asset: str) -> dict:
        try:
            s = await self._kal_sess()
            info = SERIES[asset]
            url = f"{self.api_base}/trade-api/v2/markets?series_ticker={info['series']}&status=open&limit=1"
            async with s.get(url) as r:
                if r.status != 200:
                    return {"asset": asset, "error": f"HTTP {r.status}"}
                data = await r.json()
            mkts = data.get("markets", [])
            if not mkts:
                return {"asset": asset, "error": "no_markets"}
            m = mkts[0]
            yb, ya = m.get("yes_bid"), m.get("yes_ask")
            mid, spread = None, None
            if yb and ya and yb > 0 and ya > 0:
                mid = (yb + ya) / 200.0
                spread = (ya - yb) / 100.0
            elif yb and yb > 0:
                mid = yb / 100.0
            elif ya and ya > 0:
                mid = ya / 100.0
            tte = None
            ct = m.get("close_time", "")
            if ct:
                try:
                    close_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    tte = (close_dt - datetime.now(timezone.utc)).total_seconds()
                    if tte < 0: tte = 0
                except: pass
            return {
                "asset": asset, "ticker": m.get("ticker", ""),
                "mid": mid, "spread": spread,
                "volume": m.get("volume", 0), "oi": m.get("open_interest", 0),
                "floor_strike": m.get("floor_strike"),
                "time_to_expiry": tte, "error": None,
            }
        except Exception as e:
            return {"asset": asset, "error": str(e)[:80]}

    async def poll_orderbook(self, ticker: str, asset: str) -> dict:
        if not ticker:
            return {"asset": asset, "ob_imbalance": None}
        try:
            s = await self._kal_sess()
            url = f"{self.api_base}/trade-api/v2/markets/{ticker}/orderbook?depth={OB_DEPTH}"
            async with s.get(url) as r:
                if r.status != 200:
                    return {"asset": asset, "ob_imbalance": None}
                data = await r.json()
            ob = data.get("orderbook", {})
            y_qty = sum(qty for _, qty in ob.get("yes", []))
            n_qty = sum(qty for _, qty in ob.get("no", []))
            total = y_qty + n_qty
            return {"asset": asset,
                    "ob_imbalance": round((y_qty - n_qty) / total, 4) if total > 0 else None}
        except:
            return {"asset": asset, "ob_imbalance": None}

    async def poll_prices(self) -> Dict[str, Optional[float]]:
        try:
            s = await self._cb_sess()
            url = f"{COINBASE_API}/exchange-rates?currency=USD"
            async with s.get(url) as r:
                if r.status != 200:
                    return {a: None for a in ASSETS}
                data = await r.json()
            rates = data.get("data", {}).get("rates", {})
            prices = {}
            for a in ASSETS:
                rate = rates.get(a)
                prices[a] = round(1.0 / float(rate), 4) if rate else None
            return prices
        except:
            return {a: None for a in ASSETS}

    async def poll_all(self) -> dict:
        market_tasks = [self.poll_market(a) for a in ASSETS]
        market_results = await asyncio.gather(*market_tasks, return_exceptions=True)
        markets = {}
        for i, r in enumerate(market_results):
            a = ASSETS[i]
            markets[a] = r if isinstance(r, dict) else {"asset": a, "error": str(r)[:80]}

        ob_tasks = [self.poll_orderbook(markets[a].get("ticker", ""), a) for a in ASSETS]
        ob_tasks.append(self.poll_prices())
        ob_results = await asyncio.gather(*ob_tasks, return_exceptions=True)

        for i, a in enumerate(ASSETS):
            ob = ob_results[i] if isinstance(ob_results[i], dict) else {}
            markets[a]["ob_imbalance"] = ob.get("ob_imbalance")

        prices = ob_results[-1] if isinstance(ob_results[-1], dict) else {}
        for a in ASSETS:
            rp = prices.get(a)
            markets[a]["real_price"] = rp
            fs = markets[a].get("floor_strike")
            if not fs or fs <= 0:
                fs = period_open_prices.get(a)  # fallback: captured at period open
            if rp and fs and fs > 0:
                markets[a]["price_vs_strike_pct"] = round((rp - fs) / fs * 100, 4)
            else:
                markets[a]["price_vs_strike_pct"] = None
        return markets

    async def close(self):
        for s in [self._kal_session, self._cb_session]:
            if s and not s.closed:
                try: await s.close()
                except: pass


# ==============================================================================
# Settlement lookup (uses auth for reliability)
# ==============================================================================

def lookup_settlement(client: KalshiClient, ticker: str,
                      retries: int = 5, delay: float = 3.0) -> Optional[bool]:
    for attempt in range(retries):
        try:
            data = client.get_market(ticker)
            m = data.get("market", {})
            result = m.get("result", "")
            if result == "yes":
                return True
            elif result == "no":
                return False
        except:
            pass
        if attempt < retries - 1:
            time.sleep(delay)
    return None


# ==============================================================================
# Display
# ==============================================================================

def clear_and_print(lines: List[str]):
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def format_dashboard(cycle, elapsed, duration, markets, tracker, risk,
                     last_signals, pending_settlements, mode_str) -> List[str]:
    lines = []
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    pct = elapsed / duration * 100

    osc = oscillation_score(tracker._settled)
    regime = "OSCILLATE" if osc > 0.6 else ("TREND" if osc < 0.35 else "MIXED")

    valid_sigs = {a: s for a, s in last_signals.items() if len(s) > 3 and s[3].get("valid")}
    consensus_str = ""
    if valid_sigs:
        sample = list(valid_sigs.values())[0]
        consensus_str = f"  Consensus: {sample[3].get('consensus','?')} | "

    lines.append(f"{'=' * W}")
    lines.append(f"  KALSHI LIVE TRADER v3  |  {mode_str}  |  {now_str}  |  "
                 f"cycle {cycle}  |  {elapsed:.0f}s/{duration:.0f}s ({pct:.0f}%)")
    lines.append(f"  {consensus_str}Regime: {regime} (osc={osc:.2f})  |  "
                 f"Bank: ${risk.bankroll:.2f}  PnL: ${risk.session_pnl:+.2f}")
    lines.append(f"{'=' * W}")

    # Market data
    lines.append(f"  {'Asset':<5} {'Mid':>5} {'Sprd':>5} {'TTE':>5} "
                 f"{'PVS':>8} {'Z':>6} {'ModelP':>7} {'OB':>6} {'Trend':>7} {'Price':>10}")
    lines.append(f"  {'-' * (W - 4)}")

    for a in ASSETS:
        m = markets.get(a, {})
        if m.get("error"):
            lines.append(f"  {a:<5} ERROR: {m['error']}")
            continue
        mid = m.get("mid")
        spread = m.get("spread")
        tte = m.get("time_to_expiry")
        pvs = m.get("price_vs_strike_pct")
        ob = m.get("ob_imbalance")
        rp = m.get("real_price")
        t15 = tracker.price_trend(a, 900)

        sig = last_signals.get(a)
        z_val = sig[3].get("z", 0) if sig and len(sig) > 3 else 0
        mp_val = sig[3].get("model_p", 0.5) if sig and len(sig) > 3 else 0.5

        mid_s = f"{mid:.2f}" if mid is not None else "  --"
        spr_s = f"{spread:.3f}" if spread is not None else "  --"
        tte_s = f"{tte:.0f}s" if tte is not None else " --"
        pvs_s = f"{pvs:+.4f}%" if pvs is not None else "    --"
        z_s = f"{z_val:+.2f}" if z_val else "   --"
        mp_s = f"{mp_val:.3f}" if mp_val != 0.5 else "  0.50"
        ob_s = f"{ob:+.3f}" if ob is not None else "  --"
        t15_s = f"{t15:+.3f}%" if t15 is not None else "   --"
        rp_s = f"${rp:,.1f}" if rp is not None else "   --"

        lines.append(f"  {a:<5} {mid_s:>5} {spr_s:>5} {tte_s:>5} "
                     f"{pvs_s:>8} {z_s:>6} {mp_s:>7} {ob_s:>6} {t15_s:>7} {rp_s:>10}")

    # Signals
    lines.append(f"")
    lines.append(f"  SIGNALS:")
    for a in ASSETS:
        sig = last_signals.get(a)
        if sig and len(sig) >= 4:
            side, edge, reason, info = sig
            conf = info.get("confidence", 0)
            cons = info.get("consensus", "")
            agr = info.get("agreement", 0)
            mode = info.get("edge_mode", "")
            mode_tag = f" {mode}" if mode else ""
            if side:
                lines.append(f"    {a}: {side} e={edge:.4f} conf={conf:.3f} "
                             f"[{cons} agr={agr:.0%}{mode_tag}] ({reason})")
            else:
                lines.append(f"    {a}: -- [{cons}{mode_tag}] ({reason})")
        elif sig:
            lines.append(f"    {a}: -- ({sig[2] if len(sig)>2 else '?'})")
        else:
            lines.append(f"    {a}: waiting...")

    # Positions
    lines.append(f"")
    lines.append(f"  POSITIONS:  exposure=${risk.current_exposure:.2f}")
    if risk.positions:
        for a, pos in risk.positions.items():
            m = markets.get(a, {})
            cur_mid = m.get("mid")
            tte = m.get("time_to_expiry")
            if cur_mid is not None:
                unreal = (pos["contracts"] * (cur_mid - pos["entry_price"])
                          if pos["side"] == "YES"
                          else pos["contracts"] * (pos["entry_price"] - cur_mid))
            else:
                unreal = 0
            tte_s = f"TTE={tte:.0f}s" if tte else ""
            lines.append(f"    {a}: {pos['side']} x{pos['contracts']} @ "
                         f"${pos['entry_price']:.2f}  unreal=${unreal:+.2f}  "
                         f"fee=${pos['fee']:.2f}  {tte_s}")
    else:
        lines.append(f"    (no open positions)")

    # Settlements
    if pending_settlements:
        lines.append(f"")
        lines.append(f"  EVENTS:")
        for ps in pending_settlements[-6:]:
            lines.append(f"    {ps}")

    # Trade history
    lines.append(f"")
    if risk.trades:
        lines.append(f"  TRADES ({len(risk.trades)}):")
        lines.append(f"  {'#':>3} {'Asset':<5} {'Side':<4} {'Entry':>6} {'Ctrs':>4} "
                     f"{'Fee':>5} {'Result':>7} {'PnL':>8} {'Bank':>8}")
        lines.append(f"  {'-' * 60}")
        for i, t in enumerate(risk.trades[-8:]):
            res = "WIN" if t["pnl"] > 0 else "LOSS"
            lines.append(f"  {i+1:>3} {t['asset']:<5} {t['side']:<4} "
                         f"${t['entry']:.2f} {t['contracts']:>4} "
                         f"${t['fee']:.2f} {t['settled']:>3}->{res:<4} "
                         f"${t['pnl']:>+7.2f} ${t['bankroll']:>7.2f}")
        wins = [t for t in risk.trades if t["pnl"] > 0]
        total_pnl = sum(t["pnl"] for t in risk.trades)
        total_fees = sum(t["fee"] for t in risk.trades)
        wr = len(wins) / len(risk.trades)
        lines.append(f"  >> {len(risk.trades)} trades | WR {wr:.0%} | "
                     f"PnL ${total_pnl:+.2f} | fees ${total_fees:.2f}")
    else:
        lines.append(f"  TRADES: (none yet)")

    # Stop loss / take profit
    lines.append(f"")
    lines.append(f"  SL floor: ${risk.stop_floor:.2f}  |  "
                 f"TP: +${risk.take_profit:.2f}  |  "
                 f"PnL: ${risk.session_pnl:+.2f}  |  "
                 f"Bank: ${risk.bankroll:.2f}")

    if risk.stopped:
        lines.append(f"  *** {risk.stop_reason} ***")

    lines.append(f"{'=' * W}")
    return lines


# ==============================================================================
# Log
# ==============================================================================

EVENT_LOG = "kalshi_events.log"

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(EVENT_LOG, "a") as f:
        f.write(line + "\n")


# ==============================================================================
# Main Loop
# ==============================================================================

async def main():
    duration = DEFAULT_DURATION
    demo = False
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--duration" and i + 1 < len(sys.argv) - 1:
            duration = int(sys.argv[i + 2])
        if arg == "--demo":
            demo = True

    api_base = DEMO_BASE if demo else PROD_BASE
    mode_str = "DEMO" if demo else "PRODUCTION $$$"

    # Init Kalshi client
    client = KalshiClient(KEY_ID, KEY_FILE, demo=demo)

    # Check balance
    balance = client.get_balance()
    print(f"Kalshi balance: ${balance:.2f}")
    if balance < 1.0:
        print("ERROR: Balance too low to trade. Need at least $1.00")
        return

    with open(LOG_FILE, "w") as f:
        f.write(f"=== Kalshi Live Trader v3 ({mode_str}) ===\n")
        f.write(f"Balance: ${balance:.2f}\n")
        f.write(f"Strategy: v3 consensus (no excessive guards)\n")
        f.write(f"Duration: {duration}s | Poll: {POLL_SEC}s\n\n")

    recorder = DataRecorder(DB_FILE)
    poller = Poller(api_base)
    tracker = TrendTracker()
    risk = RiskManager(bankroll=balance)

    last_tickers: Dict[str, str] = {}
    last_signals: Dict[str, tuple] = {}
    pending_settlements: List[str] = []
    deferred: List[dict] = []
    period_open_prices: Dict[str, float] = {}  # asset -> price at period open
    cycle = 0
    start_time = time.time()
    last_log_time = 0
    last_record_time = 0

    print(f"\nKalshi Live Trader v3 ({mode_str})")
    print(f"Strategy: v3 consensus trend following")
    print(f"Bankroll: ${balance:.2f}")
    print(f"Duration: {duration}s | Poll: {POLL_SEC}s")
    print(f"Data: {DB_FILE}")
    print(f"Log:  {LOG_FILE}")
    print(f"Press Ctrl+C to stop\n")

    log(f"Config: min_edge={risk.min_edge} kelly={risk.kelly_fraction} "
        f"mid_range={risk.mid_range} tte_range={risk.tte_range} "
        f"stop_loss=${risk.stop_loss:.2f} take_profit=${risk.take_profit:.2f}")

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            cycle += 1
            now = time.time()

            # Poll market data
            markets = await poller.poll_all()

            # Initialize open prices on first cycle
            if cycle == 1:
                for a in ASSETS:
                    rp = markets.get(a, {}).get("real_price")
                    if rp and rp > 0 and a not in period_open_prices:
                        period_open_prices[a] = rp
                        log(f"INIT OPEN PRICE {a}: ${rp:.2f}")

            # Feed prices
            for a in ASSETS:
                rp = markets.get(a, {}).get("real_price")
                tracker.update(a, now, rp)

            # --- Ticker transitions (settlement detection) ---
            for a in ASSETS:
                new_ticker = markets.get(a, {}).get("ticker")
                if not new_ticker:
                    continue
                if a in last_tickers and last_tickers[a] != new_ticker:
                    old_ticker = last_tickers[a]
                    msg = f"{a}: {old_ticker[:35]} -> {new_ticker[:35]}"
                    result = lookup_settlement(client, old_ticker, retries=2, delay=2.0)

                    if a in risk.positions and risk.positions[a]["ticker"] == old_ticker:
                        if result is not None:
                            trade = risk.close_position(a, result)
                            tracker.record_settlement(result)
                            res_str = "YES" if result else "NO"
                            msg += f" | {res_str} PnL=${trade['pnl']:+.2f}"
                            recorder.record_settlement(
                                now, a, old_ticker, res_str,
                                trade["pnl"], risk.bankroll)
                        else:
                            pos = risk.positions.pop(a)
                            deferred.append({"asset": a, "ticker": old_ticker,
                                             "pos": pos, "ts": now})
                            msg += f" | deferred"
                    else:
                        if result is not None:
                            tracker.record_settlement(result)
                            res_str = "YES" if result else "NO"
                            msg += f" | {res_str} (no pos)"
                            recorder.record_settlement(now, a, old_ticker, res_str)
                        else:
                            deferred.append({"asset": a, "ticker": old_ticker,
                                             "pos": None, "ts": now})
                            msg += f" | pending"

                    pending_settlements.append(
                        f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}")
                    log(f"TRANSITION: {msg}")
                    # Capture opening price for new period
                    rp = markets.get(a, {}).get("real_price")
                    if rp and rp > 0:
                        period_open_prices[a] = rp
                        log(f"OPEN PRICE {a}: ${rp:.2f} (reference for new period)")
                last_tickers[a] = new_ticker

            # --- Deferred settlement retries ---
            still_deferred = []
            for d in deferred:
                if now - d["ts"] > 300:
                    if d["pos"]:
                        log(f"DEFERRED EXPIRED: {d['asset']} {d['ticker']}")
                    continue
                result = lookup_settlement(client, d["ticker"], retries=1, delay=0)
                if result is not None:
                    tracker.record_settlement(result)
                    res_str = "YES" if result else "NO"
                    if d["pos"]:
                        pos = d["pos"]
                        side, contracts, entry = pos["side"], pos["contracts"], pos["entry_price"]
                        fee = pos["fee"]
                        if side == "YES":
                            pnl = contracts * (1 - entry) - fee if result else -(contracts * entry + fee)
                        else:
                            pnl = contracts * entry - fee if not result else -(contracts * (1 - entry) + fee)
                        risk.bankroll += pnl
                        risk.session_pnl += pnl
                        risk.peak_bankroll = max(risk.peak_bankroll, risk.bankroll)
                        trade = {
                            "asset": d["asset"], "side": side, "entry": entry,
                            "contracts": contracts, "fee": fee,
                            "settled": res_str, "pnl": round(pnl, 4),
                            "bankroll": round(risk.bankroll, 2),
                            "time": pos.get("entry_time", ""),
                        }
                        risk.trades.append(trade)
                        msg = f"DEFERRED: {d['asset']} {d['ticker'][:25]} -> {res_str} PnL=${pnl:+.2f}"
                        pending_settlements.append(
                            f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}")
                        log(msg)
                        recorder.record_settlement(
                            now, d["asset"], d["ticker"], res_str, pnl, risk.bankroll)
                    else:
                        recorder.record_settlement(now, d["asset"], d["ticker"], res_str)
                else:
                    still_deferred.append(d)
            deferred = still_deferred

            # --- 持仓中途止盈: 暂时关闭 ---

            # --- Check stop loss / take profit ---
            if risk.stopped and risk.stop_reason:
                log(f"*** {risk.stop_reason} ***")

            # --- Strategy ---
            # 共识方向全部资产入场，已有仓位的资产会被can_trade跳过
            if not risk.stopped:
                strat_results = your_strategy(tracker, markets)

                for a in ASSETS:
                    m = markets.get(a, {})
                    side, edge, info = strat_results.get(a, (None, 0, {}))

                    mid = m.get("mid")
                    tte = m.get("time_to_expiry")
                    spread = m.get("spread")

                    if side is None or edge < risk.min_edge:
                        reason = "no signal" if side is None else f"e={edge:.4f}<{risk.min_edge}"
                        last_signals[a] = (side, edge, reason, info)
                        continue

                    if mid is None or tte is None:
                        last_signals[a] = (side, edge, "no data", info)
                        continue

                    ok, reason = risk.can_trade(a, side, mid, spread, tte)
                    if not ok:
                        last_signals[a] = (side, edge, f"blocked:{reason}", info)
                        continue

                    contracts, cost = risk.calc_size(edge, mid, side)
                    if contracts <= 0:
                        last_signals[a] = (side, edge, "size=0", info)
                        continue

                    # === REAL ORDER PLACEMENT ===
                    ticker = m.get("ticker", "")

                    # 防重复: 先查 API 持仓
                    try:
                        api_positions = client.get_positions()
                        if ticker in api_positions:
                            log(f"SKIP: already have {api_positions[ticker]} contracts on {ticker}")
                            last_signals[a] = (side, edge,
                                               f"already_on_api [{info.get('consensus','?')}]", info)
                            # 同步本地状态
                            if a not in risk.positions:
                                pos_count = api_positions[ticker]
                                cp = mid if side == "YES" else (1 - mid)
                                risk.open_position(a, side, mid, pos_count,
                                                   pos_count * cp, ticker)
                            continue
                    except Exception as e:
                        log(f"POSITION CHECK ERROR: {e}")

                    # 价格: 查orderbook拿真实ask价, 直接吃单
                    try:
                        ob = client.get_orderbook(ticker)
                        if side == "YES":
                            price_cents = ob["yes_ask"]  # 对方卖YES的价
                        else:
                            # 买NO: yes_price要设低, 让NO价格=100-yes_price能吃到ask
                            price_cents = 100 - ob["no_ask"]
                        log(f"ORDERBOOK {a}: yes_ask={ob['yes_ask']}c no_ask={ob['no_ask']}c → {side} @{price_cents}c")
                    except Exception as e:
                        log(f"ORDERBOOK ERROR {a}: {e}, fallback to mid")
                        price_cents = int(mid * 100)
                    price_cents = max(1, min(99, price_cents))

                    try:
                        order_result = client.place_order(
                            ticker=ticker,
                            side=side.lower(),
                            contracts=contracts,
                            price_cents=price_cents,
                        )
                        order = order_result.get("order", {})
                        order_id = order.get("order_id", "")
                        order_status = order.get("status", "")
                        filled = order.get("fill_count", 0)

                        if order_status == "executed" or filled > 0:
                            # 成交
                            actual_contracts = filled if filled > 0 else contracts
                            actual_mid = mid
                            cp = mid if side == "YES" else (1 - mid)
                            actual_cost = actual_contracts * cp
                            risk.open_position(a, side, actual_mid, actual_contracts,
                                               actual_cost, ticker, order_id)
                            consensus_s = info.get("consensus", "?")
                            last_signals[a] = (side, edge,
                                               f"FILLED x{actual_contracts}@${actual_mid:.2f} [{consensus_s}]",
                                               info)
                            log(f"TRADE FILLED: {a} {side} x{actual_contracts} @${actual_mid:.2f} "
                                f"edge={edge:.4f} z={info.get('z',0):.2f} "
                                f"consensus={consensus_s} order_id={order_id}")
                            recorder.record_trade(
                                now, a, side, actual_contracts, actual_mid,
                                maker_fee(actual_contracts, actual_mid), ticker,
                                edge, info.get("z", 0), info.get("model_p", 0.5),
                                info.get("confidence", 0.25),
                                order_id, order_status,
                                consensus=info.get("consensus", ""),
                                avg_z=info.get("avg_z", 0),
                                agreement=info.get("agreement", 0),
                                tte_at_entry=markets.get(a, {}).get("time_to_expiry", 0),
                                edge_mode=info.get("edge_mode", ""),
                                bankroll_before=risk.bankroll)
                        else:
                            # 未成交 — 取消
                            if order_id and order_status == "resting":
                                try:
                                    client.cancel_order(order_id)
                                    log(f"ORDER CANCELED (no fill): {order_id}")
                                except:
                                    pass
                            last_signals[a] = (side, edge,
                                               f"no fill @{price_cents}c [{info.get('consensus','?')}]",
                                               info)

                    except Exception as e:
                        log(f"ORDER ERROR: {a} {side} x{contracts} @{price_cents}c: {e}")
                        last_signals[a] = (side, edge, f"order_err: {str(e)[:30]}", info)

            # --- Record data (every 10s) ---
            if now - last_record_time > 10:
                last_record_time = now
                recorder.record_tick(now, markets, risk, last_signals)

            # --- Periodic log (every 60s) ---
            if now - last_log_time > 60:
                last_log_time = now
                pos_str = ", ".join(f"{a}:{p['side']}" for a, p in risk.positions.items()) or "none"
                sig_parts = []
                for a, s in last_signals.items():
                    side_s = s[0] or "--"
                    sig_parts.append(f"{a}:{side_s}({s[1]:.3f})")
                osc = oscillation_score(tracker._settled)
                log(f"STATUS: cy={cycle} bank=${risk.bankroll:.2f} "
                    f"pnl=${risk.session_pnl:+.2f} trades={len(risk.trades)} "
                    f"osc={osc:.2f} pos=[{pos_str}] sig=[{', '.join(sig_parts)}]")

                # Refresh balance from API every minute
                try:
                    api_balance = client.get_balance()
                    log(f"API BALANCE: ${api_balance:.2f} (local: ${risk.bankroll:.2f})")
                    # Sync: if API balance higher and no open positions, trust API
                    if api_balance > risk.bankroll + 0.05 and not risk.positions:
                        diff = api_balance - risk.bankroll
                        log(f"BALANCE SYNC: +${diff:.2f} (external settlement)")
                        risk.bankroll = api_balance
                        risk.session_pnl += diff
                except:
                    pass

            # --- Display ---
            lines = format_dashboard(
                cycle, elapsed, duration, markets, tracker,
                risk, last_signals, pending_settlements, mode_str)
            clear_and_print(lines)

            # Wait
            poll_elapsed = time.time() - now
            wait = max(POLL_SEC - poll_elapsed, 0.1)
            await asyncio.sleep(wait)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    finally:
        # Cancel any resting orders
        try:
            orders = client.get("/trade-api/v2/portfolio/orders",
                                params={"status": "resting"})
            for o in orders.get("orders", []):
                try:
                    client.cancel_order(o["order_id"])
                    log(f"CLEANUP: canceled order {o['order_id']}")
                except:
                    pass
        except:
            pass

        await poller.close()
        recorder.close()

    # Final
    log(f"=== SESSION ENDED ===")
    print(f"\n{'=' * W}")
    print(f"  FINAL RESULTS — Kalshi Live Trader v3 ({mode_str})")
    print(f"{'=' * W}")
    print(f"  Duration:   {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Cycles:     {cycle}")
    print(f"  Bankroll:   ${risk.initial_bankroll:.2f} -> ${risk.bankroll:.2f}")
    print(f"  PnL:        ${risk.session_pnl:+.2f}")

    if risk.trades:
        wins = [t for t in risk.trades if t["pnl"] > 0]
        total_fees = sum(t["fee"] for t in risk.trades)
        wr = len(wins) / len(risk.trades)
        print(f"  Trades:     {len(risk.trades)} (WR {wr:.0%})")
        print(f"  Fees:       ${total_fees:.2f}")
        for i, t in enumerate(risk.trades):
            res = "WIN" if t["pnl"] > 0 else "LOSS"
            line = (f"    {i+1}. {t['asset']} {t['side']} x{t['contracts']} "
                    f"@${t['entry']:.2f} -> {t['settled']} ({res}) "
                    f"PnL=${t['pnl']:+.2f}")
            print(line)
            log(line.strip())
        log(f"FINAL: bank=${risk.bankroll:.2f} pnl=${risk.session_pnl:+.2f} "
            f"trades={len(risk.trades)} wr={wr:.0%} fees=${total_fees:.2f}")
    else:
        print(f"  Trades:     0")
        log(f"FINAL: no trades")

    # Show final API balance
    try:
        final_balance = client.get_balance()
        print(f"  API Balance: ${final_balance:.2f}")
    except:
        pass

    print(f"{'=' * W}")


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())
