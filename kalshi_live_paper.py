#!/usr/bin/env python3
"""
Kalshi Live Paper Trader v3
===========================
v3 核心改进 (在 v2 基础上):
  1. 跨资产共识趋势跟随: 4/4同向=放大, 3/4=正常, 2/4=缩小
  2. 修复 blend 公式反共识 bug: 强共识时用 z-score edge, 不用 model_p vs mid blend
  3. 校准概率模型: random walk + z-score → P(YES)
  4. 震荡检测: 识别 YES/NO 交替模式, 降低置信度
  5. 实时数据记录: SQLite + CSV 持久化

用法: python3 kalshi_live_paper.py [--duration 7200]
"""

import asyncio
import aiohttp
import math
import time
import sys
import os
import json
import sqlite3
import csv as csvmod
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
from collections import deque, defaultdict

# ==============================================================================
# Config
# ==============================================================================

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
COINBASE_API = "https://api.coinbase.com/v2"

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
W = 92

# 各资产15分钟典型波动率 (%, 用于 z-score 校准)
DEFAULT_VOL_15M = {"BTC": 0.15, "ETH": 0.20, "SOL": 0.35, "XRP": 0.30}

LOG_FILE = "kalshi_live_paper.log"
DB_FILE = "kalshi_live_paper.db"
CSV_FILE = "kalshi_live_paper_data.csv"

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
# Data Recorder (SQLite + CSV)
# ==============================================================================

class DataRecorder:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._init_tables()
        self._csv_started = False

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
                settled TEXT, pnl REAL, bankroll_after REAL
            );
            CREATE TABLE IF NOT EXISTS settlements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_unix REAL, ts_iso TEXT,
                asset TEXT, ticker TEXT, result TEXT
            );
        """)
        self.conn.commit()

    def record_tick(self, ts: float, markets: dict, risk, signals: dict):
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

    def record_trade(self, ts: float, asset: str, side: str, contracts: int,
                     entry: float, fee: float, ticker: str,
                     edge: float, z_score: float, model_p: float,
                     confidence: float):
        now_iso = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO trades VALUES (NULL, ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [ts, now_iso, asset, side, contracts, entry, fee, ticker,
             edge, z_score, model_p, confidence, None, None, None])
        self.conn.commit()

    def record_settlement(self, ts: float, asset: str, ticker: str,
                          result_str: str, pnl: float = None,
                          bankroll: float = None):
        now_iso = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO settlements VALUES (NULL, ?,?,?,?,?)",
            [ts, now_iso, asset, ticker, result_str])
        # Also update the trade record if exists
        if pnl is not None:
            self.conn.execute(
                "UPDATE trades SET settled=?, pnl=?, bankroll_after=? "
                "WHERE ticker=? AND settled IS NULL",
                [result_str, pnl, bankroll, ticker])
        self.conn.commit()

    def export_csv(self):
        cursor = self.conn.execute("SELECT * FROM ticks ORDER BY ts_unix")
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        with open(CSV_FILE, "w", newline="") as f:
            w = csvmod.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        return len(rows)

    def close(self):
        try:
            self.export_csv()
        except:
            pass
        self.conn.close()


# ==============================================================================
# Trend Tracker
# ==============================================================================

class TrendTracker:
    def __init__(self):
        self._prices: Dict[str, list] = {a: [] for a in ASSETS}
        self._settled: list = []
        # Staleness tracking: detect frozen Coinbase prices
        self._last_price: Dict[str, Optional[float]] = {a: None for a in ASSETS}
        self._last_change_ts: Dict[str, float] = {a: 0.0 for a in ASSETS}
        # Mid tracking: Kalshi market mid history (more reliable than Coinbase)
        self._mids: Dict[str, list] = {a: [] for a in ASSETS}

    def update(self, asset: str, ts: float, price: Optional[float]):
        if price is not None:
            self._prices[asset].append((ts, price))
            cutoff = ts - 3600
            self._prices[asset] = [(t, p) for t, p in self._prices[asset] if t > cutoff]
            # Track staleness
            if self._last_price[asset] is None or abs(price - self._last_price[asset]) > 0.001:
                self._last_price[asset] = price
                self._last_change_ts[asset] = ts
            # If price unchanged, _last_change_ts stays old → staleness grows

    def update_mid(self, asset: str, ts: float, mid: Optional[float]):
        """Track Kalshi market mid over time"""
        if mid is not None:
            self._mids[asset].append((ts, mid))
            cutoff = ts - 1800  # 30 min history
            self._mids[asset] = [(t, m) for t, m in self._mids[asset] if t > cutoff]

    def price_stale_sec(self, asset: str, now: float) -> float:
        """Seconds since Coinbase price last changed"""
        lc = self._last_change_ts.get(asset, 0)
        return now - lc if lc > 0 else 999

    def mid_direction(self, asset: str, window_sec: int = 120) -> Optional[float]:
        """Kalshi mid trend over last N seconds. Positive = moving toward YES."""
        hist = self._mids.get(asset, [])
        if len(hist) < 5:
            return None
        now_ts, now_mid = hist[-1]
        target_ts = now_ts - window_sec
        best, best_diff = None, float("inf")
        for ts, m in hist:
            diff = abs(ts - target_ts)
            if diff < best_diff:
                best_diff = diff
                best = m
        if best is not None and best_diff < window_sec * 0.5:
            return now_mid - best  # positive = mid rising (YES direction)
        return None

    def record_settlement(self, settled_yes: bool):
        self._settled.append(1 if settled_yes else 0)
        if len(self._settled) > 30:
            self._settled = self._settled[-30:]

    def price_trend(self, asset: str, window_sec: int = 900) -> Optional[float]:
        """过去 N 秒的价格变化百分比"""
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
        """估算过去 N 秒的已实现波动率 (% per 15min)"""
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
        # Scale to 15-min: std * sqrt(samples_per_15min)
        avg_dt = window_sec / len(returns)
        samples_per_15m = 900 / avg_dt if avg_dt > 0 else 450
        return std_per_sample * math.sqrt(samples_per_15m)


def oscillation_score(settled_list: list, n: int = 8) -> float:
    """
    衡量结算翻转频率. 1.0 = 完美交替 (YES/NO/YES/NO), 0.0 = 完全一致
    """
    recent = settled_list[-n:] if len(settled_list) >= 3 else settled_list
    if len(recent) < 3:
        return 0.5  # 未知, 假设中等
    flips = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
    return flips / (len(recent) - 1)


# ==============================================================================
# Risk Manager
# ==============================================================================

class RiskManager:
    def __init__(self, bankroll=100.0):
        self.bankroll = bankroll
        self.initial_bankroll = bankroll
        self.positions: Dict[str, dict] = {}
        self.trades: list = []
        self.session_pnl = 0.0
        self.peak_bankroll = bankroll
        self.stopped = False
        # ---- v4 调整 ----
        self.max_per_trade_pct = 0.06       # 单笔上限 6%
        self.max_exposure_pct = 0.25        # 总持仓上限 25% (4/4时需要空间)
        self.max_same_direction = 4         # 共识时允许4个同向
        self.daily_loss_limit_pct = 0.15    # 日亏损 15% 停
        self.min_edge = 0.015               # 最小 edge 1.5% (校准后)
        self.kelly_fraction = 0.50          # Kelly 50% (因为edge已经保守)
        self.mid_range = (0.20, 0.80)       # 放宽 mid 范围
        self.tte_range = (120, 420)         # v4: 收紧到 2-7 分钟 (减少被反转)
        self.max_spread = 0.05              # 最大 spread
        self.use_maker = True
        self.net_edge_floor = 0.01          # 扣费后最低net edge
        self.max_contract_price = 0.60      # v4: 合约价格上限 (控制单笔最大亏损)

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
        if spread is not None and spread > self.max_spread:
            return False, "spread"
        # v4: 合约价格上限 — 不买贵合约 (控制最大亏损)
        contract_price = mid if side == "YES" else (1 - mid)
        if contract_price > self.max_contract_price:
            return False, f"expensive(${contract_price:.2f})"
        if self.current_exposure >= self.bankroll * self.max_exposure_pct:
            return False, "exposure"
        if self.direction_count(side) >= self.max_same_direction:
            return False, f"max_{side}"
        return True, "ok"

    def calc_size(self, edge: float, mid: float, side: str) -> Tuple[int, float]:
        price = mid if side == "YES" else (1 - mid)
        odds = (1 - price) / price if price > 0 else 0
        kelly = abs(edge) / odds if odds > 0 else 0
        raw_usd = self.bankroll * self.kelly_fraction * kelly
        max_usd = self.bankroll * self.max_per_trade_pct
        remaining = self.bankroll * self.max_exposure_pct - self.current_exposure
        size_usd = max(min(raw_usd, max_usd, remaining), 0)
        if size_usd < price:
            return 0, 0
        contracts = int(size_usd / price)
        if contracts <= 0:
            return 0, 0
        fee = maker_fee(contracts, mid) if self.use_maker else taker_fee(contracts, mid)
        net_edge = abs(edge) - fee / contracts if contracts > 0 else 0
        if net_edge < self.net_edge_floor:
            return 0, 0
        return contracts, contracts * price + fee

    def open_position(self, asset: str, side: str, mid: float,
                      contracts: int, cost: float, ticker: str):
        fee = maker_fee(contracts, mid) if self.use_maker else taker_fee(contracts, mid)
        self.positions[asset] = {
            "side": side, "entry_price": mid, "contracts": contracts,
            "cost": cost, "fee": fee, "ticker": ticker,
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
        if self.bankroll < self.initial_bankroll * (1 - self.daily_loss_limit_pct):
            self.stopped = True
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
# Rule Strategy v4 — 校准概率 + 跨资产共识 + 价格冻结检测 + mid信号
# ==============================================================================

def compute_asset_signal(tracker: TrendTracker, asset: str,
                         mid: float, pvs: float, tte: float,
                         ob: float, now: float) -> dict:
    """
    单资产信号计算 (不含跨资产共识)
    v4: 增加 Coinbase 价格冻结检测 + Kalshi mid 趋势信号
    """
    info = {"z": 0, "model_p": 0.5, "raw_side": None, "raw_edge": 0,
            "vol": 0, "trend": 0, "valid": False,
            "stale_sec": 0, "mid_dir": 0, "conflict": False}

    if pvs is None or tte is None or mid is None or tte < 60:
        return info
    if not (0.10 <= mid <= 0.90):
        return info

    # 0. Coinbase 价格冻结检测
    stale_sec = tracker.price_stale_sec(asset, now)
    info["stale_sec"] = round(stale_sec, 0)

    # Kalshi mid 趋势 (过去2分钟 mid 的变化方向)
    mid_dir = tracker.mid_direction(asset, 120) or 0
    info["mid_dir"] = round(mid_dir, 4)

    # v4: 价格冻结 > 60秒 → 不信任 pvs, 标记无效
    if stale_sec > 60:
        info["valid"] = False
        info["stale_blocked"] = True
        return info

    # 1. 波动率
    rv = tracker.realized_vol(asset)
    vol_15m = rv if rv is not None else DEFAULT_VOL_15M.get(asset, 0.20)
    vol_15m = max(vol_15m, 0.03)
    info["vol"] = round(vol_15m, 4)

    # 2. Z-score
    vol_per_sec = vol_15m / math.sqrt(900)
    expected_std = vol_per_sec * math.sqrt(max(tte, 1))
    z = pvs / expected_std if expected_std > 0.001 else 0
    info["z"] = round(z, 3)

    # 3. model_p = Φ(z)
    model_p = 0.5 * (1 + math.erf(z / math.sqrt(2)))

    # 4. 趋势微调 (±2%)
    trend_15m = tracker.price_trend(asset, 900) or 0
    trend_adj = max(-0.02, min(0.02, trend_15m * 0.05))
    model_p = max(0.02, min(0.98, model_p + trend_adj))
    info["trend"] = round(trend_15m, 4)

    # 5. OB微调 (±1%)
    ob_val = ob if ob is not None else 0
    ob_adj = max(-0.01, min(0.01, ob_val * 0.015))
    model_p = max(0.02, min(0.98, model_p + ob_adj))

    info["model_p"] = round(model_p, 4)
    info["valid"] = True

    # Raw direction (before consensus adjustment)
    if model_p > 0.5:
        info["raw_side"] = "YES"
    elif model_p < 0.5:
        info["raw_side"] = "NO"
    info["raw_edge"] = abs(model_p - mid)

    # 6. v4: mid-pvs 矛盾检测
    # pvs 说 NO (model_p < 0.5) 但 Kalshi mid > 0.55 (市场说 YES) → 冲突
    # pvs 说 YES (model_p > 0.5) 但 Kalshi mid < 0.45 (市场说 NO) → 冲突
    pvs_side_yes = model_p > 0.5
    mid_side_yes = mid > 0.55
    mid_side_no = mid < 0.45
    if (pvs_side_yes and mid_side_no) or (not pvs_side_yes and mid_side_yes):
        info["conflict"] = True

    return info


def consensus_strategy(tracker: TrendTracker, markets: dict,
                       now: float = None
                       ) -> Dict[str, Tuple[Optional[str], float, dict]]:
    """
    v4 策略核心: 校准概率 + 跨资产共识 + 价格冻结保护 + mid-pvs 冲突检测

    v4 改进:
      1. Coinbase 价格冻结 > 60s → 资产标记无效, 不参与共识计算
      2. mid-pvs 矛盾 → 降低该资产置信度或跳过
      3. TTE 收紧到 2-7 分钟 (减少早期入场被反转)
      4. 合约价格上限 $0.60 (控制最大亏损)
    """
    results = {}

    # === Phase 1: 计算所有资产的原始信号 ===
    if now is None:
        now = time.time()

    signals = {}
    for a in ASSETS:
        m = markets.get(a, {})
        if m.get("error"):
            signals[a] = {"valid": False}
            continue
        mid = m.get("mid")
        pvs = m.get("price_vs_strike_pct")
        tte = m.get("time_to_expiry")
        ob = m.get("ob_imbalance") or 0
        signals[a] = compute_asset_signal(tracker, a, mid, pvs, tte, ob, now)

    # === Phase 2: 计算跨资产共识 ===
    valid_signals = {a: s for a, s in signals.items() if s.get("valid")}
    n_valid = len(valid_signals)

    # 统计方向: model_p > 0.5 → YES 倾向, < 0.5 → NO 倾向
    n_yes = sum(1 for s in valid_signals.values() if s["model_p"] > 0.5)
    n_no = sum(1 for s in valid_signals.values() if s["model_p"] < 0.5)
    # 用 z-score 的均值来衡量整体方向强度
    z_values = [s["z"] for s in valid_signals.values()]
    avg_z = sum(z_values) / len(z_values) if z_values else 0

    if n_valid >= 3:
        majority = max(n_yes, n_no)
        agreement = majority / n_valid  # 0.5 ~ 1.0
    else:
        agreement = 0.5

    # 共识方向
    consensus_side = "YES" if n_yes > n_no else ("NO" if n_no > n_yes else None)

    # === Phase 3: 给每个资产分配最终 edge ===
    osc = oscillation_score(tracker._settled)

    for a in ASSETS:
        m = markets.get(a, {})
        sig = signals[a]

        if not sig.get("valid"):
            results[a] = (None, 0, sig)
            continue

        mid = m.get("mid")
        model_p = sig["model_p"]
        asset_side = sig["raw_side"]

        # Base confidence (from oscillation)
        if osc > 0.7:
            base_conf = 0.15
        elif osc < 0.3 and len(tracker._settled) >= 6:
            base_conf = 0.35
        else:
            base_conf = 0.25

        # Consensus multiplier
        if n_valid >= 3:
            if agreement >= 0.99:  # 4/4 or 3/3
                if asset_side == consensus_side:
                    # 和共识一致 → 放大
                    consensus_mult = 1.5
                else:
                    # 和4/4共识矛盾 → 不做 (别逆势)
                    consensus_mult = 0.0
            elif agreement >= 0.74:  # 3/4
                if asset_side == consensus_side:
                    consensus_mult = 1.0
                else:
                    # 1个逆势的 → 大幅缩小
                    consensus_mult = 0.15
            else:  # 2/4 分裂
                consensus_mult = 0.3
        else:
            consensus_mult = 0.5  # 数据不足

        confidence = min(0.45, base_conf * consensus_mult)
        sig["confidence"] = round(confidence, 3)
        sig["consensus"] = f"{n_yes}Y{n_no}N"
        sig["agreement"] = round(agreement, 2)
        sig["osc"] = round(osc, 2)
        sig["avg_z"] = round(avg_z, 3)

        if confidence < 0.01:
            results[a] = (None, 0, sig)
            continue

        # v4: mid-pvs 矛盾检测 — 如果 pvs 方向和市场 mid 方向冲突, 跳过
        if sig.get("conflict"):
            sig["edge_mode"] = "conflict"
            results[a] = (None, 0, sig)
            continue

        # === STRONG CONSENSUS: trend-following mode ===
        # When 3/4 or 4/4 agree AND this asset agrees, trade consensus direction.
        # Bypass blend formula (which can produce contrarian trades when model_p < mid).
        # Edge from z-score magnitude + consensus strength.
        if (n_valid >= 3 and agreement >= 0.74
                and consensus_side is not None
                and asset_side == consensus_side):
            z_abs = abs(sig["z"])
            if agreement >= 0.99:  # 4/4 unanimous
                base_edge = 0.025
                z_bonus = min(0.03, 0.015 * z_abs)
            else:  # 3/4 majority
                base_edge = 0.015
                z_bonus = min(0.02, 0.01 * z_abs)
            edge = base_edge + z_bonus

            # Price sanity: don't buy at extreme prices
            if consensus_side == "YES" and mid > 0.80:
                results[a] = (None, 0, sig)
                continue
            elif consensus_side == "NO" and mid < 0.20:
                results[a] = (None, 0, sig)
                continue

            sig["edge_mode"] = "consensus"
            results[a] = (consensus_side, edge, sig)
        else:
            # === WEAK CONSENSUS / SPLIT: value-blend mode ===
            blended_p = confidence * model_p + (1 - confidence) * mid
            edge = abs(blended_p - mid)

            if blended_p > mid + 0.003:
                sig["edge_mode"] = "blend"
                results[a] = ("YES", edge, sig)
            elif blended_p < mid - 0.003:
                sig["edge_mode"] = "blend"
                results[a] = ("NO", edge, sig)
            else:
                results[a] = (None, 0, sig)

    return results


# ==============================================================================
# Poller
# ==============================================================================

class Poller:
    def __init__(self):
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
            url = f"{KALSHI_API}/markets?series_ticker={info['series']}&status=open&limit=1"
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
            url = f"{KALSHI_API}/markets/{ticker}/orderbook?depth={OB_DEPTH}"
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
# Settlement lookup
# ==============================================================================

def lookup_settlement(ticker: str, retries: int = 5, delay: float = 3.0) -> Optional[bool]:
    for attempt in range(retries):
        try:
            r = requests.get(f"{KALSHI_API}/markets/{ticker}", timeout=5)
            if r.status_code == 200:
                m = r.json().get("market", {})
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


def format_dashboard(cycle: int, elapsed: float, duration: float,
                     markets: dict, tracker: TrendTracker,
                     risk: RiskManager, last_signals: dict,
                     pending_settlements: list) -> List[str]:
    lines = []
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    pct = elapsed / duration * 100

    osc = oscillation_score(tracker._settled)
    regime = "OSCILLATE" if osc > 0.6 else ("TREND" if osc < 0.35 else "MIXED")
    n_settled = len(tracker._settled)

    # Compute current consensus from last_signals
    valid_sigs = {a: s for a, s in last_signals.items() if len(s) > 3 and s[3].get("valid")}
    consensus_str = ""
    if valid_sigs:
        sample = list(valid_sigs.values())[0]
        consensus_str = f"  Consensus: {sample[3].get('consensus','?')} | "

    lines.append(f"{'=' * W}")
    lines.append(f"  KALSHI LIVE PAPER v4  |  {now_str}  |  cycle {cycle}  |  "
                 f"{elapsed:.0f}s/{duration:.0f}s ({pct:.0f}%)")
    lines.append(f"  {consensus_str}Regime: {regime} (osc={osc:.2f})  |  "
                 f"Settled: {n_settled}  |  "
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
            stale = info.get("stale_sec", 0)
            stale_tag = f" STALE:{stale:.0f}s" if stale > 30 else ""
            conflict_tag = " CONFLICT" if info.get("conflict") else ""
            if side:
                lines.append(f"    {a}: {side} e={edge:.4f} conf={conf:.3f} "
                             f"[{cons} agr={agr:.0%}{mode_tag}{stale_tag}{conflict_tag}] ({reason})")
            else:
                lines.append(f"    {a}: -- [{cons}{mode_tag}{stale_tag}{conflict_tag}] ({reason})")
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
        lines.append(f"  {'-' * 56}")
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

    if risk.stopped:
        lines.append(f"  *** STOPPED ***")

    lines.append(f"{'=' * W}")
    return lines


# ==============================================================================
# Log
# ==============================================================================

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ==============================================================================
# Main Loop
# ==============================================================================

async def main():
    duration = DEFAULT_DURATION
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--duration" and i + 1 < len(sys.argv) - 1:
            duration = int(sys.argv[i + 2])

    # Init
    with open(LOG_FILE, "w") as f:
        f.write(f"=== Kalshi Live Paper Trader v4 ===\n")
        f.write(f"Strategy: consensus + stale guard + contract price cap\n")
        f.write(f"Duration: {duration}s | Poll: {POLL_SEC}s\n\n")

    recorder = DataRecorder(DB_FILE)
    poller = Poller()
    tracker = TrendTracker()
    risk = RiskManager(bankroll=100.0)

    last_tickers: Dict[str, str] = {}
    last_signals: Dict[str, tuple] = {}
    pending_settlements: List[str] = []
    deferred: List[dict] = []
    cycle = 0
    start_time = time.time()
    last_log_time = 0
    last_record_time = 0

    print(f"Kalshi Live Paper Trader v4")
    print(f"Strategy: consensus + stale guard + contract cap + mid-pvs conflict")
    print(f"Duration: {duration}s | Poll: {POLL_SEC}s")
    print(f"Data: {DB_FILE} + {CSV_FILE}")
    print(f"Log:  {LOG_FILE}")
    print(f"Press Ctrl+C to stop\n")

    log(f"Config: min_edge={risk.min_edge} kelly={risk.kelly_fraction} "
        f"mid_range={risk.mid_range} tte_range={risk.tte_range}")

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            cycle += 1
            now = time.time()

            # Poll
            markets = await poller.poll_all()

            # Feed prices + mids
            for a in ASSETS:
                rp = markets.get(a, {}).get("real_price")
                tracker.update(a, now, rp)
                mid_val = markets.get(a, {}).get("mid")
                tracker.update_mid(a, now, mid_val)

            # --- Transitions ---
            for a in ASSETS:
                new_ticker = markets.get(a, {}).get("ticker")
                if not new_ticker:
                    continue
                if a in last_tickers and last_tickers[a] != new_ticker:
                    old_ticker = last_tickers[a]
                    msg = f"{a}: {old_ticker[:30]} -> {new_ticker[:30]}"
                    result = lookup_settlement(old_ticker, retries=2, delay=2.0)

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
                last_tickers[a] = new_ticker

            # --- Deferred retries ---
            still_deferred = []
            for d in deferred:
                if now - d["ts"] > 300:
                    if d["pos"]:
                        log(f"DEFERRED EXPIRED: {d['asset']} {d['ticker']}")
                    continue
                result = lookup_settlement(d["ticker"], retries=1, delay=0)
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

            # --- Strategy (consensus-based) ---
            if not risk.stopped:
                strat_results = consensus_strategy(tracker, markets, now)

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

                    # ENTER
                    risk.open_position(a, side, mid, contracts, cost,
                                       m.get("ticker", ""))
                    consensus_s = info.get("consensus", "?")
                    last_signals[a] = (side, edge,
                                       f"ENTERED x{contracts}@${mid:.2f} [{consensus_s}]",
                                       info)
                    log(f"TRADE: {a} {side} x{contracts} @${mid:.2f} "
                        f"edge={edge:.4f} z={info.get('z',0):.2f} "
                        f"mp={info.get('model_p',0):.3f} "
                        f"conf={info.get('confidence',0):.3f} "
                        f"consensus={consensus_s} "
                        f"mode={info.get('edge_mode','?')} "
                        f"stale={info.get('stale_sec',0):.0f}s "
                        f"osc={info.get('osc',0):.2f} "
                        f"ticker={m.get('ticker','')}")
                    recorder.record_trade(
                        now, a, side, contracts, mid,
                        maker_fee(contracts, mid), m.get("ticker", ""),
                        edge, info.get("z", 0), info.get("model_p", 0.5),
                        info.get("confidence", 0.25))

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

            # --- Display ---
            lines = format_dashboard(
                cycle, elapsed, duration, markets, tracker,
                risk, last_signals, pending_settlements)
            clear_and_print(lines)

            # Wait
            poll_elapsed = time.time() - now
            wait = max(POLL_SEC - poll_elapsed, 0.1)
            await asyncio.sleep(wait)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    finally:
        await poller.close()
        n_rows = recorder.export_csv()
        recorder.close()

    # Final
    log(f"=== SESSION ENDED ===")
    print(f"\n{'=' * W}")
    print(f"  FINAL RESULTS (v2 calibrated strategy)")
    print(f"{'=' * W}")
    print(f"  Duration:   {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Cycles:     {cycle}")
    print(f"  Data:       {n_rows} ticks saved to {DB_FILE} + {CSV_FILE}")
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

    if risk.positions:
        print(f"\n  Open positions:")
        for a, pos in risk.positions.items():
            print(f"    {a}: {pos['side']} x{pos['contracts']} @${pos['entry_price']:.2f}")
    print(f"{'=' * W}")


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())
