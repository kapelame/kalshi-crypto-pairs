#!/usr/bin/env python3
"""
Kalshi Paper Trader v1
======================
Walk-forward 回测 + 手续费 + 风控

核心设计:
  1. Walk-forward CV: 按15分钟周期切分, 用过去训练/预测当前周期
  2. 两种策略对比: 规则驱动 vs ML (GradientBoosting)
  3. Taker/Maker 手续费模型
  4. 完整风控: Kelly仓位, 尾部过滤, 相关性约束, 止损

用法: python3 kalshi_paper_trader.py [csv_path]
"""

import pandas as pd
import numpy as np
import requests
import math
import sys
import os
import sqlite3
import csv as csvmod
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

warnings.filterwarnings("ignore")

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
ASSETS = ["btc", "eth", "sol", "xrp"]
CSV = "kalshi_v3_features.csv"
DB = "kalshi_v3.db"
W = 80

if len(sys.argv) > 1:
    CSV = sys.argv[1]


# ==============================================================================
# Fee Model
# ==============================================================================

def taker_fee(contracts: int, price: float) -> float:
    """Kalshi taker fee: ceil(0.07 * C * P * (1-P) * 100) / 100"""
    if contracts <= 0 or price <= 0 or price >= 1:
        return 0
    raw = 0.07 * contracts * price * (1 - price)
    return math.ceil(raw * 100) / 100


def maker_fee(contracts: int, price: float) -> float:
    """Kalshi maker fee: ceil(0.0175 * C * P * (1-P) * 100) / 100"""
    if contracts <= 0 or price <= 0 or price >= 1:
        return 0
    raw = 0.0175 * contracts * price * (1 - price)
    return math.ceil(raw * 100) / 100


# ==============================================================================
# Risk Manager
# ==============================================================================

@dataclass
class RiskConfig:
    bankroll: float = 100.0
    max_per_trade_pct: float = 0.05        # 单笔最大 bankroll 的 5%
    max_exposure_pct: float = 0.15         # 总持仓不超过 15%
    max_same_direction: int = 2            # 同方向最多2个资产
    daily_loss_limit_pct: float = 0.15     # 日亏损 15% 停
    min_edge: float = 0.04                 # 最小 edge 4% (含手续费缓冲)
    kelly_fraction: float = 0.15           # Kelly 的 15%
    mid_range: Tuple[float, float] = (0.25, 0.75)   # 只交易这个范围
    tte_range: Tuple[float, float] = (180, 700)      # 入场时间窗口(秒)
    max_spread: float = 0.04              # 最大 spread
    use_maker: bool = True                 # 用 limit order


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.cfg = config
        self.bankroll = config.bankroll
        self.initial_bankroll = config.bankroll
        self.positions: Dict[str, dict] = {}   # asset -> position
        self.session_pnl = 0.0
        self.peak_bankroll = config.bankroll
        self.stopped = False

    @property
    def current_exposure(self) -> float:
        return sum(p["cost"] for p in self.positions.values())

    def direction_count(self, direction: str) -> int:
        return sum(1 for p in self.positions.values() if p["side"] == direction)

    def can_trade(self, asset: str, side: str, mid: float,
                  spread: float, tte: float) -> Tuple[bool, str]:
        if self.stopped:
            return False, "stopped_loss_limit"
        if asset in self.positions:
            return False, "already_in_position"
        if mid is None or tte is None:
            return False, "missing_data"
        if not (self.cfg.mid_range[0] <= mid <= self.cfg.mid_range[1]):
            return False, f"mid={mid:.2f}_outside_range"
        if not (self.cfg.tte_range[0] <= tte <= self.cfg.tte_range[1]):
            return False, f"tte={tte:.0f}_outside_window"
        if spread is not None and spread > self.cfg.max_spread:
            return False, f"spread={spread:.3f}_too_wide"
        if self.current_exposure >= self.bankroll * self.cfg.max_exposure_pct:
            return False, "exposure_limit"
        if self.direction_count(side) >= self.cfg.max_same_direction:
            return False, f"max_{side}_direction"
        return True, "ok"

    def calc_position_size(self, edge: float, mid: float, side: str) -> Tuple[int, float]:
        """返回 (合约数, 总成本)"""
        if side == "YES":
            price = mid
            odds = (1 - price) / price
        else:
            price = 1 - mid
            odds = mid / (1 - mid)

        kelly = abs(edge) / odds if odds > 0 else 0
        raw_size_usd = self.bankroll * self.cfg.kelly_fraction * kelly

        max_usd = self.bankroll * self.cfg.max_per_trade_pct
        remaining = self.bankroll * self.cfg.max_exposure_pct - self.current_exposure
        size_usd = min(raw_size_usd, max_usd, remaining)
        size_usd = max(size_usd, 0)

        if size_usd < price:
            return 0, 0

        contracts = int(size_usd / price)
        if contracts <= 0:
            return 0, 0

        # Check fee eats edge
        fee = maker_fee(contracts, mid) if self.cfg.use_maker else taker_fee(contracts, mid)
        cost = contracts * price + fee
        net_edge_per_contract = abs(edge) - fee / contracts if contracts > 0 else 0

        if net_edge_per_contract < 0.02:
            return 0, 0

        return contracts, cost

    def open_position(self, asset: str, side: str, mid: float,
                      contracts: int, cost: float, ticker: str, ts: str):
        fee = maker_fee(contracts, mid) if self.cfg.use_maker else taker_fee(contracts, mid)
        self.positions[asset] = {
            "side": side,
            "entry_price": mid,
            "contracts": contracts,
            "cost": cost,
            "fee": fee,
            "ticker": ticker,
            "entry_ts": ts,
        }

    def close_position(self, asset: str, settled_yes: bool) -> dict:
        pos = self.positions.pop(asset)
        side = pos["side"]
        contracts = pos["contracts"]
        entry_price = pos["entry_price"]
        fee = pos["fee"]

        if side == "YES":
            if settled_yes:
                pnl = contracts * (1 - entry_price) - fee
            else:
                pnl = -(contracts * entry_price + fee)
        else:  # NO
            if not settled_yes:
                pnl = contracts * entry_price - fee
            else:
                pnl = -(contracts * (1 - entry_price) + fee)

        self.bankroll += pnl
        self.session_pnl += pnl
        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)

        # Check loss limit
        if self.bankroll < self.initial_bankroll * (1 - self.cfg.daily_loss_limit_pct):
            self.stopped = True

        return {
            "asset": asset,
            "side": side,
            "entry_price": entry_price,
            "contracts": contracts,
            "fee": fee,
            "settled_yes": settled_yes,
            "pnl": round(pnl, 4),
            "bankroll_after": round(self.bankroll, 2),
            "ticker": pos["ticker"],
            "entry_ts": pos["entry_ts"],
        }


# ==============================================================================
# Strategies
# ==============================================================================

class TrendTracker:
    """跨周期价格趋势追踪 — 解决策略只看当前周期的问题"""
    def __init__(self):
        self._prices: Dict[str, list] = {a: [] for a in ASSETS}
        self._settled: list = []  # 最近N个周期的结算结果

    def update(self, asset: str, ts_unix: float, real_price: Optional[float]):
        if real_price is not None:
            self._prices[asset].append((ts_unix, real_price))
            # 只保留最近60分钟
            cutoff = ts_unix - 3600
            self._prices[asset] = [(t, p) for t, p in self._prices[asset] if t > cutoff]

    def record_settlement(self, settled_yes: bool):
        self._settled.append(1 if settled_yes else 0)
        if len(self._settled) > 20:
            self._settled = self._settled[-20:]

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

    def recent_settlement_rate(self, n: int = 4) -> Optional[float]:
        """最近N个周期的YES率"""
        recent = self._settled[-n:] if self._settled else []
        return sum(recent) / len(recent) if recent else None


# 全局趋势追踪器
_trend_tracker = TrendTracker()


def rule_strategy(row: dict, asset: str) -> Tuple[Optional[str], float]:
    """
    规则驱动策略 v2:
      1. 长期趋势 (15-60min外部价格方向) — 权重最大
      2. 当前周期 pvs (价格vs参考价) — 中等权重
      3. 最近结算历史 (连续跌→偏向NO) — 修正因子
      4. Orderbook imbalance — 辅助确认
    """
    p = asset
    mid = row.get(f"{p}_mid")
    pvs = row.get(f"{p}_price_vs_strike_pct")
    tte = row.get(f"{p}_time_to_expiry")
    mom5 = row.get(f"{p}_mom_5s") or 0
    ob = row.get(f"{p}_ob_imbalance") or 0
    real_price = row.get(f"{p}_real_price")
    ts_unix = row.get("ts_unix", 0)

    if mid is None or pvs is None or tte is None:
        return None, 0

    # 趋势追踪数据由 simulate() 在每行开始时统一更新

    # === 信号1: 外部价格长期趋势 (最重要) ===
    trend_15m = _trend_tracker.price_trend(asset, 900) or 0    # 15min
    trend_30m = _trend_tracker.price_trend(asset, 1800) or 0   # 30min

    # === 信号2: 当前周期 pvs ===
    tte_frac = max(tte / 900, 0.1)
    time_factor = 1.0 / tte_frac

    # === 信号3: 最近结算历史 ===
    settle_rate = _trend_tracker.recent_settlement_rate(4)
    # settle_rate=0.25 → 最近多数跌 → 偏向NO
    # settle_rate=0.75 → 最近多数涨 → 偏向YES
    settle_bias = (settle_rate - 0.5) * 0.3 if settle_rate is not None else 0

    # === 综合信号 ===
    # 长期趋势权重最大: trend * 5
    # pvs 贡献中等: pvs * time_factor * 2
    # 结算历史: settle_bias
    # ob 贡献最小
    signal = (
        trend_15m * 5.0            # 15min价格趋势 (最重要)
        + trend_30m * 3.0          # 30min价格趋势
        + pvs * time_factor * 2.0  # 当前pvs (看当下)
        + settle_bias              # 结算历史偏移
        + ob * 0.01                # orderbook (微调)
    )

    estimated_p = 0.5 + 0.5 * math.tanh(signal)

    # === 方向一致性检查 ===
    # 如果长期趋势和pvs方向矛盾, 降低edge (不确定性高)
    trend_sign = 1 if trend_15m > 0 else (-1 if trend_15m < 0 else 0)
    pvs_sign = 1 if pvs > 0 else (-1 if pvs < 0 else 0)
    agreement = 1.0 if trend_sign == pvs_sign else 0.5  # 矛盾时edge减半

    if estimated_p > mid:
        edge = (estimated_p - mid) * agreement
        return "YES", edge
    elif estimated_p < mid:
        edge = (mid - estimated_p) * agreement
        return "NO", edge
    return None, 0


def ml_strategy(row: dict, asset: str, model, feature_cols: list) -> Tuple[Optional[str], float]:
    """
    ML 策略: 用 GradientBoosting 预测 P(YES)
    """
    if model is None:
        return None, 0

    p = asset
    mid = row.get(f"{p}_mid")
    if mid is None:
        return None, 0

    feat = {}
    for col in feature_cols:
        if col == "btc_mid_ctx":
            feat[col] = row.get("btc_mid") if asset != "btc" else 0
        elif col == "btc_lead":
            btc_mom = row.get("btc_mom_5s") or 0
            own_mom = row.get(f"{p}_mom_5s") or 0
            feat[col] = btc_mom - own_mom if asset != "btc" else 0
        elif col == "mid_decisiveness":
            feat[col] = abs(mid - 0.5) if mid is not None else 0
        else:
            feat[col] = row.get(f"{p}_{col}") or 0

    X = pd.DataFrame([feat])[feature_cols].fillna(0)

    try:
        prob_yes = model.predict_proba(X)[0, 1]
    except:
        return None, 0

    if prob_yes > mid:
        return "YES", prob_yes - mid
    elif prob_yes < mid:
        return "NO", mid - prob_yes
    return None, 0


# ==============================================================================
# Walk-Forward Engine
# ==============================================================================

def get_periods(df: pd.DataFrame) -> List[dict]:
    """提取所有15分钟周期, 按时间排序"""
    periods = {}
    for a in ASSETS:
        for ticker in df[f"{a}_ticker"].dropna().unique():
            mask = df[f"{a}_ticker"] == ticker
            rows = df[mask]
            if len(rows) == 0:
                continue
            key = ticker
            if key not in periods:
                periods[key] = {
                    "ticker": ticker,
                    "asset": a,
                    "start_ts": rows["ts_unix"].min(),
                    "end_ts": rows["ts_unix"].max(),
                    "n_rows": len(rows),
                }
    # Sort by start time
    return sorted(periods.values(), key=lambda x: (x["start_ts"], x["asset"]))


def get_settlements(tickers: set) -> Dict[str, int]:
    """从 Kalshi API 获取结算结果"""
    settlements = {}
    for ticker in tickers:
        try:
            r = requests.get(f"{KALSHI_API}/markets/{ticker}", timeout=5)
            if r.status_code == 200:
                m = r.json().get("market", {})
                result = m.get("result", "")
                if result == "yes":
                    settlements[ticker] = 1
                elif result == "no":
                    settlements[ticker] = 0
        except:
            pass
    return settlements


def build_ml_training_data(df: pd.DataFrame, train_tickers: Dict[str, int],
                           feature_cols: list) -> Tuple[pd.DataFrame, np.ndarray]:
    """从指定 ticker 集合构建训练数据"""
    rows = []
    for a in ASSETS:
        p = a
        for _, row in df.iterrows():
            ticker = row.get(f"{a}_ticker")
            if ticker not in train_tickers:
                continue
            mid = row.get(f"{p}_mid")
            tte = row.get(f"{p}_time_to_expiry")
            if pd.isna(mid) or pd.isna(tte):
                continue

            feat = {"target": train_tickers[ticker]}
            for col in feature_cols:
                if col == "btc_mid_ctx":
                    feat[col] = row.get("btc_mid") if a != "btc" else 0
                elif col == "btc_lead":
                    btc_mom = row.get("btc_mom_5s") or 0
                    own_mom = row.get(f"{p}_mom_5s") or 0
                    feat[col] = btc_mom - own_mom if a != "btc" else 0
                elif col == "mid_decisiveness":
                    feat[col] = abs(mid - 0.5)
                else:
                    feat[col] = row.get(f"{p}_{col}") or 0
            rows.append(feat)

    if not rows:
        return pd.DataFrame(), np.array([])
    feat_df = pd.DataFrame(rows).fillna(0)
    y = feat_df.pop("target").values
    return feat_df[feature_cols], y


def train_ml_model(X, y):
    """训练 GradientBoosting"""
    if len(X) < 30 or len(np.unique(y)) < 2:
        return None
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=10,
        )
        model.fit(X, y)
        return model
    except:
        return None


# ==============================================================================
# Paper Trading Simulation
# ==============================================================================

def simulate(df: pd.DataFrame, settlements: Dict[str, int],
             strategy_name: str, config: RiskConfig) -> dict:
    """
    Walk-forward paper trading 模拟

    对每一行数据:
      1. 检查是否有仓位需要结算 (ticker 变了)
      2. 如果无仓位且条件满足, 入场
    """
    risk = RiskManager(config)
    feature_cols = [
        "mid", "spread", "time_to_expiry", "time_to_expiry_pct",
        "ob_imbalance", "price_vs_strike_pct", "volume", "oi",
        "mom_5s", "mom_15s", "mom_30s",
        "btc_mid_ctx", "btc_lead", "mid_decisiveness",
    ]

    # Reset global trend tracker for each strategy run
    global _trend_tracker
    _trend_tracker = TrendTracker()

    # Track which tickers we've seen settle (for ML training)
    seen_settled = {}
    current_tickers = {}
    ml_model = None
    trades = []
    skipped = defaultdict(int)
    bankroll_history = [config.bankroll]

    # Determine unique time periods for retraining
    retrain_tickers = set()

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        ts_unix = row_dict.get("ts_unix", 0)

        # 0. Feed real prices into trend tracker (every row, every asset)
        for a in ASSETS:
            real_price = row_dict.get(f"{a}_real_price")
            _trend_tracker.update(a, ts_unix, real_price)

        # 1. Check for settlements (ticker changed)
        for a in ASSETS:
            new_ticker = row_dict.get(f"{a}_ticker")
            if a in current_tickers and current_tickers[a] != new_ticker:
                old_ticker = current_tickers[a]
                settled_yes = None
                # Settle position if we have one
                if a in risk.positions and risk.positions[a]["ticker"] == old_ticker:
                    if old_ticker in settlements:
                        settled_yes = settlements[old_ticker] == 1
                        trade = risk.close_position(a, settled_yes)
                        trades.append(trade)
                        bankroll_history.append(risk.bankroll)

                # Record settlement in trend tracker
                if old_ticker in settlements:
                    s_yes = settlements[old_ticker] == 1
                    _trend_tracker.record_settlement(s_yes)
                    seen_settled[old_ticker] = settlements[old_ticker]
                    retrain_tickers.add(old_ticker)

            current_tickers[a] = new_ticker

        # 2. Retrain ML model when we have new settled data
        if strategy_name == "ML" and retrain_tickers:
            if len(seen_settled) >= 4 and len(np.unique(list(seen_settled.values()))) >= 2:
                X_train, y_train = build_ml_training_data(
                    df.iloc[:idx], seen_settled, feature_cols)
                if len(X_train) >= 30:
                    ml_model = train_ml_model(X_train, y_train)
            retrain_tickers.clear()

        if risk.stopped:
            continue

        # 3. Check entries for each asset
        for a in ASSETS:
            p = a
            mid = row_dict.get(f"{p}_mid")
            tte = row_dict.get(f"{p}_time_to_expiry")
            spread = row_dict.get(f"{p}_spread")
            ticker = row_dict.get(f"{p}_ticker")

            if mid is None or tte is None or ticker is None:
                continue

            # Get strategy signal
            if strategy_name == "rule":
                side, edge = rule_strategy(row_dict, a)
            elif strategy_name == "ML":
                side, edge = ml_strategy(row_dict, a, ml_model, feature_cols)
            else:
                continue

            if side is None or edge < config.min_edge:
                continue

            # Risk check
            ok, reason = risk.can_trade(a, side, mid, spread, tte)
            if not ok:
                skipped[reason] += 1
                continue

            # Position sizing
            contracts, cost = risk.calc_position_size(edge, mid, side)
            if contracts <= 0:
                skipped["size_too_small"] += 1
                continue

            # Open position
            risk.open_position(a, side, mid, contracts, cost, ticker,
                               row_dict.get("ts", ""))

    # Settle any remaining positions
    for a in list(risk.positions.keys()):
        ticker = risk.positions[a]["ticker"]
        if ticker in settlements:
            trade = risk.close_position(a, settlements[ticker] == 1)
            trades.append(trade)
            bankroll_history.append(risk.bankroll)

    return {
        "strategy": strategy_name,
        "trades": trades,
        "bankroll_history": bankroll_history,
        "final_bankroll": risk.bankroll,
        "skipped": dict(skipped),
        "stopped": risk.stopped,
    }


# ==============================================================================
# Performance Analytics
# ==============================================================================

def analyze(result: dict, config: RiskConfig):
    trades = result["trades"]
    bh = result["bankroll_history"]
    strategy = result["strategy"]

    print(f"\n{'=' * W}")
    print(f"  STRATEGY: {strategy.upper()}")
    print(f"{'=' * W}")

    if not trades:
        print(f"  No trades executed.")
        if result["skipped"]:
            print(f"  Skipped reasons:")
            for reason, count in sorted(result["skipped"].items(),
                                        key=lambda x: -x[1]):
                print(f"    {reason}: {count}")
        return

    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    total_fees = sum(t["fee"] for t in trades)
    total_contracts = sum(t["contracts"] for t in trades)

    # Win rate
    win_rate = len(wins) / n if n > 0 else 0

    # Average PnL
    avg_pnl = total_pnl / n if n > 0 else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0

    # Max drawdown
    peak = bh[0]
    max_dd = 0
    for b in bh:
        peak = max(peak, b)
        dd = (peak - b) / peak
        max_dd = max(max_dd, dd)

    # Profit factor
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # ROI
    roi = (result["final_bankroll"] - config.bankroll) / config.bankroll * 100

    print(f"\n  TRADES:")
    print(f"    total:     {n}")
    print(f"    wins:      {len(wins)} ({win_rate:.0%})")
    print(f"    losses:    {len(losses)}")
    print(f"    contracts: {total_contracts}")

    print(f"\n  PnL:")
    print(f"    total PnL:    ${total_pnl:+.2f}")
    print(f"    total fees:   ${total_fees:.2f}")
    print(f"    PnL pre-fee:  ${total_pnl + total_fees:+.2f}")
    print(f"    avg per trade: ${avg_pnl:+.4f}")
    print(f"    avg win:      ${avg_win:+.4f}")
    print(f"    avg loss:     ${avg_loss:+.4f}")

    print(f"\n  RISK:")
    print(f"    bankroll:      ${config.bankroll:.2f} → ${result['final_bankroll']:.2f}")
    print(f"    ROI:           {roi:+.2f}%")
    print(f"    max drawdown:  {max_dd:.2%}")
    print(f"    profit factor: {profit_factor:.2f}")
    print(f"    stopped:       {result['stopped']}")

    # Per-asset breakdown
    print(f"\n  PER ASSET:")
    for a in ASSETS:
        asset_trades = [t for t in trades if t["asset"] == a]
        if not asset_trades:
            continue
        pnl = sum(t["pnl"] for t in asset_trades)
        w = sum(1 for t in asset_trades if t["pnl"] > 0)
        print(f"    {a.upper()}: {len(asset_trades)} trades, "
              f"PnL=${pnl:+.2f}, win={w}/{len(asset_trades)}")

    # Per-side breakdown
    print(f"\n  PER SIDE:")
    for side in ["YES", "NO"]:
        side_trades = [t for t in trades if t["side"] == side]
        if not side_trades:
            continue
        pnl = sum(t["pnl"] for t in side_trades)
        w = sum(1 for t in side_trades if t["pnl"] > 0)
        print(f"    {side}: {len(side_trades)} trades, "
              f"PnL=${pnl:+.2f}, win={w}/{len(side_trades)}")

    # Trade log
    print(f"\n  TRADE LOG:")
    print(f"  {'#':>3} {'Asset':<5} {'Side':<4} {'Entry':>6} {'Ctrs':>5} "
          f"{'Fee':>5} {'Result':>7} {'PnL':>8} {'Bankroll':>9}")
    print(f"  {'-' * 60}")
    for i, t in enumerate(trades):
        result_str = "WIN" if t["pnl"] > 0 else "LOSS"
        settled = "YES" if t["settled_yes"] else "NO"
        print(f"  {i+1:>3} {t['asset'].upper():<5} {t['side']:<4} "
              f"${t['entry_price']:.2f} {t['contracts']:>5} "
              f"${t['fee']:.2f} {settled:>3}→{result_str:<4} "
              f"${t['pnl']:>+7.2f} ${t['bankroll_after']:>8.2f}")

    if result["skipped"]:
        print(f"\n  SKIP REASONS:")
        for reason, count in sorted(result["skipped"].items(),
                                    key=lambda x: -x[1])[:10]:
            print(f"    {reason}: {count}")


# ==============================================================================
# Cross-Validation Report
# ==============================================================================

def cv_report(df, settlements, config):
    """Walk-forward CV: 逐周期训练/测试"""
    print(f"\n{'#' * W}")
    print(f"  WALK-FORWARD CROSS-VALIDATION")
    print(f"{'#' * W}")

    # Get chronological periods
    period_times = {}
    for a in ASSETS:
        for ticker in df[f"{a}_ticker"].dropna().unique():
            mask = df[f"{a}_ticker"] == ticker
            if mask.sum() > 0:
                start = df.loc[mask, "ts_unix"].min()
                period_times[ticker] = start

    # Group by time period (all 4 assets share same 15-min window)
    # Tickers like KXBTC15M-26FEB242330-30 → time key "26FEB242330"
    time_groups = defaultdict(set)
    for ticker in period_times:
        parts = ticker.split("-")
        if len(parts) >= 2:
            time_key = parts[1]
            time_groups[time_key].add(ticker)

    sorted_groups = sorted(time_groups.items(),
                           key=lambda x: min(period_times.get(t, 0) for t in x[1]))

    print(f"  Found {len(sorted_groups)} time periods:")
    for i, (key, tickers) in enumerate(sorted_groups):
        settled = sum(1 for t in tickers if t in settlements)
        results = [settlements.get(t) for t in tickers if t in settlements]
        yes_n = sum(1 for r in results if r == 1)
        no_n = sum(1 for r in results if r == 0)
        print(f"    Period {i+1} [{key}]: {len(tickers)} tickers, "
              f"settled={settled} (Y={yes_n} N={no_n})")

    if len(sorted_groups) < 3:
        print(f"\n  Need at least 3 periods for walk-forward CV.")
        print(f"  Currently have {len(sorted_groups)}. Collect more data.")
        return

    # Walk-forward: train on periods 0..i-1, test on period i
    feature_cols = [
        "mid", "spread", "time_to_expiry", "time_to_expiry_pct",
        "ob_imbalance", "price_vs_strike_pct", "volume", "oi",
        "mom_5s", "mom_15s", "mom_30s",
        "btc_mid_ctx", "btc_lead", "mid_decisiveness",
    ]

    all_preds = []
    all_actuals = []

    for fold_idx in range(2, len(sorted_groups)):
        # Training tickers: all settled tickers from periods 0..fold_idx-1
        train_tickers = {}
        for j in range(fold_idx):
            for t in sorted_groups[j][1]:
                if t in settlements:
                    train_tickers[t] = settlements[t]

        # Test tickers: period fold_idx
        test_tickers = {}
        for t in sorted_groups[fold_idx][1]:
            if t in settlements:
                test_tickers[t] = settlements[t]

        if not train_tickers or not test_tickers:
            continue
        if len(np.unique(list(train_tickers.values()))) < 2:
            continue

        # Build train/test data
        X_train, y_train = build_ml_training_data(df, train_tickers, feature_cols)
        X_test, y_test = build_ml_training_data(df, test_tickers, feature_cols)

        if len(X_train) < 30 or len(X_test) < 10:
            continue

        model = train_ml_model(X_train, y_train)
        if model is None:
            continue

        preds = model.predict_proba(X_test)[:, 1]
        all_preds.extend(preds)
        all_actuals.extend(y_test)

        # Fold stats
        fold_acc = ((preds >= 0.5).astype(int) == y_test).mean()
        fold_brier = np.mean((preds - y_test) ** 2)
        print(f"\n  Fold {fold_idx}: train={len(X_train)} test={len(X_test)} "
              f"acc={fold_acc:.3f} brier={fold_brier:.4f}")

    if all_preds:
        all_preds = np.array(all_preds)
        all_actuals = np.array(all_actuals)
        overall_acc = ((all_preds >= 0.5).astype(int) == all_actuals).mean()
        overall_brier = np.mean((all_preds - all_actuals) ** 2)
        print(f"\n  OVERALL CV: acc={overall_acc:.3f} brier={overall_brier:.4f} "
              f"(random=0.2500)")
        print(f"  Samples: {len(all_preds)} "
              f"(YES={all_actuals.sum():.0f} NO={(1-all_actuals).sum():.0f})")
    else:
        print(f"\n  Not enough data for CV folds.")


# ==============================================================================
# Main
# ==============================================================================

def main():
    # Export fresh from DB
    if os.path.exists(DB):
        conn = sqlite3.connect(DB)
        cursor = conn.execute("SELECT * FROM features ORDER BY ts_unix")
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        with open(CSV, "w", newline="", encoding="utf-8") as f:
            w = csvmod.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        conn.close()
        print(f"  Exported {len(rows)} rows from {DB}")

    df = pd.read_csv(CSV)
    print(f"  Loaded {len(df)} rows")

    # Get all tickers
    all_tickers = set()
    for a in ASSETS:
        for t in df[f"{a}_ticker"].dropna().unique():
            all_tickers.add(t)

    print(f"  Found {len(all_tickers)} unique tickers, looking up settlements...")
    settlements = get_settlements(all_tickers)
    yes_n = sum(1 for v in settlements.values() if v == 1)
    no_n = sum(1 for v in settlements.values() if v == 0)
    print(f"  Settlements: {len(settlements)}/{len(all_tickers)} "
          f"(YES={yes_n}, NO={no_n})")

    if len(settlements) < 4:
        print(f"\n  Need at least 4 settled tickers. Wait for more data.")
        return

    config = RiskConfig()
    print(f"\n  RISK CONFIG:")
    print(f"    bankroll:        ${config.bankroll}")
    print(f"    max per trade:   {config.max_per_trade_pct:.0%}")
    print(f"    max exposure:    {config.max_exposure_pct:.0%}")
    print(f"    max same dir:    {config.max_same_direction}")
    print(f"    daily loss:      -{config.daily_loss_limit_pct:.0%}")
    print(f"    min edge:        {config.min_edge:.0%}")
    print(f"    kelly fraction:  {config.kelly_fraction:.0%}")
    print(f"    mid range:       {config.mid_range}")
    print(f"    TTE range:       {config.tte_range}")
    print(f"    fee mode:        {'maker' if config.use_maker else 'taker'}")

    # Cross-validation report
    cv_report(df, settlements, config)

    # Run both strategies
    print(f"\n{'#' * W}")
    print(f"  PAPER TRADING SIMULATION")
    print(f"{'#' * W}")

    for strategy in ["rule", "ML"]:
        result = simulate(df, settlements, strategy, config)
        analyze(result, config)

    # Comparison summary
    print(f"\n{'=' * W}")
    print(f"  STRATEGY COMPARISON")
    print(f"{'=' * W}")
    for strategy in ["rule", "ML"]:
        result = simulate(df, settlements, strategy, config)
        n_trades = len(result["trades"])
        pnl = sum(t["pnl"] for t in result["trades"])
        fees = sum(t["fee"] for t in result["trades"])
        wins = sum(1 for t in result["trades"] if t["pnl"] > 0)
        wr = wins / n_trades if n_trades > 0 else 0
        print(f"  {strategy.upper():>5}: {n_trades:>3} trades, "
              f"PnL=${pnl:>+7.2f}, fees=${fees:.2f}, "
              f"win={wr:.0%}, final=${result['final_bankroll']:.2f}")


if __name__ == "__main__":
    main()
