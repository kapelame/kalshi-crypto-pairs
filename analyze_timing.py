#!/usr/bin/env python3
"""
分析历史数据，找到最佳入场时机 (TTE)
用 kalshi_live_paper.db 的tick数据模拟不同TTE窗口的表现
"""

import sqlite3
import math
from collections import defaultdict

DB_FILE = "kalshi_live_paper.db"
ASSETS = ["BTC", "ETH", "SOL", "XRP"]
DEFAULT_VOL_15M = {"BTC": 0.15, "ETH": 0.20, "SOL": 0.35, "XRP": 0.30}

MIN_EDGE = 0.015
KELLY_FRACTION = 0.50
MID_RANGE = (0.20, 0.80)


def maker_fee(contracts, price):
    if contracts <= 0 or price <= 0 or price >= 1:
        return 0
    return math.ceil(0.0175 * contracts * price * (1 - price) * 100) / 100


def compute_signal(pvs, tte, mid, vol_15m):
    if pvs is None or tte is None or mid is None or tte < 60:
        return None, None
    if not (0.10 <= mid <= 0.90):
        return None, None
    vol_per_sec = vol_15m / math.sqrt(900)
    expected_std = vol_per_sec * math.sqrt(max(tte, 1))
    z = pvs / expected_std if expected_std > 0.001 else 0
    model_p = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return z, model_p


def simulate_tte_window(tte_min, tte_max, min_avg_z=0.5, require_4_4=True):
    """模拟一个TTE窗口的表现"""
    db = sqlite3.connect(DB_FILE)
    db.row_factory = sqlite3.Row

    # Load settlements
    settlements = {}
    for r in db.execute("SELECT * FROM settlements ORDER BY ts_unix"):
        tk = r['ticker']
        parts = tk.split('-')
        period = '-'.join(parts[1:])
        if period not in settlements:
            settlements[period] = {}
        settlements[period][r['asset']] = r['result']

    # Load ticks grouped by period
    periods_data = defaultdict(list)
    for r in db.execute("SELECT * FROM ticks ORDER BY ts_unix"):
        btc_tk = r['btc_ticker']
        if btc_tk:
            parts = btc_tk.split('-')
            period = '-'.join(parts[1:])
            periods_data[period].append(r)

    trades = []

    for period in sorted(periods_data.keys()):
        ticks = periods_data[period]
        settle = settlements.get(period, {})
        if len(settle) < 4:
            continue

        entered = False
        for tick in ticks:
            if entered:
                break

            btc_tte = tick['btc_tte']
            if btc_tte is None:
                continue
            if not (tte_min <= btc_tte <= tte_max):
                continue

            # Compute signals
            signals = {}
            for a in ASSETS:
                al = a.lower()
                mid = tick[f'{al}_mid']
                pvs = tick[f'{al}_pvs']
                tte = tick[f'{al}_tte']
                vol = DEFAULT_VOL_15M.get(a, 0.20)
                if mid is None or pvs is None or tte is None:
                    continue
                z, model_p = compute_signal(pvs, tte, mid, vol)
                if z is None:
                    continue
                if not (MID_RANGE[0] <= mid <= MID_RANGE[1]):
                    continue
                raw_side = "YES" if model_p > 0.5 else "NO"
                signals[a] = {"z": z, "model_p": model_p, "mid": mid,
                              "raw_side": raw_side, "tte": tte}

            if len(signals) < 3:
                continue

            # Consensus
            n_yes = sum(1 for s in signals.values() if s["raw_side"] == "YES")
            n_no = sum(1 for s in signals.values() if s["raw_side"] == "NO")
            n_valid = len(signals)
            majority = max(n_yes, n_no)
            agreement = majority / n_valid
            consensus_side = "YES" if n_yes > n_no else ("NO" if n_no > n_yes else None)

            avg_z = sum(s["z"] for s in signals.values()) / n_valid

            if consensus_side is None:
                continue

            # 4/4 unanimous check
            if require_4_4 and agreement < 0.99:
                continue
            elif not require_4_4 and agreement < 0.74:
                continue

            # avg|z| threshold
            if abs(avg_z) < min_avg_z:
                continue

            # Enter all consensus-aligned positions
            for a, sig in signals.items():
                if sig["raw_side"] != consensus_side:
                    continue

                mid = sig["mid"]
                cp = mid if consensus_side == "YES" else (1 - mid)

                # Dynamic sizing
                if cp <= 0.35:
                    contracts = 2
                elif cp <= 0.65:
                    contracts = 1
                else:
                    if abs(sig["z"]) < 0.5:
                        continue
                    contracts = 1

                fee = maker_fee(contracts, mid)
                settled_yes = settle.get(a) == "YES"

                if consensus_side == "YES":
                    pnl = contracts * (1 - mid) - fee if settled_yes else -(contracts * mid + fee)
                else:
                    pnl = contracts * mid - fee if not settled_yes else -(contracts * (1 - mid) + fee)

                trades.append({
                    "period": period, "asset": a, "side": consensus_side,
                    "contracts": contracts, "mid": mid, "cp": cp,
                    "fee": fee, "z": sig["z"], "avg_z": avg_z,
                    "tte": sig["tte"], "consensus": f"{n_yes}Y{n_no}N",
                    "settled": settle.get(a, "?"), "pnl": pnl,
                    "agreement": agreement,
                })

            entered = True

    db.close()
    return trades


# ============================================================
# 测试不同TTE窗口
# ============================================================
print("=" * 100)
print("  最佳入场时机分析 (历史数据回测)")
print("=" * 100)

windows = [
    # (name, tte_min, tte_max, min_avg_z, require_4_4)
    ("TTE 1-2min (60-120)",    60,  120, 0.5, True),
    ("TTE 2-3min (120-180)",   120, 180, 0.5, True),
    ("TTE 3-5min (180-300)",   180, 300, 0.5, True),
    ("TTE 2-5min (120-300)",   120, 300, 0.5, True),
    ("TTE 5-7min (300-420)",   300, 420, 0.5, True),
    ("TTE 7-10min (420-600)",  420, 600, 0.5, True),
    ("TTE 10-13min (600-780)", 600, 780, 0.5, True),
    ("TTE 5-10min (300-600)",  300, 600, 0.5, True),
    ("TTE 3-7min (180-420)",   180, 420, 0.5, True),
    # 放宽到 3/4 共识
    ("TTE 2-5min 3/4ok",       120, 300, 0.5, False),
    ("TTE 5-10min 3/4ok",      300, 600, 0.5, False),
    # 不同 avg_z 门槛
    ("TTE 2-5min z>0.3",       120, 300, 0.3, True),
    ("TTE 2-5min z>0.7",       120, 300, 0.7, True),
    ("TTE 5-10min z>0.3",      300, 600, 0.3, True),
]

print(f"\n{'Window':<25} {'Trades':>6} {'Wins':>5} {'WR':>6} {'PnL':>8} {'AvgPnL':>8} {'MaxLoss':>8} {'MaxWin':>8} {'Periods':>8}")
print("-" * 95)

best_pnl = -999
best_name = ""

for name, tmin, tmax, maz, r44 in windows:
    tl = simulate_tte_window(tmin, tmax, maz, r44)
    if not tl:
        print(f"  {name:<23} {'no trades':>6}")
        continue

    wins = sum(1 for t in tl if t["pnl"] > 0)
    losses = sum(1 for t in tl if t["pnl"] < 0)
    total_pnl = sum(t["pnl"] for t in tl)
    wr = wins / len(tl) * 100 if tl else 0
    avg_pnl = total_pnl / len(tl) if tl else 0
    max_loss = min(t["pnl"] for t in tl)
    max_win = max(t["pnl"] for t in tl)
    periods = len(set(t["period"] for t in tl))

    marker = " <<<" if total_pnl == max(total_pnl, best_pnl) and total_pnl > best_pnl else ""
    if total_pnl > best_pnl:
        best_pnl = total_pnl
        best_name = name

    print(f"  {name:<23} {len(tl):>6} {wins:>5} {wr:>5.1f}% ${total_pnl:>+7.2f} ${avg_pnl:>+7.3f} ${max_loss:>+7.2f} ${max_win:>+7.2f} {periods:>5}/15{marker}")

print(f"\n  BEST: {best_name} (PnL=${best_pnl:+.2f})")

# ============================================================
# 最佳窗口的详细交易
# ============================================================
print(f"\n{'=' * 100}")
print(f"  每分钟粒度 TTE 分析 (4/4 unanimous, avg|z|>0.5)")
print(f"{'=' * 100}")

print(f"\n{'TTE Window':<20} {'Trades':>6} {'Wins':>5} {'WR':>6} {'PnL':>8}")
print("-" * 50)

for minute in range(1, 14):
    tmin = minute * 60
    tmax = (minute + 1) * 60
    tl = simulate_tte_window(tmin, tmax, 0.5, True)
    if not tl:
        print(f"  {minute}-{minute+1}min ({tmin}-{tmax}s)  {'--':>6}")
        continue
    wins = sum(1 for t in tl if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in tl)
    wr = wins / len(tl) * 100
    print(f"  {minute}-{minute+1}min ({tmin}-{tmax}s)  {len(tl):>6} {wins:>5} {wr:>5.1f}% ${total_pnl:>+7.2f}")

# ============================================================
# 每个周期的TTE分布 - 什么时候有信号
# ============================================================
print(f"\n{'=' * 100}")
print(f"  信号出现时间分析 (每个周期第一次出现4/4共识的TTE)")
print(f"{'=' * 100}")

db = sqlite3.connect(DB_FILE)
db.row_factory = sqlite3.Row
settlements = {}
for r in db.execute("SELECT * FROM settlements ORDER BY ts_unix"):
    tk = r['ticker']
    parts = tk.split('-')
    period = '-'.join(parts[1:])
    if period not in settlements:
        settlements[period] = {}
    settlements[period][r['asset']] = r['result']

periods_data = defaultdict(list)
for r in db.execute("SELECT * FROM ticks ORDER BY ts_unix"):
    btc_tk = r['btc_ticker']
    if btc_tk:
        parts = btc_tk.split('-')
        period = '-'.join(parts[1:])
        periods_data[period].append(r)

print(f"\n{'Period':<22} {'First4/4':>9} {'First3/4':>9} {'MaxAvgZ':>8} {'Settlement':>10}")
print("-" * 65)

for period in sorted(periods_data.keys()):
    ticks = periods_data[period]
    settle = settlements.get(period, {})
    if len(settle) < 4:
        continue

    n_yes_s = sum(1 for v in settle.values() if v == "YES")
    n_no_s = sum(1 for v in settle.values() if v == "NO")

    first_4_4 = None
    first_3_4 = None
    max_avg_z = 0

    for tick in ticks:
        btc_tte = tick['btc_tte']
        if btc_tte is None:
            continue

        signals = {}
        for a in ASSETS:
            al = a.lower()
            mid = tick[f'{al}_mid']
            pvs = tick[f'{al}_pvs']
            tte = tick[f'{al}_tte']
            vol = DEFAULT_VOL_15M.get(a, 0.20)
            if mid is None or pvs is None or tte is None:
                continue
            z, model_p = compute_signal(pvs, tte, mid, vol)
            if z is None:
                continue
            if not (MID_RANGE[0] <= mid <= MID_RANGE[1]):
                continue
            raw_side = "YES" if model_p > 0.5 else "NO"
            signals[a] = {"z": z, "model_p": model_p, "raw_side": raw_side}

        if len(signals) < 3:
            continue

        n_yes = sum(1 for s in signals.values() if s["raw_side"] == "YES")
        n_no = sum(1 for s in signals.values() if s["raw_side"] == "NO")
        n_valid = len(signals)
        majority = max(n_yes, n_no)
        agreement = majority / n_valid
        avg_z = sum(s["z"] for s in signals.values()) / n_valid

        if abs(avg_z) > abs(max_avg_z):
            max_avg_z = avg_z

        if agreement >= 0.99 and abs(avg_z) >= 0.5 and first_4_4 is None:
            first_4_4 = btc_tte
        if agreement >= 0.74 and abs(avg_z) >= 0.5 and first_3_4 is None:
            first_3_4 = btc_tte

    f44 = f"{first_4_4:.0f}s" if first_4_4 else "--"
    f34 = f"{first_3_4:.0f}s" if first_3_4 else "--"
    print(f"  {period:<20} {f44:>9} {f34:>9} {max_avg_z:>+7.2f}   {n_yes_s}Y{n_no_s}N")

db.close()
