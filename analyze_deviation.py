#!/usr/bin/env python3
"""
偏差交易策略分析:
1. 每个周期中，哪个资产的mid和其他资产偏差最大?
2. 交易这个outlier是否稳定盈利?
3. 资产之间的mid相关性如何?
4. 什么程度的偏差值得交易?
"""

import sqlite3
import math
import statistics
from collections import defaultdict

DB_FILE = "kalshi_live_paper.db"
ASSETS = ["BTC", "ETH", "SOL", "XRP"]
DEFAULT_VOL_15M = {"BTC": 0.15, "ETH": 0.20, "SOL": 0.35, "XRP": 0.30}


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

# ============================================================
# 1. 资产mid之间的相关性
# ============================================================
print("=" * 100)
print("  1. 资产mid价格相关性分析")
print("=" * 100)

# Collect mid snapshots across all ticks
all_mids = {a: [] for a in ASSETS}
for period in sorted(periods_data.keys()):
    for tick in periods_data[period]:
        valid = True
        mids = {}
        for a in ASSETS:
            m = tick[f'{a.lower()}_mid']
            if m is None or m <= 0:
                valid = False
                break
            mids[a] = m
        if valid:
            for a in ASSETS:
                all_mids[a].append(mids[a])

n = len(all_mids["BTC"])
print(f"\n  样本量: {n} ticks")
print(f"\n  {'Pair':<12} {'Correlation':>12}")
print(f"  {'-'*25}")
for i, a1 in enumerate(ASSETS):
    for a2 in ASSETS[i+1:]:
        if len(all_mids[a1]) == len(all_mids[a2]) and len(all_mids[a1]) > 2:
            # Pearson correlation
            mean1 = statistics.mean(all_mids[a1])
            mean2 = statistics.mean(all_mids[a2])
            std1 = statistics.stdev(all_mids[a1])
            std2 = statistics.stdev(all_mids[a2])
            if std1 > 0 and std2 > 0:
                cov = sum((all_mids[a1][j] - mean1) * (all_mids[a2][j] - mean2)
                          for j in range(len(all_mids[a1]))) / (len(all_mids[a1]) - 1)
                corr = cov / (std1 * std2)
                print(f"  {a1}-{a2:<7} {corr:>12.4f}")

# ============================================================
# 2. 每个周期的outlier分析
# ============================================================
print(f"\n{'=' * 100}")
print("  2. 每个周期 Outlier 分析 (TTE 480-780, 跳过600-660)")
print("=" * 100)

outlier_trades = []
consensus_all_trades = []

print(f"\n  {'Period':<22} {'BTC':>6} {'ETH':>6} {'SOL':>6} {'XRP':>6} | {'Outlier':<5} {'Dev':>6} {'Side':>4} {'CP':>5} | {'Settled':>8} {'PnL':>7}")
print(f"  {'-'*95}")

for period in sorted(periods_data.keys()):
    ticks = periods_data[period]
    settle = settlements.get(period, {})
    if len(settle) < 4:
        continue

    n_yes_s = sum(1 for v in settle.values() if v == "YES")
    n_no_s = sum(1 for v in settle.values() if v == "NO")

    # Find best tick in TTE window
    best_tick = None
    best_signals = None
    for tick in ticks:
        btc_tte = tick['btc_tte']
        if btc_tte is None:
            continue
        if not (480 <= btc_tte <= 780):
            continue
        if 600 <= btc_tte <= 660:
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
            signals[a] = {"z": z, "model_p": model_p, "mid": mid, "tte": tte}

        if len(signals) >= 3 and (best_signals is None or len(signals) > len(best_signals)):
            best_tick = tick
            best_signals = signals

    if best_signals is None or len(best_signals) < 3:
        continue

    # Calculate consensus
    mids = {a: s["mid"] for a, s in best_signals.items()}
    n_yes = sum(1 for s in best_signals.values() if s["model_p"] > 0.5)
    n_no = sum(1 for s in best_signals.values() if s["model_p"] < 0.5)
    consensus_side = "YES" if n_yes > n_no else ("NO" if n_no > n_yes else None)
    avg_z = sum(s["z"] for s in best_signals.values()) / len(best_signals)

    if consensus_side is None:
        continue

    # Find outlier: asset whose mid deviates most from the group average
    avg_mid = sum(mids.values()) / len(mids)
    deviations = {}
    for a, mid in mids.items():
        # For consensus NO: high mid = outlier (market thinks YES, we think NO)
        # For consensus YES: low mid = outlier (market thinks NO, we think YES)
        dev = mid - avg_mid  # positive = higher than average
        deviations[a] = dev

    # Sort by absolute deviation
    sorted_devs = sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True)
    outlier_asset = sorted_devs[0][0]
    outlier_dev = sorted_devs[0][1]
    outlier_mid = mids[outlier_asset]

    # Should we trade the outlier?
    # If consensus is NO and outlier has HIGH mid → buy NO on outlier (market overpricing YES)
    # If consensus is YES and outlier has LOW mid → buy YES on outlier (market underpricing YES)
    if consensus_side == "NO" and outlier_dev > 0:
        trade_side = "NO"
        cp = 1 - outlier_mid
    elif consensus_side == "YES" and outlier_dev < 0:
        trade_side = "YES"
        cp = outlier_mid
    elif consensus_side == "NO" and outlier_dev < 0:
        # Outlier is MORE no than average - actually agrees MORE with consensus
        # Trade the asset that disagrees most = highest mid
        # Re-sort: for NO consensus, find highest mid
        sorted_by_mid = sorted(mids.items(), key=lambda x: x[1], reverse=True)
        outlier_asset = sorted_by_mid[0][0]
        outlier_mid = sorted_by_mid[0][1]
        outlier_dev = deviations[outlier_asset]
        trade_side = "NO"
        cp = 1 - outlier_mid
    else:
        sorted_by_mid = sorted(mids.items(), key=lambda x: x[1])
        outlier_asset = sorted_by_mid[0][0]
        outlier_mid = sorted_by_mid[0][1]
        outlier_dev = deviations[outlier_asset]
        trade_side = "YES"
        cp = outlier_mid

    # Simulate trade
    contracts = 2 if cp <= 0.35 else 1
    fee = maker_fee(contracts, outlier_mid)
    settled_yes = settle.get(outlier_asset) == "YES"

    if trade_side == "YES":
        pnl = contracts * (1 - outlier_mid) - fee if settled_yes else -(contracts * outlier_mid + fee)
    else:
        pnl = contracts * outlier_mid - fee if not settled_yes else -(contracts * (1 - outlier_mid) + fee)

    outlier_trades.append({
        "period": period, "asset": outlier_asset, "side": trade_side,
        "contracts": contracts, "mid": outlier_mid, "cp": cp,
        "dev": outlier_dev, "avg_mid": avg_mid, "pnl": pnl,
        "consensus": f"{n_yes}Y{n_no}N", "avg_z": avg_z,
        "settled": settle.get(outlier_asset, "?"),
    })

    # Also simulate consensus-all (trade all aligned assets)
    for a, sig in best_signals.items():
        raw_side = "YES" if sig["model_p"] > 0.5 else "NO"
        if raw_side != consensus_side:
            continue
        mid = sig["mid"]
        c_cp = mid if consensus_side == "YES" else (1 - mid)
        c_contracts = 2 if c_cp <= 0.35 else 1
        c_fee = maker_fee(c_contracts, mid)
        c_settled_yes = settle.get(a) == "YES"
        if consensus_side == "YES":
            c_pnl = c_contracts * (1 - mid) - c_fee if c_settled_yes else -(c_contracts * mid + c_fee)
        else:
            c_pnl = c_contracts * mid - c_fee if not c_settled_yes else -(c_contracts * (1 - mid) + c_fee)
        consensus_all_trades.append({
            "period": period, "asset": a, "pnl": c_pnl,
        })

    mid_strs = {a: f"{mids.get(a, 0):.2f}" for a in ASSETS}
    res = "WIN" if pnl > 0 else "LOSS"
    print(f"  {period:<22} {mid_strs.get('BTC','--'):>6} {mid_strs.get('ETH','--'):>6} "
          f"{mid_strs.get('SOL','--'):>6} {mid_strs.get('XRP','--'):>6} | "
          f"{outlier_asset:<5} {outlier_dev:>+.3f} {trade_side:>4} ${cp:.2f} | "
          f"{settle.get(outlier_asset,'?'):>4}={res:<4} ${pnl:>+.2f}")


# ============================================================
# 3. Outlier策略 vs 全共识策略 对比
# ============================================================
print(f"\n{'=' * 100}")
print("  3. 策略对比: Outlier-Only vs Consensus-All")
print("=" * 100)

# Outlier stats
o_wins = sum(1 for t in outlier_trades if t["pnl"] > 0)
o_pnl = sum(t["pnl"] for t in outlier_trades)
o_wr = o_wins / len(outlier_trades) * 100 if outlier_trades else 0

# Consensus-all stats
c_wins = sum(1 for t in consensus_all_trades if t["pnl"] > 0)
c_pnl = sum(t["pnl"] for t in consensus_all_trades)
c_wr = c_wins / len(consensus_all_trades) * 100 if consensus_all_trades else 0

print(f"\n  {'Strategy':<20} {'Trades':>7} {'Wins':>5} {'WR':>6} {'PnL':>8} {'Avg PnL':>8}")
print(f"  {'-'*55}")
print(f"  {'Outlier-Only':<20} {len(outlier_trades):>7} {o_wins:>5} {o_wr:>5.1f}% ${o_pnl:>+7.2f} ${o_pnl/max(len(outlier_trades),1):>+7.3f}")
print(f"  {'Consensus-All':<20} {len(consensus_all_trades):>7} {c_wins:>5} {c_wr:>5.1f}% ${c_pnl:>+7.2f} ${c_pnl/max(len(consensus_all_trades),1):>+7.3f}")

# ============================================================
# 4. 偏差大小 vs 胜率
# ============================================================
print(f"\n{'=' * 100}")
print("  4. 偏差大小 vs 胜率 (多大的偏差值得交易?)")
print("=" * 100)

dev_buckets = [
    ("dev < 0.05", 0, 0.05),
    ("dev 0.05-0.10", 0.05, 0.10),
    ("dev 0.10-0.15", 0.10, 0.15),
    ("dev 0.15-0.20", 0.15, 0.20),
    ("dev >= 0.20", 0.20, 1.0),
]

print(f"\n  {'Deviation':<16} {'Trades':>7} {'Wins':>5} {'WR':>6} {'PnL':>8} {'Assets'}")
print(f"  {'-'*60}")

for name, lo, hi in dev_buckets:
    bucket = [t for t in outlier_trades if lo <= abs(t["dev"]) < hi]
    if not bucket:
        print(f"  {name:<16} {'--':>7}")
        continue
    bw = sum(1 for t in bucket if t["pnl"] > 0)
    bpnl = sum(t["pnl"] for t in bucket)
    bwr = bw / len(bucket) * 100
    assets = ", ".join(sorted(set(t["asset"] for t in bucket)))
    print(f"  {name:<16} {len(bucket):>7} {bw:>5} {bwr:>5.1f}% ${bpnl:>+7.2f}  {assets}")

# ============================================================
# 5. 合约价格 vs 胜率 (cheap vs expensive)
# ============================================================
print(f"\n{'=' * 100}")
print("  5. 合约价格 vs 胜率")
print("=" * 100)

price_buckets = [
    ("cp <= $0.20", 0, 0.20),
    ("cp $0.20-0.35", 0.20, 0.35),
    ("cp $0.35-0.50", 0.35, 0.50),
    ("cp $0.50-0.65", 0.50, 0.65),
    ("cp > $0.65", 0.65, 1.0),
]

print(f"\n  {'Price Range':<16} {'Trades':>7} {'Wins':>5} {'WR':>6} {'PnL':>8}")
print(f"  {'-'*45}")

for name, lo, hi in price_buckets:
    bucket = [t for t in outlier_trades if lo <= t["cp"] < hi]
    if not bucket:
        print(f"  {name:<16} {'--':>7}")
        continue
    bw = sum(1 for t in bucket if t["pnl"] > 0)
    bpnl = sum(t["pnl"] for t in bucket)
    bwr = bw / len(bucket) * 100
    print(f"  {name:<16} {len(bucket):>7} {bw:>5} {bwr:>5.1f}% ${bpnl:>+7.2f}")

# ============================================================
# 6. avg|z| vs outlier交易胜率
# ============================================================
print(f"\n{'=' * 100}")
print("  6. avg|z| 门槛 vs Outlier胜率")
print("=" * 100)

z_thresholds = [0.0, 0.3, 0.5, 0.7, 1.0]
print(f"\n  {'Min avg|z|':<12} {'Trades':>7} {'Wins':>5} {'WR':>6} {'PnL':>8}")
print(f"  {'-'*40}")

for zt in z_thresholds:
    bucket = [t for t in outlier_trades if abs(t["avg_z"]) >= zt]
    if not bucket:
        print(f"  >={zt:<10} {'--':>7}")
        continue
    bw = sum(1 for t in bucket if t["pnl"] > 0)
    bpnl = sum(t["pnl"] for t in bucket)
    bwr = bw / len(bucket) * 100
    print(f"  >={zt:<10} {len(bucket):>7} {bw:>5} {bwr:>5.1f}% ${bpnl:>+7.2f}")

# ============================================================
# 7. Combo: outlier + consensus-all 混合策略
# ============================================================
print(f"\n{'=' * 100}")
print("  7. 混合策略: Outlier加仓 + Consensus其余标准仓")
print("=" * 100)
print("  (对偏差最大的outlier买2份, 其余共识资产买1份)")

combo_trades = []
for period in sorted(periods_data.keys()):
    ticks = periods_data[period]
    settle = settlements.get(period, {})
    if len(settle) < 4:
        continue

    best_signals = None
    for tick in ticks:
        btc_tte = tick['btc_tte']
        if btc_tte is None:
            continue
        if not (480 <= btc_tte <= 780):
            continue
        if 600 <= btc_tte <= 660:
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
            if not (0.20 <= mid <= 0.80):
                continue
            signals[a] = {"z": z, "model_p": model_p, "mid": mid}
        if len(signals) >= 3 and (best_signals is None or len(signals) > len(best_signals)):
            best_signals = signals
    if best_signals is None or len(best_signals) < 3:
        continue

    n_yes = sum(1 for s in best_signals.values() if s["model_p"] > 0.5)
    n_no = sum(1 for s in best_signals.values() if s["model_p"] < 0.5)
    n_valid = len(best_signals)
    majority = max(n_yes, n_no)
    agreement = majority / n_valid
    avg_z = sum(s["z"] for s in best_signals.values()) / n_valid
    consensus_side = "YES" if n_yes > n_no else ("NO" if n_no > n_yes else None)

    if consensus_side is None or agreement < 0.74 or abs(avg_z) < 0.5:
        continue

    mids = {a: s["mid"] for a, s in best_signals.items()}
    avg_mid = sum(mids.values()) / len(mids)

    # Find outlier (most deviant from consensus direction)
    if consensus_side == "NO":
        sorted_assets = sorted(mids.items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_assets = sorted(mids.items(), key=lambda x: x[1])
    outlier = sorted_assets[0][0]

    for a, sig in best_signals.items():
        raw_side = "YES" if sig["model_p"] > 0.5 else "NO"
        if raw_side != consensus_side:
            continue
        mid = sig["mid"]
        cp = mid if consensus_side == "YES" else (1 - mid)
        # Outlier gets 2 contracts, rest get 1
        contracts = 2 if a == outlier else 1
        fee = maker_fee(contracts, mid)
        settled_yes = settle.get(a) == "YES"
        if consensus_side == "YES":
            pnl = contracts * (1 - mid) - fee if settled_yes else -(contracts * mid + fee)
        else:
            pnl = contracts * mid - fee if not settled_yes else -(contracts * (1 - mid) + fee)
        combo_trades.append({"period": period, "asset": a, "pnl": pnl, "contracts": contracts})

combo_wins = sum(1 for t in combo_trades if t["pnl"] > 0)
combo_pnl = sum(t["pnl"] for t in combo_trades)
combo_wr = combo_wins / len(combo_trades) * 100 if combo_trades else 0

print(f"\n  {'Strategy':<25} {'Trades':>7} {'Wins':>5} {'WR':>6} {'PnL':>8} {'Avg PnL':>8}")
print(f"  {'-'*60}")
print(f"  {'Outlier-Only':<25} {len(outlier_trades):>7} {o_wins:>5} {o_wr:>5.1f}% ${o_pnl:>+7.2f} ${o_pnl/max(len(outlier_trades),1):>+7.3f}")
print(f"  {'Consensus-All (1 each)':<25} {len(consensus_all_trades):>7} {c_wins:>5} {c_wr:>5.1f}% ${c_pnl:>+7.2f} ${c_pnl/max(len(consensus_all_trades),1):>+7.3f}")
print(f"  {'Combo (outlier=2, rest=1)':<25} {len(combo_trades):>7} {combo_wins:>5} {combo_wr:>5.1f}% ${combo_pnl:>+7.2f} ${combo_pnl/max(len(combo_trades),1):>+7.3f}")

db.close()
