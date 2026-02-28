#!/usr/bin/env python3
"""
分析更多因子对共识策略胜率的影响:
1. 波动率 (mid变动幅度) vs 胜率
2. 加权市场波动率 (orderbook spread) vs 胜率
3. mid分散度 (4个资产mid的std) vs 胜率
4. price_vs_strike 动量 vs 胜率
5. ob_imbalance vs 胜率
6. 多因子组合
"""

import sqlite3
import math
import statistics
from collections import defaultdict

DB_FILE = "kalshi_live_paper.db"
ASSETS = ["BTC", "ETH", "SOL", "XRP"]
DEFAULT_VOL_15M = {"BTC": 0.15, "ETH": 0.20, "SOL": 0.35, "XRP": 0.30}

TTE_MIN, TTE_MAX = 480, 780
TTE_BLACKOUT = (600, 660)


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
# Collect per-period features at entry point
# ============================================================
period_features = []

for period in sorted(periods_data.keys()):
    ticks = periods_data[period]
    settle = settlements.get(period, {})
    if len(settle) < 4:
        continue

    # Collect mid history for volatility calculation
    mid_history = {a: [] for a in ASSETS}
    entry_tick = None
    entry_signals = None

    for tick in ticks:
        btc_tte = tick['btc_tte']
        if btc_tte is None:
            continue

        # Collect mid history
        for a in ASSETS:
            m = tick[f'{a.lower()}_mid']
            if m and m > 0:
                mid_history[a].append(m)

        # Find entry point
        if not (TTE_MIN <= btc_tte <= TTE_MAX):
            continue
        if TTE_BLACKOUT[0] <= btc_tte <= TTE_BLACKOUT[1]:
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
            signals[a] = {
                "z": z, "model_p": model_p, "mid": mid, "tte": tte,
                "spread": tick[f'{al}_spread'],
                "ob": tick[f'{al}_ob'],
                "pvs": pvs,
            }

        if len(signals) >= 3 and entry_tick is None:
            entry_tick = tick
            entry_signals = signals

    if entry_signals is None:
        continue

    # Consensus
    n_yes = sum(1 for s in entry_signals.values() if s["model_p"] > 0.5)
    n_no = sum(1 for s in entry_signals.values() if s["model_p"] < 0.5)
    n_valid = len(entry_signals)
    majority = max(n_yes, n_no)
    agreement = majority / n_valid
    consensus_side = "YES" if n_yes > n_no else ("NO" if n_no > n_yes else None)
    avg_z = sum(s["z"] for s in entry_signals.values()) / n_valid

    if consensus_side is None or agreement < 0.74:
        continue

    # === FACTOR 1: Mid volatility (how much did mids move before entry?) ===
    mid_vols = {}
    for a in ASSETS:
        hist = mid_history[a]
        if len(hist) >= 5:
            # Standard deviation of mid changes
            changes = [hist[i] - hist[i-1] for i in range(1, len(hist))]
            mid_vols[a] = statistics.stdev(changes) if len(changes) > 1 else 0
        else:
            mid_vols[a] = 0
    avg_mid_vol = sum(mid_vols.values()) / max(len(mid_vols), 1)

    # === FACTOR 2: Average spread (market liquidity) ===
    spreads = [s.get("spread") or 0 for s in entry_signals.values() if s.get("spread")]
    avg_spread = sum(spreads) / len(spreads) if spreads else 0

    # === FACTOR 3: Mid dispersion (how different are the 4 mids?) ===
    mids = [s["mid"] for s in entry_signals.values()]
    mid_dispersion = statistics.stdev(mids) if len(mids) > 1 else 0

    # === FACTOR 4: Average |pvs| (price momentum strength) ===
    pvs_values = [abs(s["pvs"]) for s in entry_signals.values()]
    avg_pvs = sum(pvs_values) / len(pvs_values) if pvs_values else 0

    # === FACTOR 5: OB imbalance consistency ===
    obs = [s.get("ob") or 0 for s in entry_signals.values()]
    avg_ob = sum(obs) / len(obs) if obs else 0
    # Are OB imbalances aligned with consensus?
    if consensus_side == "YES":
        ob_aligned = sum(1 for o in obs if o > 0) / len(obs)  # positive OB = more YES bids
    else:
        ob_aligned = sum(1 for o in obs if o < 0) / len(obs)  # negative OB = more NO bids

    # === Simulate trades ===
    period_pnl = 0
    n_trades = 0
    n_wins = 0
    for a, sig in entry_signals.items():
        raw_side = "YES" if sig["model_p"] > 0.5 else "NO"
        if raw_side != consensus_side:
            continue
        mid = sig["mid"]
        cp = mid if consensus_side == "YES" else (1 - mid)
        contracts = 2 if cp <= 0.35 else 1
        fee = maker_fee(contracts, mid)
        settled_yes = settle.get(a) == "YES"
        if consensus_side == "YES":
            pnl = contracts * (1 - mid) - fee if settled_yes else -(contracts * mid + fee)
        else:
            pnl = contracts * mid - fee if not settled_yes else -(contracts * (1 - mid) + fee)
        period_pnl += pnl
        n_trades += 1
        if pnl > 0:
            n_wins += 1

    period_features.append({
        "period": period,
        "consensus": f"{n_yes}Y{n_no}N",
        "avg_z": avg_z,
        "agreement": agreement,
        "avg_mid_vol": avg_mid_vol,
        "avg_spread": avg_spread,
        "mid_dispersion": mid_dispersion,
        "avg_pvs": avg_pvs,
        "avg_ob": avg_ob,
        "ob_aligned": ob_aligned,
        "pnl": period_pnl,
        "n_trades": n_trades,
        "n_wins": n_wins,
        "won": period_pnl > 0,
    })


# ============================================================
# Analysis
# ============================================================
total = len(period_features)
wins = sum(1 for p in period_features if p["won"])
print("=" * 100)
print(f"  因子分析: {total} 个周期, {wins} 赢 ({wins/total*100:.0f}%)")
print("=" * 100)

# Helper to analyze a factor
def analyze_factor(name, key, buckets):
    print(f"\n  --- {name} ---")
    print(f"  {'Bucket':<25} {'Periods':>8} {'Win':>5} {'WR':>6} {'PnL':>8} {'AvgPnL':>8}")
    print(f"  {'-'*60}")
    for bname, lo, hi in buckets:
        bucket = [p for p in period_features if lo <= p[key] < hi]
        if not bucket:
            print(f"  {bname:<25} {'--':>8}")
            continue
        bw = sum(1 for p in bucket if p["won"])
        bpnl = sum(p["pnl"] for p in bucket)
        bwr = bw / len(bucket) * 100
        avg_pnl = bpnl / len(bucket)
        marker = " <<<" if bwr >= 80 else ""
        print(f"  {bname:<25} {len(bucket):>8} {bw:>5} {bwr:>5.0f}% ${bpnl:>+7.2f} ${avg_pnl:>+7.3f}{marker}")


# Factor 1: Mid volatility
vals = sorted(p["avg_mid_vol"] for p in period_features)
if vals:
    q25 = vals[len(vals)//4]
    q50 = vals[len(vals)//2]
    q75 = vals[3*len(vals)//4]
    analyze_factor("Mid波动率 (入场前mid变动幅度)", "avg_mid_vol", [
        (f"低波动 (< {q25:.4f})", 0, q25),
        (f"中低 ({q25:.4f}-{q50:.4f})", q25, q50),
        (f"中高 ({q50:.4f}-{q75:.4f})", q50, q75),
        (f"高波动 (>= {q75:.4f})", q75, 999),
    ])

# Factor 2: Spread
analyze_factor("市场Spread (流动性)", "avg_spread", [
    ("tight (<= 0.03)", 0, 0.031),
    ("normal (0.03-0.06)", 0.031, 0.061),
    ("wide (0.06-0.10)", 0.061, 0.101),
    ("very wide (> 0.10)", 0.101, 999),
])

# Factor 3: Mid dispersion
analyze_factor("Mid分散度 (4资产mid差异)", "mid_dispersion", [
    ("low (< 0.05)", 0, 0.05),
    ("medium (0.05-0.10)", 0.05, 0.10),
    ("high (0.10-0.15)", 0.10, 0.15),
    ("very high (>= 0.15)", 0.15, 999),
])

# Factor 4: PVS strength
analyze_factor("|PVS| 动量强度", "avg_pvs", [
    ("weak (< 0.002)", 0, 0.002),
    ("moderate (0.002-0.005)", 0.002, 0.005),
    ("strong (0.005-0.010)", 0.005, 0.010),
    ("very strong (>= 0.010)", 0.010, 999),
])

# Factor 5: OB alignment
analyze_factor("OB Imbalance 与共识一致性", "ob_aligned", [
    ("0% aligned", -0.01, 0.01),
    ("25% aligned", 0.01, 0.30),
    ("50% aligned", 0.30, 0.60),
    ("75%+ aligned", 0.60, 1.01),
])

# Factor 6: avg|z|
analyze_factor("avg|z| 信号强度", "avg_z", [
    ("|z| < 0.3", -0.30, 0.30),
    ("|z| 0.3-0.5 (mod-)", -0.50, -0.30),
    ("|z| 0.3-0.5 (mod+)", 0.30, 0.50),
    ("|z| 0.5-0.8 (NO)", -0.80, -0.50),
    ("|z| 0.5-0.8 (YES)", 0.50, 0.80),
    ("|z| 0.8-1.2 (NO)", -1.20, -0.80),
    ("|z| 0.8-1.2 (YES)", 0.80, 1.20),
    ("|z| >= 1.2 (any)", -999, -1.20),
])

# Rewrite avg_z analysis more simply
print(f"\n  --- avg|z| 简化 ---")
for zt in [0.0, 0.3, 0.5, 0.7, 0.9]:
    bucket = [p for p in period_features if abs(p["avg_z"]) >= zt]
    if bucket:
        bw = sum(1 for p in bucket if p["won"])
        bpnl = sum(p["pnl"] for p in bucket)
        print(f"  avg|z| >= {zt:.1f}: {len(bucket)} periods, {bw}W, WR={bw/len(bucket)*100:.0f}%, PnL=${bpnl:+.2f}")

# ============================================================
# Combo: best factor combination
# ============================================================
print(f"\n{'=' * 100}")
print("  多因子组合筛选")
print("=" * 100)

combos = [
    ("baseline (no filter)", lambda p: True),
    ("avg|z|>=0.5", lambda p: abs(p["avg_z"]) >= 0.5),
    ("avg|z|>=0.5 + low disp", lambda p: abs(p["avg_z"]) >= 0.5 and p["mid_dispersion"] < 0.10),
    ("avg|z|>=0.5 + OB aligned>=50%", lambda p: abs(p["avg_z"]) >= 0.5 and p["ob_aligned"] >= 0.50),
    ("avg|z|>=0.3 + OB aligned>=50%", lambda p: abs(p["avg_z"]) >= 0.3 and p["ob_aligned"] >= 0.50),
    ("avg|z|>=0.5 + tight spread", lambda p: abs(p["avg_z"]) >= 0.5 and p["avg_spread"] <= 0.06),
    ("avg|z|>=0.3 + OB>=50% + tight sprd", lambda p: abs(p["avg_z"]) >= 0.3 and p["ob_aligned"] >= 0.50 and p["avg_spread"] <= 0.06),
    ("4/4 only", lambda p: p["agreement"] >= 0.99),
    ("4/4 + avg|z|>=0.5", lambda p: p["agreement"] >= 0.99 and abs(p["avg_z"]) >= 0.5),
    ("4/4 + avg|z|>=0.5 + OB>=50%", lambda p: p["agreement"] >= 0.99 and abs(p["avg_z"]) >= 0.5 and p["ob_aligned"] >= 0.50),
    ("low vol + avg|z|>=0.5", lambda p: p["avg_mid_vol"] < (q50 if vals else 0.01) and abs(p["avg_z"]) >= 0.5),
]

print(f"\n  {'Combo':<40} {'Periods':>8} {'Win':>5} {'WR':>6} {'PnL':>8} {'AvgPnL':>8}")
print(f"  {'-'*75}")

for name, filt in combos:
    bucket = [p for p in period_features if filt(p)]
    if not bucket:
        print(f"  {name:<40} {'--':>8}")
        continue
    bw = sum(1 for p in bucket if p["won"])
    bpnl = sum(p["pnl"] for p in bucket)
    bwr = bw / len(bucket) * 100
    marker = " <<<" if bwr >= 80 and len(bucket) >= 3 else ""
    print(f"  {name:<40} {len(bucket):>8} {bw:>5} {bwr:>5.0f}% ${bpnl:>+7.2f} ${bpnl/len(bucket):>+7.3f}{marker}")

# ============================================================
# Raw data dump for each period
# ============================================================
print(f"\n{'=' * 100}")
print("  每周期详情")
print("=" * 100)
print(f"\n  {'Period':<22} {'Cons':>5} {'avgZ':>6} {'MidVol':>7} {'Spread':>7} {'Disp':>6} {'OB%':>5} {'PnL':>7} {'Result'}")
print(f"  {'-'*80}")
for p in period_features:
    res = "WIN" if p["won"] else "LOSS"
    print(f"  {p['period']:<22} {p['consensus']:>5} {p['avg_z']:>+.3f} {p['avg_mid_vol']:>.5f} "
          f"{p['avg_spread']:>.4f}  {p['mid_dispersion']:>.4f} {p['ob_aligned']:>.2f} "
          f"${p['pnl']:>+.2f}  {res}")

db.close()
