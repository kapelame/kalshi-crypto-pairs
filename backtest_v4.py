#!/usr/bin/env python3
"""
Backtest v4 strategy on historical tick data from kalshi_live_paper.db
Simulates: stale guard, contract price cap, TTE window, mid-pvs conflict, consensus edge
"""

import sqlite3
import math
from collections import defaultdict
from datetime import datetime, timezone

DB_FILE = "kalshi_live_paper.db"
ASSETS = ["BTC", "ETH", "SOL", "XRP"]
DEFAULT_VOL_15M = {"BTC": 0.15, "ETH": 0.20, "SOL": 0.35, "XRP": 0.30}

# ============================================================
# Config (mirrors v4 live)
# ============================================================
TTE_MIN = 120
TTE_MAX = 420
MAX_CONTRACT_PRICE = 0.60
MID_RANGE = (0.20, 0.80)
MIN_EDGE = 0.015
KELLY_FRACTION = 0.50
MAX_PER_TRADE_PCT = 0.06
MAX_EXPOSURE_PCT = 0.25
STALE_THRESHOLD = 60  # seconds
NET_EDGE_FLOOR = 0.01


def maker_fee(contracts, price):
    if contracts <= 0 or price <= 0 or price >= 1:
        return 0
    return math.ceil(0.0175 * contracts * price * (1 - price) * 100) / 100


def compute_signal(pvs, tte, mid, vol_15m):
    """Compute z-score and model_p"""
    if pvs is None or tte is None or mid is None or tte < 60:
        return None, None
    if not (0.10 <= mid <= 0.90):
        return None, None

    vol_per_sec = vol_15m / math.sqrt(900)
    expected_std = vol_per_sec * math.sqrt(max(tte, 1))
    z = pvs / expected_std if expected_std > 0.001 else 0
    model_p = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return z, model_p


def run_backtest(config_name, tte_range, max_cp, stale_guard, conflict_guard):
    """Run one backtest variant"""
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

    bankroll = 100.0
    total_trades = 0
    total_wins = 0
    results = []
    all_trades = []

    # Track price staleness across periods
    last_prices = {a: None for a in ASSETS}
    last_change_ts = {a: 0.0 for a in ASSETS}

    for period in sorted(periods_data.keys()):
        ticks = periods_data[period]
        settle = settlements.get(period, {})
        if len(settle) < 4:
            continue  # incomplete settlement data

        # Reset staleness tracking at period boundary
        last_prices = {a: None for a in ASSETS}
        last_change_ts = {a: 0.0 for a in ASSETS}

        positions = {}
        exposure = 0

        for tick_idx, tick in enumerate(ticks):
            ts = tick['ts_unix']

            # Update staleness tracking
            for a in ASSETS:
                col = f"{a.lower()}_price"
                price = tick[col]
                if price and price > 0:
                    if last_prices[a] is None or abs(price - last_prices[a]) > 0.001:
                        last_prices[a] = price
                        last_change_ts[a] = ts

            # Already have positions? Skip trading
            if positions:
                continue

            # Check TTE
            btc_tte = tick['btc_tte']
            if btc_tte is None:
                continue
            if not (tte_range[0] <= btc_tte <= tte_range[1]):
                continue

            # Compute signals for all assets
            signals = {}
            for a in ASSETS:
                al = a.lower()
                mid = tick[f'{al}_mid']
                pvs = tick[f'{al}_pvs']
                tte = tick[f'{al}_tte']
                price = tick[f'{al}_price']
                vol = DEFAULT_VOL_15M.get(a, 0.20)

                if mid is None or pvs is None or tte is None:
                    continue

                # Stale check
                if stale_guard:
                    stale_sec = ts - last_change_ts[a] if last_change_ts[a] > 0 else 999
                    if stale_sec > STALE_THRESHOLD:
                        continue

                z, model_p = compute_signal(pvs, tte, mid, vol)
                if z is None:
                    continue

                # Mid-pvs conflict check
                if conflict_guard:
                    pvs_yes = model_p > 0.5
                    mid_yes = mid > 0.55
                    mid_no = mid < 0.45
                    if (pvs_yes and mid_no) or (not pvs_yes and mid_yes):
                        continue

                raw_side = "YES" if model_p > 0.5 else "NO"
                signals[a] = {"z": z, "model_p": model_p, "mid": mid,
                              "raw_side": raw_side, "tte": tte}

            if len(signals) < 2:
                continue

            # Consensus
            n_yes = sum(1 for s in signals.values() if s["raw_side"] == "YES")
            n_no = sum(1 for s in signals.values() if s["raw_side"] == "NO")
            n_valid = len(signals)
            majority = max(n_yes, n_no)
            agreement = majority / n_valid if n_valid >= 3 else 0.5
            consensus_side = "YES" if n_yes > n_no else ("NO" if n_no > n_yes else None)

            # Only trade with consensus
            if consensus_side is None or agreement < 0.74:
                continue

            # Enter positions
            for a, sig in signals.items():
                if sig["raw_side"] != consensus_side:
                    continue

                mid = sig["mid"]
                side = consensus_side

                # Contract price check
                cp = mid if side == "YES" else (1 - mid)
                if max_cp and cp > max_cp:
                    continue

                # Mid range check
                if not (MID_RANGE[0] <= mid <= MID_RANGE[1]):
                    continue

                # Compute edge
                z_abs = abs(sig["z"])
                if agreement >= 0.99:
                    edge = 0.025 + min(0.03, 0.015 * z_abs)
                else:
                    edge = 0.015 + min(0.02, 0.01 * z_abs)

                if edge < MIN_EDGE:
                    continue

                # Kelly sizing
                odds = (1 - cp) / cp if cp > 0 else 0
                kelly = edge / odds if odds > 0 else 0
                raw_usd = bankroll * KELLY_FRACTION * kelly
                max_usd = bankroll * MAX_PER_TRADE_PCT
                remaining = bankroll * MAX_EXPOSURE_PCT - exposure
                size_usd = max(min(raw_usd, max_usd, remaining), 0)
                if size_usd < cp:
                    continue
                contracts = int(size_usd / cp)
                if contracts <= 0:
                    continue
                fee = maker_fee(contracts, mid)
                net_edge = edge - fee / contracts if contracts > 0 else 0
                if net_edge < NET_EDGE_FLOOR:
                    continue

                cost = contracts * cp + fee
                positions[a] = {
                    "side": side, "mid": mid, "contracts": contracts,
                    "cost": cost, "fee": fee, "cp": cp, "edge": edge,
                    "z": sig["z"], "consensus": f"{n_yes}Y{n_no}N"
                }
                exposure += cost

            break  # Only enter once per period

        # Settle all positions
        period_pnl = 0
        for a, pos in positions.items():
            settled_yes = settle.get(a) == "YES"
            side = pos["side"]
            contracts = pos["contracts"]
            entry = pos["mid"]
            fee = pos["fee"]

            if side == "YES":
                pnl = contracts * (1 - entry) - fee if settled_yes else -(contracts * entry + fee)
            else:
                pnl = contracts * entry - fee if not settled_yes else -(contracts * (1 - entry) + fee)

            bankroll += pnl
            period_pnl += pnl
            total_trades += 1
            won = pnl > 0
            if won:
                total_wins += 1

            all_trades.append({
                "period": period, "asset": a, "side": side,
                "contracts": contracts, "mid": entry, "cp": pos["cp"],
                "fee": fee, "edge": pos["edge"],
                "settled": "YES" if settled_yes else "NO",
                "pnl": round(pnl, 4), "z": pos["z"],
                "consensus": pos["consensus"],
            })

        n_yes_settle = sum(1 for v in settle.values() if v == "YES")
        n_no_settle = sum(1 for v in settle.values() if v == "NO")

        results.append({
            "period": period,
            "settlement": f"{n_yes_settle}Y{n_no_settle}N",
            "n_trades": len(positions),
            "pnl": round(period_pnl, 4),
            "bankroll": round(bankroll, 2),
        })

    db.close()
    return config_name, bankroll, total_trades, total_wins, results, all_trades


# ============================================================
# Run all variants
# ============================================================

variants = [
    # name, tte_range, max_contract_price, stale_guard, conflict_guard
    ("v3 (no guard)",      (120, 780), None,  False, False),
    ("+ TTE 2-7min",       (120, 420), None,  False, False),
    ("+ contract cap $0.60", (120, 420), 0.60, False, False),
    ("+ stale guard 60s",  (120, 420), 0.60, True,  False),
    ("v4 FULL (all guards)", (120, 420), 0.60, True,  True),
]

print("=" * 95)
print("  BACKTEST: v4 STRATEGY ON 15 HISTORICAL PERIODS")
print("=" * 95)

all_results = []
for name, tte, mcp, sg, cg in variants:
    cname, bank, trades, wins, results, trade_list = run_backtest(name, tte, mcp, sg, cg)
    wr = wins / trades * 100 if trades > 0 else 0
    pnl = bank - 100
    max_loss = min((r["pnl"] for r in results), default=0)
    max_win = max((r["pnl"] for r in results), default=0)
    active_periods = sum(1 for r in results if r["n_trades"] > 0)
    all_results.append((cname, bank, trades, wins, wr, pnl, max_loss, max_win,
                        active_periods, results, trade_list))

# Summary table
print(f"\n{'Strategy':<25} {'PnL':>8} {'Bank':>8} {'Trades':>7} {'WR':>6} {'MaxLoss':>8} {'MaxWin':>8} {'Active':>7}")
print("-" * 85)
for name, bank, trades, wins, wr, pnl, ml, mw, ap, _, _ in all_results:
    print(f"  {name:<23} {pnl:>+8.2f} ${bank:>7.2f} {trades:>5}   {wr:>5.1f}% {ml:>+8.2f} {mw:>+8.2f} {ap:>4}/15")

# Detailed period-by-period comparison for v3 vs v4 FULL
print(f"\n{'=' * 95}")
print(f"  PERIOD-BY-PERIOD: v3 (no guard) vs v4 FULL")
print(f"{'=' * 95}")

v3_results = all_results[0][9]
v4_results = all_results[-1][9]

print(f"\n  {'Period':<20} {'Settle':>6} | {'v3 trades':>9} {'v3 PnL':>8} {'v3 Bank':>8} | {'v4 trades':>9} {'v4 PnL':>8} {'v4 Bank':>8} | {'Delta':>8}")
print(f"  {'-' * 90}")

for i in range(len(v3_results)):
    v3 = v3_results[i]
    v4 = v4_results[i] if i < len(v4_results) else {"n_trades": 0, "pnl": 0, "bankroll": 100}
    delta = v4["pnl"] - v3["pnl"]
    marker = " <<<" if abs(delta) > 1 else ""
    print(f"  {v3['period']:<20} {v3['settlement']:>6} | {v3['n_trades']:>5}     {v3['pnl']:>+8.2f} ${v3['bankroll']:>7.2f} | "
          f"{v4['n_trades']:>5}     {v4['pnl']:>+8.2f} ${v4['bankroll']:>7.2f} | {delta:>+8.2f}{marker}")

# Show v4 FULL trade details
print(f"\n{'=' * 95}")
print(f"  v4 FULL TRADE DETAILS")
print(f"{'=' * 95}")

v4_trades = all_results[-1][10]
if v4_trades:
    print(f"\n  {'Period':<20} {'Asset':<5} {'Side':<4} {'Ctrs':>5} {'Mid':>6} {'CP':>6} {'Edge':>6} {'Z':>6} {'Cons':>6} {'Settle':>7} {'PnL':>8}")
    print(f"  {'-' * 90}")
    for t in v4_trades:
        res = "WIN" if t["pnl"] > 0 else "LOSS"
        print(f"  {t['period']:<20} {t['asset']:<5} {t['side']:<4} {t['contracts']:>5} "
              f"${t['mid']:>.2f}  ${t['cp']:>.2f}  {t['edge']:>.3f}  {t['z']:>+.2f}  "
              f"{t['consensus']:>5}  {t['settled']:>4}={res:<4} {t['pnl']:>+8.2f}")
else:
    print("  (no trades)")

# Show all variant trade details for comparison
print(f"\n{'=' * 95}")
print(f"  TRADE COUNT BY PERIOD (ALL VARIANTS)")
print(f"{'=' * 95}")
print(f"\n  {'Period':<20} {'Settle':>6}", end="")
for name, _, _, _, _, _, _, _, _, _, _ in all_results:
    short = name[:15]
    print(f" | {short:>15}", end="")
print()
print(f"  {'-' * (28 + 18 * len(all_results))}")

for i in range(len(v3_results)):
    period = v3_results[i]["period"]
    settle = v3_results[i]["settlement"]
    print(f"  {period:<20} {settle:>6}", end="")
    for _, _, _, _, _, _, _, _, _, results, _ in all_results:
        if i < len(results):
            r = results[i]
            if r["n_trades"] > 0:
                print(f" | {r['n_trades']}t {r['pnl']:>+7.2f}", end="")
            else:
                print(f" |       --    ", end="")
        else:
            print(f" |       --    ", end="")
    print()
