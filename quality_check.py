#!/usr/bin/env python3
"""
Kalshi Data Quality Check v2 (for collector v3 data)
=====================================================
检查 v3 采集数据质量, 决定是否可以训练 ML

用法: python3 kalshi_quality_check_v2.py [csv_path]
"""

import pandas as pd
import numpy as np
import sqlite3
import sys
import os
import requests

CSV = "kalshi_v3_features.csv"
DB = "kalshi_v3.db"

# Allow CLI override
if len(sys.argv) > 1:
    CSV = sys.argv[1]

# Auto-detect path
if not os.path.exists(CSV) and os.path.exists(f"/content/{CSV}"):
    CSV = f"/content/{CSV}"

# If CSV doesn't exist, try to export from DB
if not os.path.exists(CSV) and os.path.exists(DB):
    print(f"  CSV not found, exporting from {DB}...")
    import csv as csvmod
    conn = sqlite3.connect(DB)
    cursor = conn.execute("SELECT * FROM features ORDER BY ts_unix")
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csvmod.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    conn.close()
    print(f"  Exported {len(rows)} rows to {CSV}")

df = pd.read_csv(CSV)

ASSETS = ["btc", "eth", "sol", "xrp"]

print(f"\n{'=' * 70}")
print(f"  DATA QUALITY REPORT (v2 - for collector v3)")
print(f"{'=' * 70}")
print(f"  rows:     {len(df)}")
print(f"  columns:  {len(df.columns)}")

# === Time span ===
span = df["ts_unix"].iloc[-1] - df["ts_unix"].iloc[0]
print(f"  timespan: {span:.0f}s = {span/60:.1f}min = {span/3600:.2f}h")
print(f"  15m cycles: {span/900:.1f}")

# === Gap detection ===
diffs = df["ts_unix"].diff().dropna()
gaps = diffs[diffs > 5]
total_gap = gaps.sum() if len(gaps) > 0 else 0
gap_pct = total_gap / span * 100 if span > 0 else 0

print(f"\n  GAPS:")
print(f"    gaps > 5s:  {len(gaps)}")
print(f"    total gap:  {total_gap:.0f}s ({gap_pct:.1f}%)")
if len(gaps) > 0:
    for idx in list(gaps.index[:5]):
        print(f"    row {idx}: {diffs[idx]:.0f}s")
    if len(gaps) > 5:
        print(f"    ... and {len(gaps)-5} more")

# === Data completeness ===
print(f"\n  DATA COMPLETENESS:")
for a in ASSETS:
    mid_ok = df[f"{a}_mid"].notna().sum()
    ob_ok = df[f"{a}_ob_imbalance"].notna().sum()
    rp_ok = df[f"{a}_real_price"].notna().sum()
    pvs_ok = df[f"{a}_price_vs_strike_pct"].notna().sum()
    fs_ok = df[f"{a}_floor_strike"].notna().sum()
    tte_ok = df[f"{a}_time_to_expiry"].notna().sum()
    err = df[f"{a}_error"].notna().sum()
    total = len(df)
    print(f"    {a.upper()}: mid={mid_ok}/{total} ob={ob_ok}/{total} "
          f"price={rp_ok}/{total} strike={fs_ok}/{total} "
          f"tte={tte_ok}/{total} errors={err}")

# === Time-to-expiry distribution ===
print(f"\n  TIME-TO-EXPIRY (BTC):")
tte = df["btc_time_to_expiry"].dropna()
if len(tte) > 0:
    print(f"    min={tte.min():.0f}s  max={tte.max():.0f}s  "
          f"mean={tte.mean():.0f}s  std={tte.std():.0f}s")
    # Check coverage: do we see the full 0-900s range?
    bins = [0, 60, 180, 300, 450, 600, 750, 900]
    hist, _ = np.histogram(tte, bins=bins)
    bin_labels = ["0-1m", "1-3m", "3-5m", "5-7.5m", "7.5-10m", "10-12.5m", "12.5-15m"]
    print(f"    distribution:")
    for label, count in zip(bin_labels, hist):
        bar = "#" * min(count // max(1, len(df) // 200), 40)
        print(f"      {label:>10}: {count:>5} {bar}")

# === Market transitions ===
print(f"\n  MARKET TRANSITIONS:")
for a in ASSETS:
    tickers = df[f"{a}_ticker"].dropna().unique()
    print(f"    {a.upper()}: {len(tickers)} unique tickers")
    for t in tickers[:8]:
        count = (df[f"{a}_ticker"] == t).sum()
        print(f"      {t}: {count} rows")

# === Orderbook imbalance ===
print(f"\n  ORDERBOOK IMBALANCE:")
for a in ASSETS:
    obi = df[f"{a}_ob_imbalance"].dropna()
    if len(obi) > 0:
        print(f"    {a.upper()}: mean={obi.mean():+.3f}  std={obi.std():.3f}  "
              f"min={obi.min():+.3f}  max={obi.max():+.3f}")

# === Price vs Strike ===
print(f"\n  PRICE VS STRIKE (%):")
for a in ASSETS:
    pvs = df[f"{a}_price_vs_strike_pct"].dropna()
    if len(pvs) > 0:
        print(f"    {a.upper()}: mean={pvs.mean():+.4f}%  std={pvs.std():.4f}%  "
              f"range=[{pvs.min():+.4f}%, {pvs.max():+.4f}%]")

# === Mid probability stats ===
print(f"\n  MID PROBABILITY:")
for a in ASSETS:
    mid = df[f"{a}_mid"].dropna()
    if len(mid) > 0:
        print(f"    {a.upper()}: mean={mid.mean():.3f}  std={mid.std():.3f}  "
              f"range=[{mid.min():.3f}, {mid.max():.3f}]")

# === Momentum availability ===
print(f"\n  MOMENTUM DATA:")
for a in ASSETS:
    for w in [5, 15, 30]:
        m = df[f"{a}_mom_{w}s"].notna().sum()
        pct = m / len(df) * 100
        if w == 5:
            print(f"    {a.upper()}: 5s={m}({pct:.0f}%)  ", end="")
        elif w == 15:
            print(f"15s={m}({pct:.0f}%)  ", end="")
        else:
            print(f"30s={m}({pct:.0f}%)")

# === Settlement lookup (try to get outcomes for ML target) ===
print(f"\n  SETTLEMENT OUTCOMES:")
all_tickers = set()
for a in ASSETS:
    tickers = df[f"{a}_ticker"].dropna().unique()
    all_tickers.update(tickers)

settled_count = 0
yes_count = 0
no_count = 0
try:
    for ticker in all_tickers:
        url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            m = r.json().get("market", {})
            result = m.get("result", "")
            if result in ("yes", "no"):
                settled_count += 1
                if result == "yes":
                    yes_count += 1
                else:
                    no_count += 1
    print(f"    total unique tickers: {len(all_tickers)}")
    print(f"    settled: {settled_count}  (YES={yes_count}, NO={no_count})")
    if settled_count > 0:
        print(f"    YES rate: {yes_count/settled_count:.0%}")
    unsettled = len(all_tickers) - settled_count
    if unsettled > 0:
        print(f"    unsettled: {unsettled} (still active or pending)")
except Exception as e:
    print(f"    lookup failed: {e}")

# === VERDICT ===
print(f"\n{'=' * 70}")
print(f"  VERDICT")
print(f"{'=' * 70}")

issues = []
warnings = []

if len(df) < 1500:
    issues.append(f"rows too few: {len(df)} < 1500 minimum")
if span < 3600:
    issues.append(f"timespan too short: {span/60:.0f}min < 60min minimum")
if gap_pct > 10:
    issues.append(f"gap too high: {gap_pct:.1f}% > 10% limit")
elif gap_pct > 5:
    warnings.append(f"gap elevated: {gap_pct:.1f}%")

# Check data completeness
for a in ASSETS:
    mid_pct = df[f"{a}_mid"].notna().sum() / len(df) * 100
    if mid_pct < 90:
        issues.append(f"{a.upper()} mid data only {mid_pct:.0f}% complete")

rp_pct = df["btc_real_price"].notna().sum() / len(df) * 100
if rp_pct < 50:
    warnings.append(f"Real price data only {rp_pct:.0f}% (Coinbase may be intermittent)")

# Check TTE coverage
if len(tte) > 0 and tte.max() < 600:
    warnings.append(f"TTE max only {tte.max():.0f}s, may be missing early-period data")

# Check transitions
total_tickers = sum(len(df[f"{a}_ticker"].dropna().unique()) for a in ASSETS)
if total_tickers < 8:
    warnings.append(f"Only {total_tickers} unique tickers, "
                    f"need more 15-min cycles for ML training")

if settled_count < 4:
    warnings.append(f"Only {settled_count} settled markets, "
                    f"ML target labels may be limited")

if not issues:
    print(f"  [OK] Data ready for kalshi_ml_v3.py!")
    if warnings:
        print(f"  Notes:")
        for w in warnings:
            print(f"    - {w}")
    print(f"\n  Next: python3 kalshi_ml_v3.py")
else:
    print(f"  [FAIL] Data not ready:")
    for i in issues:
        print(f"    X {i}")
    if warnings:
        for w in warnings:
            print(f"    ! {w}")
    print(f"\n  Action: continue collecting data")
