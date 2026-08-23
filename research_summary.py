#!/usr/bin/env python3
"""Descriptive-only completed-window summary; no profit or threshold optimization."""

import argparse
import collections
import json
import sqlite3
from datetime import datetime, timezone

from signals.export import checkpoint_rows
from signals.research import lagged_correlations


def pct(n, d): return 0 if not d else n / d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-db", default="kalshi_features_v3.db")
    args = parser.parse_args()
    connection = sqlite3.connect(f"file:{args.features_db}?mode=ro", uri=True)
    outcomes = connection.execute("SELECT market_ticker,asset,result,settlement_timestamp FROM contract_outcomes").fetchall()
    checkpoints = checkpoint_rows(connection)
    print(f"completed windows: {len(outcomes)}")
    for asset in ("BTC", "ETH", "SOL", "XRP", "DOGE"):
        rows = [row for row in outcomes if row[1] == asset]
        yes = sum(row[2] == "yes" for row in rows)
        print(f"{asset}: windows={len(rows)} UP={pct(yes,len(rows)):.1%} DOWN={pct(len(rows)-yes,len(rows)):.1%}")
    regimes = collections.Counter(); reset_regimes = collections.Counter()
    breadth = collections.Counter()
    quarter_volatility = collections.defaultdict(list)
    series = collections.defaultdict(list)
    for row in connection.execute("SELECT timestamp,asset,features_json,basket_json FROM feature_snapshots"):
        features, basket = json.loads(row[2]), json.loads(row[3])
        regimes[features.get("regime_label")] += 1
        reset_regimes[features.get("reset_regime_label")] += 1
        breadth[basket.get("breadth_direction")] += 1
        minute = datetime.fromtimestamp(row[0], timezone.utc).minute
        volatility = features.get("realized_volatility_300s")
        if volatility is not None: quarter_volatility[(minute // 15) * 15].append(volatility)
        velocity = features.get("prob_velocity_30s")
        if velocity is not None: series[row[1]].append((row[0], velocity))
    total_breadth = sum(breadth.values())
    sync5 = sum(v for k,v in breadth.items() if k in ('5/5 UP','5/5 DOWN'))
    sync4 = sum(v for k,v in breadth.items() if k in ('4/5 UP','4/5 DOWN'))
    print(f"5/5 synchronized snapshots: {sync5} ({pct(sync5,total_breadth):.1%})")
    print(f"4/5 synchronized snapshots: {sync4} ({pct(sync4,total_breadth):.1%})")
    print(f"regime counts: {dict(regimes)}")
    continuation = sum(v for k,v in reset_regimes.items() if k and k.startswith('RESET_'))
    reversal = sum(v for k,v in reset_regimes.items() if k and k.startswith('REVERSAL_'))
    classified_resets = continuation + reversal
    print(f"reset continuation candidates: {continuation} ({pct(continuation,classified_resets):.1%})")
    print(f"reset reversal candidates: {reversal} ({pct(reversal,classified_resets):.1%})")
    print("mean realized volatility by quarter-hour UTC minute: " + str({
        quarter: sum(values)/len(values) for quarter,values in sorted(quarter_volatility.items())}))
    for asset in ("ETH", "SOL", "XRP", "DOGE"):
        print(f"BTC->{asset} lag correlations: {lagged_correlations(series['BTC'],series[asset])}")
    for checkpoint in (30, 60, 120):
        rows = [row for row in checkpoints if row["checkpoint_seconds"] == checkpoint]
        counts = collections.Counter(row["basket"].get("breadth_direction") for row in rows if row["basket"])
        print(f"T+{checkpoint}s breadth: {dict(counts)}")
    connection.close()


if __name__ == "__main__": main()
