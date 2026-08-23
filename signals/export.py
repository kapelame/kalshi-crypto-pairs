"""Leakage-safe checkpoint selection and post-freeze outcome joining."""

import csv
import json


def checkpoint_rows(connection, checkpoints=(30, 60, 120, 180, 300, 600)):
    markets = connection.execute(
        "SELECT DISTINCT market_ticker,asset,contract_open FROM feature_snapshots "
        "WHERE market_ticker IS NOT NULL AND contract_open IS NOT NULL").fetchall()
    rows = []
    for ticker, asset, opened in markets:
        for checkpoint in checkpoints:
            frozen_at = opened + checkpoint
            row = connection.execute(
                "SELECT timestamp,features_json,basket_json,raw_watermark_event_id "
                "FROM feature_snapshots WHERE market_ticker=? AND asset=? AND timestamp>=? "
                "AND json_extract(features_json,'$.midpoint_up_probability') IS NOT NULL "
                "ORDER BY timestamp LIMIT 1", (ticker, asset, frozen_at)).fetchone()
            if row is None:
                continue
            outcome = connection.execute(
                "SELECT result,settlement_timestamp FROM contract_outcomes "
                "WHERE market_ticker=? AND settlement_timestamp>=?", (ticker, row[0])).fetchone()
            rows.append({"market_ticker": ticker, "asset": asset, "checkpoint_seconds": checkpoint,
                         "snapshot_timestamp": row[0], "raw_watermark_event_id": row[3],
                         "features": json.loads(row[1]), "basket": json.loads(row[2]),
                         "outcome": None if outcome is None else outcome[0],
                         "settlement_timestamp": None if outcome is None else outcome[1]})
        final = connection.execute(
            "SELECT result,settlement_timestamp FROM contract_outcomes WHERE market_ticker=?",
            (ticker,)).fetchone()
        if final:
            frozen = connection.execute(
                "SELECT timestamp,features_json,basket_json,raw_watermark_event_id "
                "FROM feature_snapshots WHERE market_ticker=? AND asset=? AND timestamp<=? "
                "ORDER BY timestamp DESC LIMIT 1", (ticker, asset, final[1])).fetchone()
            rows.append({"market_ticker": ticker, "asset": asset,
                         "checkpoint_seconds": "settlement", "snapshot_timestamp": final[1],
                         "raw_watermark_event_id": None if frozen is None else frozen[3],
                         "features": None if frozen is None else json.loads(frozen[1]),
                         "basket": None if frozen is None else json.loads(frozen[2]),
                         "outcome": final[0], "settlement_timestamp": final[1]})
    return rows


def export_csv(connection, path):
    rows = checkpoint_rows(connection)
    fields = ("market_ticker", "asset", "checkpoint_seconds", "snapshot_timestamp",
              "raw_watermark_event_id", "features", "basket", "outcome", "settlement_timestamp")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            row = dict(row)
            for key in ("features", "basket"):
                row[key] = None if row[key] is None else json.dumps(row[key], sort_keys=True)
            writer.writerow(row)
    return len(rows)
