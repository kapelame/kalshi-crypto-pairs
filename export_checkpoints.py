#!/usr/bin/env python3
"""Export frozen checkpoint features, joining outcomes only after selection."""

import argparse
import sqlite3

from signals.export import export_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-db", default="kalshi_features_v3.db")
    parser.add_argument("--output", default="phase3_checkpoints.csv")
    args = parser.parse_args()
    connection = sqlite3.connect(f"file:{args.features_db}?mode=ro", uri=True)
    try: count = export_csv(connection, args.output)
    finally: connection.close()
    print(f"Exported {count} immutable checkpoint/outcome rows to {args.output}")


if __name__ == "__main__": main()
