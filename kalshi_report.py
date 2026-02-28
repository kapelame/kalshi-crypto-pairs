#!/usr/bin/env python3
"""
Kalshi Trading Report - sends summary email every 4 hours via AWS SES
"""

import sqlite3
import json
import os
import boto3
from datetime import datetime, timezone, timedelta

DB_FILE = os.environ.get("TRADER_DB", "/home/ec2-user/kalshi_live_trader.db")
LOG_FILE = os.environ.get("TRADER_LOG", "/home/ec2-user/kalshi_live_trader.log")
RECIPIENT = os.environ.get("REPORT_EMAIL", "you@example.com")
SENDER = os.environ.get("REPORT_SENDER", RECIPIENT)  # SES sandbox: sender must also be verified
REGION = os.environ.get("AWS_REGION", "us-east-1")


def get_report():
    db = sqlite3.connect(DB_FILE)
    db.row_factory = sqlite3.Row

    now = datetime.now(timezone.utc)
    h4_ago = (now - timedelta(hours=4)).timestamp()
    h24_ago = (now - timedelta(hours=24)).timestamp()

    # --- All-time stats ---
    all_trades = db.execute("SELECT * FROM trades WHERE settled IS NOT NULL").fetchall()
    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if (t['pnl'] or 0) > 0)
    total_pnl = sum(t['pnl'] or 0 for t in all_trades)
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    # --- Last 4 hours ---
    recent = db.execute(
        "SELECT * FROM trades WHERE ts_unix >= ? ORDER BY ts_unix", [h4_ago]
    ).fetchall()
    recent_settled = [t for t in recent if t['settled'] is not None]
    recent_pending = [t for t in recent if t['settled'] is None]
    recent_pnl = sum(t['pnl'] or 0 for t in recent_settled)
    recent_wins = sum(1 for t in recent_settled if (t['pnl'] or 0) > 0)
    recent_wr = (recent_wins / len(recent_settled) * 100) if recent_settled else 0

    # --- Last 24 hours ---
    day_trades = db.execute(
        "SELECT * FROM trades WHERE ts_unix >= ? AND settled IS NOT NULL ORDER BY ts_unix",
        [h24_ago]
    ).fetchall()
    day_pnl = sum(t['pnl'] or 0 for t in day_trades)
    day_wins = sum(1 for t in day_trades if (t['pnl'] or 0) > 0)
    day_wr = (day_wins / len(day_trades) * 100) if day_trades else 0

    # --- Current balance ---
    latest_balance = db.execute(
        "SELECT bankroll_after FROM trades WHERE bankroll_after IS NOT NULL "
        "ORDER BY ts_unix DESC LIMIT 1"
    ).fetchone()
    balance = latest_balance['bankroll_after'] if latest_balance else 0

    # --- Settlements ---
    recent_settlements = db.execute(
        "SELECT * FROM settlements WHERE ts_unix >= ? ORDER BY ts_unix DESC",
        [h4_ago]
    ).fetchall()

    # --- Last log lines ---
    try:
        with open(LOG_FILE) as f:
            lines = f.readlines()
            last_lines = lines[-15:] if len(lines) >= 15 else lines
    except Exception:
        last_lines = ["(log unavailable)"]

    # --- Build report ---
    report = []
    report.append(f"{'='*50}")
    report.append(f"  KALSHI TRADER REPORT")
    report.append(f"  {now.strftime('%Y-%m-%d %H:%M UTC')}")
    report.append(f"{'='*50}")
    report.append("")
    report.append(f"  BALANCE: ${balance:.2f}")
    report.append("")

    report.append(f"--- Last 4 Hours ---")
    report.append(f"  Trades: {len(recent_settled)} settled, {len(recent_pending)} pending")
    report.append(f"  PnL: ${recent_pnl:+.2f}")
    report.append(f"  Win Rate: {recent_wr:.0f}% ({recent_wins}/{len(recent_settled)})")
    report.append("")

    if recent_settled:
        report.append(f"  {'Asset':<6} {'Side':<4} {'Entry':>6} {'Settled':>8} {'PnL':>8} {'Z':>6} {'Cons':>6} {'AvgZ':>6}")
        report.append(f"  {'-'*55}")
        for t in recent_settled:
            res = "WIN" if (t['pnl'] or 0) > 0 else "LOSS"
            consensus = t['consensus'] if t['consensus'] else '?'
            avg_z = t['avg_z'] if t['avg_z'] else 0
            z = t['z_score'] if t['z_score'] else 0
            report.append(
                f"  {t['asset']:<6} {t['side']:<4} ${t['entry_price']:.2f}"
                f"  {t['settled']:>4}={res:<4} ${t['pnl']:>+.2f}"
                f"  {z:>+.2f}  {consensus:>5}  {avg_z:>+.2f}"
            )
    report.append("")

    report.append(f"--- Last 24 Hours ---")
    report.append(f"  Trades: {len(day_trades)}")
    report.append(f"  PnL: ${day_pnl:+.2f}")
    report.append(f"  Win Rate: {day_wr:.0f}% ({day_wins}/{len(day_trades)})")
    report.append("")

    report.append(f"--- All Time ---")
    report.append(f"  Trades: {total_trades}")
    report.append(f"  PnL: ${total_pnl:+.2f}")
    report.append(f"  Win Rate: {win_rate:.0f}% ({total_wins}/{total_trades})")
    report.append("")

    report.append(f"--- Recent Log ---")
    for line in last_lines:
        report.append(f"  {line.rstrip()}")

    report.append("")
    report.append(f"--- Config ---")
    report.append(f"  TTE: (120, 300) | 4/4: avg|z|>0.5 | 3/4: avg|z|>1.0")
    report.append(f"  1 contract/asset | trailing stop: bank-$5")

    db.close()
    return "\n".join(report), balance, recent_pnl


def send_email(subject, body):
    ses = boto3.client('ses', region_name=REGION)
    ses.send_email(
        Source=SENDER,
        Destination={'ToAddresses': [RECIPIENT]},
        Message={
            'Subject': {'Data': subject, 'Charset': 'UTF-8'},
            'Body': {'Text': {'Data': body, 'Charset': 'UTF-8'}}
        }
    )


if __name__ == "__main__":
    report_text, balance, recent_pnl = get_report()
    pnl_str = f"+${recent_pnl:.2f}" if recent_pnl >= 0 else f"-${abs(recent_pnl):.2f}"
    subject = f"Kalshi Report | Balance: ${balance:.2f} | 4h PnL: {pnl_str}"
    send_email(subject, report_text)
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Report sent to {RECIPIENT}")
