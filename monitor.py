#!/usr/bin/env python3
"""
Kalshi Live Trader Dashboard
Usage: python3 monitor.py
"""
import subprocess, sqlite3, os, time
from datetime import datetime, timezone
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

EC2_HOST = os.environ.get("EC2_HOST", "ec2-user@YOUR_EC2_IP")
EC2_KEY = os.environ.get("EC2_KEY", os.path.expanduser("~/kalshi-crypto-pairs/kalshi-ec2.pem"))
EC2_DB = os.environ.get("EC2_DB", "/home/ec2-user/kalshi_live_trader.db")
LOCAL_DB = "/tmp/ec2_monitor.db"
POLL_SEC = 10

TRADE_COLS = ("id, ts_unix, asset, side, contracts, entry_price, fee, ticker, "
              "edge, z_score, consensus, avg_z, agreement, tte_at_entry, "
              "settled, pnl, bankroll_before, bankroll_after")

def sync_db():
    subprocess.run(["scp", "-q", "-i", EC2_KEY, f"{EC2_HOST}:{EC2_DB}", LOCAL_DB],
                   capture_output=True, timeout=10)

def fmt_time(ts):
    if ts is None: return "---"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")

def sparkline(values, width=40):
    if len(values) < 2: return ""
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    mn, mx = min(sampled), max(sampled)
    chars = "▁▂▃▄▅▆▇█"
    if mx == mn: return chars[4] * len(sampled)
    return "".join(chars[int((v - mn) / (mx - mn) * 7)] for v in sampled)

def taker_fee(ct, mid):
    if ct <= 0 or mid <= 0 or mid >= 1: return 0
    import math
    return math.ceil(0.07 * ct * mid * (1 - mid) * 100) / 100

def compute_pnl(side, contracts, mid, fee, result_yes):
    """Given settlement result (YES/NO), compute trade PnL."""
    if result_yes is None:
        return None
    if side == "YES":
        if result_yes:
            return contracts * (1 - mid) - fee
        else:
            return -(contracts * mid + fee)
    else:  # NO
        if not result_yes:
            return contracts * mid - fee
        else:
            return -(contracts * (1 - mid) + fee)

def enrich_trades(db):
    """Load trades and fill in PnL from settlements table where possible."""
    trades = db.execute(f"SELECT {TRADE_COLS} FROM trades ORDER BY id").fetchall()

    # Build settlement lookup: ticker -> result
    settle_map = {}
    for row in db.execute("SELECT ticker, asset, result FROM settlements"):
        settle_map[(row[0], row[1])] = row[2]

    # Also build by period: (period_part, asset) -> result
    # Using trade ticker to match (e.g. KXBTC15M-26FEB271730-30 -> lookup same ticker)
    settle_by_period = {}
    for row in db.execute("SELECT ticker, asset, result FROM settlements"):
        parts = row[0].split('-')
        period = '-'.join(parts[1:])
        settle_by_period[(period, row[1])] = row[2]

    enriched = []
    for t in trades:
        t = list(t)
        tid, ts, asset, side, ct, price, fee, ticker = t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]
        pnl = t[15]

        if pnl is None and ticker:
            # Try exact ticker match
            result = settle_map.get((ticker, asset))
            if result is None:
                # Try period match
                parts = ticker.split('-')
                period = '-'.join(parts[1:])
                result = settle_by_period.get((period, asset))

            if result is not None:
                result_yes = (result == "YES")
                pnl = compute_pnl(side, ct, price, fee, result_yes)
                t[15] = pnl
                t[14] = result  # settled field

        enriched.append(tuple(t))
    return enriched


def build_dashboard(db):
    trades = enrich_trades(db)
    if not trades:
        return Text("No trades yet")

    total = len(trades)
    wins = sum(1 for t in trades if t[15] is not None and t[15] > 0)
    losses = sum(1 for t in trades if t[15] is not None and t[15] <= 0)
    pending = sum(1 for t in trades if t[15] is None)
    settled = wins + losses
    wr = wins / settled * 100 if settled > 0 else 0
    total_pnl = sum(t[15] for t in trades if t[15] is not None)
    total_fee = sum(t[6] for t in trades)

    # Balance tracking: replay all settled trades sequentially
    bk_first = trades[0][16] if trades[0][16] else 11.94
    bk = bk_first
    bk_history = []
    by_period = defaultdict(list)
    for t in trades:
        tk = t[7]
        if tk:
            parts = tk.split('-')
            period = '-'.join(parts[1:])
            by_period[period].append(t)

    period_pnls = []
    for p in sorted(by_period.keys()):
        pp = sum(t[15] for t in by_period[p] if t[15] is not None)
        period_pnls.append(pp)
        bk += pp
        bk_history.append(bk)

    bk_now = bk

    bk_ret = (bk_now - bk_first) / bk_first * 100

    # Drawdown
    peak = bk_first; mdd_pct = 0; running = bk_first
    for pp in period_pnls:
        running += pp; peak = max(peak, running)
        dd = (peak - running) / peak * 100 if peak > 0 else 0
        mdd_pct = max(mdd_pct, dd)

    # Sharpe
    import statistics as st
    if len(period_pnls) > 1:
        mean_p = sum(period_pnls) / len(period_pnls)
        std_p = st.stdev(period_pnls) if len(period_pnls) > 1 else 0
        sharpe = mean_p / std_p if std_p > 0 else 0
    else:
        sharpe = 0

    # Streaks
    streak_w = streak_l = max_w = max_l = 0
    for t in trades:
        if t[15] is None: continue
        if t[15] > 0: streak_w += 1; streak_l = 0; max_w = max(max_w, streak_w)
        else: streak_l += 1; streak_w = 0; max_l = max(max_l, streak_l)

    avg_win = sum(t[15] for t in trades if t[15] is not None and t[15] > 0) / wins if wins else 0
    avg_loss = sum(t[15] for t in trades if t[15] is not None and t[15] <= 0) / losses if losses else 0
    gross_w = sum(t[15] for t in trades if t[15] is not None and t[15] > 0)
    gross_l = abs(sum(t[15] for t in trades if t[15] is not None and t[15] <= 0))
    pf = gross_w / gross_l if gross_l > 0 else 99

    # Per-asset
    asset_data = {}
    for a in ["BTC", "ETH", "SOL", "XRP"]:
        at = [t for t in trades if t[2] == a]
        aw = sum(1 for t in at if t[15] is not None and t[15] > 0)
        al = sum(1 for t in at if t[15] is not None and t[15] <= 0)
        ap = sum(1 for t in at if t[15] is None)
        apnl = sum(t[15] for t in at if t[15] is not None)
        awr = aw / (aw + al) * 100 if (aw + al) > 0 else 0
        asset_data[a] = (len(at), aw, al, ap, awr, apnl)

    # ══════════════════════════════════════
    # RENDER
    # ══════════════════════════════════════
    out = Table.grid(padding=(0, 0))

    # ── Balance ──
    pc = "green" if total_pnl >= 0 else "red"
    rc = "green" if bk_ret >= 0 else "red"
    bal = Text()
    bal.append("  BALANCE ", style="bold white on blue")
    bal.append(f"  ${bk_now:.2f} ", style=f"bold {pc}")
    bal.append(f"({bk_ret:+.1f}%) ", style=rc)
    bal.append(f"  PnL ", style="dim")
    bal.append(f"${total_pnl:+.2f} ", style=pc)
    bal.append(f"  Fees ${total_fee:.2f} ", style="dim")
    bal.append(f"  Start ${bk_first:.2f}", style="dim")
    out.add_row(bal)
    out.add_row(Text(""))

    # ── Sparkline ──
    if bk_history:
        spark = Text()
        spark.append(f"  ${min(bk_history):.0f} ", style="red")
        spark.append(sparkline(bk_history), style="bright_cyan")
        spark.append(f" ${max(bk_history):.0f}", style="green")
        out.add_row(spark)
        out.add_row(Text(""))

    # ── Metrics ──
    wr_c = "green" if wr >= 65 else "yellow"
    mdd_c = "green" if mdd_pct < 25 else ("yellow" if mdd_pct < 40 else "red")
    sh_c = "green" if sharpe > 0.2 else "yellow"

    lines = [
        f"  [dim]Trades[/] {total} ({pending}open)   [{wr_c}]{wr:.0f}%[/] {wins}W/{losses}L   [dim]Sharpe[/] [{sh_c}]{sharpe:.2f}[/]   [dim]MaxDD[/] [{mdd_c}]{mdd_pct:.1f}%[/]",
        f"  [dim]AvgWin[/] [green]${avg_win:.2f}[/]   [dim]AvgLoss[/] [red]${avg_loss:.2f}[/]   [dim]PF[/] {pf:.2f}x   [dim]Streak[/] [green]{max_w}W[/]/[red]{max_l}L[/]",
    ]
    for l in lines:
        out.add_row(Text.from_markup(l))
    out.add_row(Text(""))

    # ── Per-asset ──
    for a in ["BTC", "ETH", "SOL", "XRP"]:
        n, w, l, p, wr_a, pnl_a = asset_data[a]
        pc_a = "green" if pnl_a >= 0 else "red"
        wc_a = "green" if wr_a >= 70 else ("yellow" if wr_a >= 55 else "red")
        pend_str = f" ({p}open)" if p > 0 else ""
        line = f"  [bold]{a}[/]  {n}笔  {w}W/{l}L  [{wc_a}]{wr_a:.0f}%[/]  [{pc_a}]${pnl_a:+.2f}[/]{pend_str}"
        out.add_row(Text.from_markup(line))
    out.add_row(Text(""))

    # ── Recent trades ──
    tt = Table(box=box.SIMPLE, padding=(0, 0), show_edge=False, expand=False)
    tt.add_column("Time", min_width=8, no_wrap=True)
    tt.add_column("Ast", min_width=3, no_wrap=True)
    tt.add_column("Side", min_width=5, no_wrap=True)
    tt.add_column("Ct", justify="right", min_width=2, no_wrap=True)
    tt.add_column("Price", justify="right", min_width=5, no_wrap=True)
    tt.add_column("Cost", justify="right", min_width=5, no_wrap=True)
    tt.add_column("Edge", justify="right", min_width=6, no_wrap=True)
    tt.add_column("Z", justify="right", min_width=5, no_wrap=True)
    tt.add_column("Cons", min_width=4, no_wrap=True)
    tt.add_column("TTE", justify="right", min_width=4, no_wrap=True)
    tt.add_column("Result", justify="right", min_width=7, no_wrap=True)
    tt.add_column("Bal", justify="right", min_width=6, no_wrap=True)

    # Rebuild running balance for display
    running_bk = bk_first
    trade_bals = {}
    for p in sorted(by_period.keys()):
        for t in by_period[p]:
            if t[15] is not None:
                running_bk += t[15]
                trade_bals[t[0]] = running_bk

    for t in trades[-20:]:
        tid, ts, asset, side, ct, price, fee, tk, edge, z, cons, avgz, agr, tte, stl, pnl, bkb, bka = t
        cost = ct * price
        s_str = "[green]Y ^[/]" if side == "YES" else "[red]N v[/]"
        if pnl is not None:
            r_str = f"[green]+{pnl:.2f}[/]" if pnl > 0 else f"[red]{pnl:.2f}[/]"
        else:
            r_str = "[yellow]...[/]"
        # Use enriched balance or original
        bal_val = trade_bals.get(tid)
        if bal_val is None and bka is not None:
            bal_val = bka
        b_str = f"{bal_val:.2f}" if bal_val else "[dim]---[/]"
        c_str = (cons or "")[:4]
        tt.add_row(
            fmt_time(ts), asset, s_str, str(ct),
            f"{price:.2f}", f"{cost:.2f}", f"{edge:+.3f}", f"{z:+.2f}",
            c_str, f"{tte:.0f}" if tte else "", r_str, b_str
        )
    out.add_row(Panel(tt, title="[bold]Recent Trades[/]", border_style="cyan", padding=(0, 0)))

    # ── Periods ──
    pt = Table(box=box.SIMPLE, padding=(0, 1), show_edge=False, expand=False)
    pt.add_column("Period", min_width=20, no_wrap=True)
    pt.add_column("N", justify="right", min_width=2, no_wrap=True)
    pt.add_column("W/L", justify="center", min_width=4, no_wrap=True)
    pt.add_column("PnL", justify="right", min_width=7, no_wrap=True)
    pt.add_column("Dir", justify="center", min_width=3, no_wrap=True)

    for p in sorted(by_period.keys())[-10:]:
        ptrades = by_period[p]
        pw = sum(1 for t in ptrades if t[15] is not None and t[15] > 0)
        pl = sum(1 for t in ptrades if t[15] is not None and t[15] <= 0)
        pp_pending = sum(1 for t in ptrades if t[15] is None)
        ppnl = sum(t[15] for t in ptrades if t[15] is not None)
        if pp_pending > 0:
            pnl_s = f"[yellow]open({pp_pending})[/]"
        else:
            pnl_s = f"[green]${ppnl:+.2f}[/]" if ppnl >= 0 else f"[red]${ppnl:+.2f}[/]"
        sides = set(t[3] for t in ptrades)
        d = "[green]YES[/]" if sides == {"YES"} else ("[red]NO[/]" if sides == {"NO"} else "[yellow]MX[/]")
        pt.add_row(p, str(len(ptrades)), f"{pw}/{pl}", pnl_s, d)
    out.add_row(Panel(pt, title="[bold]Periods[/]", border_style="blue", padding=(0, 0)))

    # ── Footer ──
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out.add_row(Text(f"  {now_str}  |  $1.25 cost<=0.60 C1 50%exp dir_lock  |  {POLL_SEC}s", style="dim"))

    return out


def main():
    console = Console()
    console.print("[dim]Syncing...[/]")
    sync_db()
    if not os.path.exists(LOCAL_DB):
        console.print("[red]ERROR: Cannot sync DB[/]"); return

    while True:
        try:
            sync_db()
        except Exception:
            pass
        try:
            db = sqlite3.connect(LOCAL_DB)
            dashboard = build_dashboard(db)
            db.close()
            console.clear()
            console.print(dashboard)
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        Console().print("\n[dim]Stopped.[/]")
