#!/usr/bin/env python3
"""
Kalshi Trader — Terminal Dashboard
Live market data + account balance/positions + DB trade history
Usage: python3 dashboard.py [db_path]
"""
import asyncio, aiohttp, sqlite3, os, sys, time, statistics, base64, requests
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional, Dict
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich import box

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TRADER_DB", "kalshi_live_trader.db")
POLL_SEC = 2

PROD_BASE = "https://api.elections.kalshi.com"
COINBASE_API = "https://api.coinbase.com/v2"

# Kalshi API auth (optional — dashboard works without it, but balance/positions require auth)
# 1. Create account at https://kalshi.com
# 2. Go to Settings → API Keys → generate RSA key pair
# 3. Set env vars: KALSHI_KEY_ID=<your-key-id>  KALSHI_KEY_FILE=<path-to-private.pem>
#    Or copy .env.example to .env and fill in your values
KEY_ID = os.environ.get("KALSHI_KEY_ID", "")
KEY_FILE = os.environ.get("KALSHI_KEY_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "kalshi_private.pem"))

SERIES = {
    "BTC": {"series": "KXBTC15M"},
    "ETH": {"series": "KXETH15M"},
    "SOL": {"series": "KXSOL15M"},
    "XRP": {"series": "KXXRP15M"},
}
ASSETS = ["BTC", "ETH", "SOL", "XRP"]
OB_DEPTH = 10

# DB column indices
TRADE_COLS = ("id, ts_unix, ts_iso, asset, side, contracts, entry_price, fee, ticker, "
              "edge, z_score, model_p, confidence, order_id, order_status, "
              "settled, pnl, bankroll_after")
I_ID, I_TS, I_ISO, I_ASSET, I_SIDE, I_CT, I_PRICE, I_FEE, I_TICKER = range(9)
I_EDGE, I_Z, I_MODELP, I_CONF, I_OID, I_OSTATUS = range(9, 15)
I_SETTLED, I_PNL, I_BK_AFTER = range(15, 18)

# Ticker -> asset lookup
TICKER_TO_ASSET = {v["series"]: k for k, v in SERIES.items()}


# ==============================================================================
# Kalshi Auth Client (for balance + positions)
# ==============================================================================

class KalshiClient:
    def __init__(self, key_id: str, key_file: str):
        self.key_id = key_id
        self.base = PROD_BASE
        self.ok = False
        try:
            with open(key_file, "rb") as f:
                self.private_key = serialization.load_pem_private_key(f.read(), password=None)
            self.ok = bool(key_id)
        except Exception:
            self.private_key = None

    def _sign(self, method: str, path: str) -> dict:
        ts_ms = str(int(time.time() * 1000))
        path_clean = path.split("?")[0]
        msg = ts_ms + method + path_clean
        sig = self.private_key.sign(
            msg.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256())
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }

    def get(self, path: str, params: dict = None) -> dict:
        r = requests.get(self.base + path, headers=self._sign("GET", path),
                         params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_balance(self) -> Optional[float]:
        if not self.ok: return None
        try:
            data = self.get("/trade-api/v2/portfolio/balance")
            return data.get("balance", 0) / 100.0
        except Exception:
            return None

    def get_positions(self) -> dict:
        """Returns {ticker: {side, qty, avg_price}} for open positions."""
        if not self.ok: return {}
        try:
            data = self.get("/trade-api/v2/portfolio/positions",
                            params={"count_filter": "position", "limit": 100})
            positions = {}
            for mp in data.get("market_positions", []):
                ticker = mp.get("ticker", "")
                yes_pos = mp.get("market_position", {}).get("yes", 0) if isinstance(mp.get("market_position"), dict) else 0
                no_pos = mp.get("market_position", {}).get("no", 0) if isinstance(mp.get("market_position"), dict) else 0
                pos = mp.get("position", 0)
                if pos != 0:
                    # Determine asset from ticker prefix
                    asset = "?"
                    for prefix, a in TICKER_TO_ASSET.items():
                        if ticker.startswith(prefix):
                            asset = a
                            break
                    positions[ticker] = {"qty": abs(pos), "asset": asset, "ticker": ticker}
            return positions
        except Exception:
            return {}

    def fetch_account(self) -> dict:
        """Fetch balance + positions in one call cycle."""
        bal = self.get_balance()
        pos = self.get_positions()
        return {"balance": bal, "positions": pos}


# ==============================================================================
# Live Market Poller (no auth needed)
# ==============================================================================

class Poller:
    def __init__(self):
        self._kal_session: Optional[aiohttp.ClientSession] = None
        self._cb_session: Optional[aiohttp.ClientSession] = None
        self._kal_created: float = 0

    async def _kal_sess(self) -> aiohttp.ClientSession:
        now = time.time()
        if self._kal_session is None or self._kal_session.closed or now - self._kal_created > 600:
            if self._kal_session and not self._kal_session.closed:
                try: await self._kal_session.close()
                except: pass
            self._kal_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10, connect=5, sock_read=5),
                headers={"Accept": "application/json"})
            self._kal_created = now
        return self._kal_session

    async def _cb_sess(self) -> aiohttp.ClientSession:
        if self._cb_session is None or self._cb_session.closed:
            self._cb_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5))
        return self._cb_session

    async def poll_market(self, asset: str) -> dict:
        try:
            s = await self._kal_sess()
            url = f"{PROD_BASE}/trade-api/v2/markets?series_ticker={SERIES[asset]['series']}&status=open&limit=1"
            async with s.get(url) as r:
                if r.status != 200:
                    return {"asset": asset, "error": f"HTTP {r.status}"}
                data = await r.json()
            mkts = data.get("markets", [])
            if not mkts:
                return {"asset": asset, "error": "no_market"}
            m = mkts[0]
            yb, ya = m.get("yes_bid"), m.get("yes_ask")
            mid, spread = None, None
            if yb and ya and yb > 0 and ya > 0:
                mid = (yb + ya) / 200.0
                spread = (ya - yb) / 100.0
            elif yb and yb > 0:
                mid = yb / 100.0
            elif ya and ya > 0:
                mid = ya / 100.0
            tte = None
            ct = m.get("close_time", "")
            if ct:
                try:
                    close_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    tte = (close_dt - datetime.now(timezone.utc)).total_seconds()
                    if tte < 0: tte = 0
                except: pass
            return {
                "asset": asset, "ticker": m.get("ticker", ""),
                "mid": mid, "spread": spread,
                "yes_bid": yb, "yes_ask": ya,
                "volume": m.get("volume", 0), "oi": m.get("open_interest", 0),
                "floor_strike": m.get("floor_strike"),
                "time_to_expiry": tte, "error": None,
            }
        except Exception as e:
            return {"asset": asset, "error": str(e)[:60]}

    async def poll_orderbook(self, ticker: str, asset: str) -> dict:
        if not ticker:
            return {"asset": asset, "ob_imbalance": None, "y_qty": 0, "n_qty": 0}
        try:
            s = await self._kal_sess()
            url = f"{PROD_BASE}/trade-api/v2/markets/{ticker}/orderbook?depth={OB_DEPTH}"
            async with s.get(url) as r:
                if r.status != 200:
                    return {"asset": asset, "ob_imbalance": None, "y_qty": 0, "n_qty": 0}
                data = await r.json()
            ob = data.get("orderbook", {})
            y_qty = sum(qty for _, qty in ob.get("yes", []))
            n_qty = sum(qty for _, qty in ob.get("no", []))
            total = y_qty + n_qty
            return {"asset": asset, "y_qty": y_qty, "n_qty": n_qty,
                    "ob_imbalance": round((y_qty - n_qty) / total, 4) if total > 0 else None}
        except:
            return {"asset": asset, "ob_imbalance": None, "y_qty": 0, "n_qty": 0}

    async def poll_prices(self) -> Dict[str, Optional[float]]:
        try:
            s = await self._cb_sess()
            url = f"{COINBASE_API}/exchange-rates?currency=USD"
            async with s.get(url) as r:
                if r.status != 200:
                    return {a: None for a in ASSETS}
                data = await r.json()
            rates = data.get("data", {}).get("rates", {})
            prices = {}
            for a in ASSETS:
                rate = rates.get(a)
                prices[a] = round(1.0 / float(rate), 4) if rate else None
            return prices
        except:
            return {a: None for a in ASSETS}

    async def poll_all(self) -> dict:
        market_tasks = [self.poll_market(a) for a in ASSETS]
        market_results = await asyncio.gather(*market_tasks, return_exceptions=True)
        markets = {}
        for i, r in enumerate(market_results):
            a = ASSETS[i]
            markets[a] = r if isinstance(r, dict) else {"asset": a, "error": str(r)[:60]}

        ob_tasks = [self.poll_orderbook(markets[a].get("ticker", ""), a) for a in ASSETS]
        ob_tasks.append(self.poll_prices())
        ob_results = await asyncio.gather(*ob_tasks, return_exceptions=True)

        for i, a in enumerate(ASSETS):
            ob = ob_results[i] if isinstance(ob_results[i], dict) else {}
            markets[a]["ob_imbalance"] = ob.get("ob_imbalance")
            markets[a]["y_qty"] = ob.get("y_qty", 0)
            markets[a]["n_qty"] = ob.get("n_qty", 0)

        prices = ob_results[-1] if isinstance(ob_results[-1], dict) else {}
        for a in ASSETS:
            rp = prices.get(a)
            markets[a]["real_price"] = rp
            fs = markets[a].get("floor_strike")
            if rp and fs and fs > 0:
                markets[a]["price_vs_strike_pct"] = round((rp - fs) / fs * 100, 4)
            else:
                markets[a]["price_vs_strike_pct"] = None
        return markets

    async def close(self):
        for s in [self._kal_session, self._cb_session]:
            if s and not s.closed:
                try: await s.close()
                except: pass


# ==============================================================================
# DB + render helpers
# ==============================================================================

def fmt_time(ts):
    if ts is None: return "---"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")

def fmt_tte(seconds):
    if seconds is None: return "---"
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"

def sparkline(values, width=50):
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

def compute_pnl(side, contracts, mid, fee, result_yes):
    if result_yes is None: return None
    if side == "YES":
        return contracts * (1 - mid) - fee if result_yes else -(contracts * mid + fee)
    else:
        return contracts * mid - fee if not result_yes else -(contracts * (1 - mid) + fee)

def enrich_trades(db):
    trades = db.execute(f"SELECT {TRADE_COLS} FROM trades ORDER BY id").fetchall()
    settle_map = {}
    settle_by_period = {}
    for row in db.execute("SELECT ticker, asset, result FROM settlements"):
        settle_map[(row[0], row[1])] = row[2]
        parts = row[0].split('-')
        period = '-'.join(parts[1:])
        settle_by_period[(period, row[1])] = row[2]

    enriched = []
    for t in trades:
        t = list(t)
        if t[I_PNL] is None and t[I_TICKER]:
            result = settle_map.get((t[I_TICKER], t[I_ASSET]))
            if result is None:
                parts = t[I_TICKER].split('-')
                period = '-'.join(parts[1:])
                result = settle_by_period.get((period, t[I_ASSET]))
            if result is not None:
                t[I_PNL] = compute_pnl(t[I_SIDE], t[I_CT], t[I_PRICE], t[I_FEE], result == "YES")
                t[I_SETTLED] = result
        enriched.append(tuple(t))
    return enriched


# ==============================================================================
# Build dashboard
# ==============================================================================

def build_dashboard(live_data=None, account=None):
    out = Table.grid(padding=(0, 0))

    # ── Header ──
    now_str = datetime.now().strftime("%H:%M:%S")
    header = Text()
    header.append("  KALSHI TRADER ", style="bold white on blue")
    header.append(f"  {now_str}  ", style="dim")
    header.append(f"  {POLL_SEC}s  ", style="dim")
    out.add_row(header)
    out.add_row(Text(""))

    # ── Account (live from API) ──
    if account:
        api_bal = account.get("balance")
        positions = account.get("positions", {})

        acct_line = Text()
        if api_bal is not None:
            acct_line.append("  BALANCE ", style="bold white on blue")
            acct_line.append(f"  ${api_bal:.2f}", style="bold green" if api_bal > 0 else "bold red")
        if positions:
            acct_line.append(f"   POSITIONS ", style="bold white on magenta")
            acct_line.append(f"  {len(positions)} open", style="yellow")
        elif api_bal is not None:
            acct_line.append("   [dim]No open positions[/]")
        out.add_row(acct_line)

        # Position details
        if positions and live_data:
            for ticker, pinfo in positions.items():
                asset = pinfo.get("asset", "?")
                qty = pinfo.get("qty", 0)
                # Find current mid for this position's market
                mkt = live_data.get(asset, {})
                mid = mkt.get("mid")
                tte = mkt.get("time_to_expiry")
                pos_line = Text()
                pos_line.append(f"    {asset} ", style="bold")
                pos_line.append(f"x{qty} ", style="yellow")
                pos_line.append(f" {ticker} ", style="dim")
                if mid is not None:
                    pos_line.append(f" mid={mid:.1%}", style="cyan")
                if tte is not None:
                    pos_line.append(f"  tte={fmt_tte(tte)}", style="yellow" if tte < 300 else "dim")
                out.add_row(pos_line)
        out.add_row(Text(""))

    # ── Live Markets ──
    if live_data:
        mt = Table(box=box.SIMPLE, padding=(0, 1), show_edge=False, expand=False)
        mt.add_column("Asset", min_width=3, no_wrap=True)
        mt.add_column("Mid", justify="right", min_width=5, no_wrap=True)
        mt.add_column("Sprd", justify="right", min_width=5, no_wrap=True)
        mt.add_column("TTE", justify="right", min_width=5, no_wrap=True)
        mt.add_column("Strike", justify="right", min_width=10, no_wrap=True)
        mt.add_column("Price", justify="right", min_width=10, no_wrap=True)
        mt.add_column("PvS%", justify="right", min_width=7, no_wrap=True)
        mt.add_column("OB", justify="right", min_width=6, no_wrap=True)
        mt.add_column("Y/N", justify="right", min_width=8, no_wrap=True)
        mt.add_column("Vol", justify="right", min_width=5, no_wrap=True)

        for a in ASSETS:
            m = live_data.get(a, {})
            if m.get("error"):
                mt.add_row(f"[bold]{a}[/]", f"[red]{m['error']}[/]", "", "", "", "", "", "", "", "")
                continue
            mid = m.get("mid")
            spread = m.get("spread")
            tte = m.get("time_to_expiry")
            fs = m.get("floor_strike")
            rp = m.get("real_price")
            pvs = m.get("price_vs_strike_pct")
            obi = m.get("ob_imbalance")
            y_qty = m.get("y_qty", 0)
            n_qty = m.get("n_qty", 0)
            vol = m.get("volume", 0)

            mid_s = f"{mid:.1%}" if mid else "---"
            mid_c = "green" if mid and mid > 0.55 else ("red" if mid and mid < 0.45 else "yellow")
            pvs_s = f"{pvs:+.3f}" if pvs is not None else "---"
            pvs_c = "green" if pvs and pvs > 0 else ("red" if pvs and pvs < 0 else "dim")
            obi_s = f"{obi:+.2f}" if obi is not None else "---"
            obi_c = "green" if obi and obi > 0.1 else ("red" if obi and obi < -0.1 else "dim")
            tte_s = fmt_tte(tte)
            tte_c = "red" if tte is not None and tte < 120 else ("yellow" if tte is not None and tte < 300 else "dim")

            mt.add_row(
                f"[bold]{a}[/]",
                f"[{mid_c}]{mid_s}[/]",
                f"{spread:.3f}" if spread else "---",
                f"[{tte_c}]{tte_s}[/]",
                f"${fs:,.0f}" if fs else "---",
                f"${rp:,.0f}" if rp else "---",
                f"[{pvs_c}]{pvs_s}[/]",
                f"[{obi_c}]{obi_s}[/]",
                f"{y_qty}/{n_qty}",
                str(vol),
            )
        out.add_row(Panel(mt, title="[bold]Live Markets[/]", border_style="green", padding=(0, 0)))
        out.add_row(Text(""))

    # ── DB Trade Stats ──
    if not os.path.exists(DB_PATH):
        out.add_row(Text(f"  DB not found: {DB_PATH}", style="dim"))
        return out

    try:
        db = sqlite3.connect(DB_PATH)
        trades = enrich_trades(db)
        db.close()
    except Exception as e:
        out.add_row(Text(f"  DB error: {e}", style="red"))
        return out

    if not trades:
        out.add_row(Text("  No trades yet", style="dim"))
        return out

    total = len(trades)
    wins = sum(1 for t in trades if t[I_PNL] is not None and t[I_PNL] > 0)
    losses = sum(1 for t in trades if t[I_PNL] is not None and t[I_PNL] <= 0)
    pending = sum(1 for t in trades if t[I_PNL] is None)
    settled_ct = wins + losses
    wr = wins / settled_ct * 100 if settled_ct > 0 else 0
    total_pnl = sum(t[I_PNL] for t in trades if t[I_PNL] is not None)
    total_fee = sum(t[I_FEE] for t in trades)

    # Balance from API if available, else from DB
    api_bal = account.get("balance") if account else None
    bk_after_vals = [t[I_BK_AFTER] for t in trades if t[I_BK_AFTER] is not None]
    if bk_after_vals:
        first_pnl = trades[0][I_PNL] or 0
        bk_first = bk_after_vals[0] - first_pnl
        bk_now = api_bal if api_bal is not None else bk_after_vals[-1]
    else:
        bk_first = 10.0
        bk_now = api_bal if api_bal is not None else (bk_first + total_pnl)
    bk_ret = (bk_now - bk_first) / bk_first * 100 if bk_first > 0 else 0

    # Group by period
    by_period = defaultdict(list)
    for t in trades:
        tk = t[I_TICKER]
        if tk:
            parts = tk.split('-')
            period = '-'.join(parts[1:])
            by_period[period].append(t)

    bk = bk_first
    bk_history = [bk_first]
    period_pnls = []
    for p in sorted(by_period.keys()):
        pp = sum(t[I_PNL] for t in by_period[p] if t[I_PNL] is not None)
        period_pnls.append(pp)
        bk += pp
        bk_history.append(bk)

    # Drawdown
    peak = bk_first; mdd_pct = 0; running = bk_first
    for pp in period_pnls:
        running += pp; peak = max(peak, running)
        dd = (peak - running) / peak * 100 if peak > 0 else 0
        mdd_pct = max(mdd_pct, dd)

    # Sharpe
    if len(period_pnls) > 1:
        std_p = statistics.stdev(period_pnls)
        sharpe = (sum(period_pnls) / len(period_pnls)) / std_p if std_p > 0 else 0
    else:
        sharpe = 0

    # Streaks
    streak_w = streak_l = max_w = max_l = 0
    for t in trades:
        if t[I_PNL] is None: continue
        if t[I_PNL] > 0: streak_w += 1; streak_l = 0; max_w = max(max_w, streak_w)
        else: streak_l += 1; streak_w = 0; max_l = max(max_l, streak_l)

    avg_win = sum(t[I_PNL] for t in trades if t[I_PNL] is not None and t[I_PNL] > 0) / wins if wins else 0
    avg_loss = sum(t[I_PNL] for t in trades if t[I_PNL] is not None and t[I_PNL] <= 0) / losses if losses else 0
    gross_w = sum(t[I_PNL] for t in trades if t[I_PNL] is not None and t[I_PNL] > 0)
    gross_l = abs(sum(t[I_PNL] for t in trades if t[I_PNL] is not None and t[I_PNL] <= 0))
    pf = gross_w / gross_l if gross_l > 0 else 99

    # ── PnL line ──
    pc = "green" if total_pnl >= 0 else "red"
    rc = "green" if bk_ret >= 0 else "red"
    bal_line = Text()
    bal_line.append(f"  ${bk_now:.2f} ", style=f"bold {pc}")
    bal_line.append(f"({bk_ret:+.1f}%) ", style=rc)
    bal_line.append(f"  PnL ", style="dim")
    bal_line.append(f"${total_pnl:+.2f} ", style=pc)
    bal_line.append(f"  Fees ${total_fee:.2f} ", style="dim")
    bal_line.append(f"  Start ${bk_first:.2f}", style="dim")
    out.add_row(bal_line)
    out.add_row(Text(""))

    # ── Sparkline ──
    if len(bk_history) >= 2:
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

    out.add_row(Text.from_markup(
        f"  [dim]Trades[/] {total} ({pending}open)   [{wr_c}]{wr:.0f}%[/] {wins}W/{losses}L   "
        f"[dim]Sharpe[/] [{sh_c}]{sharpe:.2f}[/]   [dim]MaxDD[/] [{mdd_c}]{mdd_pct:.1f}%[/]   "
        f"[dim]PF[/] {pf:.2f}x"
    ))
    out.add_row(Text.from_markup(
        f"  [dim]AvgWin[/] [green]${avg_win:.2f}[/]   [dim]AvgLoss[/] [red]${avg_loss:.2f}[/]   "
        f"[dim]Streak[/] [green]{max_w}W[/]/[red]{max_l}L[/]"
    ))
    out.add_row(Text(""))

    # ── Per-asset ──
    for a in ASSETS:
        at = [t for t in trades if t[I_ASSET] == a]
        if not at: continue
        aw = sum(1 for t in at if t[I_PNL] is not None and t[I_PNL] > 0)
        al = sum(1 for t in at if t[I_PNL] is not None and t[I_PNL] <= 0)
        ap = sum(1 for t in at if t[I_PNL] is None)
        apnl = sum(t[I_PNL] for t in at if t[I_PNL] is not None)
        awr = aw / (aw + al) * 100 if (aw + al) > 0 else 0
        pc_a = "green" if apnl >= 0 else "red"
        wc_a = "green" if awr >= 70 else ("yellow" if awr >= 55 else "red")
        pend_str = f" ({ap}open)" if ap > 0 else ""
        out.add_row(Text.from_markup(
            f"  [bold]{a:3s}[/]  {len(at):3d}trades  {aw}W/{al}L  [{wc_a}]{awr:.0f}%[/]  [{pc_a}]${apnl:+.2f}[/]{pend_str}"
        ))
    out.add_row(Text(""))

    # ── Recent trades ──
    tt = Table(box=box.SIMPLE, padding=(0, 0), show_edge=False, expand=False)
    for col, kw in [("Time",{}), ("Ast",{}), ("Side",{}), ("Ct",{"justify":"right"}),
                    ("Price",{"justify":"right"}), ("Cost",{"justify":"right"}),
                    ("Edge",{"justify":"right"}), ("Z",{"justify":"right"}),
                    ("P(m)",{"justify":"right"}), ("Result",{"justify":"right"}),
                    ("Bal",{"justify":"right"})]:
        tt.add_column(col, no_wrap=True, **kw)

    running_bk = bk_first
    trade_bals = {}
    for p in sorted(by_period.keys()):
        for t in by_period[p]:
            if t[I_PNL] is not None:
                running_bk += t[I_PNL]
                trade_bals[t[I_ID]] = running_bk

    for t in trades[-25:]:
        cost = t[I_CT] * t[I_PRICE]
        s_str = "[green]Y ^[/]" if t[I_SIDE] == "YES" else "[red]N v[/]"
        pnl = t[I_PNL]
        r_str = f"[green]+{pnl:.2f}[/]" if pnl is not None and pnl > 0 else (
            f"[red]{pnl:.2f}[/]" if pnl is not None else "[yellow]...[/]")
        bal_val = trade_bals.get(t[I_ID]) or t[I_BK_AFTER]
        b_str = f"{bal_val:.2f}" if bal_val else "[dim]---[/]"
        tt.add_row(
            fmt_time(t[I_TS]), t[I_ASSET], s_str, str(t[I_CT]),
            f"{t[I_PRICE]:.2f}", f"{cost:.2f}",
            f"{t[I_EDGE]:+.3f}" if t[I_EDGE] is not None else "",
            f"{t[I_Z]:+.2f}" if t[I_Z] is not None else "",
            f"{t[I_MODELP]:.2f}" if t[I_MODELP] is not None else "",
            r_str, b_str
        )
    out.add_row(Panel(tt, title="[bold]Recent Trades (last 25)[/]", border_style="cyan", padding=(0, 0)))

    # ── Periods ──
    pt = Table(box=box.SIMPLE, padding=(0, 1), show_edge=False, expand=False)
    pt.add_column("Period", min_width=20, no_wrap=True)
    pt.add_column("N", justify="right", min_width=2, no_wrap=True)
    pt.add_column("W/L", justify="center", min_width=4, no_wrap=True)
    pt.add_column("PnL", justify="right", min_width=7, no_wrap=True)
    pt.add_column("Dir", justify="center", min_width=3, no_wrap=True)

    for p in sorted(by_period.keys())[-10:]:
        ptrades = by_period[p]
        pw = sum(1 for t in ptrades if t[I_PNL] is not None and t[I_PNL] > 0)
        pl = sum(1 for t in ptrades if t[I_PNL] is not None and t[I_PNL] <= 0)
        pp_pend = sum(1 for t in ptrades if t[I_PNL] is None)
        ppnl = sum(t[I_PNL] for t in ptrades if t[I_PNL] is not None)
        if pp_pend > 0:
            pnl_s = f"[yellow]open({pp_pend})[/]"
        else:
            pnl_s = f"[green]${ppnl:+.2f}[/]" if ppnl >= 0 else f"[red]${ppnl:+.2f}[/]"
        sides = set(t[I_SIDE] for t in ptrades)
        d = "[green]YES[/]" if sides == {"YES"} else ("[red]NO[/]" if sides == {"NO"} else "[yellow]MX[/]")
        pt.add_row(p, str(len(ptrades)), f"{pw}/{pl}", pnl_s, d)
    out.add_row(Panel(pt, title="[bold]Periods (last 10)[/]", border_style="blue", padding=(0, 0)))

    return out


# ==============================================================================
# Main loop
# ==============================================================================

async def main():
    console = Console()
    poller = Poller()
    client = KalshiClient(KEY_ID, KEY_FILE)
    live_data = None
    account = None
    acct_tick = 0  # poll account every 5 cycles (10s) to avoid rate limit

    auth_str = "[green]auth OK[/]" if client.ok else "[yellow]no auth (market data only)[/]"
    console.print(f"[dim]Dashboard: {DB_PATH} | {POLL_SEC}s | {auth_str} | Ctrl+C to quit[/]")

    with Live(build_dashboard(), console=console, refresh_per_second=1, screen=True) as live:
        try:
            while True:
                # Live market data (async, every cycle)
                try:
                    live_data = await poller.poll_all()
                except Exception:
                    pass

                # Account data (sync/auth, every 5 cycles = 10s)
                if client.ok and acct_tick % 5 == 0:
                    try:
                        account = await asyncio.get_event_loop().run_in_executor(
                            None, client.fetch_account)
                    except Exception:
                        pass
                acct_tick += 1

                live.update(build_dashboard(live_data, account))
                await asyncio.sleep(POLL_SEC)
        finally:
            await poller.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        Console().print("\n[dim]Stopped.[/]")
