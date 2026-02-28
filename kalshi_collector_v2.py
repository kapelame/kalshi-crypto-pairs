#!/usr/bin/env python3
"""
================================================================================
Kalshi 15-Min Crypto Data Collector v2 (长时间稳定版)
================================================================================

v2升级 (解决v1的749秒断裂问题):
  1. 4小时运行时间 (v1=1小时)
  2. 轮询间隔1.5s (v1=1s, 降低限流风险)
  3. Session自动刷新 (每10分钟重建HTTP连接, 防止连接老化)
  4. 指数退避重试 (连续错误→等待1s→2s→4s→8s, 不crash)
  5. 每5分钟进度报告 (行数/gap数/体制估算/剩余时间)
  6. 断连自动恢复 (不因单次API失败产生大gap)

数据流:
  Kalshi API → 原始快照 → 特征工程 → SQLite/CSV

存储:
  SQLite: kalshi_data.db (实时写入, 断电不丢失)
  CSV: kalshi_features.csv (结束时导出)

运行 (Colab):
  Cell 1 - 防睡眠:
    from IPython.display import display, Javascript
    display(Javascript('setInterval(()=>{document.querySelector("colab-toolbar-button#connect")?.click()},60000)'))
  Cell 2 - 安装:
    pip install aiohttp nest_asyncio
  Cell 3 - 粘贴此文件运行
================================================================================
"""

import asyncio
import aiohttp
import sqlite3
import csv
import os
import sys
import math
import time
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple
from collections import deque
from enum import Enum


# ==============================================================================
# 配置 (v2: 4小时 + 1.5s轮询)
# ==============================================================================

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

POLL_SEC = 1.5             # v2: 1.5s (v1=1.0s, 降低限流风险)
MAX_RUNTIME = 14400        # v2: 4小时 (v1=3600=1小时)

SERIES = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
}
ASSETS = list(SERIES.keys())

# Kelly公式参数
BANKROLL = 100.0
KELLY_FRAC = 0.25

# 偏差信号阈值
DEVIATION_THRESHOLD = 0.05

# 动量窗口 (秒)
MOMENTUM_WINDOWS = [5, 15, 30, 60]

# v2: 稳定性参数
SESSION_REFRESH_SEC = 600  # 每10分钟刷新HTTP session
MAX_CONSECUTIVE_ERRORS = 20  # 连续20次失败才报警
PROGRESS_INTERVAL = 300    # 每5分钟打印进度
BACKOFF_BASE = 1.0         # 退避基数(秒)
BACKOFF_MAX = 16.0         # 最大退避(秒)

DB_PATH = "kalshi_data.db"
CSV_EXPORT = "kalshi_features.csv"
W = 105


# ==============================================================================
# 数据模型
# ==============================================================================

@dataclass
class Tick:
    """单个市场的原始快照"""
    ts: str
    ts_unix: float
    asset: str
    ticker: str
    status: str
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    mid: Optional[float] = None
    spread: Optional[float] = None
    last_price: Optional[float] = None
    volume: int = 0
    close_time: str = ""
    error: Optional[str] = None


@dataclass
class TickFeatures:
    """每秒一行的ML特征向量"""
    ts: str
    ts_unix: float
    n_valid: int

    btc_mid: Optional[float] = None
    btc_spread: Optional[float] = None
    btc_vol: int = 0
    eth_mid: Optional[float] = None
    eth_spread: Optional[float] = None
    eth_vol: int = 0
    sol_mid: Optional[float] = None
    sol_spread: Optional[float] = None
    sol_vol: int = 0
    xrp_mid: Optional[float] = None
    xrp_spread: Optional[float] = None
    xrp_vol: int = 0

    vw_mid: Optional[float] = None
    equal_mid: Optional[float] = None

    btc_dev: Optional[float] = None
    eth_dev: Optional[float] = None
    sol_dev: Optional[float] = None
    xrp_dev: Optional[float] = None

    dispersion: Optional[float] = None
    range_spread: Optional[float] = None

    max_abs_dev: Optional[float] = None
    lagging_asset: str = ""

    btc_mom_5s: Optional[float] = None
    btc_mom_15s: Optional[float] = None
    eth_mom_5s: Optional[float] = None
    eth_mom_15s: Optional[float] = None
    sol_mom_5s: Optional[float] = None
    sol_mom_15s: Optional[float] = None
    xrp_mom_5s: Optional[float] = None
    xrp_mom_15s: Optional[float] = None
    vw_mom_5s: Optional[float] = None
    vw_mom_15s: Optional[float] = None

    kelly_frac: Optional[float] = None
    kelly_usd: Optional[float] = None

    signal: str = "none"


# ==============================================================================
# MarketPoller v2 (自动刷新 + 指数退避)
# ==============================================================================

class MarketPoller:
    """
    v2改进:
    - Session每10分钟自动重建 (防止连接池老化导致的静默断连)
    - 单次请求失败 -> 返回error Tick (不crash)
    - 连续错误计数 -> 触发指数退避等待
    - 连接异常 -> 强制刷新session
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_created: float = 0
        self._consecutive_errors: int = 0
        self._total_errors: int = 0
        self._total_polls: int = 0

    async def _sess(self) -> aiohttp.ClientSession:
        now = time.time()
        needs_refresh = (
            self._session is None
            or self._session.closed
            or (now - self._session_created > SESSION_REFRESH_SEC)
        )
        if needs_refresh:
            if self._session and not self._session.closed:
                try:
                    await self._session.close()
                except Exception:
                    pass
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10, connect=5, sock_read=5),
                headers={"Accept": "application/json"}
            )
            self._session_created = now
        return self._session

    async def poll_one(self, asset: str, series: str) -> Tick:
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat(timespec="milliseconds")
        now_unix = now_dt.timestamp()

        try:
            s = await self._sess()
            url = f"{KALSHI_API}/markets?series_ticker={series}&status=open&limit=5"
            async with s.get(url) as r:
                if r.status == 429:
                    return Tick(now_iso, now_unix, asset, "", "rate_limit",
                                error="HTTP 429 rate limit")
                if r.status != 200:
                    body = await r.text()
                    return Tick(now_iso, now_unix, asset, "", "error",
                                error=f"HTTP {r.status}: {body[:150]}")
                data = await r.json()

            mkts = data.get("markets", [])
            if not mkts:
                return Tick(now_iso, now_unix, asset, series, "no_markets",
                            error="no active markets")

            m = mkts[0]
            rb, ra = m.get("yes_bid"), m.get("yes_ask")
            yb = (rb / 100.0) if rb and rb > 0 else None
            ya = (ra / 100.0) if ra and ra > 0 else None

            mid, spread = None, None
            if yb is not None and ya is not None:
                mid = (yb + ya) / 2.0
                spread = ya - yb
            elif yb is not None:
                mid = yb
            elif ya is not None:
                mid = ya

            lp = m.get("last_price")
            last_p = (lp / 100.0) if lp and lp > 0 else None

            return Tick(
                ts=now_iso, ts_unix=now_unix, asset=asset,
                ticker=m.get("ticker", ""), status=m.get("status", ""),
                yes_bid=yb, yes_ask=ya, mid=mid, spread=spread,
                last_price=last_p, volume=m.get("volume", 0),
                close_time=m.get("close_time", ""), error=None
            )
        except asyncio.TimeoutError:
            return Tick(now_iso, now_unix, asset, "", "timeout", error="timeout")
        except (aiohttp.ClientError, aiohttp.ServerDisconnectedError) as e:
            # v2: 连接错误 -> 强制刷新session下次重建
            if self._session and not self._session.closed:
                try:
                    await self._session.close()
                except Exception:
                    pass
            self._session = None
            return Tick(now_iso, now_unix, asset, "", "conn_error",
                        error=f"conn: {type(e).__name__}")
        except Exception as e:
            return Tick(now_iso, now_unix, asset, "", "error",
                        error=f"{type(e).__name__}: {str(e)[:120]}")

    async def poll_all(self) -> Dict[str, Tick]:
        self._total_polls += 1
        tasks = [self.poll_one(a, s) for a, s in SERIES.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        ticks = {}
        has_error = False
        for i, r in enumerate(results):
            a = ASSETS[i]
            if isinstance(r, Exception):
                now = datetime.now(timezone.utc)
                ticks[a] = Tick(now.isoformat(), now.timestamp(), a, "", "error",
                                error=str(r)[:150])
                has_error = True
            else:
                ticks[a] = r
                if r.error:
                    has_error = True

        if has_error:
            self._consecutive_errors += 1
            self._total_errors += 1
        else:
            self._consecutive_errors = 0

        return ticks

    def get_backoff(self) -> float:
        """指数退避: 连续错误越多等待越长, 上限16秒"""
        if self._consecutive_errors <= 1:
            return 0
        return min(BACKOFF_BASE * (2 ** (self._consecutive_errors - 2)), BACKOFF_MAX)

    @property
    def stats_str(self) -> str:
        return (f"polls:{self._total_polls} errs:{self._total_errors} "
                f"consec:{self._consecutive_errors}")

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# ==============================================================================
# FeatureEngine (与v1完全相同)
# ==============================================================================

class FeatureEngine:
    """
    从原始Tick数据计算ML特征

    1. Volume-Weighted Mid: vw_mid = sum(mid_i * vol_i) / sum(vol_i)
    2. Deviation: dev_i = mid_i - vw_mid
    3. Dispersion: std(mid_1, ..., mid_4)
    4. Momentum: mom_Ns = mid_now - mid_(now-N)
    5. Kelly Criterion: f* = (p*b - q) / b
    """

    def __init__(self):
        max_window = max(MOMENTUM_WINDOWS) + 5
        self._history: Dict[str, deque] = {
            a: deque(maxlen=max_window * 2) for a in ASSETS
        }
        self._vw_history: deque = deque(maxlen=max_window * 2)

    def compute(self, ticks: Dict[str, Tick]) -> TickFeatures:
        first = next(iter(ticks.values()))
        ts = first.ts
        ts_unix = first.ts_unix

        valid = {}
        for a, t in ticks.items():
            if t.mid is not None and t.error is None:
                valid[a] = (t.mid, t.volume)

        n_valid = len(valid)
        feat = TickFeatures(ts=ts, ts_unix=ts_unix, n_valid=n_valid)

        for a, t in ticks.items():
            prefix = a.lower()
            setattr(feat, f"{prefix}_mid", t.mid)
            setattr(feat, f"{prefix}_spread", t.spread)
            setattr(feat, f"{prefix}_vol", t.volume)

        if n_valid == 0:
            return feat

        # 1. Volume-Weighted Mid
        mids = [m for m, v in valid.values()]
        vols = [v for m, v in valid.values()]
        total_vol = sum(vols)

        if total_vol > 0:
            feat.vw_mid = sum(m * v for m, v in valid.values()) / total_vol
        else:
            feat.vw_mid = sum(mids) / len(mids)

        feat.equal_mid = sum(mids) / len(mids)

        # 2. Deviation
        devs = {}
        for a in ASSETS:
            t = ticks[a]
            if t.mid is not None and t.error is None and feat.vw_mid is not None:
                dev = t.mid - feat.vw_mid
                devs[a] = dev
                setattr(feat, f"{a.lower()}_dev", dev)

        # 3. Dispersion
        if len(mids) >= 2:
            mean_m = sum(mids) / len(mids)
            variance = sum((m - mean_m) ** 2 for m in mids) / len(mids)
            feat.dispersion = math.sqrt(variance)
            feat.range_spread = max(mids) - min(mids)

        if devs:
            abs_devs = {a: abs(d) for a, d in devs.items()}
            max_a = max(abs_devs, key=abs_devs.get)
            feat.max_abs_dev = abs_devs[max_a]
            feat.lagging_asset = max_a

        # 4. Momentum
        for a, t in ticks.items():
            if t.mid is not None and t.error is None:
                self._history[a].append((ts_unix, t.mid))
        if feat.vw_mid is not None:
            self._vw_history.append((ts_unix, feat.vw_mid))

        for a in ASSETS:
            for win in [5, 15]:
                mom = self._calc_momentum(self._history[a], ts_unix, win)
                setattr(feat, f"{a.lower()}_mom_{win}s", mom)

        for win in [5, 15]:
            setattr(feat, f"vw_mom_{win}s",
                    self._calc_momentum(self._vw_history, ts_unix, win))

        # 5. Signal Detection
        feat.signal = self._detect_signal(feat, devs)

        # 6. Kelly
        self._calc_kelly(feat, ticks, devs)

        return feat

    def _calc_momentum(self, history: deque, now: float, window: int) -> Optional[float]:
        target_ts = now - window
        best = None
        best_diff = float("inf")

        for ts, mid in history:
            diff = abs(ts - target_ts)
            if diff < best_diff:
                best_diff = diff
                best = mid

        if best is not None and best_diff <= window * 0.5:
            current = None
            for ts, mid in reversed(history):
                current = mid
                break
            if current is not None:
                return current - best
        return None

    def _detect_signal(self, feat: TickFeatures, devs: Dict[str, float]) -> str:
        if feat.vw_mid is None or feat.n_valid < 4:
            return "none"

        if devs:
            if feat.vw_mid >= 0.70:
                min_a = min(devs, key=devs.get)
                if devs[min_a] < -DEVIATION_THRESHOLD:
                    return "bullish_lag"
            elif feat.vw_mid <= 0.30:
                max_a = max(devs, key=devs.get)
                if devs[max_a] > DEVIATION_THRESHOLD:
                    return "bearish_lag"

        if feat.dispersion is not None and feat.dispersion > DEVIATION_THRESHOLD * 2:
            return "high_dispersion"
        return "none"

    def _calc_kelly(self, feat: TickFeatures, ticks: Dict[str, Tick],
                    devs: Dict[str, float]):
        if not devs or feat.lagging_asset == "" or feat.vw_mid is None:
            return

        lag_a = feat.lagging_asset
        lag_tick = ticks.get(lag_a)
        if lag_tick is None or lag_tick.yes_ask is None:
            return

        entry = lag_tick.yes_ask
        if entry <= 0 or entry >= 1:
            return

        b = (1.0 - entry) / entry
        abs_dev = abs(devs.get(lag_a, 0))
        p = 0.50 + min(abs_dev * 2.0, 0.30)
        q = 1.0 - p

        edge = p * b - q
        if edge <= 0:
            feat.kelly_frac = 0.0
            feat.kelly_usd = 0.0
            return

        f_star = edge / b
        f_actual = KELLY_FRAC * f_star
        f_actual = max(0, min(f_actual, 0.20))
        feat.kelly_frac = f_actual
        feat.kelly_usd = round(BANKROLL * f_actual, 2)


# ==============================================================================
# DataStore (与v1相同)
# ==============================================================================

class DataStore:
    def __init__(self, db_path: str = DB_PATH):
        self._path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def init(self):
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_ticks (
                ts TEXT, ts_unix REAL, asset TEXT, ticker TEXT, status TEXT,
                yes_bid REAL, yes_ask REAL, mid REAL, spread REAL,
                last_price REAL, volume INTEGER, close_time TEXT, error TEXT
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tick_features (
                ts TEXT, ts_unix REAL, n_valid INTEGER,
                btc_mid REAL, btc_spread REAL, btc_vol INTEGER,
                eth_mid REAL, eth_spread REAL, eth_vol INTEGER,
                sol_mid REAL, sol_spread REAL, sol_vol INTEGER,
                xrp_mid REAL, xrp_spread REAL, xrp_vol INTEGER,
                vw_mid REAL, equal_mid REAL,
                btc_dev REAL, eth_dev REAL, sol_dev REAL, xrp_dev REAL,
                dispersion REAL, range_spread REAL,
                max_abs_dev REAL, lagging_asset TEXT,
                btc_mom_5s REAL, btc_mom_15s REAL,
                eth_mom_5s REAL, eth_mom_15s REAL,
                sol_mom_5s REAL, sol_mom_15s REAL,
                xrp_mom_5s REAL, xrp_mom_15s REAL,
                vw_mom_5s REAL, vw_mom_15s REAL,
                kelly_frac REAL, kelly_usd REAL,
                signal TEXT
            )
        """)

        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticks_ts ON raw_ticks(ts_unix)")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feat_ts ON tick_features(ts_unix)")
        self._conn.commit()

    def save_ticks(self, ticks: Dict[str, Tick]):
        rows = []
        for a, t in ticks.items():
            rows.append((
                t.ts, t.ts_unix, t.asset, t.ticker, t.status,
                t.yes_bid, t.yes_ask, t.mid, t.spread,
                t.last_price, t.volume, t.close_time, t.error
            ))
        self._conn.executemany(
            "INSERT INTO raw_ticks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        self._conn.commit()

    def save_features(self, feat: TickFeatures):
        self._conn.execute(
            "INSERT INTO tick_features VALUES ("
            + ",".join(["?"] * 38) + ")",
            (
                feat.ts, feat.ts_unix, feat.n_valid,
                feat.btc_mid, feat.btc_spread, feat.btc_vol,
                feat.eth_mid, feat.eth_spread, feat.eth_vol,
                feat.sol_mid, feat.sol_spread, feat.sol_vol,
                feat.xrp_mid, feat.xrp_spread, feat.xrp_vol,
                feat.vw_mid, feat.equal_mid,
                feat.btc_dev, feat.eth_dev, feat.sol_dev, feat.xrp_dev,
                feat.dispersion, feat.range_spread,
                feat.max_abs_dev, feat.lagging_asset,
                feat.btc_mom_5s, feat.btc_mom_15s,
                feat.eth_mom_5s, feat.eth_mom_15s,
                feat.sol_mom_5s, feat.sol_mom_15s,
                feat.xrp_mom_5s, feat.xrp_mom_15s,
                feat.vw_mom_5s, feat.vw_mom_15s,
                feat.kelly_frac, feat.kelly_usd,
                feat.signal,
            )
        )
        self._conn.commit()

    def export_csv(self, path: str = CSV_EXPORT):
        cursor = self._conn.execute("SELECT * FROM tick_features ORDER BY ts_unix")
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        return len(rows)

    def row_count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM tick_features").fetchone()[0]

    def stats(self) -> dict:
        raw_n = self._conn.execute("SELECT COUNT(*) FROM raw_ticks").fetchone()[0]
        feat_n = self._conn.execute("SELECT COUNT(*) FROM tick_features").fetchone()[0]
        sig_n = self._conn.execute(
            "SELECT COUNT(*) FROM tick_features WHERE signal != 'none'"
        ).fetchone()[0]
        return {"raw_ticks": raw_n, "features": feat_n, "signals": sig_n}

    def close(self):
        if self._conn:
            self._conn.close()


# ==============================================================================
# Console v2 (紧凑输出 + 进度报告 + 实时gap检测)
# ==============================================================================

class Console:
    """
    v2改进:
    - 每50行才打印一次完整状态 (v1每行都打印, 刷屏严重)
    - 其他行只打印紧凑单行
    - 实时检测gap并警告
    - 每5分钟自动输出进度报告 (行数/gap/剩余时间/ML就绪评估)
    """

    def __init__(self):
        self._n = 0
        self._sig = 0
        self._last_progress = time.time()
        self._start_time = time.time()
        self._gap_count = 0
        self._gap_total_sec = 0
        self._last_ts_unix = 0

    def show(self, ticks: Dict[str, Tick], feat: TickFeatures):
        self._n += 1
        if feat.signal != "none":
            self._sig += 1

        # v2: 实时gap检测
        if self._last_ts_unix > 0:
            gap = feat.ts_unix - self._last_ts_unix
            if gap > 5.0:
                self._gap_count += 1
                self._gap_total_sec += gap
                print(f"\n  !! GAP #{self._gap_count}: {gap:.0f}s "
                      f"(total gap: {self._gap_total_sec:.0f}s)")
        self._last_ts_unix = feat.ts_unix

        # v2: 每50行完整输出, 其他行紧凑单行
        if self._n % 50 == 1 or self._n <= 3:
            self._show_full(ticks, feat)
        else:
            vw = f"{feat.vw_mid:.4f}" if feat.vw_mid else "---"
            dev = f"{feat.max_abs_dev:.4f}" if feat.max_abs_dev else "---"
            sig = f" [{feat.signal}]" if feat.signal != "none" else ""
            elapsed = time.time() - self._start_time
            print(f"\r  #{self._n:>6} VW={vw} Dev={dev} "
                  f"{elapsed/60:.0f}m/{MAX_RUNTIME/60:.0f}m "
                  f"gaps:{self._gap_count}{sig}     ",
                  end="", flush=True)

        # v2: 每5分钟进度报告
        now = time.time()
        if now - self._last_progress >= PROGRESS_INTERVAL:
            self._show_progress()
            self._last_progress = now

    def _show_full(self, ticks: Dict[str, Tick], feat: TickFeatures):
        elapsed = time.time() - self._start_time
        print(f"\n{'=' * W}")
        print(f"  #{self._n}  {feat.ts}  valid:{feat.n_valid}/4  "
              f"sigs:{self._sig}  {elapsed/60:.0f}m")
        print(f"{'=' * W}")

        print(f"  {'ASSET':<5}{'Mid':>7}{'Bid':>7}{'Ask':>7}{'Sprd':>7}"
              f"{'Vol':>8}{'Dev':>8}{'Mom5s':>8} FLAG")
        print(f"  {'-'*65}")

        for a in ASSETS:
            t = ticks[a]
            dev = getattr(feat, f"{a.lower()}_dev", None)
            mom = getattr(feat, f"{a.lower()}_mom_5s", None)

            if t.error:
                print(f"  {a:<5}{'---':>7}{'---':>7}{'---':>7}{'---':>7}"
                      f"{'---':>8}{'---':>8}{'---':>8} ERR:{t.error[:25]}")
            else:
                m_s = f"{t.mid:.4f}" if t.mid is not None else "---"
                b_s = f"{t.yes_bid:.2f}" if t.yes_bid is not None else "---"
                a_s = f"{t.yes_ask:.2f}" if t.yes_ask is not None else "---"
                sp_s = f"{t.spread:.3f}" if t.spread is not None else "---"
                v_s = str(t.volume)
                d_s = f"{dev:+.4f}" if dev is not None else "---"
                mm_s = f"{mom:+.4f}" if mom is not None else "---"

                flag = ""
                if dev is not None:
                    if abs(dev) > DEVIATION_THRESHOLD:
                        flag = "DEVN" if dev < 0 else "DEVP"
                    elif t.mid is not None and t.mid >= 0.80:
                        flag = "HIGH"
                    elif t.mid is not None and t.mid <= 0.20:
                        flag = "LOW"

                print(f"  {a:<5}{m_s:>7}{b_s:>7}{a_s:>7}{sp_s:>7}"
                      f"{v_s:>8}{d_s:>8}{mm_s:>8} {flag}")

        vw_s = f"{feat.vw_mid:.4f}" if feat.vw_mid is not None else "---"
        disp_s = f"{feat.dispersion:.4f}" if feat.dispersion is not None else "---"
        print(f"  VW={vw_s}  disp={disp_s}  signal={feat.signal}")

    def _show_progress(self):
        elapsed = time.time() - self._start_time
        remaining = MAX_RUNTIME - elapsed
        pct = elapsed / MAX_RUNTIME * 100
        gap_pct = (self._gap_total_sec / elapsed * 100) if elapsed > 0 else 0

        print(f"\n\n{'#' * 70}")
        print(f"  PROGRESS: {elapsed/60:.0f}m / {MAX_RUNTIME/60:.0f}m ({pct:.0f}%)")
        print(f"  rows={self._n}  signals={self._sig}  "
              f"gaps={self._gap_count} ({self._gap_total_sec:.0f}s = {gap_pct:.1f}%)")
        print(f"  remaining: {remaining/60:.0f} min")

        # 质量判定
        if gap_pct > 10:
            print(f"  !! GAP WARNING: {gap_pct:.1f}% > 10% -- check connection!")
        elif gap_pct > 5:
            print(f"  !  gap elevated: {gap_pct:.1f}%")
        else:
            print(f"  OK data quality (gap {gap_pct:.1f}%)")

        # ML就绪度
        if self._n >= 7200:
            print(f"  ** READY: {self._n} rows sufficient, can stop anytime")
        elif self._n >= 3600:
            print(f"  *  USABLE: {self._n} rows, more is better")
        else:
            need = 3600 - self._n
            eta = need * POLL_SEC
            print(f"  .. NEED MORE: {need} rows to go (~{eta/60:.0f} min)")
        print(f"{'#' * 70}\n")


# ==============================================================================
# Main Loop v2 (指数退避 + 自动恢复)
# ==============================================================================

async def main():
    print(f"\n{'=' * W}")
    print(f"  Kalshi Collector v2 | Long-Duration Stable Collection")
    print(f"{'=' * W}")
    print(f"  API:       {KALSHI_API}")
    print(f"  assets:    {', '.join(f'{k}={v}' for k,v in SERIES.items())}")
    print(f"  poll:      {POLL_SEC}s (v2: reduced rate limit risk)")
    print(f"  runtime:   {MAX_RUNTIME}s ({MAX_RUNTIME/3600:.1f} hours)")
    print(f"  session:   refresh every {SESSION_REFRESH_SEC}s")
    print(f"  bankroll:  ${BANKROLL}  kelly={KELLY_FRAC}")
    print(f"  storage:   {DB_PATH} + {CSV_EXPORT}")
    print(f"  Ctrl+C to stop early\n")

    poller = MarketPoller()
    engine = FeatureEngine()
    store = DataStore()
    store.init()
    console = Console()

    start_time = time.time()

    try:
        while True:
            t0 = asyncio.get_event_loop().time()

            # v2: 指数退避
            backoff = poller.get_backoff()
            if backoff > 0:
                print(f"\n  backoff {backoff:.1f}s "
                      f"(consecutive errors: {poller._consecutive_errors})")
                await asyncio.sleep(backoff)

            # 1. Poll
            ticks = await poller.poll_all()

            # 2. Check total failure
            all_error = all(t.error is not None for t in ticks.values())
            if all_error:
                if poller._consecutive_errors <= 3:
                    print(f"\n  all-fail, retrying... "
                          f"(consec:{poller._consecutive_errors})")
                continue

            # 3. Compute features
            feat = engine.compute(ticks)

            # 4. Store
            store.save_ticks(ticks)
            store.save_features(feat)

            # 5. Display
            console.show(ticks, feat)

            # 6. Time check
            if time.time() - start_time >= MAX_RUNTIME:
                print(f"\n\nTIME: reached {MAX_RUNTIME}s "
                      f"({MAX_RUNTIME/3600:.1f}h)")
                break

            # Sleep precisely
            elapsed = asyncio.get_event_loop().time() - t0
            await asyncio.sleep(max(0, POLL_SEC - elapsed))

    except KeyboardInterrupt:
        print(f"\nSTOPPED by user")
    except Exception as e:
        print(f"\nFATAL: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
    finally:
        n_rows = store.export_csv()
        stats = store.stats()
        elapsed = time.time() - start_time

        await poller.close()
        store.close()

        print(f"\n{'=' * W}")
        print(f"  COLLECTION COMPLETE (v2)")
        print(f"  runtime:   {elapsed:.0f}s ({elapsed/60:.1f}m)")
        print(f"  raw ticks: {stats['raw_ticks']}")
        print(f"  features:  {stats['features']} rows")
        print(f"  signals:   {stats['signals']}")
        print(f"  poller:    {poller.stats_str}")
        print(f"  gaps:      {console._gap_count} "
              f"({console._gap_total_sec:.0f}s total)")
        print(f"  sqlite:    {os.path.abspath(DB_PATH)}")
        print(f"  csv:       {os.path.abspath(CSV_EXPORT)} ({n_rows} rows)")
        print(f"{'=' * W}")

        if n_rows > 0:
            gap_pct = (console._gap_total_sec / elapsed * 100
                       if elapsed > 0 else 0)
            print(f"\n  QUALITY:")
            if n_rows >= 3600 and gap_pct < 5:
                print(f"  [OK] {n_rows} rows, gap {gap_pct:.1f}%")
                print(f"  -> Ready for kalshi_ml_v3.py!")
            elif n_rows >= 2000:
                print(f"  [PARTIAL] {n_rows} rows, gap {gap_pct:.1f}%")
                print(f"  -> Can try ML, interpret carefully")
            else:
                print(f"  [INSUFFICIENT] {n_rows} rows")
                print(f"  -> Need more data, continue collecting")


# ==============================================================================
# Entry
# ==============================================================================

def _in_nb():
    try:
        return get_ipython().__class__.__name__ in (
            "ZMQInteractiveShell", "Shell", "TerminalInteractiveShell")
    except NameError:
        return False

if _in_nb():
    import nest_asyncio; nest_asyncio.apply()
    asyncio.run(main())
else:
    if __name__ == "__main__":
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            sys.exit(0)
