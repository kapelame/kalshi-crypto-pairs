"""Authenticated, read-only Kalshi WebSocket transport."""

import asyncio
import json
import logging

import aiohttp

from .auth import auth_headers_from_environment
from .sequence import SequenceValidator, channel_family
from .timeutil import iso_utc


LOGGER = logging.getLogger("kalshi.streaming.websocket")
WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"


def subscription_messages(tickers, start_id=1):
    tickers = list(tickers)
    return [
        {"id": start_id, "cmd": "subscribe", "params": {
            "channels": ["ticker", "trade"], "market_tickers": tickers}},
        {"id": start_id + 1, "cmd": "subscribe", "params": {
            "channels": ["orderbook_delta"], "market_tickers": tickers,
            "use_yes_price": True}},
        {"id": start_id + 2, "cmd": "subscribe", "params": {
            "channels": ["market_lifecycle_v2"]}},
    ]


class KalshiWebSocketClient:
    def __init__(self, tickers, on_message, on_state=None, stale_seconds=30,
                 dotenv_path=".env", session=None):
        self.tickers = list(tickers)
        self.on_message = on_message
        self.on_state = on_state
        self.stale_seconds = stale_seconds
        self.dotenv_path = dotenv_path
        self.session = session
        self._owns_session = session is None
        self._stop = asyncio.Event()
        self._ws = None
        self.connected = False
        self.reconnects = 0
        self.sequence = SequenceValidator()

    async def _state(self, connected):
        self.connected = connected
        if self.on_state:
            result = self.on_state(connected)
            if asyncio.iscoroutine(result):
                await result

    async def _subscribe(self, ws):
        for message in subscription_messages(self.tickers):
            await ws.send_str(json.dumps(message, separators=(",", ":")))

    async def run_connection(self, ws):
        self.sequence.reset()
        generation = self.sequence.generation
        await self._state(True)
        await self._subscribe(ws)
        try:
            while not self._stop.is_set():
                try:
                    message = await asyncio.wait_for(ws.receive(), self.stale_seconds)
                except asyncio.TimeoutError as exc:
                    raise ConnectionError(
                        f"no WebSocket message for {self.stale_seconds}s") from exc
                received_at = iso_utc()
                if message.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = json.loads(message.data)
                    except json.JSONDecodeError:
                        LOGGER.error("discarding malformed WebSocket JSON")
                        continue
                    healthy = self.sequence.observe(
                        payload.get("sid"), payload.get("seq"),
                        channel_family(payload.get("type")), generation)
                    await self.on_message(payload, received_at, healthy, generation)
                elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE,
                                      aiohttp.WSMsgType.ERROR):
                    raise ConnectionError(f"WebSocket closed: {message.type}")
        finally:
            await self._state(False)

    async def run_forever(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        backoff = 1.0
        first = True
        while not self._stop.is_set():
            if not first:
                self.reconnects += 1
            first = False
            try:
                headers = auth_headers_from_environment(self.dotenv_path)
                async with self.session.ws_connect(
                        WS_URL, headers=headers, autoping=True, heartbeat=10) as ws:
                    self._ws = ws
                    backoff = 1.0
                    await self.run_connection(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stop.is_set():
                    break
                LOGGER.warning("WebSocket reconnect after %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
        if self._owns_session and self.session and not self.session.closed:
            await self.session.close()

    async def replace_tickers(self, tickers):
        updated = list(tickers)
        if updated == self.tickers:
            return
        self.tickers = updated
        if self._ws is not None and not self._ws.closed:
            await self._ws.close(code=1000, message=b"contract rollover")

    async def request_reconnect(self, reason=b"recovery requested"):
        if self._ws is not None and not self._ws.closed:
            await self._ws.close(code=1012, message=reason)

    async def stop(self):
        self._stop.set()
        if self._ws is not None and not self._ws.closed:
            await self._ws.close(code=1000, message=b"graceful shutdown")
