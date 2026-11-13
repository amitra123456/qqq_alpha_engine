"""
Alpaca Markets Equity Bars & Snapshot Client
===========================================
Provides historical daily OHLCV bars (/v2/stocks/bars) and live equity
snapshots (/v2/stocks/snapshots) from Alpaca Markets.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv


class AlpacaClient:
    """Alpaca daily equity bars and snapshot client."""

    def __init__(self, env_path: Optional[Path] = None):
        if env_path and env_path.exists():
            load_dotenv(env_path)
        else:
            proj_root = Path(__file__).resolve().parents[2]
            load_dotenv(proj_root / ".env")
            load_dotenv(proj_root.parent / ".env")

        self.api_key = (
            os.getenv("APCA_API_KEY_ID")
            or os.getenv("ALPACA_API_KEY")
            or ""
        ).strip()
        self.secret_key = (
            os.getenv("APCA_API_SECRET_KEY")
            or os.getenv("ALPACA_SECRET_KEY")
            or ""
        ).strip()

        base = (
            os.getenv("APCA_API_BASE_URL")
            or os.getenv("ALPACA_BASE_URL")
            or "https://paper-api.alpaca.markets"
        ).strip().rstrip("/")
        if base.endswith("/v2"):
            base = base[:-3]
        self.trading_url = base
        self.data_url = "https://data.alpaca.markets"

        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Accept": "application/json",
            }
        )

    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def get_multi_bars(
        self,
        symbols: List[str],
        lookback_calendar_days: int = 400,
        timeframe: str = "1Day",
    ) -> Dict[str, pd.DataFrame]:
        """Fetches daily OHLCV bars for multiple symbols from Alpaca /v2/stocks/bars."""
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=lookback_calendar_days)
        start_iso = start_dt.strftime("%Y-%m-%dT00:00:00Z")

        result: Dict[str, List[Dict[str, Any]]] = {s: [] for s in symbols}
        batch_size = 50

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            page_token = None
            while True:
                params: Dict[str, Any] = {
                    "symbols": ",".join(batch),
                    "timeframe": timeframe,
                    "start": start_iso,
                    "limit": 10000,
                    "adjustment": "all",
                    "feed": "sip",
                }
                if page_token:
                    params["page_token"] = page_token

                resp = self.session.get(
                    f"{self.data_url}/v2/stocks/bars", params=params, timeout=20
                )
                if resp.status_code != 200:
                    params["feed"] = "iex"
                    resp = self.session.get(
                        f"{self.data_url}/v2/stocks/bars", params=params, timeout=20
                    )
                    if resp.status_code != 200:
                        break

                payload = resp.json()
                bars_map = payload.get("bars", {})
                for sym, bar_list in bars_map.items():
                    if sym in result:
                        result[sym].extend(bar_list)

                page_token = payload.get("next_page_token")
                if not page_token:
                    break

        dfs: Dict[str, pd.DataFrame] = {}
        for sym, rows in result.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["t"] = pd.to_datetime(df["t"])
            df = df.sort_values("t").drop_duplicates(subset=["t"]).reset_index(drop=True)
            df = df.rename(
                columns={
                    "t": "timestamp",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "vw": "vwap",
                    "n": "trade_count",
                }
            )
            dfs[sym] = df
        return dfs

    def get_stock_snapshots(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetches real-time quote, trade, and daily/prevDaily bar snapshots."""
        out: Dict[str, Dict[str, Any]] = {}
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            resp = self.session.get(
                f"{self.data_url}/v2/stocks/snapshots",
                params={"symbols": ",".join(batch)},
                timeout=15,
            )
            if resp.status_code == 200:
                out.update(resp.json())
        return out
