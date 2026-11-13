"""
SEC EDGAR XBRL Share-Count Client
=================================
Extracts point-in-time share counts strictly filtered by filing date
(`filed <= cutoff_filed`) directly from US SEC EDGAR XBRL `data.sec.gov`
to avoid restatement and look-ahead bias in historical market-cap ranking.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class SecEdgarClient:
    """SEC EDGAR XBRL share-count client with local disk caching."""

    SHARES_TAGS = [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ]

    def __init__(self, cache_dir: Optional[Path] = None):
        proj_root = Path(__file__).resolve().parents[2]
        self.cache_dir = cache_dir or (proj_root / "data" / "cache" / "sec_edgar")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        ua = os.getenv("SEC_USER_AGENT", "QQQAlphaEngine research@antsox.com")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": ua,
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self._cik_map: Dict[str, str] = {}
        self._facts_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def _load_cik_map(self) -> Dict[str, str]:
        if self._cik_map:
            return self._cik_map

        map_file = self.cache_dir / "company_tickers.json"
        if map_file.exists() and (
            datetime.now() - datetime.fromtimestamp(map_file.stat().st_mtime)
        ) < timedelta(days=7):
            raw = json.loads(map_file.read_text(encoding="utf-8"))
        elif map_file.exists():
            try:
                resp = self.session.get("https://www.sec.gov/files/company_tickers.json", timeout=15)
                if resp.status_code == 200:
                    raw = resp.json()
                    map_file.write_text(json.dumps(raw), encoding="utf-8")
                else:
                    raw = json.loads(map_file.read_text(encoding="utf-8"))
            except Exception:
                raw = json.loads(map_file.read_text(encoding="utf-8"))
        else:
            try:
                resp = self.session.get("https://www.sec.gov/files/company_tickers.json", timeout=15)
                if resp.status_code == 200:
                    raw = resp.json()
                    map_file.write_text(json.dumps(raw), encoding="utf-8")
                else:
                    raw = {}
            except Exception:
                raw = {}

        for entry in raw.values():
            ticker = str(entry.get("ticker", "")).upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            if ticker and cik:
                self._cik_map[ticker] = cik
        return self._cik_map

    def get_cik(self, symbol: str) -> Optional[str]:
        cik_map = self._load_cik_map()
        return cik_map.get(symbol.upper())

    def get_company_facts(
        self, symbol: str, max_cache_days: int = 90, offline_only: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Fetches or loads cached SEC XBRL companyfacts JSON for `symbol`."""
        sym_u = symbol.upper()
        if sym_u in self._facts_cache:
            return self._facts_cache[sym_u]

        cik = self.get_cik(sym_u)
        if not cik:
            self._facts_cache[sym_u] = None
            return None

        cache_file = self.cache_dir / f"facts_{sym_u}_{cik}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                self._facts_cache[sym_u] = data
                return data
            except Exception:
                pass

        if offline_only:
            self._facts_cache[sym_u] = None
            return None

        time.sleep(0.12)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                cache_file.write_text(json.dumps(data), encoding="utf-8")
                self._facts_cache[sym_u] = data
                return data
        except Exception:
            pass
        self._facts_cache[sym_u] = None
        return None

    @staticmethod
    def _extract_merged_quarterly_series(
        us_gaap: Dict[str, Any],
        tags: List[str],
        unit_key: str = "shares",
        cutoff_filed: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts single-quarter (~70-108 day) values filtered by `filed <= cutoff_filed`,
        retaining the earliest filing for each period `end` to avoid restatement leakage.
        """
        by_end: Dict[str, Dict[str, Any]] = {}
        for tag in reversed(tags):
            concept = us_gaap.get(tag, {})
            items = concept.get("units", {}).get(unit_key, [])
            for item in items:
                start = item.get("start")
                end = item.get("end")
                filed = item.get("filed", "")
                val = item.get("val")
                form = item.get("form", "")
                if not start or not end or val is None or form not in ("10-Q", "10-K"):
                    continue
                if cutoff_filed and filed and filed > cutoff_filed:
                    continue
                try:
                    d_start = datetime.strptime(start, "%Y-%m-%d")
                    d_end = datetime.strptime(end, "%Y-%m-%d")
                    days = (d_end - d_start).days
                    if 70 <= days <= 108:
                        existing = by_end.get(end)
                        if existing is None or (filed and filed < existing.get("filed", "9999")):
                            by_end[end] = {"end": end, "val": float(val), "filed": filed}
                except Exception:
                    continue

        series = sorted(by_end.values(), key=lambda x: x["end"])
        return series[-40:]

    @staticmethod
    def _extract_point_or_avg_series(
        us_gaap: Dict[str, Any],
        dei: Dict[str, Any],
        tags: List[str],
        unit_key: str = "shares",
        cutoff_filed: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Extracts point-in-time share count series filtered by `filed <= cutoff_filed`."""
        by_end: Dict[str, Dict[str, Any]] = {}
        for source in (dei, us_gaap):
            for tag in reversed(tags):
                concept = source.get(tag, {})
                items = concept.get("units", {}).get(unit_key, [])
                for item in items:
                    end = item.get("end")
                    filed = item.get("filed", "")
                    val = item.get("val")
                    form = item.get("form", "")
                    if not end or val is None or form not in ("10-Q", "10-K") or float(val) <= 0:
                        continue
                    if cutoff_filed and filed and filed > cutoff_filed:
                        continue
                    existing = by_end.get(end)
                    if existing is None or (filed and filed < existing.get("filed", "9999")):
                        by_end[end] = {"end": end, "val": float(val), "filed": filed}
        series = sorted(by_end.values(), key=lambda x: x["end"])
        return series[-40:]
