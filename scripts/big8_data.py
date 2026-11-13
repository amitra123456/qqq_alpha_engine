"""
Data helpers for the Big Eight strategy (8 largest Nasdaq-100 members, cap-weighted, quarterly).

Files in data/big8/:
  quarterly_ranking.csv               every true Nasdaq-100 member at each quarter-end with its market cap and rank
                                      (close on/before the quarter-end x point-in-time share count; GOOG dropped, GOOGL = all Alphabet)
  prices/{SYM}.csv                    split-adjusted daily open/close + cash dividends (Yahoo) for every stock that ever made the top 8
  nasdaq100_membership_2010_2026.json point-in-time membership rebuilt from the index change log
  pit_shares_2010_2026.csv            share counts from SEC filings made on or before each quarter-end (ADR-equivalent for ADRs)
  current_members.json, n100_changes.csv, adr_equivalent_shares.json   inputs for extending membership with update_big8_lineup.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIG8 = PROJECT_ROOT / "data" / "big8"
PRICES = BIG8 / "prices"
RANKING = BIG8 / "quarterly_ranking.csv"


def load_ranking() -> pd.DataFrame:
    return pd.read_csv(RANKING, dtype={"quarter_date": str})


def load_lineups(top_n: int = 8, ranking: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """{quarter_date: DataFrame(symbol, market_cap_bn, weight)} for the top_n members by market cap, weights summing to 1."""
    r = load_ranking() if ranking is None else ranking
    out = {}
    for qd, g in r.groupby("quarter_date"):
        t = g.nsmallest(top_n, "rank")[["symbol", "market_cap_bn"]].reset_index(drop=True)
        t["weight"] = t.market_cap_bn / t.market_cap_bn.sum()
        out[qd] = t
    return dict(sorted(out.items()))


def load_prices(symbols: Iterable[str]) -> Dict[str, pd.DataFrame]:
    out = {}
    for s in symbols:
        p = PRICES / f"{s}.csv"
        if not p.exists():
            raise FileNotFoundError(f"missing price file {p}; run scripts/update_big8_lineup.py --refresh-prices")
        out[s] = pd.read_csv(p, dtype={"date_str": str}).set_index("date_str").sort_index()
    return out


def total_return_index(px: pd.DataFrame, calendar: List[str]) -> tuple[np.ndarray, np.ndarray]:
    """Total-return index (dividends reinvested) valued at each open and close of the calendar."""
    b = px.reindex(calendar)
    c = b["close"].ffill()
    o = b["open"].fillna(c)
    dv = b["dividends"].fillna(0.0)
    g = ((c + dv) / c.shift(1)).fillna(1.0)
    tc = g.cumprod()
    to = tc.shift(1) * (o + dv) / c.shift(1)
    to.iloc[0] = o.iloc[0] / c.iloc[0]
    return to.values, tc.values


def refresh_prices(symbols: Iterable[str], start: str = "2010-06-01") -> List[str]:
    """Re-download split-adjusted prices with dividends from Yahoo (needs the yfinance package and internet)."""
    import yfinance as yf
    PRICES.mkdir(parents=True, exist_ok=True)
    done = []
    for s in symbols:
        d = yf.Ticker(s).history(start=start, auto_adjust=False, actions=True)
        if len(d) < 50:
            print(f"  ! no prices for {s}")
            continue
        d.index = d.index.tz_localize(None)
        pd.DataFrame({"date_str": d.index.strftime("%Y-%m-%d"), "open": d["Open"].round(6).values, "close": d["Close"].round(6).values,
                      "dividends": d["Dividends"].values, "splits": d["Stock Splits"].values}).to_csv(PRICES / f"{s}.csv", index=False)
        done.append(s)
    return done


ALIAS = {"FB": "META", "GOOG": "GOOGL", "PCLN": "BKNG"}


def members_at(quarter_date: str) -> List[str]:
    """True Nasdaq-100 members on a date: current members, undoing every index change dated after it."""
    cur = json.loads((BIG8 / "current_members.json").read_text(encoding="utf-8"))
    ch = pd.read_csv(BIG8 / "n100_changes.csv", parse_dates=["date"]).sort_values("date", ascending=False)
    m = set(cur)
    for r in ch.itertuples():
        if r.date > pd.Timestamp(quarter_date):
            if isinstance(r.add, str) and r.add.strip():
                m.discard(r.add.strip())
            if isinstance(r.rem, str) and r.rem.strip():
                m.add(r.rem.strip())
    return sorted({ALIAS.get(x, x) for x in m})
