"""
Rebuild data/big8/quarterly_ranking.csv from point-in-time membership, share counts and prices.

  market cap at quarter-end = last close on/before the quarter-end x share count from SEC filings made on/before it
  rules: GOOG dropped (GOOGL carries all Alphabet shares); a stock needs 60+ trading days of history and a close
         within 7 days of the quarter-end to be ranked (keeps fresh IPOs out until they have a track record).

Prices for all ~180 historical members are not committed (about 80 MB). Point --prices-dir at a folder of Yahoo
CSVs (Date, Close, ...) or let --download fetch them into data/cache/yahoo_long/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.big8_data import BIG8, RANKING

DEFAULT_PRICES = PROJECT_ROOT / "data" / "cache" / "yahoo_long"


def build(prices_dir: Path, download: bool = False) -> pd.DataFrame:
    mem = json.loads((BIG8 / "nasdaq100_membership_2010_2026.json").read_text(encoding="utf-8"))
    shs = pd.read_csv(BIG8 / "pit_shares_2010_2026.csv").set_index(["quarter_date", "symbol"])["shares_bn"]
    cache = {}

    def bars(s):
        p = prices_dir / f"{s}.csv"
        if not p.exists() and download:
            import yfinance as yf
            d = yf.Ticker(s).history(start="2009-06-01", auto_adjust=False, actions=True)
            if len(d) >= 50:
                d.index = d.index.tz_localize(None)
                prices_dir.mkdir(parents=True, exist_ok=True)
                d.to_csv(p)
        if not p.exists():
            return None
        x = pd.read_csv(p, index_col=0, parse_dates=True)
        x.index = x.index.strftime("%Y-%m-%d")
        return x

    rows = []
    for qd in sorted(mem):
        for s in mem[qd]:
            if s == "GOOG":
                continue
            if s not in cache:
                cache[s] = bars(s)
            b = cache[s]
            if b is None:
                continue
            bb = b[b.index <= qd]
            if len(bb) < 60 or bb.index[-1] < (pd.Timestamp(qd) - pd.Timedelta(days=7)).strftime("%Y-%m-%d"):
                continue
            n = shs.get((qd, s), np.nan)
            if np.isfinite(n):
                c = float(bb.Close.iloc[-1])
                rows.append((qd, s, round(c, 4), float(n), round(c * n, 3)))
    r = pd.DataFrame(rows, columns=["quarter_date", "symbol", "close", "shares_bn", "market_cap_bn"])
    r["rank"] = r.groupby("quarter_date")["market_cap_bn"].rank(ascending=False, method="first").astype(int)
    return r.sort_values(["quarter_date", "rank"])


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild the Big Eight quarterly ranking")
    p.add_argument("--prices-dir", type=Path, default=DEFAULT_PRICES)
    p.add_argument("--download", action="store_true", help="Download missing price files from Yahoo")
    a = p.parse_args()
    r = build(a.prices_dir, a.download)
    r.to_csv(RANKING, index=False)
    print(f"{len(r)} rows, {r.quarter_date.nunique()} quarters -> {RANKING}")


if __name__ == "__main__":
    main()
