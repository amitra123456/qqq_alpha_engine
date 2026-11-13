"""
Quarterly line-up update — Big Eight Weather Plan
=================================================
Run once after each quarter-end (e.g. on the first trading day of January, April, July, October).

  1. Members on the quarter-end date = data/big8/current_members.json, undoing changes in data/big8/n100_changes.csv
     dated after it. Keep both files current with the index's announced changes before running.
  2. Market cap on the quarter-end = Yahoo's current market cap scaled by (quarter-end close / latest close).
     Yahoo's market cap covers every share class (e.g. all of Alphabet) and ADRs, so no manual adjustment is needed.
     The historical rows came from SEC share counts filed on or before each quarter-end; this live method is an
     approximation that is accurate when run shortly after the quarter-end.
  3. Appends the ranking to data/big8/quarterly_ranking.csv (refuses to overwrite a quarter unless --force)
     and downloads prices for the new top 8 into data/big8/prices.

Needs the yfinance package and internet access. Use --dry-run to print the ranking without writing anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.big8_data import RANKING, load_ranking, members_at, refresh_prices

SKIP = {"GOOG"}   # second Alphabet share class; GOOGL's market cap already covers the whole company


def rank_quarter(quarter_date: str) -> pd.DataFrame:
    import yfinance as yf
    rows = []
    for s in members_at(quarter_date):
        if s in SKIP:
            continue
        try:
            t = yf.Ticker(s)
            h = t.history(start=(pd.Timestamp(quarter_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), auto_adjust=False)
            h.index = h.index.tz_localize(None)
            at_q = h.loc[:quarter_date, "Close"]
            mcap_now = float(t.fast_info["market_cap"])
            if at_q.empty or not mcap_now:
                raise ValueError("no data")
            mcap_q = mcap_now * float(at_q.iloc[-1]) / float(h["Close"].iloc[-1])
            rows.append((quarter_date, s, round(float(at_q.iloc[-1]), 4), mcap_q / 1e9 / float(at_q.iloc[-1]), round(mcap_q / 1e9, 3)))
        except Exception as ex:
            print(f"  ! skipped {s}: {type(ex).__name__}")
    r = pd.DataFrame(rows, columns=["quarter_date", "symbol", "close", "shares_bn", "market_cap_bn"])
    r["rank"] = r["market_cap_bn"].rank(ascending=False, method="first").astype(int)
    return r.sort_values("rank")


def main() -> None:
    p = argparse.ArgumentParser(description="Add a quarter-end line-up for the Big Eight plan")
    p.add_argument("--quarter-date", required=True, help="Last trading day of the quarter, YYYY-MM-DD")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Replace an existing quarter")
    a = p.parse_args()
    r = rank_quarter(a.quarter_date)
    print(r.head(12).to_string(index=False))
    if a.dry_run:
        return
    old = load_ranking()
    if a.quarter_date in set(old.quarter_date):
        if not a.force:
            sys.exit(f"{a.quarter_date} already in {RANKING.name}; use --force to replace it")
        old = old[old.quarter_date != a.quarter_date]
    pd.concat([old, r]).sort_values(["quarter_date", "rank"]).to_csv(RANKING, index=False)
    top = list(r.nsmallest(8, "rank").symbol)
    refresh_prices(top)
    print(f"saved {a.quarter_date}: top 8 = {', '.join(top)}")


if __name__ == "__main__":
    main()
