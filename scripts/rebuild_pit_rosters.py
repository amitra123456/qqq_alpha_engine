"""
Rebuild Point-in-Time Nasdaq-100 (`QQQ`) Quarterly Cap-Weighted Rosters (`2016-Q2` -> `2026-Q3`)
==================================================================================================
Applies strict point-in-time constituent filtering and causal SEC EDGAR share counts:
  1. Multi-Interval Index Membership (`config/nasdaq100_pit_membership.json`):
     - Every candidate MUST have a verified Nasdaq-100 `[inclusion_date, removal_date]` interval
       containing `q_date` (`inc <= q_date <= rem`).
     - Dual share classes (`GOOG`) are deduplicated so Alphabet is counted once (`GOOGL`).
  2. Causal Market-Cap Ranking (`Close(q_date) * SplitAdj_Shares_filed_on_or_before(q_date)`):
     - Ranks constituents strictly from SEC EDGAR filings with `filed <= q_date`.
     - Records both Top-8 (`weight_top8_pct`) and Top-16 (`weight_pct`) proportional weights,
       quarter-end close (`close_q_date`), and PIT coverage (`pit_true_members`, `pit_members_with_data`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.clients.sec_edgar_client import SecEdgarClient

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
SIP_DIR = DATA_DIR / "cache" / "alpaca_sip_daily"

pit_json_path = CONFIG_DIR / "nasdaq100_pit_membership.json"
pit_db: Dict[str, dict] = json.loads(pit_json_path.read_text(encoding="utf-8")) if pit_json_path.exists() else {}


def is_constituent_on_date(sym: str, date_str: str) -> bool:
    meta = pit_db.get(sym)
    if not meta:
        return False
    return any(s <= date_str <= e for s, e in meta.get("intervals", []))


KNOWN_SPLITS: Dict[str, List[Tuple[str, float]]] = {
    "AAPL": [("2020-08-31", 4.0)],
    "TSLA": [("2020-08-31", 5.0), ("2022-08-25", 3.0)],
    "NVDA": [("2021-07-20", 4.0), ("2024-06-10", 10.0)],
    "AMZN": [("2022-06-06", 20.0)],
    "GOOGL": [("2022-07-18", 20.0)],
    "GOOG": [("2022-07-18", 20.0)],
    "AVGO": [("2024-07-15", 10.0)],
    "ODFL": [("2020-03-25", 1.5), ("2024-03-28", 2.0)],
    "CTAS": [("2024-09-12", 4.0)],
    "CPRT": [("2017-04-11", 2.0)],
    "FAST": [("2019-05-23", 2.0)],
}
SPLITS_CACHE: Dict[str, pd.Series] = {}


def get_split_factor(sym: str, as_of_date: str) -> float:
    if sym in KNOWN_SPLITS:
        fac = 1.0
        for s_date, s_ratio in KNOWN_SPLITS[sym]:
            if s_date > as_of_date:
                fac *= s_ratio
        return fac
    if sym not in SPLITS_CACHE:
        try:
            t = yf.Ticker(sym)
            SPLITS_CACHE[sym] = t.splits
        except Exception:
            SPLITS_CACHE[sym] = pd.Series(dtype=float)
    splits = SPLITS_CACHE[sym]
    if splits.empty:
        return 1.0
    try:
        ts = pd.Timestamp(as_of_date, tz=splits.index.tz)
        after = splits[splits.index > ts]
        fac = 1.0
        for s in after:
            if s > 0:
                fac *= s
        return fac
    except Exception:
        return 1.0


def compute_causal_mcap_billions(
    sec: SecEdgarClient,
    sym: str,
    meta: dict,
    close_px: float,
    as_of_date: str,
    offline_only: bool = True,
) -> float:
    """Computes split-adjusted causal market cap ($B) using SEC EDGAR filings with filed <= as_of_date."""
    cik = sec.get_cik(sym)
    if cik:
        facts = sec.get_company_facts(sym, offline_only=offline_only)
        if facts:
            us_gaap = facts.get("facts", {}).get("us-gaap", {})
            sh_series = sec._extract_merged_quarterly_series(
                us_gaap, sec.SHARES_TAGS, unit_key="shares", cutoff_filed=as_of_date
            )
            if sh_series:
                sh_val = sh_series[-1]["val"]
                as_of_sh_date = sh_series[-1].get("end") or sh_series[-1].get("filed") or as_of_date
                fac = get_split_factor(sym, as_of_sh_date)
                return float(close_px * (sh_val * fac) / 1e9)
    sh_b = float(meta.get("shares_split_adj_billions", 1.0))
    return float(close_px * sh_b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild PIT Nasdaq-100 quarterly cap-weighted rosters.")
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Read only cached SEC EDGAR files on disk without querying data.sec.gov.",
    )
    args = parser.parse_args()

    sec = SecEdgarClient()

    qqq_df = pd.read_csv(SIP_DIR / "QQQ.csv")
    qqq_df["dt"] = pd.to_datetime(qqq_df["date_str"])
    qqq_df = qqq_df.sort_values("dt").reset_index(drop=True)
    latest_bar_date = str(qqq_df["date_str"].max())

    quarter_dates: List[Tuple[str, str]] = []
    for y in range(2016, 2027):
        for m in [3, 6, 9, 12]:
            if y == 2016 and m == 3:
                continue
            if y == 2026 and m == 12:
                continue
            sub = qqq_df[(qqq_df["dt"].dt.year == y) & (qqq_df["dt"].dt.month == m)]
            if not sub.empty:
                quarter_dates.append((f"{y}-Q{m // 3}", str(sub["date_str"].iloc[-1])))

    bars_by_sym: Dict[str, pd.DataFrame] = {}
    for sym in set(pit_db.keys()) | {"QQQ"}:
        p = SIP_DIR / f"{sym}.csv"
        if p.exists():
            df_s = pd.read_csv(p)
            df_s = df_s.sort_values("date_str").reset_index(drop=True)
            df_s["timestamp"] = pd.to_datetime(df_s["date_str"])
            bars_by_sym[sym] = df_s

    print(f"Loaded {len(bars_by_sym)} price bar series. Rebuilding {len(quarter_dates)} quarters with causal PIT data...")

    roster_rows = []
    for q_idx, (q_label, q_date) in enumerate(quarter_dates):
        q_ts = pd.to_datetime(q_date)
        min_active_bar_date = (q_ts - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

        pit_true_members_count = sum(
            1
            for sym, meta in pit_db.items()
            if is_constituent_on_date(sym, q_date) and not (meta.get("is_dual_class_duplicate") or sym == "GOOG")
        )

        eligible = []
        for sym, meta in pit_db.items():
            if not is_constituent_on_date(sym, q_date):
                continue
            if meta.get("is_dual_class_duplicate") or sym == "GOOG":
                continue
            df_s = bars_by_sym.get(sym)
            if df_s is None:
                continue
            sub = df_s[df_s["date_str"] <= q_date].tail(380).reset_index(drop=True)
            if len(sub) < 60:
                continue
            if str(sub["date_str"].iloc[-1]) < min_active_bar_date:
                continue
            if (q_ts - sub["timestamp"].iloc[-60]).days > 105:
                continue

            close_q = float(sub["close"].iloc[-1])
            mcap_b = compute_causal_mcap_billions(
                sec, sym, meta, close_q, q_date, offline_only=args.offline
            )
            name = meta.get("name", sym)
            sector = meta.get("sector", "Technology")
            eligible.append((mcap_b, close_q, sym, name, sector))

        eligible.sort(key=lambda x: x[0], reverse=True)
        pit_members_with_data = len(eligible)

        top16 = eligible[:16]
        assert len(top16) == 16, f"Quarter {q_label} produced only {len(top16)} eligible stocks!"
        tot_top8_mcap = sum(x[0] for x in top16[:8])
        tot_top16_mcap = sum(x[0] for x in top16)

        for idx, (mcap_b, close_q, sym, name, sector) in enumerate(top16):
            rk = idx + 1
            w_top8 = round((mcap_b / tot_top8_mcap) * 100.0, 4) if rk <= 8 else 0.0
            w_top16 = round((mcap_b / tot_top16_mcap) * 100.0, 4)
            assert is_constituent_on_date(sym, q_date), f"PIT leak: {sym} on {q_date}"
            roster_rows.append({
                "quarter": q_label,
                "date": q_date,
                "rank": rk,
                "ticker": sym,
                "company": name,
                "sector": sector,
                "mcap_billions": round(mcap_b, 4),
                "close_q_date": round(close_q, 4),
                "weight_top8_pct": w_top8,
                "weight_pct": w_top16,
                "bucket": "PIT_TOP_8_MCAP" if rk <= 8 else "PIT_TOP_9_16_MCAP",
                "pit_true_members": pit_true_members_count,
                "pit_members_with_data": pit_members_with_data,
            })

        if (q_idx + 1) % 5 == 0 or q_idx == len(quarter_dates) - 1:
            leaders = [r["ticker"] for r in roster_rows[-16:-8]]
            print(
                f"  [{q_idx+1:2d}/{len(quarter_dates)}] {q_label} ({q_date}): "
                f"TrueMembers={pit_true_members_count} | WithData={pit_members_with_data} | Top-8={leaders}",
                flush=True,
            )

    df_roster = pd.DataFrame(roster_rows)
    out_csv = DATA_DIR / "point_in_time_rosters.csv"
    df_roster.to_csv(out_csv, index=False)
    print(f"\nExported {len(df_roster)} rows to {out_csv}")

    active_today = []
    min_latest_bar = (pd.to_datetime(latest_bar_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    for sym, meta in pit_db.items():
        if not is_constituent_on_date(sym, latest_bar_date):
            continue
        if meta.get("is_dual_class_duplicate") or sym == "GOOG":
            continue
        df_s = bars_by_sym.get(sym)
        if df_s is None or df_s.empty or str(df_s["date_str"].iloc[-1]) < min_latest_bar:
            continue
        c_now = float(df_s["close"].iloc[-1])
        mcap_b = compute_causal_mcap_billions(
            sec, sym, meta, c_now, latest_bar_date, offline_only=args.offline
        )
        active_today.append({
            "symbol": sym,
            "name": meta.get("name", sym),
            "sector": meta.get("sector", "Technology"),
            "mcap_b": mcap_b,
            "pit_intervals": meta.get("intervals", []),
        })

    total_mcap = sum(x["mcap_b"] for x in active_today) or 1.0
    active_sorted = sorted(active_today, key=lambda x: x["mcap_b"], reverse=True)
    mega8_live = [
        {
            "symbol": x["symbol"],
            "name": x["name"],
            "qqq_weight_pct": round(x["mcap_b"] / total_mcap * 100.0, 2),
            "sector": x["sector"],
            "pit_intervals": x["pit_intervals"],
        }
        for x in active_sorted[:8]
    ]
    alpha_live = [
        {
            "symbol": x["symbol"],
            "name": x["name"],
            "qqq_weight_pct": round(x["mcap_b"] / total_mcap * 100.0, 2),
            "sector": x["sector"],
            "pit_intervals": x["pit_intervals"],
        }
        for x in active_sorted[8:]
    ]

    uni_obj = {
        "benchmark": "QQQ",
        "as_of_constituent_date": latest_bar_date,
        "description": "Point-in-Time Nasdaq-100 Constituent Universe (Dual-Class Deduplicated, Causal Market Cap)",
        "mega_cap_top8": mega8_live,
        "alpha_universe": alpha_live,
    }
    uni_path = CONFIG_DIR / "universe_qqq.json"
    uni_path.write_text(json.dumps(uni_obj, indent=2), encoding="utf-8")
    print(f"Updated {uni_path} as of {latest_bar_date}: {len(mega8_live)} Top-8 + {len(alpha_live)} Remaining Constituents")


if __name__ == "__main__":
    main()
