"""
QQQ Alpha Engine — Live Daily Production Signal & Order Ticket Generator
========================================================================
Executes the un-fitted Strategy #1 rule configured in `config/strategy_config.yaml`:
  - Quarterly Point-in-Time Nasdaq-100 Cap-Weighted Equity Basket (`Top-8` default; `Top-16` supported)
    with dual-class deduplication (`GOOG` excluded, `GOOGL` kept) and intra-quarter cap-weight drift.
  - Daily QQQ 200-Day Simple Moving Average cash rule evaluated at the last COMPLETED daily close:
    * When QQQ Close > QQQ 200-Day SMA: 100% in the cap-weighted PIT equity basket.
    * When QQQ Close <= QQQ 200-Day SMA: 100% in US Treasury Bill ETF (`SGOV` / `BIL` fallback).
  - Zero leverage (`0% TQQQ`), zero option collars, zero margin borrowing; transitions execute at next open.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.rebuild_pit_rosters import compute_causal_mcap_billions, is_constituent_on_date
from src.clients.alpaca_client import AlpacaClient
from src.clients.sec_edgar_client import SecEdgarClient


def _add_date_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date_str" not in df.columns:
        dt_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
        ts = pd.to_datetime(df[dt_col])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("America/New_York")
        df["date_str"] = ts.dt.strftime("%Y-%m-%d")
    df = df.sort_values("date_str").drop_duplicates(subset=["date_str"]).reset_index(drop=True)
    return df


def _load_offline_cached_bars(symbols: List[str], as_of_override: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Loads daily bars from `data/cache/alpaca_sip_daily/` and pre-warmed QQQ history if needed."""
    cache_dir = PROJECT_ROOT / "data" / "cache" / "alpaca_sip_daily"
    tr_qqq_path = PROJECT_ROOT / "data" / "cache" / "qqq_total_return_1999_2026.csv"
    out: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        p = tr_qqq_path if (sym == "QQQ" and tr_qqq_path.exists()) else (cache_dir / f"{sym}.csv")
        if not p.exists():
            continue
        df = _add_date_str(pd.read_csv(p))
        if as_of_override:
            df = df[df["date_str"] <= as_of_override].reset_index(drop=True)
        df = df.tail(450).reset_index(drop=True)
        if not df.empty:
            out[sym] = df
    return out


def _latest_calendar_quarter_end(as_of_date: str) -> str:
    """Returns the most recent standard calendar quarter-end date (YYYY-MM-DD) <= as_of_date."""
    dt = pd.Timestamp(as_of_date)
    q_ends = []
    for y in (dt.year - 1, dt.year):
        for m, d in [(3, 31), (6, 30), (9, 30), (12, 31)]:
            cand = pd.Timestamp(year=y, month=m, day=d)
            if cand <= dt:
                q_ends.append(cand.strftime("%Y-%m-%d"))
    return max(q_ends)


def _resolve_quarter_roster(
    bar_as_of: str,
    bars_map: Dict[str, pd.DataFrame],
    pit_db: Dict[str, Any],
    sec: SecEdgarClient,
) -> Tuple[str, str, int, int, pd.DataFrame]:
    """
    Returns (quarter_label, quarter_end_date, pit_true_members, pit_members_with_data, roster_df_for_quarter)
    using `point_in_time_rosters.csv` or causal SEC EDGAR computation if a newer quarter-end has passed.
    """
    roster_path = PROJECT_ROOT / "data" / "point_in_time_rosters.csv"
    latest_cal_qe = _latest_calendar_quarter_end(bar_as_of)

    if roster_path.exists():
        rost = pd.read_csv(roster_path)
        rost["date_str"] = pd.to_datetime(rost["date"]).dt.strftime("%Y-%m-%d")
        sub = rost[rost["date_str"] <= bar_as_of]
        if not sub.empty:
            last_csv_qe = str(sub["date_str"].max())
            if last_csv_qe[:7] >= latest_cal_qe[:7]:
                q_rows = sub[sub["date_str"] == last_csv_qe].sort_values("rank").reset_index(drop=True)
                q_lbl = str(q_rows["quarter"].iloc[0])
                n_true = int(q_rows["pit_true_members"].iloc[0]) if "pit_true_members" in q_rows.columns else len(pit_db)
                n_data = int(q_rows["pit_members_with_data"].iloc[0]) if "pit_members_with_data" in q_rows.columns else 0
                return q_lbl, last_csv_qe, n_true, n_data, q_rows

    q_bars = bars_map["QQQ"]
    q_dates_before = q_bars[q_bars["date_str"] <= latest_cal_qe]["date_str"]
    qe_trading_date = str(q_dates_before.iloc[-1]) if not q_dates_before.empty else bar_as_of
    dt_qe = pd.Timestamp(qe_trading_date)
    q_lbl = f"{dt_qe.year}-Q{(dt_qe.month - 1) // 3 + 1}"

    n_true = sum(
        1 for sym, meta in pit_db.items()
        if is_constituent_on_date(sym, qe_trading_date) and not (meta.get("is_dual_class_duplicate") or sym == "GOOG")
    )
    eligible = []
    for sym, meta in pit_db.items():
        if not is_constituent_on_date(sym, qe_trading_date):
            continue
        if meta.get("is_dual_class_duplicate") or sym == "GOOG":
            continue
        s_df = bars_map.get(sym)
        if s_df is None:
            continue
        s_sub = s_df[s_df["date_str"] <= qe_trading_date]
        if s_sub.empty:
            continue
        close_q = float(s_sub["close"].iloc[-1])
        mcap_b = compute_causal_mcap_billions(sec, sym, meta, close_q, qe_trading_date, offline_only=True)
        eligible.append((mcap_b, close_q, sym, meta.get("name", sym), meta.get("sector", "Technology")))

    eligible.sort(key=lambda x: x[0], reverse=True)
    top16 = eligible[:16]
    tot8 = sum(x[0] for x in top16[:8]) or 1.0
    tot16 = sum(x[0] for x in top16) or 1.0
    rows = []
    for idx, (mcap_b, close_q, sym, name, sector) in enumerate(top16):
        rk = idx + 1
        rows.append({
            "quarter": q_lbl,
            "date_str": qe_trading_date,
            "rank": rk,
            "ticker": sym,
            "company": name,
            "sector": sector,
            "mcap_billions": round(mcap_b, 4),
            "close_q_date": round(close_q, 4),
            "weight_top8_pct": round((mcap_b / tot8) * 100.0, 4) if rk <= 8 else 0.0,
            "weight_pct": round((mcap_b / tot16) * 100.0, 4),
        })
    return q_lbl, qe_trading_date, n_true, len(eligible), pd.DataFrame(rows)


def _build_equity_holdings(
    q_rows: pd.DataFrame,
    top_k: int,
    capital: float,
    bars_map: Dict[str, pd.DataFrame],
    pit_db: Dict[str, Any],
    bar_as_of: str,
    action_status: str,
) -> List[Dict[str, Any]]:
    """Computes intra-quarter drifted market-cap weights and next-open share targets for Top-K."""
    sub = q_rows.sort_values("rank").iloc[:top_k].copy()
    base_col = "weight_top8_pct" if (top_k == 8 and "weight_top8_pct" in sub.columns) else "weight_pct"
    base_sum = float(sub[base_col].sum()) or 1.0

    raw_items = []
    bar_ts = pd.to_datetime(bar_as_of)
    for _, row in sub.iterrows():
        sym = str(row["ticker"])
        if pit_db and not is_constituent_on_date(sym, bar_as_of):
            continue
        s_bars = bars_map.get(sym)
        if s_bars is None or s_bars.empty:
            continue
        last_ts = pd.to_datetime(s_bars["date_str"].iloc[-1])
        if (bar_ts - last_ts).days > 7:
            continue

        spot = float(s_bars["close"].iloc[-1])
        qe_w_pct = (float(row[base_col]) / base_sum) * 100.0
        close_qe = float(row["close_q_date"]) if ("close_q_date" in row and pd.notna(row["close_q_date"]) and float(row["close_q_date"]) > 0) else spot
        drifted_raw = qe_w_pct * (spot / close_qe)
        raw_items.append({
            "symbol": sym,
            "name": str(row.get("company", sym)),
            "sector": str(row.get("sector", "Technology")),
            "rank": int(row["rank"]),
            "qe_base_weight_pct": round(qe_w_pct, 2),
            "drifted_raw": drifted_raw,
            "spot_price": round(spot, 4),
        })

    tot_drift = sum(x["drifted_raw"] for x in raw_items) or 1.0
    holdings: List[Dict[str, Any]] = []
    for item in raw_items:
        tgt_w_pct = round((item["drifted_raw"] / tot_drift) * 100.0, 2)
        tgt_dollars = round(capital * (tgt_w_pct / 100.0), 2)
        tgt_shares = int(tgt_dollars / max(item["spot_price"], 0.01))
        holdings.append({
            "symbol": item["symbol"],
            "name": item["name"],
            "bucket": f"PIT_TOP_{top_k}_MCAP",
            "sector": item["sector"],
            "rank": item["rank"],
            "qe_base_weight_pct": item["qe_base_weight_pct"],
            "target_weight_pct": tgt_w_pct,
            "target_dollars": tgt_dollars,
            "target_shares": tgt_shares,
            "spot_price": item["spot_price"],
            "execution_order_type": "09:30 ET Official Open",
            "status": action_status,
            "routing_destination": f"Exit to T-Bills ({action_status}) if QQQ Close <= SMA200",
        })
    return holdings


def run_production(
    capital: float = 100000.0,
    top_k: Optional[int] = None,
    cash_etf: Optional[str] = None,
    output_dir: Optional[Path] = None,
    db_path: Optional[Path] = None,
    as_of_override: Optional[str] = None,
    allow_intraday_bar: bool = False,
) -> Dict[str, Any]:
    output_dir = output_dir or (PROJECT_ROOT / "data" / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_path or (PROJECT_ROOT / "data" / "qqq_alpha_engine.db")
    config_dir = PROJECT_ROOT / "config"

    strat_cfg = yaml.safe_load((config_dir / "strategy_config.yaml").read_text(encoding="utf-8"))
    port_cfg = strat_cfg.get("portfolio_rules", {})
    cash_cfg = strat_cfg.get("cash_management", {})
    trend_cfg = strat_cfg.get("trend_filter", {})

    top_k = int(top_k if top_k is not None else port_cfg.get("default_top_k", 8))
    if top_k not in (8, 16):
        raise ValueError(f"Unsupported top_k={top_k}. Supported values: 8, 16.")
    cash_etf = (cash_etf or cash_cfg.get("live_tbill_etf", "SGOV")).upper()
    fallback_cash_etf = str(cash_cfg.get("fallback_tbill_etf", "BIL")).upper()

    pit_path = config_dir / "nasdaq100_pit_membership.json"
    pit_db: Dict[str, Any] = json.loads(pit_path.read_text(encoding="utf-8")) if pit_path.exists() else {}

    alpaca = AlpacaClient()
    sec = SecEdgarClient()

    active_candidates = [
        sym for sym, meta in pit_db.items()
        if not (meta.get("is_dual_class_duplicate") or sym == "GOOG")
        and is_constituent_on_date(sym, as_of_override or datetime.now().strftime("%Y-%m-%d"))
    ]
    all_syms = sorted(set(active_candidates) | {"QQQ", cash_etf, fallback_cash_etf})

    bars_map: Dict[str, pd.DataFrame] = {}
    data_source_mode = "LIVE_ALPACA_SIP"
    if alpaca.is_configured() and not as_of_override:
        try:
            raw_map = alpaca.get_multi_bars(all_syms, lookback_calendar_days=420)
            for sym, df_raw in raw_map.items():
                if not df_raw.empty:
                    bars_map[sym] = _add_date_str(df_raw)
        except Exception:
            bars_map = {}

    if "QQQ" not in bars_map or len(bars_map.get("QQQ", [])) < 200:
        data_source_mode = "OFFLINE_CACHED_ALPACA_SIP"
        bars_map = _load_offline_cached_bars(all_syms, as_of_override=as_of_override)

    if as_of_override:
        for sym in list(bars_map.keys()):
            df_s = bars_map[sym]
            bars_map[sym] = df_s[df_s["date_str"] <= as_of_override].reset_index(drop=True)

    intraday_preview: Optional[Dict[str, Any]] = None
    if as_of_override is None and not allow_intraday_bar and "QQQ" in bars_map and len(bars_map["QQQ"]) >= 201:
        now_et = datetime.now(ZoneInfo("America/New_York"))
        today_et = now_et.strftime("%Y-%m-%d")
        q_raw = bars_map["QQQ"]
        if str(q_raw["date_str"].iloc[-1]) == today_et and now_et.time() < dtime(16, 5):
            prov_sma200 = float(q_raw["close"].rolling(200, min_periods=200).mean().iloc[-1])
            prov_close = float(q_raw["close"].iloc[-1])
            intraday_preview = {
                "intraday_date": today_et,
                "et_timestamp": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
                "qqq_last_price": round(prov_close, 2),
                "provisional_sma200": round(prov_sma200, 2),
                "provisional_above_sma200": bool(prov_close > prov_sma200),
                "note": "Incomplete intraday session (<16:05 ET); official signal uses last completed close.",
            }
            for sym in list(bars_map.keys()):
                df_s = bars_map[sym]
                bars_map[sym] = df_s[df_s["date_str"] != today_et].reset_index(drop=True)

    q_bars = bars_map.get("QQQ")
    if q_bars is None or len(q_bars) < 200:
        raise RuntimeError(f"Insufficient QQQ daily bars for 200-day SMA calculation (found {0 if q_bars is None else len(q_bars)}).")

    ma_win = int(trend_cfg.get("ma_window_days", 200))
    q_sma200_series = q_bars["close"].rolling(ma_win, min_periods=ma_win).mean()

    bar_as_of = str(q_bars["date_str"].iloc[-1])
    prev_bar_date = str(q_bars["date_str"].iloc[-2])

    q_close = float(q_bars["close"].iloc[-1])
    q_sma200 = float(q_sma200_series.iloc[-1])
    curr_above = bool(q_close > q_sma200)

    q_prev_close = float(q_bars["close"].iloc[-2])
    q_prev_sma200 = float(q_sma200_series.iloc[-2])
    prev_above = bool(q_prev_close > q_prev_sma200)

    q_lbl, qe_date, pit_true_members, pit_members_with_data, q_rows = _resolve_quarter_roster(
        bar_as_of, bars_map, pit_db, sec
    )
    is_new_quarter_rebal = (bar_as_of == qe_date)

    if curr_above and not prev_above:
        transition_action = "ENTER_EQUITIES_AT_NEXT_OPEN"
    elif (not curr_above) and prev_above:
        transition_action = "EXIT_EQUITIES_TO_TBILLS_AT_NEXT_OPEN"
    elif curr_above and is_new_quarter_rebal:
        transition_action = "QUARTERLY_RECONSTITUTION_AT_NEXT_OPEN"
    elif curr_above and prev_above:
        transition_action = "HOLD_EQUITIES_NO_TRADE"
    else:
        transition_action = "HOLD_TBILLS_NO_TRADE"

    regime_state = f"EQUITIES_PIT_TOP{top_k}" if curr_above else f"TBILLS_100PCT_{cash_etf}"

    eq_holdings_k = _build_equity_holdings(q_rows, top_k, capital, bars_map, pit_db, bar_as_of, transition_action)
    alt_k = 16 if top_k == 8 else 8
    companion_holdings = _build_equity_holdings(q_rows, alt_k, capital, bars_map, pit_db, bar_as_of, transition_action)

    if curr_above:
        final_holdings = eq_holdings_k
    else:
        use_etf = cash_etf if (cash_etf in bars_map and not bars_map[cash_etf].empty) else fallback_cash_etf
        c_bars = bars_map.get(use_etf)
        c_spot = float(c_bars["close"].iloc[-1]) if (c_bars is not None and not c_bars.empty) else 100.0
        final_holdings = [{
            "symbol": use_etf,
            "name": "iShares 0-3 Month Treasury Bond ETF" if use_etf == "SGOV" else "SPDR Bloomberg 1-3 Month T-Bill ETF",
            "bucket": "TBILL_CASH_SHIELD",
            "sector": "US Treasury Bills",
            "rank": 1,
            "qe_base_weight_pct": 100.0,
            "target_weight_pct": 100.0,
            "target_dollars": round(capital, 2),
            "target_shares": int(capital / max(c_spot, 0.01)),
            "spot_price": round(c_spot, 4),
            "execution_order_type": "09:30 ET Official Open",
            "status": transition_action,
            "routing_destination": f"Re-enter PIT Top-{top_k} at next open if QQQ Close > SMA200",
        }]

    dist_pct = ((q_close / q_sma200) - 1.0) * 100.0
    print("=" * 116)
    print(f"  QQQ ALPHA ENGINE — QUARTERLY PIT TOP-{top_k} CAP-WEIGHT + QQQ 200-DAY MA CASH RULE")
    print(f"  Completed Close As-Of: {bar_as_of} ({data_source_mode}) | Capital: ${capital:,.2f}")
    print("=" * 116)
    print(f"  QQQ Close / SMA200  : ${q_close:.2f} vs ${q_sma200:.2f} ({dist_pct:+.2f}%) | Above SMA200: {curr_above} (Prev {prev_bar_date}: {prev_above})")
    print(f"  Active Regime / Act : {regime_state} | Action: {transition_action}")
    print(f"  Active PIT Quarter  : {q_lbl} (Quarter-End: {qe_date}) | PIT True Members: {pit_true_members} (With Data: {pit_members_with_data})")
    if intraday_preview:
        print(
            f"  Intraday Preview    : [{intraday_preview['et_timestamp']}] QQQ ${intraday_preview['qqq_last_price']:.2f} "
            f"vs Prov SMA200 ${intraday_preview['provisional_sma200']:.2f} (Above={intraday_preview['provisional_above_sma200']}) — NOT ACTED ON BEFORE 16:05 ET"
        )
    print("-" * 116)
    print(f"{'Symbol':<6} {'Bucket':<18} {'Sector':<20} {'QE Wt%':>7} {'DriftWt%':>9} {'Allocation':>12} {'Shares':>8} {'Spot':>9} {'Status'}")
    print("-" * 116)
    for h in final_holdings:
        print(
            f"{h['symbol']:<6} {h['bucket']:<18} {h['sector'][:19]:<20} "
            f"{h['qe_base_weight_pct']:>6.2f}% {h['target_weight_pct']:>8.2f}% "
            f"${h['target_dollars']:>10,.2f} {h['target_shares']:>8} ${h['spot_price']:>8.2f} {h['status']}"
        )
    print("=" * 116)

    out_json = output_dir / "production_daily_signals.json"
    payload = {
        "timestamp": datetime.now().isoformat(),
        "architecture": f"PIT_TOP{top_k}_CAPWEIGHT_QQQ_200DMA_TBILLS",
        "strategy_name": strat_cfg.get("strategy_name", "Quarterly Point-in-Time Cap-Weight + QQQ 200-Day Moving Average Cash Rule"),
        "data_source_mode": data_source_mode,
        "bar_as_of_date": bar_as_of,
        "prev_bar_date": prev_bar_date,
        "intraday_preview": intraday_preview,
        "active_quarter": q_lbl,
        "active_quarter_end_date": qe_date,
        "pit_true_members": pit_true_members,
        "pit_members_with_data": pit_members_with_data,
        "top_k": top_k,
        "cash_etf": cash_etf,
        "portfolio_equity": capital,
        "regime_state": regime_state,
        "transition_action": transition_action,
        "execution_order_type": "09:30 ET Official Open",
        "qqq_macro": {
            "spot": round(q_close, 4),
            "sma200": round(q_sma200, 4),
            "dist_sma200_pct": round(dist_pct, 2),
            "prev_spot": round(q_prev_close, 4),
            "prev_sma200": round(q_prev_sma200, 4),
            "prev_above_sma200": prev_above,
            "above_sma200": curr_above,
            "bull_regime": curr_above,
            "cash_reserve_pct": 0.0 if curr_above else 100.0,
        },
        "holdings": final_holdings,
        "companion_variant_top_k": alt_k,
        "companion_variant_holdings": companion_holdings,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS live_target_portfolio")
    cur.execute("""
        CREATE TABLE live_target_portfolio (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            bucket TEXT,
            sector TEXT,
            rank INTEGER,
            qe_base_weight_pct REAL,
            target_weight_pct REAL,
            target_dollars REAL,
            target_shares INTEGER,
            spot_price REAL,
            execution_order_type TEXT,
            status TEXT,
            routing_destination TEXT
        )
    """)
    for h in final_holdings:
        cur.execute(
            "INSERT INTO live_target_portfolio VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                h["symbol"],
                h["name"],
                h["bucket"],
                h["sector"],
                h["rank"],
                h["qe_base_weight_pct"],
                h["target_weight_pct"],
                h["target_dollars"],
                h["target_shares"],
                h["spot_price"],
                h["execution_order_type"],
                h["status"],
                h["routing_destination"],
            ),
        )
    conn.commit()
    conn.close()
    print(f"Saved JSON -> {out_json}\nUpdated SQLite live_target_portfolio -> {db_path} ({len(final_holdings)} rows)")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Quarterly PIT Cap-Weight + QQQ 200-Day MA Daily Production Signal")
    parser.add_argument("--capital", type=float, default=100000.0, help="Total portfolio equity ($)")
    parser.add_argument("--top-k", type=int, choices=[8, 16], default=None, help="PIT Top-K basket size (8 or 16)")
    parser.add_argument("--cash-etf", type=str, default=None, help="Live T-Bill ETF ticker (default SGOV)")
    parser.add_argument("--as-of", type=str, default=None, help="Optional YYYY-MM-DD historical replay date")
    parser.add_argument("--allow-intraday-bar", action="store_true", default=False, help="Use incomplete intraday bar if market open")
    args = parser.parse_args()

    run_production(
        capital=args.capital,
        top_k=args.top_k,
        cash_etf=args.cash_etf,
        as_of_override=args.as_of,
        allow_intraday_bar=args.allow_intraday_bar,
    )


if __name__ == "__main__":
    main()
