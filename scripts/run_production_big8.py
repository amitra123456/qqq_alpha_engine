"""
Daily Signal & Order Ticket — Big Eight Weather Plan (no leverage)
==================================================================
Run after the 16:05 ET close. Uses the last COMPLETED daily close only.

  invested fraction = min(1.0, 20% / QQQ's 21-day realised volatility)   (config: big_eight)
  the invested part is split across the current quarter's 8 stocks by their (drifted) cap weights;
  the rest goes to a T-bill ETF (SGOV, BIL fallback).
  Trade at the next open only when the target differs from the fraction held by 10+ points.

Line-up: the latest quarter in data/big8/quarterly_ranking.csv. After each quarter-end, run
scripts/update_big8_lineup.py first; this script warns when the line-up is out of date.
Stock prices: data/big8/prices (refresh with --refresh-prices; needs yfinance). QQQ: Alpaca if configured, else the cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.big8_data import load_lineups, load_prices, refresh_prices
from scripts.run_production_voltarget import _offline_bars, target_exposure_from_closes

SIGNAL_FILE = "production_big8_signal.json"


def last_quarter_end(as_of: str) -> str:
    d = pd.Timestamp(as_of)
    return (d - pd.offsets.QuarterEnd(0) if d.is_quarter_end else d - pd.offsets.QuarterEnd(1)).strftime("%Y-%m-%d")


def drifted_weights(lineup: pd.DataFrame, prices: Dict[str, pd.DataFrame], trade_date: str, as_of: str) -> Dict[str, float]:
    """Cap weights set at the line-up's first open, drifted by each stock's total return to the as-of close."""
    w = {}
    for x in lineup.itertuples():
        p = prices[x.symbol]
        p_open = float(p.loc[p.index >= trade_date, "open"].iloc[0])
        sub = p.loc[(p.index >= trade_date) & (p.index <= as_of)]
        divs = float((sub["dividends"] / sub["close"]).sum())          # dividends reinvested, approximately
        w[x.symbol] = x.weight * float(sub["close"].iloc[-1]) / p_open * (1 + divs)
    tot = sum(w.values())
    return {s: v / tot for s, v in w.items()}


def run_production(capital: float = 100000.0, current_fraction: Optional[float] = None, as_of_override: Optional[str] = None,
                   refresh: bool = False, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    output_dir = output_dir or (PROJECT_ROOT / "data" / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((PROJECT_ROOT / "config" / "strategy_config.yaml").read_text(encoding="utf-8"))
    b8 = cfg["big_eight"]
    tv, cap, lb, band, top_n = float(b8["target_vol"]), float(b8["max_exposure"]), int(b8["lookback_days"]), float(b8["rebalance_deadband"]), int(b8["top_n"])
    cash_etf, cash_fb = str(b8.get("tbill_etf", "SGOV")), str(b8.get("tbill_etf_fallback", "BIL"))

    lineups = load_lineups(top_n)
    qd = list(lineups)[-1]
    lineup = lineups[qd]
    if refresh:
        refresh_prices(list(lineup.symbol))
    prices = load_prices(lineup.symbol)

    mode = "OFFLINE_CACHE"
    q = None
    if not as_of_override:
        try:
            from src.clients.alpaca_client import AlpacaClient
            al = AlpacaClient()
            if al.is_configured():
                d = al.get_multi_bars(["QQQ"], lookback_calendar_days=90).get("QQQ")
                if d is not None and len(d) > lb + 2:
                    from scripts.run_production_voltarget import _date_str
                    q, mode = _date_str(d), "LIVE_ALPACA_SIP"
        except Exception:
            q = None
    if q is None:
        q = _offline_bars("QQQ")
    if as_of_override:
        q = q[q["date_str"] <= as_of_override].reset_index(drop=True)
    as_of = str(q["date_str"].iloc[-1])
    stock_asof = min(str(p.index[-1]) for p in prices.values())
    sig = target_exposure_from_closes(q["close"].astype(float), tv, cap, lb)
    w = sig["target_exposure"]

    trade_dates = [d for d in q["date_str"] if d > qd] or [as_of]
    weights = drifted_weights(lineup, prices, trade_dates[0], min(as_of, stock_asof))
    warnings = []
    if last_quarter_end(as_of) > qd:
        warnings.append(f"Line-up is from {qd} but the quarter ended {last_quarter_end(as_of)}: run scripts/update_big8_lineup.py.")
    if stock_asof < as_of:
        warnings.append(f"Stock prices end {stock_asof}, QQQ ends {as_of}: run with --refresh-prices for current share counts.")

    prev = output_dir / SIGNAL_FILE
    source = "argument"
    if current_fraction is None and prev.exists():
        current_fraction = json.loads(prev.read_text(encoding="utf-8")).get("invested_after_orders")
        source = "previous saved signal"
    if current_fraction is None:
        source = "unknown (first run)"
    if current_fraction is None or abs(w - current_fraction) >= band:
        action, after = "REBALANCE_AT_NEXT_OPEN", w
    else:
        action, after = "HOLD_NO_TRADE", float(current_fraction)

    holdings = []
    for s, wt in sorted(weights.items(), key=lambda kv: -kv[1]):
        px = float(prices[s].loc[:as_of, "close"].iloc[-1])
        dollars = capital * after * wt
        holdings.append({"symbol": s, "role": "STOCK", "basket_weight_pct": round(wt * 100, 2), "target_weight_pct": round(after * wt * 100, 2),
                         "target_dollars": round(dollars, 2), "target_shares": round(dollars / px, 4), "spot_price": round(px, 4)})
    cb = _offline_bars(cash_etf)
    cash_sym = cash_etf if cb is not None else cash_fb
    cb = cb if cb is not None else _offline_bars(cash_fb)
    cpx = float(cb[cb["date_str"] <= as_of]["close"].iloc[-1]) if cb is not None else float("nan")
    holdings.append({"symbol": cash_sym, "role": "TBILL", "basket_weight_pct": None, "target_weight_pct": round((1 - after) * 100, 2),
                     "target_dollars": round(capital * (1 - after), 2), "target_shares": round(capital * (1 - after) / cpx, 4) if cpx == cpx else None,
                     "spot_price": round(cpx, 4) if cpx == cpx else None})
    payload = {"timestamp": datetime.now().isoformat(), "strategy": "Big Eight Weather Plan (no leverage)", "data_source_mode": mode,
               "bar_as_of_date": as_of, "stock_prices_as_of": stock_asof, "lineup_quarter": qd,
               "parameters": {"top_n": top_n, "target_vol": tv, "max_exposure": cap, "lookback_days": lb, "rebalance_deadband": band},
               "qqq_realised_vol_pct": round(sig["realised_vol"] * 100, 2), "target_invested": round(w, 4),
               "current_invested": None if current_fraction is None else round(float(current_fraction), 4), "current_invested_source": source,
               "action": action, "invested_after_orders": round(after, 4), "portfolio_equity": capital, "holdings": holdings, "warnings": warnings}
    (output_dir / SIGNAL_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 90)
    print(f"  BIG EIGHT WEATHER PLAN — signal from completed close {as_of} ({mode}); line-up {qd}")
    print("=" * 90)
    print(f"  QQQ {lb}-day vol {payload['qqq_realised_vol_pct']:.1f}% | target invested {w:.0%} | held "
          f"{'?' if current_fraction is None else f'{float(current_fraction):.0%}'} ({source}) | {action} -> {after:.0%}")
    for h in holdings:
        print(f"  {h['symbol']:<6} {h['target_weight_pct']:>6.2f}%  ${h['target_dollars']:>12,.2f}  shares {h['target_shares']}  @ {h['spot_price']}")
    for m in warnings:
        print(f"  ! {m}")
    print("=" * 90)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Big Eight Weather Plan daily signal")
    p.add_argument("--capital", type=float, default=100000.0)
    p.add_argument("--current-invested", type=float, default=None, help="Fraction currently in stocks, e.g. 0.65")
    p.add_argument("--as-of", type=str, default=None, help="Replay a historical close YYYY-MM-DD (offline cache)")
    p.add_argument("--refresh-prices", action="store_true", help="Download fresh prices for the 8 stocks first (yfinance)")
    a = p.parse_args()
    run_production(a.capital, a.current_invested, a.as_of, a.refresh_prices)


if __name__ == "__main__":
    main()
