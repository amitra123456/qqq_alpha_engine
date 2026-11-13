"""
Daily Signal & Order Ticket Generator — Volatility-Target QQQ (strategy v6.0.0)
=============================================================================
Run after the 16:05 ET close. Uses the last COMPLETED daily close only.

Rule (config/strategy_config.yaml -> volatility_target):
  target exposure = min(max_exposure, target_vol / trailing realised volatility of QQQ daily returns)
  - exposure <= 1.0x : that fraction in QQQ, the rest in a T-bill ETF (SGOV, BIL fallback)
  - exposure  > 1.0x : (2 - exposure) in QQQ and (exposure - 1) in QLD (2x daily Nasdaq-100 ETF); no margin
  - trade at the next open only if the target differs from the exposure currently held by at least the
    deadband (10 points), or crosses 1.0x; otherwise hold.

Current exposure comes from --current-exposure, or from holdings given as dollar values
(--qqq-value / --qld-value / --cash-value), or else from the previous run's saved target.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.clients.alpaca_client import AlpacaClient

CACHE = PROJECT_ROOT / "data" / "cache"
SIGNAL_FILE = "production_voltarget_signal.json"


def _date_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date_str" not in df.columns:
        col = "timestamp" if "timestamp" in df.columns else df.columns[0]
        ts = pd.to_datetime(df[col])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("America/New_York")
        df["date_str"] = ts.dt.strftime("%Y-%m-%d")
    return df.sort_values("date_str").drop_duplicates("date_str").reset_index(drop=True)


def _offline_bars(sym: str) -> Optional[pd.DataFrame]:
    for p in ((CACHE / "qqq_total_return_1999_2026.csv") if sym == "QQQ" else None,
              CACHE / "alpaca_sip_daily" / f"{sym}.csv", CACHE / "yahoo_daily" / f"{sym}.csv"):
        if p is not None and p.exists():
            return _date_str(pd.read_csv(p))
    return None


def target_exposure_from_closes(closes: pd.Series, target_vol: float, max_exposure: float, lookback: int) -> Dict[str, float]:
    rets = closes.pct_change().dropna()
    if len(rets) < lookback:
        raise RuntimeError(f"Need at least {lookback + 1} daily closes, found {len(closes)}")
    vol = float(rets.iloc[-lookback:].std() * np.sqrt(252.0))
    return {"realised_vol": vol, "target_exposure": float(min(max_exposure, target_vol / vol))}


def split_exposure(w: float) -> Dict[str, float]:
    """Portfolio weights (fractions of equity) that deliver exposure w using QQQ, QLD (2x) and T-bills."""
    if w <= 1.0:
        return {"QQQ": w, "QLD": 0.0, "TBILL": 1.0 - w}
    a = w - 1.0
    return {"QQQ": 1.0 - a, "QLD": a, "TBILL": 0.0}


def run_production(capital: float = 100000.0, current_exposure: Optional[float] = None,
                   holdings_values: Optional[Dict[str, float]] = None, output_dir: Optional[Path] = None,
                   db_path: Optional[Path] = None, as_of_override: Optional[str] = None,
                   allow_intraday_bar: bool = False) -> Dict[str, Any]:
    output_dir = output_dir or (PROJECT_ROOT / "data" / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_path or (PROJECT_ROOT / "data" / "qqq_alpha_engine.db")
    cfg = yaml.safe_load((PROJECT_ROOT / "config" / "strategy_config.yaml").read_text(encoding="utf-8"))
    vt = cfg["volatility_target"]
    tv, cap, lb, band = float(vt["target_vol"]), float(vt["max_exposure"]), int(vt["lookback_days"]), float(vt["rebalance_deadband"])
    lev_etf = str(vt.get("leverage_etf", "QLD")).upper()
    cash_etf = str(vt.get("tbill_etf", "SGOV")).upper()
    cash_fb = str(vt.get("tbill_etf_fallback", "BIL")).upper()

    syms = ["QQQ", lev_etf, cash_etf, cash_fb]
    bars: Dict[str, pd.DataFrame] = {}
    mode = "LIVE_ALPACA_SIP"
    alpaca = AlpacaClient()
    if alpaca.is_configured() and not as_of_override:
        try:
            for s, d in alpaca.get_multi_bars(syms, lookback_calendar_days=90).items():
                if not d.empty:
                    bars[s] = _date_str(d)
        except Exception:
            bars = {}
    if "QQQ" not in bars or len(bars["QQQ"]) < lb + 2:
        mode = "OFFLINE_CACHE"
        bars = {s: b for s in syms if (b := _offline_bars(s)) is not None}
    if as_of_override:
        bars = {s: b[b["date_str"] <= as_of_override].reset_index(drop=True) for s, b in bars.items()}

    intraday_note = None
    if as_of_override is None and not allow_intraday_bar:
        now_et = datetime.now(ZoneInfo("America/New_York"))
        today = now_et.strftime("%Y-%m-%d")
        if str(bars["QQQ"]["date_str"].iloc[-1]) == today and now_et.time() < dtime(16, 5):
            intraday_note = f"Dropped incomplete {today} bar (run before 16:05 ET); signal uses the previous completed close."
            bars = {s: b[b["date_str"] != today].reset_index(drop=True) for s, b in bars.items()}

    q = bars["QQQ"]
    as_of = str(q["date_str"].iloc[-1])
    sig = target_exposure_from_closes(q["close"].astype(float), tv, cap, lb)
    w = sig["target_exposure"]

    prev_path = output_dir / SIGNAL_FILE
    source = "argument"
    if holdings_values:
        tot = sum(holdings_values.values()) or 1.0
        current_exposure = (holdings_values.get("QQQ", 0.0) + 2.0 * holdings_values.get(lev_etf, 0.0)) / tot
        source = "holdings values"
    elif current_exposure is None and prev_path.exists():
        current_exposure = float(json.loads(prev_path.read_text(encoding="utf-8")).get("exposure_after_orders", np.nan))
        source = "previous saved signal"
    if current_exposure is None or not np.isfinite(current_exposure):
        current_exposure, source = None, "unknown (first run)"

    crossed = current_exposure is not None and ((current_exposure <= 1.0 < w) or (w <= 1.0 < current_exposure))
    if current_exposure is None or abs(w - current_exposure) >= band or crossed:
        action, exposure_after = "REBALANCE_AT_NEXT_OPEN", w
    else:
        action, exposure_after = "HOLD_NO_TRADE", current_exposure

    weights = split_exposure(exposure_after)
    use_cash = cash_etf if cash_etf in bars and not bars[cash_etf].empty else cash_fb
    tickers = {"QQQ": "QQQ", "QLD": lev_etf, "TBILL": use_cash}
    names = {"QQQ": "Invesco QQQ Trust (Nasdaq-100)", "QLD": "ProShares Ultra QQQ (2x daily)", "TBILL": "T-bill ETF"}
    holdings = []
    for k in ("QQQ", "QLD", "TBILL"):
        t = tickers[k]
        b = bars.get(t)
        spot = float(b["close"].iloc[-1]) if b is not None and not b.empty else float("nan")
        dollars = capital * weights[k]
        holdings.append({"symbol": t, "name": names[k], "role": k, "target_weight_pct": round(weights[k] * 100.0, 2),
                         "target_dollars": round(dollars, 2),
                         "target_shares": int(dollars / spot) if np.isfinite(spot) and spot > 0 else None,
                         "spot_price": round(spot, 4) if np.isfinite(spot) else None, "status": action,
                         "execution_order_type": "09:30 ET next open"})

    payload = {
        "timestamp": datetime.now().isoformat(), "strategy_version": cfg.get("version"),
        "strategy_name": cfg.get("strategy_name"), "data_source_mode": mode, "bar_as_of_date": as_of,
        "intraday_note": intraday_note, "parameters": {"target_vol": tv, "max_exposure": cap, "lookback_days": lb, "rebalance_deadband": band},
        "qqq_close": round(float(q["close"].iloc[-1]), 4), "realised_vol_pct": round(sig["realised_vol"] * 100.0, 2),
        "target_exposure": round(w, 4), "current_exposure": None if current_exposure is None else round(current_exposure, 4),
        "current_exposure_source": source, "action": action, "exposure_after_orders": round(exposure_after, 4),
        "portfolio_equity": capital, "holdings": holdings,
    }
    (output_dir / SIGNAL_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS live_target_portfolio_voltarget")
    con.execute("CREATE TABLE live_target_portfolio_voltarget (symbol TEXT PRIMARY KEY, role TEXT, target_weight_pct REAL, "
                "target_dollars REAL, target_shares INTEGER, spot_price REAL, status TEXT, as_of TEXT)")
    for h in holdings:
        con.execute("INSERT OR REPLACE INTO live_target_portfolio_voltarget VALUES (?,?,?,?,?,?,?,?)",
                    (h["symbol"], h["role"], h["target_weight_pct"], h["target_dollars"], h["target_shares"], h["spot_price"], action, as_of))
    con.commit()
    con.close()

    print("=" * 100)
    print(f"  VOLATILITY-TARGET QQQ — signal from completed close {as_of} ({mode})")
    print("=" * 100)
    print(f"  QQQ close ${payload['qqq_close']:.2f} | {lb}-day realised vol {payload['realised_vol_pct']:.1f}% | "
          f"target exposure {w:.2f}x (cap {cap:.2f}x) | held {('?' if current_exposure is None else f'{current_exposure:.2f}x')} ({source})")
    print(f"  Action: {action} -> exposure {exposure_after:.2f}x")
    if intraday_note:
        print(f"  Note: {intraday_note}")
    print("-" * 100)
    for h in holdings:
        print(f"  {h['symbol']:<5} {h['role']:<6} {h['target_weight_pct']:>7.2f}%  ${h['target_dollars']:>12,.2f}  shares {h['target_shares']}  @ {h['spot_price']}")
    print("=" * 100)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Volatility-target QQQ daily signal (v6.0.0)")
    p.add_argument("--capital", type=float, default=100000.0)
    p.add_argument("--current-exposure", type=float, default=None, help="Exposure currently held, e.g. 1.2")
    p.add_argument("--qqq-value", type=float, default=None)
    p.add_argument("--qld-value", type=float, default=None)
    p.add_argument("--cash-value", type=float, default=None)
    p.add_argument("--as-of", type=str, default=None, help="Replay a historical close YYYY-MM-DD (offline cache)")
    p.add_argument("--allow-intraday-bar", action="store_true", default=False)
    a = p.parse_args()
    hv = None
    if any(v is not None for v in (a.qqq_value, a.qld_value, a.cash_value)):
        hv = {"QQQ": a.qqq_value or 0.0, "QLD": a.qld_value or 0.0, "TBILL": a.cash_value or 0.0}
    run_production(capital=a.capital, current_exposure=a.current_exposure, holdings_values=hv,
                   as_of_override=a.as_of, allow_intraday_bar=a.allow_intraday_bar)


if __name__ == "__main__":
    main()
