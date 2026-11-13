"""
Backtest — Big Eight Weather Plan (no leverage)
===============================================
Basket : the 8 largest true Nasdaq-100 members by point-in-time market cap at each quarter-end, cap-weighted,
         rebuilt at the first open after the quarter-end, weights drifting in between. Total return (dividends
         reinvested), 10 bps on reconstitution turnover.
Overlay: invested fraction = min(1.0, 20% / QQQ's trailing 21-day realised volatility), decided at the close,
         traded at the next open only when it differs from the fraction held by 10+ points; the rest earns the
         T-bill rate (FRED DTB3, lagged one day). 10 bps on traded dollars. No margin, no 2x funds.
Window : first trade 2011-02-01 (first line-up is 2010-12-31). 2011 to mid-2016 was never used to design the rules.

Outputs (data/outputs/): big8_daily_nav.csv, big8_trade_log.csv, big8_lineups.csv, big8_period_stats.csv, big8_calendar_years.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.big8_data import load_lineups, load_prices, total_return_index
from scripts.run_voltarget_backtest import load_data

OUT = PROJECT_ROOT / "data" / "outputs"
START = "2011-02-01"
TOP_N = 8
TARGET_VOL = 0.20
MAX_EXPOSURE = 1.0
LOOKBACK = 21
DEADBAND = 0.10
COST_BPS = 10.0
CAL_START = "2010-12-01"


def regime(vol: float) -> str:
    return "Stormy" if vol >= 0.40 else "Choppy" if vol >= 0.25 else "Normal" if vol >= 0.15 else "Calm"


def qqq_vol(close: pd.Series, lookback: int = LOOKBACK) -> pd.Series:
    return close.pct_change().rolling(lookback).std() * np.sqrt(252.0)


def build_basket(lineups: Dict[str, pd.DataFrame], calendar: List[str], cost_bps: float = COST_BPS) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, np.ndarray]]:
    """Open/close values of the cap-weighted basket. Returns (basket, units held at the end, total-return close index per symbol)."""
    syms = sorted({s for g in lineups.values() for s in g.symbol})
    px = load_prices(syms)
    TRo, TRc = {}, {}
    for s in syms:
        TRo[s], TRc[s] = total_return_index(px[s], calendar)
    pos = {d: i for i, d in enumerate(calendar)}
    reb = {}
    for qd, g in lineups.items():
        nxt = [d for d in calendar if d > qd]
        if nxt:
            reb[pos[nxt[0]]] = dict(zip(g.symbol, g.weight))
    units: Dict[str, float] = {}
    bo = np.full(len(calendar), np.nan); bc = np.full(len(calendar), np.nan)
    for i in range(min(reb), len(calendar)):
        if i in reb:
            v_open = sum(u * TRo[s][i] for s, u in units.items()) if units else 1.0
            old = {s: u * TRo[s][i] / v_open for s, u in units.items()} if units else {}
            new = reb[i]
            turnover = sum(abs(new.get(s, 0) - old.get(s, 0)) for s in set(new) | set(old)) if units else 1.0
            v_open *= 1 - cost_bps / 1e4 * turnover
            units = {s: v_open * w / TRo[s][i] for s, w in new.items()}
            bo[i] = v_open
        else:
            bo[i] = sum(u * TRo[s][i] for s, u in units.items())
        bc[i] = sum(u * TRc[s][i] for s, u in units.items())
    return pd.DataFrame({"open": bo, "close": bc}, index=calendar).dropna(), units, TRc


def simulate_overlay(asset: pd.DataFrame, target: pd.Series, rf: pd.Series, start: str = START,
                     cost_bps: float = COST_BPS, deadband: float = DEADBAND) -> Tuple[pd.Series, pd.Series, List[dict]]:
    """Hold `target` (known at each close, capped at 1.0) of the asset from the next open, rest in T-bills."""
    idx = asset.index
    o, c = asset.open.values, asset.close.values
    on = np.r_[0.0, o[1:] / c[:-1] - 1.0]
    inn = c / o - 1.0
    r_f = rf.reindex(idx).fillna(0.0).values
    t = target.reindex(idx).fillna(1.0).clip(upper=1.0).values
    i0 = int(np.searchsorted(idx.values, start))
    vq, cash = 0.0, 1.0
    navs, expo, log = [], [], []
    for i in range(i0, len(idx)):
        vq *= 1 + on[i]
        nav = vq + cash
        held = vq / nav
        w = float(t[i - 1])
        if i == i0 or abs(w - held) >= deadband:
            traded = abs(nav * w - vq)
            net = nav - traded * cost_bps / 1e4
            vq, cash = net * w, net - net * w
            log.append({"date": idx[i], "held_before": round(held, 4), "target": round(w, 4)})
        vq *= 1 + inn[i]
        cash *= 1 + r_f[i]
        navs.append(vq + cash)
        expo.append(vq / navs[-1])
    ix = pd.to_datetime(idx[i0:])
    return pd.Series(navs, ix), pd.Series(expo, ix), log


def run(top_n: int = TOP_N, cost_bps: float = COST_BPS) -> Dict[str, object]:
    df = load_data()
    cal = [d for d in df.index if d >= CAL_START]
    lineups = load_lineups(top_n)
    basket, units, trc = build_basket(lineups, cal, cost_bps)
    vol = qqq_vol(df["close"])
    target = (TARGET_VOL / vol).clip(upper=MAX_EXPOSURE)
    nav, expo, log = simulate_overlay(basket, target, df["rf"], START, cost_bps)
    q = df.loc[basket.index[0]:, ["open", "close"]]
    bh, _, _ = simulate_overlay(q, pd.Series(1.0, q.index), df["rf"], START, cost_bps)
    for e in log:
        v = float(vol.shift(1).loc[e["date"]])
        e["qqq_vol_pct"] = round(v * 100, 2)
        e["regime"] = regime(v)
    return {"df": df, "calendar": cal, "lineups": lineups, "basket": basket, "units": units, "trc": trc,
            "vol": vol, "target": target, "nav": nav, "exposure": expo, "log": log, "qqq_bh": bh}


def period_stats(nav: pd.Series, a=None, b=None) -> Dict[str, float]:
    n = nav.loc[a:b]
    n = n / n.iloc[0]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    return {"cagr_pct": (n.iloc[-1] ** (1 / yrs) - 1) * 100, "max_dd_pct": (n / n.cummax() - 1).min() * 100}


PERIODS = [("Feb 2011 to latest", None, None), ("2011 to mid-2016 (never used in design)", None, "2016-06-30"),
           ("Mid-2016 to 2019", "2016-07-01", "2019-12-31"), ("2020 to 2022", "2020-01-01", "2022-12-31"), ("2023 to latest", "2023-01-01", None)]


def main() -> None:
    r = run()
    nav, bh, log = r["nav"], r["qqq_bh"], r["log"]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"big8_weather_plan": nav, "qqq_buy_and_hold": bh, "invested_fraction": r["exposure"]}).to_csv(OUT / "big8_daily_nav.csv")
    pd.DataFrame(log).to_csv(OUT / "big8_trade_log.csv", index=False)
    rows = []
    for qd, g in r["lineups"].items():
        for k, x in enumerate(g.itertuples(), 1):
            rows.append({"quarter_date": qd, "rank": k, "symbol": x.symbol, "market_cap_bn": x.market_cap_bn, "weight_pct": round(x.weight * 100, 2)})
    pd.DataFrame(rows).to_csv(OUT / "big8_lineups.csv", index=False)
    ps = []
    for lab, a, b in PERIODS:
        for name, s in (("Big Eight Weather Plan", nav), ("QQQ buy & hold", bh)):
            ps.append({"period": lab, "strategy": name, **{k: round(v, 2) for k, v in period_stats(s, a, b).items()}})
    pd.DataFrame(ps).to_csv(OUT / "big8_period_stats.csv", index=False)
    y = pd.DataFrame({"Big Eight Weather Plan": nav, "QQQ buy & hold": bh}).resample("YE").last()
    yr = y.pct_change()
    yr.iloc[0] = y.iloc[0] - 1
    yr.index = yr.index.year
    (yr * 100).round(2).to_csv(OUT / "big8_calendar_years.csv")
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    print("=" * 90)
    print(f"  BIG EIGHT WEATHER PLAN (no leverage) — {nav.index[0].date()} to {nav.index[-1].date()}")
    print("=" * 90)
    print(pd.DataFrame(ps).pivot(index="period", columns="strategy", values=["cagr_pct", "max_dd_pct"]).round(1).to_string())
    print(f"\n  weather trades: {len(log)} ({len(log) / yrs:.1f}/yr) | line-ups: {len(r['lineups'])} | "
          f"years beating QQQ: {(yr['Big Eight Weather Plan'] > yr['QQQ buy & hold']).sum()} of {len(yr)} | avg invested {r['exposure'].mean():.2f}")
    print(f"  outputs written to {OUT}")


if __name__ == "__main__":
    main()
