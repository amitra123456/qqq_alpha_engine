"""
Volatility-Target QQQ Backtest (strategy v6.0.0)
================================================
Rule: each day, target QQQ exposure = min(1.5, 20% / trailing 21-day realised volatility of QQQ).
  - Exposure up to 1.0x is held in QQQ, the rest in T-bills.
  - Exposure above 1.0x is obtained by moving part of the QQQ position into QLD (2x daily Nasdaq-100 ETF):
    holding a fraction `a` in QLD and `1 - a` in QQQ gives exposure 1 + a, so 1.5x = 50% QQQ + 50% QLD. No margin.
  - The target is computed from data up to the close of day t-1 and traded at the open of day t.
  - The portfolio is only rebalanced when the target differs from the exposure actually held by 10 percentage
    points or more (deadband), or 1.0x is crossed.

Simulation is share-level: holdings drift with prices between rebalances; every traded dollar pays the cost
assumption; idle cash earns the 1-day-lagged FRED 3-month T-bill rate (DTB3).
QLD exists from June 2006. Before that, a synthetic 2x fund is used (2 x QQQ daily return minus a 0.95% fee and
T-bill + 0.5% financing on the borrowed half); the script reports how closely the synthetic matches real QLD after 2006.

Outputs (data/outputs/):
  voltarget_daily_nav.csv        daily NAV of the strategy and baselines
  voltarget_period_stats.csv     CAGR / max drawdown / Sharpe / average exposure by period
  voltarget_calendar_years.csv   calendar-year returns
  voltarget_parameter_grid.csv   all 48 target / cap / lookback combinations tried
  voltarget_summary.json         headline numbers, sensitivities and data checks
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CACHE = PROJECT_ROOT / "data" / "cache"
OUT = PROJECT_ROOT / "data" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

START = "1999-04-12"          # first open after 21 trading days of QQQ history
INITIAL = 100000.0
QLD_FEE = 0.0095              # QLD expense ratio
QLD_FIN_SPREAD = 0.005        # financing spread assumed inside a 2x fund
TARGET_VOL, MAX_EXPOSURE, LOOKBACK, DEADBAND = 0.20, 1.5, 21, 0.10


def load_data() -> pd.DataFrame:
    q = pd.read_csv(CACHE / "qqq_total_return_1999_2026.csv").set_index("date_str").sort_index()[["open", "close"]]
    dtb = pd.read_csv(CACHE / "fred_macro" / "DTB3.csv").set_index("observation_date")["DTB3"]
    dtb = pd.to_numeric(dtb, errors="coerce")
    df = q.copy()
    df["rf"] = dtb.reindex(df.index).ffill().bfill().shift(1).fillna(0.0) / 100.0 / 252.0
    qld = pd.read_csv(CACHE / "yahoo_daily" / "QLD.csv").set_index("date_str")[["open", "close"]]
    df["qld_open"] = qld["open"].reindex(df.index)
    df["qld_close"] = qld["close"].reindex(df.index)
    return df


def target_exposure(close: pd.Series, target_vol: float = TARGET_VOL, cap: float = MAX_EXPOSURE, lookback: int = LOOKBACK) -> pd.Series:
    """Exposure wanted at the close of each day, using only returns up to that close."""
    vol = close.pct_change().rolling(lookback).std() * np.sqrt(252.0)
    return (target_vol / vol).clip(upper=cap).fillna(1.0)


def leg_returns(df: pd.DataFrame, synthetic_only: bool = False) -> Tuple[np.ndarray, ...]:
    """Overnight (close->open) and intraday (open->close) returns for QQQ and the 2x leg."""
    q_on = (df["open"] / df["close"].shift(1) - 1.0).fillna(0.0).values
    q_in = (df["close"] / df["open"] - 1.0).values
    rf = df["rf"].values
    syn_on = 2.0 * q_on
    syn_in = 2.0 * q_in - (rf + QLD_FIN_SPREAD / 252.0) - QLD_FEE / 252.0      # one day of fund costs, charged on the intraday leg
    if synthetic_only:
        return q_on, q_in, syn_on, syn_in, rf
    real_on = (df["qld_open"] / df["qld_close"].shift(1) - 1.0).values
    real_in = (df["qld_close"] / df["qld_open"] - 1.0).values
    use_real = ~np.isnan(real_on) & ~np.isnan(real_in)
    return q_on, q_in, np.where(use_real, real_on, syn_on), np.where(use_real, real_in, syn_in), rf


def simulate(df: pd.DataFrame, target: pd.Series, cost_bps: float = 10.0, deadband: float = DEADBAND,
             leverage: str = "qld", margin_spread: float = 0.01, synthetic_only: bool = False,
             start: str = START) -> Tuple[pd.Series, pd.Series, float]:
    """Share-level simulation. leverage='qld' (QQQ + QLD + T-bills) or 'margin' (QQQ + borrowing at T-bill + spread)."""
    q_on, q_in, l_on, l_in, rf = leg_returns(df, synthetic_only)
    tgt = target.reindex(df.index).fillna(1.0).values
    cost = cost_bps / 10000.0
    i0 = int(np.searchsorted(df.index.values, start))
    v_q = v_l = 0.0
    cash = INITIAL
    navs, expo = [], []
    traded = 0.0
    for i in range(i0, len(df)):
        # overnight move on yesterday's holdings
        v_q *= 1.0 + q_on[i]
        v_l *= 1.0 + l_on[i]
        nav_open = v_q + v_l + cash
        held = (v_q + 2.0 * v_l) / nav_open if nav_open > 0 else 0.0
        w = float(tgt[i - 1])
        crossed_one = (held <= 1.0 < w) or (w <= 1.0 < held)
        if i == i0 or abs(w - held) >= deadband or crossed_one:
            def targets(equity: float) -> Tuple[float, float]:
                if leverage == "qld":
                    a = max(w - 1.0, 0.0)
                    return equity * (min(w, 1.0) - a), equity * a
                return equity * w, 0.0
            new_q, new_l = targets(nav_open)
            trade = abs(new_q - v_q) + abs(new_l - v_l)
            nav_net = nav_open - trade * cost            # pay costs first, then size positions on what is left
            new_q, new_l = targets(nav_net)
            traded += trade / nav_open
            cash = nav_net - new_q - new_l
            v_q, v_l = new_q, new_l
        # intraday move, then one day of interest on cash (or financing on borrowed cash)
        v_q *= 1.0 + q_in[i]
        v_l *= 1.0 + l_in[i]
        cash *= 1.0 + (rf[i] if cash >= 0 else rf[i] + margin_spread / 252.0)
        nav = v_q + v_l + cash
        navs.append(nav)
        expo.append((v_q + 2.0 * v_l) / nav if nav > 0 else 0.0)
    idx = pd.to_datetime(df.index[i0:])
    yrs = (idx[-1] - idx[0]).days / 365.25
    return pd.Series(navs, idx), pd.Series(expo, idx), traded / yrs


def stats(nav: pd.Series, expo: pd.Series, rf: pd.Series, a=None, b=None) -> Dict[str, float]:
    n = nav.loc[a:b]
    n = n / n.iloc[0]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    r = n.pct_change().dropna()
    ex = r - rf.reindex(r.index).fillna(0.0)
    dd = n / n.cummax() - 1.0
    return {"cagr_pct": ((n.iloc[-1]) ** (1.0 / yrs) - 1.0) * 100.0, "max_dd_pct": dd.min() * 100.0,
            "sharpe": ex.mean() / r.std() * np.sqrt(252.0), "avg_exposure": float(expo.loc[a:b].mean())}


def calendar_years(nav: pd.Series) -> pd.Series:
    ye = nav.resample("YE").last()
    out = ye.pct_change()
    out.iloc[0] = ye.iloc[0] / nav.iloc[0] - 1.0
    out.index = out.index.year
    return out * 100.0


PERIODS = [
    ("Full 1999-04 to latest", None, None),
    ("1999-2002 dot-com crash", None, "2002-12-31"),
    ("2003-2007 recovery", "2003-01-01", "2007-12-31"),
    ("2008-2009 financial crisis", "2008-01-01", "2009-12-31"),
    ("2010-2014", "2010-01-01", "2014-12-31"),
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("2020-2022 Covid and inflation bear", "2020-01-01", "2022-12-31"),
    ("2023-latest", "2023-01-01", None),
]


def main() -> None:
    df = load_data()
    close = df["close"]
    rf = pd.Series(df["rf"].values, pd.to_datetime(df.index))
    one = pd.Series(1.0, df.index)
    sma200 = close.rolling(200).mean()
    strategies = {
        "QQQ buy & hold (total return)": one,
        "QQQ 200-day rule (T-bills below)": (close > sma200).astype(float),
        "Vol target 20%, max 1.0x": target_exposure(close, 0.20, 1.0),
        "Vol target 20%, max 1.25x": target_exposure(close, 0.20, 1.25),
        "STRATEGY: Vol target 20%, max 1.5x": target_exposure(close, 0.20, 1.5),
    }
    runs = {k: simulate(df, t) for k, t in strategies.items()}
    main_key = "STRATEGY: Vol target 20%, max 1.5x"

    rows = []
    for lab, a, b in PERIODS:
        for k, (nav, expo, _) in runs.items():
            rows.append({"period": lab, "strategy": k, **stats(nav, expo, rf, a, b)})
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "voltarget_period_stats.csv", index=False)

    years = pd.DataFrame({k: calendar_years(v[0]) for k, v in runs.items()}).round(2)
    years.to_csv(OUT / "voltarget_calendar_years.csv")
    beat = {k: int((years[k] > years["QQQ buy & hold (total return)"]).sum()) for k in years.columns if k != "QQQ buy & hold (total return)"}

    pd.DataFrame({k: v[0] for k, v in runs.items()}).assign(
        **{"Exposure: " + main_key: runs[main_key][1]}).rename_axis("Date").to_csv(OUT / "voltarget_daily_nav.csv")

    # cost, leverage-method and financing sensitivity for the strategy
    t_main = strategies[main_key]
    sens = {
        "QLD leverage, 10 bps (base case)": runs[main_key],
        "QLD leverage, 20 bps": simulate(df, t_main, cost_bps=20.0),
        "QLD leverage, 30 bps": simulate(df, t_main, cost_bps=30.0),
        "Margin at T-bill + 0.3% (futures-like)": simulate(df, t_main, leverage="margin", margin_spread=0.003),
        "Margin at T-bill + 1%": simulate(df, t_main, leverage="margin", margin_spread=0.01),
        "Margin at T-bill + 2%": simulate(df, t_main, leverage="margin", margin_spread=0.02),
        "No deadband (trade every day)": simulate(df, t_main, deadband=0.0),
    }
    sens_tab = {k: {**stats(v[0], v[1], rf), "turnover_x_per_year": v[2]} for k, v in sens.items()}

    # synthetic 2x model vs real QLD, from QLD's first full day
    qld_r = (df["qld_close"].pct_change()).dropna()
    syn_r = (2.0 * close.pct_change() - (df["rf"] + QLD_FIN_SPREAD / 252.0) - QLD_FEE / 252.0).reindex(qld_r.index)
    qld_check = {"real_qld_cagr_pct": float(((1 + qld_r).prod() ** (252 / len(qld_r)) - 1) * 100),
                 "synthetic_cagr_pct": float(((1 + syn_r).prod() ** (252 / len(syn_r)) - 1) * 100),
                 "daily_tracking_error_bps": float((qld_r - syn_r).std() * 1e4), "from": str(qld_r.index[0])}

    # every parameter combination tried (margin leverage for speed and comparability)
    grid = []
    for tv in (0.15, 0.20, 0.25, 0.30):
        for cap in (1.0, 1.25, 1.5, 2.0):
            for lb in (10, 21, 63):
                nav, expo, _ = simulate(df, target_exposure(close, tv, cap, lb), leverage="margin")
                grid.append({"target_vol": tv, "max_exposure": cap, "lookback_days": lb, **stats(nav, expo, rf),
                             **{f"{k}_1999_2002": v for k, v in stats(nav, expo, rf, None, "2002-12-31").items() if k in ("cagr_pct", "max_dd_pct")}})
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "voltarget_parameter_grid.csv", index=False)
    bh_full = stats(*runs["QQQ buy & hold (total return)"][:2], rf)

    summary = {
        "strategy": "QQQ volatility target 20%, max 1.5x (QLD for exposure above 1x), 10-point deadband, next-open execution",
        "start": str(runs[main_key][0].index[0].date()), "end": str(runs[main_key][0].index[-1].date()),
        "full_period": {k: stats(v[0], v[1], rf) for k, v in runs.items()},
        "years_beating_qqq": beat, "n_years": int(len(years)),
        "turnover_x_per_year": {k: v[2] for k, v in runs.items()},
        "sensitivity": sens_tab, "qld_model_check": qld_check,
        "parameter_grid": {"combinations": int(len(grid)), "cagr_min": float(grid.cagr_pct.min()), "cagr_median": float(grid.cagr_pct.median()),
                           "cagr_max": float(grid.cagr_pct.max()), "beat_qqq_cagr": int((grid.cagr_pct > bh_full["cagr_pct"]).sum()),
                           "beat_qqq_cagr_and_drawdown": int(((grid.cagr_pct > bh_full["cagr_pct"]) & (grid.max_dd_pct > bh_full["max_dd_pct"])).sum())},
    }
    (OUT / "voltarget_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    pd.set_option("display.width", 220)
    order = [p[0] for p in PERIODS]
    print("=" * 120)
    print(f"  VOLATILITY-TARGET QQQ BACKTEST  {summary['start']} -> {summary['end']}  (next-open execution, 10 bps, T-bill cash)")
    print("=" * 120)
    for m, lab in (("cagr_pct", "CAGR %"), ("max_dd_pct", "Max drawdown %"), ("sharpe", "Sharpe"), ("avg_exposure", "Average exposure")):
        print(f"\n{lab}")
        print(per.pivot(index="period", columns="strategy", values=m).reindex(order)[list(strategies)].round(2).to_string())
    print("\nCalendar-year returns %")
    print(years.round(1).to_string())
    print("\nYears beating QQQ:", beat, f"of {len(years)}")
    print("\nSensitivity (full period):")
    print(pd.DataFrame(sens_tab).T.round(2).to_string())
    print("\nSynthetic 2x model vs real QLD:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in qld_check.items()})
    print("Parameter grid:", summary["parameter_grid"])
    print(f"\nSaved outputs to {OUT}")


if __name__ == "__main__":
    main()
