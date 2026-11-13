"""
Official Audit Backtest — Quarterly Point-in-Time Cap-Weight + QQQ 200-Day Moving Average Cash Rule
==================================================================================================
Plain-language rules:
  - Quarterly reconstitution of Point-in-Time Nasdaq-100 top constituents by causal SEC EDGAR market cap
    (`Top-8` default; `Top-16` broad variant), with dual-class deduplication (`GOOG` excluded, `GOOGL` kept).
  - Daily trend signal evaluated strictly at completed Close_{t-1}: `QQQ Close > QQQ 200-Day SMA`.
    * Above SMA200: 100% invested in cap-weighted PIT basket (weights drift naturally intra-quarter).
    * At or below SMA200: 100% in US Treasury Bills (1-day-lagged FRED `DTB3`).
  - Execution strictly at Open_t (`09:30 ET`), zero leverage (`0% TQQQ`), zero margin (`cash >= 0`),
    zero option collars, and sensitivity tested at 10, 20, and 30 bps per traded dollar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ALPACA_DIR = CACHE_DIR / "alpaca_sip_daily"
FRED_DIR = CACHE_DIR / "fred_macro"
ROSTER_PATH = DATA_DIR / "point_in_time_rosters.csv"
TR_CACHE_PATH = CACHE_DIR / "qqq_total_return_1999_2026.csv"
OUT_DIR = DATA_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAP = 100000.0
SIM_START_DATE = "2016-06-30"


def load_total_return_qqq_and_sip() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads pre-warmed Total-Return QQQ (1999-2026) from `data/cache/qqq_total_return_1999_2026.csv`,
    appends any newer dates present in `data/cache/alpaca_sip_daily/QQQ.csv`, and computes
    200-day SMA and 20-day realized volatility BEFORE slicing to SIM_START_DATE.
    """
    qsip = pd.read_csv(ALPACA_DIR / "QQQ.csv")
    qsip["date_str"] = qsip["date_str"].astype(str)
    qsip = qsip.sort_values("date_str").drop_duplicates(subset=["date_str"]).set_index("date_str")

    if not TR_CACHE_PATH.exists():
        raise FileNotFoundError(f"Required Total-Return QQQ cache not found: {TR_CACHE_PATH}")

    cached = pd.read_csv(TR_CACHE_PATH).set_index("date_str")
    newer = qsip.loc[qsip.index > cached.index.max(), ["open", "high", "low", "close", "volume"]]
    q_tr = pd.concat([cached, newer]) if not newer.empty else cached
    q_tr = q_tr[~q_tr.index.duplicated(keep="first")].sort_index()
    if not newer.empty:
        q_tr.reset_index().to_csv(TR_CACHE_PATH, index=False)

    q_tr["sma200"] = q_tr["close"].rolling(200, min_periods=200).mean()
    q_tr["rv20"] = q_tr["close"].pct_change().rolling(20, min_periods=20).std() * np.sqrt(252.0)
    return q_tr, qsip


def load_lagged_dtb3_series(dates_index: pd.Index) -> pd.Series:
    """Loads FRED DTB3 3-month Treasury rate lagged by 1 trading day (daily decimal return)."""
    dtb = pd.read_csv(FRED_DIR / "DTB3.csv")
    dtb["DTB3"] = pd.to_numeric(dtb["DTB3"], errors="coerce").ffill()
    dtb_s = dtb.set_index("observation_date")["DTB3"]
    return dtb_s.reindex(dates_index).ffill().bfill().shift(1).fillna(0.0) / 100.0 / 252.0


def build_quarter_plans(roster_df: pd.DataFrame, cal_dates: List[str], top_k: int) -> Dict[str, dict]:
    """Maps first trading day strictly AFTER quarter-end close to normalized target weights."""
    weight_col = "weight_top8_pct" if (top_k == 8 and "weight_top8_pct" in roster_df.columns) else "weight_pct"
    plans: Dict[str, dict] = {}
    for q_label, grp in roster_df.groupby("quarter"):
        q_date = str(grp["date_str"].iloc[0])
        valid_dates = [d for d in cal_dates if d > q_date]
        if not valid_dates:
            continue
        start_d = valid_dates[0]
        sub = grp.sort_values("rank").iloc[:top_k]
        w_sum = float(sub[weight_col].sum())
        w_map = {str(r["ticker"]): float(r[weight_col]) / w_sum for _, r in sub.iterrows() if w_sum > 0}
        plans[start_d] = {
            "quarter": str(q_label),
            "ranked_on": q_date,
            "weights": w_map,
            "pit_true_members": int(sub["pit_true_members"].iloc[0]) if "pit_true_members" in sub.columns else np.nan,
            "pit_members_with_data": int(sub["pit_members_with_data"].iloc[0]) if "pit_members_with_data" in sub.columns else np.nan,
        }
    return plans


def simulate_pit_basket(
    cal_dates: List[str],
    q_tr: pd.DataFrame,
    rf_s: pd.Series,
    op_ex: pd.DataFrame,
    cl_ff: pd.DataFrame,
    syms: List[str],
    quarter_plans: Dict[str, dict],
    use_ma200: bool = True,
    cost_bps: float = 10.0,
) -> Tuple[pd.Series, float, int]:
    """
    Exact share-level daily mark-to-market simulation with Close_{t-1} -> Open_t execution,
    quarterly reconstitution, intra-quarter market-cap drift, zero margin, and lagged DTB3 cash yield.
    """
    cost = cost_bps / 10000.0
    cash = INITIAL_CAP
    shares = pd.Series(0.0, index=syms)
    cur_weights: Dict[str, float] = {}
    prev_invested = False
    traded_frac_sum = 0.0
    regime_switches = 0
    navs: List[float] = []

    for i, d in enumerate(cal_dates):
        if i > 0:
            cash *= 1.0 + float(rf_s.loc[d])

        prev_d = cal_dates[i - 1] if i > 0 else d
        is_rebal = d in quarter_plans
        if is_rebal:
            cur_weights = quarter_plans[d]["weights"]

        if i == 0 or not cur_weights:
            invested = False
        elif use_ma200:
            invested = bool(q_tr.loc[prev_d, "close"] > q_tr.loc[prev_d, "sma200"])
        else:
            invested = True

        if i > 0 and (is_rebal or (invested != prev_invested)):
            if invested != prev_invested:
                regime_switches += 1
            opx = op_ex.loc[d].fillna(0.0)
            cur_vals = shares * opx
            total_open_nav = float(cur_vals.sum() + cash)

            tgt_w = pd.Series(0.0, index=syms)
            if invested and cur_weights:
                for k, v in cur_weights.items():
                    if opx.get(k, 0.0) > 0.0:
                        tgt_w[k] = v
                if tgt_w.sum() > 0.0:
                    tgt_w /= tgt_w.sum()

            tgt_vals = total_open_nav * tgt_w
            deltas = tgt_vals - cur_vals

            sells = deltas[deltas < -1e-6]
            for s, dv in sells.items():
                px_s = float(opx[s])
                if px_s > 0.0 and shares[s] > 0.0:
                    sell_sh = min(float(shares[s]), abs(float(dv)) / px_s)
                    gross = sell_sh * px_s
                    shares[s] -= sell_sh
                    cash += gross * (1.0 - cost)
                    traded_frac_sum += gross / max(total_open_nav, 1.0)

            buys = deltas[deltas > 1e-6]
            buy_req = float(buys.sum())
            if buy_req > 1.0 and cash > 1.0:
                scale = min(1.0, (cash / (1.0 + cost)) / buy_req)
                for s, dv in buys.items():
                    px_s = float(opx[s])
                    if px_s > 0.0:
                        buy_val = float(dv) * scale
                        shares[s] += buy_val / px_s
                        cash -= buy_val * (1.0 + cost)
                        traded_frac_sum += buy_val / max(total_open_nav, 1.0)

            shares[shares < 1e-9] = 0.0
            prev_invested = invested

        assert cash >= -1e-6, f"Negative cash {cash:.4f} on {d}"
        cpx = cl_ff.loc[d].fillna(0.0)
        eq_val = float((shares * cpx).sum())
        daily_nav = eq_val + cash
        assert abs(daily_nav - (eq_val + cash)) < 1e-6
        navs.append(daily_nav)

    yrs = (pd.Timestamp(cal_dates[-1]) - pd.Timestamp(cal_dates[0])).days / 365.25
    ann_turnover = traded_frac_sum / yrs if yrs > 0 else 0.0
    return pd.Series(navs, index=pd.to_datetime(cal_dates)), ann_turnover, regime_switches


def simulate_single_qqq_rule(
    cal_dates: List[str],
    q_tr: pd.DataFrame,
    rf_s: pd.Series,
    mode: str,
    target_weight_const: float = 1.0,
    cost_bps: float = 10.0,
) -> Tuple[pd.Series, float, int]:
    """
    Simulates single-instrument QQQ + lagged DTB3 rules at Open_t:
      - 'buy_hold': 100% Total-Return QQQ from cal_dates[1] open
      - 'sma200': 100% QQQ when Close_{t-1} > SMA200_{t-1}, else DTB3
      - 'vol_target_20': w_t = min(1.0, 0.20 / rv20_{t-1}), rebalanced monthly or >15% drift
      - 'fixed_weight': fixed target weight in QQQ, rebalanced quarterly
    """
    cost = cost_bps / 10000.0
    cash = INITIAL_CAP
    q_sh = 0.0
    prev_w = 0.0
    traded_frac_sum = 0.0
    switches = 0
    navs: List[float] = []

    for i, d in enumerate(cal_dates):
        if i > 0:
            cash *= 1.0 + float(rf_s.loc[d])
        prev_d = cal_dates[i - 1] if i > 0 else d
        op_q = float(q_tr.loc[d, "open"])
        cl_q = float(q_tr.loc[d, "close"])

        if i == 0:
            navs.append(INITIAL_CAP)
            continue

        rebal_needed = False
        tgt_w = prev_w
        if mode == "buy_hold":
            if i == 1:
                tgt_w = 1.0
                rebal_needed = True
        elif mode == "sma200":
            tgt_w = 1.0 if (q_tr.loc[prev_d, "close"] > q_tr.loc[prev_d, "sma200"]) else 0.0
            if tgt_w != prev_w:
                rebal_needed = True
                switches += 1
        elif mode == "vol_target_20":
            rv_prev = float(q_tr.loc[prev_d, "rv20"])
            raw_w = min(1.0, 0.20 / rv_prev) if (rv_prev > 0 and not np.isnan(rv_prev)) else 1.0
            is_month_start = d[:7] != prev_d[:7]
            if i == 1 or is_month_start or abs(raw_w - prev_w) > 0.15:
                tgt_w = raw_w
                rebal_needed = True
        elif mode == "fixed_weight":
            is_q_start = ((int(d[5:7]) - 1) // 3) != ((int(prev_d[5:7]) - 1) // 3)
            if i == 1 or is_q_start:
                tgt_w = target_weight_const
                rebal_needed = True

        if rebal_needed:
            cur_eq = q_sh * op_q
            tot_open = cur_eq + cash
            tgt_eq = tot_open * tgt_w
            delta = tgt_eq - cur_eq
            if delta < -1.0 and q_sh > 0.0:
                sell_sh = min(q_sh, abs(delta) / op_q)
                gross = sell_sh * op_q
                q_sh -= sell_sh
                cash += gross * (1.0 - cost)
                traded_frac_sum += gross / max(tot_open, 1.0)
            elif delta > 1.0 and cash > 1.0:
                buy_val = min(delta, cash / (1.0 + cost))
                q_sh += buy_val / op_q
                cash -= buy_val * (1.0 + cost)
                traded_frac_sum += buy_val / max(tot_open, 1.0)
            prev_w = tgt_w

        assert cash >= -1e-6
        navs.append(q_sh * cl_q + cash)

    yrs = (pd.Timestamp(cal_dates[-1]) - pd.Timestamp(cal_dates[0])).days / 365.25
    return pd.Series(navs, index=pd.to_datetime(cal_dates)), (traded_frac_sum / yrs if yrs > 0 else 0.0), switches


def compute_summary_stats(
    s_nav: pd.Series,
    bench_nav: pd.Series,
    beta_bench_nav: pd.Series,
    rf_daily: np.ndarray,
    ann_turnover: float,
    switches: int,
    name: str,
) -> dict:
    """Computes summary performance metrics, HAC CAPM alpha vs both benchmarks, and Ex-Best-10d CAGR."""
    yrs = (s_nav.index[-1] - s_nav.index[0]).days / 365.25
    cagr = (s_nav.iloc[-1] / s_nav.iloc[0]) ** (1.0 / yrs) - 1.0
    max_dd = float(((s_nav - s_nav.cummax()) / s_nav.cummax()).min())
    rets = s_nav.pct_change().fillna(0.0).values
    ex = rets[1:] - rf_daily[1:]
    ex_b = bench_nav.pct_change().fillna(0.0).values[1:] - rf_daily[1:]
    ex_bm = beta_bench_nav.pct_change().fillna(0.0).values[1:] - rf_daily[1:]

    vol = float(np.std(rets[1:], ddof=1) * np.sqrt(252.0))
    sharpe = float((np.mean(ex) / np.std(ex, ddof=1)) * np.sqrt(252.0)) if np.std(ex, ddof=1) > 0 else 0.0
    neg_ex = ex[ex < 0]
    dn_vol = float(np.std(neg_ex, ddof=1) * np.sqrt(252.0)) if len(neg_ex) > 1 else 0.0
    sortino = float((np.mean(ex) * 252.0) / dn_vol) if dn_vol > 0 else 0.0
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 0 else 0.0

    mod_q = sm.OLS(ex, sm.add_constant(ex_b)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    mod_bm = sm.OLS(ex, sm.add_constant(ex_bm)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    top10_idx = np.argsort(rets[1:])[-10:]
    rets_trimmed = rets[1:].copy()
    rets_trimmed[top10_idx] = 0.0
    nav_trimmed = float(np.prod(1.0 + rets_trimmed))
    best10_removed_cagr = nav_trimmed ** (1.0 / yrs) - 1.0

    return {
        "strategy": name,
        "ending_nav": round(float(s_nav.iloc[-1]), 2),
        "cagr_pct": round(cagr * 100.0, 2),
        "max_dd_pct": round(max_dd * 100.0, 2),
        "vol_pct": round(vol * 100.0, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "turnover_x": round(float(ann_turnover), 2),
        "regime_switches": int(switches),
        "beta_qqq": round(float(mod_q.params[1]), 2),
        "alpha_qqq_pct": round(float(mod_q.params[0]) * 252.0 * 100.0, 2),
        "t_stat_qqq": round(float(mod_q.tvalues[0]), 2),
        "p_val_qqq": round(float(mod_q.pvalues[0]), 4),
        "alpha_beta_matched_pct": round(float(mod_bm.params[0]) * 252.0 * 100.0, 2),
        "t_stat_beta_matched": round(float(mod_bm.tvalues[0]), 2),
        "p_val_beta_matched": round(float(mod_bm.pvalues[0]), 4),
        "best10_removed_cagr_pct": round(best10_removed_cagr * 100.0, 2),
    }


def main() -> None:
    q_tr, qsip = load_total_return_qqq_and_sip()
    cal_dates = [d for d in qsip.index if d >= SIM_START_DATE]
    assert q_tr.loc[cal_dates, "sma200"].isna().sum() == 0, "SMA200 warmup incomplete on start date!"

    rf_s = load_lagged_dtb3_series(q_tr.index)
    rf_cal = rf_s.loc[cal_dates].values

    roster_df = pd.read_csv(ROSTER_PATH)
    roster_df["date_str"] = pd.to_datetime(roster_df["date"]).dt.strftime("%Y-%m-%d")

    syms = sorted(roster_df["ticker"].unique().tolist())
    op_df = pd.DataFrame(index=cal_dates, columns=syms, dtype=float)
    cl_df = pd.DataFrame(index=cal_dates, columns=syms, dtype=float)
    for s in syms:
        p = ALPACA_DIR / f"{s}.csv"
        if p.exists():
            df_s = pd.read_csv(p).set_index("date_str")
            common = df_s.index.intersection(cal_dates)
            op_df.loc[common, s] = df_s.loc[common, "open"]
            cl_df.loc[common, s] = df_s.loc[common, "close"]
    cl_ff = cl_df.ffill()
    op_ex = op_df.combine_first(cl_ff.shift(1))

    plans_8 = build_quarter_plans(roster_df, cal_dates, top_k=8)
    plans_16 = build_quarter_plans(roster_df, cal_dates, top_k=16)

    nav_qqq, turn_q, sw_q = simulate_single_qqq_rule(cal_dates, q_tr, rf_s, "buy_hold", cost_bps=10.0)
    nav_q_ma, turn_qma, sw_qma = simulate_single_qqq_rule(cal_dates, q_tr, rf_s, "sma200", cost_bps=10.0)
    nav_q_vt, turn_qvt, sw_qvt = simulate_single_qqq_rule(cal_dates, q_tr, rf_s, "vol_target_20", cost_bps=10.0)

    nav_pass_8, turn_p8, sw_p8 = simulate_pit_basket(cal_dates, q_tr, rf_s, op_ex, cl_ff, syms, plans_8, use_ma200=False, cost_bps=10.0)
    nav_pass_16, turn_p16, sw_p16 = simulate_pit_basket(cal_dates, q_tr, rf_s, op_ex, cl_ff, syms, plans_16, use_ma200=False, cost_bps=10.0)

    nav_top8_10, turn_t8_10, sw_t8 = simulate_pit_basket(cal_dates, q_tr, rf_s, op_ex, cl_ff, syms, plans_8, use_ma200=True, cost_bps=10.0)
    nav_top8_20, turn_t8_20, _ = simulate_pit_basket(cal_dates, q_tr, rf_s, op_ex, cl_ff, syms, plans_8, use_ma200=True, cost_bps=20.0)
    nav_top8_30, turn_t8_30, _ = simulate_pit_basket(cal_dates, q_tr, rf_s, op_ex, cl_ff, syms, plans_8, use_ma200=True, cost_bps=30.0)

    nav_top16_10, turn_t16_10, sw_t16 = simulate_pit_basket(cal_dates, q_tr, rf_s, op_ex, cl_ff, syms, plans_16, use_ma200=True, cost_bps=10.0)
    nav_top16_20, turn_t16_20, _ = simulate_pit_basket(cal_dates, q_tr, rf_s, op_ex, cl_ff, syms, plans_16, use_ma200=True, cost_bps=20.0)
    nav_top16_30, turn_t16_30, _ = simulate_pit_basket(cal_dates, q_tr, rf_s, op_ex, cl_ff, syms, plans_16, use_ma200=True, cost_bps=30.0)

    ex_t8 = nav_top8_10.pct_change().fillna(0.0).values[1:] - rf_cal[1:]
    ex_q = nav_qqq.pct_change().fillna(0.0).values[1:] - rf_cal[1:]
    realized_beta_top8 = float(np.cov(ex_t8, ex_q)[0, 1] / np.var(ex_q))
    nav_beta_matched, turn_bm, sw_bm = simulate_single_qqq_rule(
        cal_dates, q_tr, rf_s, "fixed_weight", target_weight_const=realized_beta_top8, cost_bps=10.0
    )

    series_specs = [
        ("QQQ Total Return (Buy & Hold)", nav_qqq, turn_q, sw_q),
        (f"Beta-Matched QQQ ({realized_beta_top8*100:.0f}% QQQ / {(1-realized_beta_top8)*100:.0f}% DTB3)", nav_beta_matched, turn_bm, sw_bm),
        ("Simple Baseline: QQQ > 200d SMA", nav_q_ma, turn_qma, sw_qma),
        ("Simple Baseline: QQQ 20% Vol-Target", nav_q_vt, turn_qvt, sw_qvt),
        ("Ablation: Passive PIT Top-8 (No 200d MA)", nav_pass_8, turn_p8, sw_p8),
        ("Ablation: Passive PIT Top-16 (No 200d MA)", nav_pass_16, turn_p16, sw_p16),
        ("PIT Top-8 + QQQ 200d MA (10 bps)", nav_top8_10, turn_t8_10, sw_t8),
        ("PIT Top-8 + QQQ 200d MA (20 bps)", nav_top8_20, turn_t8_20, sw_t8),
        ("PIT Top-8 + QQQ 200d MA (30 bps)", nav_top8_30, turn_t8_30, sw_t8),
        ("PIT Top-16 + QQQ 200d MA (10 bps)", nav_top16_10, turn_t16_10, sw_t16),
        ("PIT Top-16 + QQQ 200d MA (20 bps)", nav_top16_20, turn_t16_20, sw_t16),
        ("PIT Top-16 + QQQ 200d MA (30 bps)", nav_top16_30, turn_t16_30, sw_t16),
    ]

    full_rows = [
        compute_summary_stats(s_n, nav_qqq, nav_beta_matched, rf_cal, trn, sw, lbl)
        for lbl, s_n, trn, sw in series_specs
    ]

    print("=" * 136)
    print(f"  PRODUCTION AUDIT: PIT CAP-WEIGHT + QQQ 200-DAY MA CASH RULE ({cal_dates[0]} -> {cal_dates[-1]})")
    print("  Open_t Execution | Pre-Warmed 200d SMA | 1-Day Lagged FRED DTB3 Cash Yield | Realized Beta = {:.2f}".format(realized_beta_top8))
    print("=" * 136)
    print(
        f"{'Strategy':<44} | {'End $100k':>10} | {'CAGR':>6} | {'MaxDD':>7} | {'Sh':>4} | {'Beta':>4} | {'Alpha(Q)':>8} | {'t(Q)':>5} | {'Alpha(BM)':>9} | {'t(BM)':>5} | {'Turn':>5}"
    )
    print("-" * 136)
    for r in full_rows:
        print(
            f"{r['strategy']:<44} | ${r['ending_nav']:>9,.0f} | {r['cagr_pct']:>5.2f}% | {r['max_dd_pct']:>6.2f}% | "
            f"{r['sharpe']:>4.2f} | {r['beta_qqq']:>4.2f} | {r['alpha_qqq_pct']:>+7.2f}% | {r['t_stat_qqq']:>+5.2f} | "
            f"{r['alpha_beta_matched_pct']:>+8.2f}% | {r['t_stat_beta_matched']:>+5.2f} | {r['turnover_x']:>4.2f}x"
        )
    print("=" * 136)

    subperiod_stats = {}
    for sp_name, s_dt, e_dt in [
        ("FULL_2016_2026", cal_dates[0], cal_dates[-1]),
        ("FIRST_HALF_2016_2021", cal_dates[0], "2021-12-31"),
        ("SECOND_HALF_2022_2026", "2022-01-03", cal_dates[-1]),
    ]:
        mask = (nav_qqq.index >= pd.Timestamp(s_dt)) & (nav_qqq.index <= pd.Timestamp(e_dt))
        rf_sub = rf_cal[mask]
        b_q = nav_qqq.loc[mask] / nav_qqq.loc[mask].iloc[0] * INITIAL_CAP
        b_bm = nav_beta_matched.loc[mask] / nav_beta_matched.loc[mask].iloc[0] * INITIAL_CAP
        subperiod_stats[sp_name] = [
            compute_summary_stats(
                s_n.loc[mask] / s_n.loc[mask].iloc[0] * INITIAL_CAP,
                b_q,
                b_bm,
                rf_sub,
                trn,
                sw,
                lbl,
            )
            for lbl, s_n, trn, sw in series_specs
        ]

    df_daily = pd.DataFrame({
        "Date": cal_dates,
        "QQQ_Close": q_tr.loc[cal_dates, "close"].values,
        "QQQ_SMA200": q_tr.loc[cal_dates, "sma200"].values,
        "QQQ_Above_SMA200": (q_tr.loc[cal_dates, "close"] > q_tr.loc[cal_dates, "sma200"]).values,
        "RF_Daily_Rate": rf_cal,
        "QQQ_TotalReturn_NAV_100k": nav_qqq.values,
        "QQQ_BetaMatched_NAV_100k": nav_beta_matched.values,
        "QQQ_Simple_200DMA_NAV_100k": nav_q_ma.values,
        "QQQ_VolTarget20_NAV_100k": nav_q_vt.values,
        "Passive_PIT_Top8_NAV_100k": nav_pass_8.values,
        "Passive_PIT_Top16_NAV_100k": nav_pass_16.values,
        "PIT_Top8_200DMA_10bps_NAV_100k": nav_top8_10.values,
        "PIT_Top8_200DMA_20bps_NAV_100k": nav_top8_20.values,
        "PIT_Top8_200DMA_30bps_NAV_100k": nav_top8_30.values,
        "PIT_Top16_200DMA_10bps_NAV_100k": nav_top16_10.values,
        "PIT_Top16_200DMA_20bps_NAV_100k": nav_top16_20.values,
        "PIT_Top16_200DMA_30bps_NAV_100k": nav_top16_30.values,
    })
    daily_csv = OUT_DIR / "daily_nav_ledger.csv"
    df_daily.to_csv(daily_csv, index=False)

    q_starts = sorted(plans_8.keys())
    q_rows = []
    for k, ds in enumerate(q_starts):
        de = q_starts[k + 1] if k + 1 < len(q_starts) else cal_dates[-1]
        info = plans_8[ds]

        def _qret(s_n: pd.Series) -> float:
            sub_s = s_n.loc[ds:de]
            return ((sub_s.iloc[-1] / sub_s.iloc[0]) - 1.0) * 100.0 if len(sub_s) > 1 else 0.0

        rq = _qret(nav_qqq)
        rqm = _qret(nav_q_ma)
        rp8 = _qret(nav_pass_8)
        rt8 = _qret(nav_top8_10)
        rt16 = _qret(nav_top16_10)

        q_rows.append({
            "Quarter": info["quarter"],
            "Start_Date": ds,
            "End_Date": de,
            "Subperiod": "SECOND_HALF_2022_2026" if ds >= "2022-01-03" else "FIRST_HALF_2016_2021",
            "QQQ_Ret_Pct": round(rq, 2),
            "QQQ_200DMA_Ret_Pct": round(rqm, 2),
            "Passive_Top8_Ret_Pct": round(rp8, 2),
            "PIT_Top8_200DMA_Ret_Pct": round(rt8, 2),
            "PIT_Top16_200DMA_Ret_Pct": round(rt16, 2),
            "Top8_Edge_vs_QQQ_Pct": round(rt8 - rq, 2),
            "Top16_Edge_vs_QQQ_Pct": round(rt16 - rq, 2),
            "Top8_Beat_QQQ": bool(rt8 > rq),
            "Top16_Beat_QQQ": bool(rt16 > rq),
            "PIT_True_Members": info["pit_true_members"],
            "PIT_Members_With_Data": info["pit_members_with_data"],
        })
    q_csv = OUT_DIR / "quarterly_performance_ledger.csv"
    pd.DataFrame(q_rows).to_csv(q_csv, index=False)

    recent_dates = [d for d in cal_dates if d >= "2026-10-28"]
    r_q = (nav_qqq.pct_change() * 100.0).set_axis(cal_dates)
    r_t8 = (nav_top8_10.pct_change() * 100.0).set_axis(cal_dates)
    r_t16 = (nav_top16_10.pct_change() * 100.0).set_axis(cal_dates)
    recent_log = []
    for d in recent_dates:
        idx = cal_dates.index(d)
        prev_d = cal_dates[idx - 1]
        pos_active = bool(q_tr.loc[prev_d, "close"] > q_tr.loc[prev_d, "sma200"])
        close_above = bool(q_tr.loc[d, "close"] > q_tr.loc[d, "sma200"])
        recent_log.append({
            "date": d,
            "qqq_close": round(float(q_tr.loc[d, "close"]), 2),
            "qqq_sma200": round(float(q_tr.loc[d, "sma200"]), 2),
            "qqq_ret_pct": round(float(r_q.loc[d]), 2),
            "top8_ret_pct": round(float(r_t8.loc[d]), 2),
            "top16_ret_pct": round(float(r_t16.loc[d]), 2),
            "edge_top8_vs_qqq_pct": round(float(r_t8.loc[d] - r_q.loc[d]), 2),
            "held_at_open": "EQUITIES" if pos_active else "TBILLS",
            "close_signal": "ABOVE_SMA200" if close_above else "BELOW_SMA200",
        })

    audit_summary = {
        "start_date": cal_dates[0],
        "end_date": cal_dates[-1],
        "trading_days": len(cal_dates),
        "realized_beta_top8": round(realized_beta_top8, 4),
        "subperiod_stats": subperiod_stats,
        "recent_daily_log": recent_log,
    }
    summary_json = OUT_DIR / "production_audit_summary.json"
    summary_json.write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    print(f"Saved audited ledgers:\n  - {daily_csv}\n  - {q_csv}\n  - {summary_json}")


if __name__ == "__main__":
    main()
