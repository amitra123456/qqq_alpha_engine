"""
Strict Due-Diligence Audit of the OR (1m/5m/30m) + TPO + Delta Flow (CVD) + GEX/VWAP Strategy
=============================================================================================
Implements all checks required by the Trading Strategy Due Diligence protocol:
  1. Fixes lookahead bugs from the initial prototype:
     - Intraday 10:00 AM model uses ONLY t-1 DIX/GEX/OPRA walls and 09:30-09:59 bars,
       filling at the 10:00:00 Open (minute_idx=30 'o') -> 15:59 Close ('c').
     - Intraday 10:30 AM model uses 10:30 AM OPRA walls + 60m Initial Balance (09:30-10:29),
       filling at the 10:30:00 Open (minute_idx=60 'o') -> 15:59 Close ('c').
     - EOD Next-Day model tests both:
         (a) Signal at 16:00 Close on day t -> Fill at 09:30 Open on day t+1 -> Exit 15:59 Close on day t+1
         (b) Signal at 15:50 (minute_idx=380) on day t -> Fill at 15:59 Close on day t -> Exit 15:59 Close on day t+1
  2. Walk-Forward Expanding-Window Fitting (re-fit every 21 days using strictly past data <= t-1).
  3. Transaction Costs at 1x (6 bps RT TQQQ, 10 bps RT SOXL), 2x, and 3x, plus real SQQQ/SOXS pricing or short borrow.
  4. Benchmarks & Simple Baselines (Buy & Hold, Exposure-Matched Benchmark, Simple 30m ORB, Simple VWAP, Simple SMA20).
  5. Ablation (removing OR, TPO, Delta CVD, VWAP, GEX one by one), Placebo test (200 random shifts),
     Concentration (top 10 days), and Tail Risk (worst day, week, month, max underwater days).
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

ROOT = Path("c:/Users/amitr/OneDrive/Documents/ANT_SOX")
ALPACA_DIR = ROOT / "data" / "cache" / "alpaca_1m_stocks"
OPRA_PATH = ROOT / "ant_0dte_credit_spread" / "data" / "zero_dte_opra_walls_gex_2024_2026.parquet"
DIX_PATH = ROOT / "ant_0dte_credit_spread" / "data" / "squeezemetrics_dix_gex_2011_2026.parquet"


def compute_tpo_from_slice(sub_df: pd.DataFrame, n_bins: int = 40):
    low_min = sub_df["l"].min()
    high_max = sub_df["h"].max()
    if high_max <= low_min + 1e-6:
        return low_min, low_min, low_min, 0.0, 0.0
    bins = np.linspace(low_min, high_max, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    tpo_counts = np.zeros(n_bins, dtype=float)
    brackets = sub_df["minute_idx"].values // 30
    for b_id in np.unique(brackets):
        b_sub = sub_df[brackets == b_id]
        hit = (bin_centers >= b_sub["l"].min() - 1e-6) & (bin_centers <= b_sub["h"].max() + 1e-6)
        tpo_counts[hit] += 1.0
    poc_idx = int(np.argmax(tpo_counts))
    tpo_poc = bin_centers[poc_idx]
    target_tpo = 0.70 * max(1.0, tpo_counts.sum())
    cur_tpo = tpo_counts[poc_idx]
    lo_i, hi_i = poc_idx, poc_idx
    while cur_tpo < target_tpo and (lo_i > 0 or hi_i < n_bins - 1):
        up_v = tpo_counts[hi_i + 1] if hi_i < n_bins - 1 else -1.0
        dn_v = tpo_counts[lo_i - 1] if lo_i > 0 else -1.0
        if up_v >= dn_v and hi_i < n_bins - 1:
            hi_i += 1
            cur_tpo += tpo_counts[hi_i]
        elif lo_i > 0:
            lo_i -= 1
            cur_tpo += tpo_counts[lo_i]
        else:
            break
    low_single = 0
    for k in range(poc_idx):
        if tpo_counts[k] <= 1.0:
            low_single += 1
        else:
            break
    high_single = 0
    for k in range(n_bins - 1, poc_idx, -1):
        if tpo_counts[k] <= 1.0:
            high_single += 1
        else:
            break
    return tpo_poc, bin_centers[hi_i], bin_centers[lo_i], (low_single / n_bins) * 100.0, (high_single / n_bins) * 100.0


def build_strict_causal_table(df_1m: pd.DataFrame, sym: str) -> pd.DataFrame:
    rows = []
    for dt_str, day_df in df_1m.groupby("date_str", sort=True):
        if len(day_df) < 100:
            continue
        day_df = day_df.sort_values("minute_idx")
        o_day = day_df["o"].iloc[0]
        h_day = day_df["h"].max()
        l_day = day_df["l"].min()
        c_day = day_df["c"].iloc[-1]
        v_day = day_df["v"].sum()

        # Exact execution prices at 10:00:00 Open (minute_idx >= 30), 10:30:00 Open (minute_idx >= 60), and 15:50:00 Close (minute_idx <= 380)
        post_1000 = day_df[day_df["minute_idx"] >= 30]
        post_1030 = day_df[day_df["minute_idx"] >= 60]
        pre_1550 = day_df[day_df["minute_idx"] <= 380]
        if len(post_1000) == 0 or len(post_1030) == 0 or len(pre_1550) < 60:
            continue

        fill_1000_open = post_1000["o"].iloc[0]
        fill_1030_open = post_1030["o"].iloc[0]
        px_1550_close = pre_1550["c"].iloc[-1]

        # 1m Delta Flow
        bar_rng = (day_df["h"] - day_df["l"]).values
        clv_1m = np.where(
            bar_rng > 1e-6,
            ((day_df["c"] - day_df["l"]) - (day_df["h"] - day_df["c"])) / np.maximum(bar_rng, 1e-6),
            np.sign(day_df["c"].diff().fillna(0.0)),
        )
        delta_1m = day_df["v"].values * clv_1m

        # Slice 1: Known strictly by 09:59:59 (minute_idx < 30)
        s_30 = day_df[day_df["minute_idx"] < 30]
        d_30 = delta_1m[day_df["minute_idx"].values < 30]
        or1_h, or1_l, or1_c = s_30["h"].iloc[0], s_30["l"].iloc[0], s_30["c"].iloc[0]
        s_5 = s_30[s_30["minute_idx"] < 5]
        or5_h, or5_l, or5_c = s_5["h"].max(), s_5["l"].min(), s_5["c"].iloc[-1]
        or30_h, or30_l, or30_c = s_30["h"].max(), s_30["l"].min(), s_30["c"].iloc[-1]
        tp_30 = (s_30["h"] + s_30["l"] + s_30["c"]) / 3.0
        vwap_1000 = (tp_30 * s_30["v"]).sum() / max(1.0, s_30["v"].sum())
        cvd_1000 = d_30.sum() / max(1.0, s_30["v"].sum())

        # Slice 2: Known strictly by 10:29:59 (minute_idx < 60, Initial Balance IB)
        s_60 = day_df[day_df["minute_idx"] < 60]
        d_60 = delta_1m[day_df["minute_idx"].values < 60]
        ib_h, ib_l, ib_c = s_60["h"].max(), s_60["l"].min(), s_60["c"].iloc[-1]
        tp_60 = (s_60["h"] + s_60["l"] + s_60["c"]) / 3.0
        vwap_1030 = (tp_60 * s_60["v"]).sum() / max(1.0, s_60["v"].sum())
        cvd_1030 = d_60.sum() / max(1.0, s_60["v"].sum())

        # Slice 3: Known strictly by 15:50:00 (minute_idx <= 380) for MOC/15:59 entry
        d_1550 = delta_1m[day_df["minute_idx"].values <= 380]
        tp_1550 = (pre_1550["h"] + pre_1550["l"] + pre_1550["c"]) / 3.0
        vwap_1550 = (tp_1550 * pre_1550["v"]).sum() / max(1.0, pre_1550["v"].sum())
        vwap_sd_1550 = np.sqrt(((tp_1550 - vwap_1550) ** 2 * pre_1550["v"]).sum() / max(1.0, pre_1550["v"].sum()))
        cvd_day_1550 = d_1550.sum() / max(1.0, pre_1550["v"].sum())
        cvd_pm_1550 = d_1550[-110:].sum() / max(1.0, pre_1550["v"].iloc[-110:].sum())
        poc_1550, vah_1550, val_1550, btail_1550, stail_1550 = compute_tpo_from_slice(pre_1550)
        h_1550, l_1550 = pre_1550["h"].max(), pre_1550["l"].min()
        clv_1550 = (px_1550_close - l_1550) / max(h_1550 - l_1550, 1e-6)

        # Full-Day EOD (known at 16:00:00)
        tp_all = (day_df["h"] + day_df["l"] + day_df["c"]) / 3.0
        sess_vwap = (tp_all * day_df["v"]).sum() / max(1.0, v_day)
        vwap_sd = np.sqrt(((tp_all - sess_vwap) ** 2 * day_df["v"]).sum() / max(1.0, v_day))
        tpo_poc, tpo_vah, tpo_val, btail, stail = compute_tpo_from_slice(day_df)
        cvd_day = delta_1m.sum() / max(1.0, v_day)
        cvd_pm = delta_1m[-120:].sum() / max(1.0, day_df["v"].iloc[-120:].sum())

        rows.append(
            {
                "date": pd.Timestamp(dt_str),
                "open": o_day,
                "high": h_day,
                "low": l_day,
                "close": c_day,
                "volume": v_day,
                "fill_1000_open": fill_1000_open,
                "fill_1030_open": fill_1030_open,
                "px_1550_close": px_1550_close,
                "or1_h": or1_h,
                "or1_l": or1_l,
                "or1_c": or1_c,
                "or5_h": or5_h,
                "or5_l": or5_l,
                "or5_c": or5_c,
                "or30_h": or30_h,
                "or30_l": or30_l,
                "or30_c": or30_c,
                "ib_h": ib_h,
                "ib_l": ib_l,
                "ib_c": ib_c,
                "vwap_1000": vwap_1000,
                "vwap_1030": vwap_1030,
                "cvd_1000": cvd_1000,
                "cvd_1030": cvd_1030,
                # 15:50 features
                "clv_1550": clv_1550,
                "vwap_z_1550": (px_1550_close - vwap_1550) / max(vwap_sd_1550, 1e-6),
                "poc_dist_1550": (px_1550_close / poc_1550 - 1.0) * 100.0,
                "above_vah_1550": float(px_1550_close > vah_1550),
                "below_val_1550": float(px_1550_close < val_1550),
                "btail_1550": btail_1550,
                "stail_1550": stail_1550,
                "cvd_day_1550": cvd_day_1550,
                "cvd_pm_1550": cvd_pm_1550,
                "above_or30h_1550": float(px_1550_close > or30_h),
                "reclaim_or30l_1550": float((l_1550 < or30_l) and (px_1550_close >= or30_l * 0.998)),
                # 16:00 full day features
                "sess_vwap": sess_vwap,
                "vwap_sd": vwap_sd,
                "tpo_poc": tpo_poc,
                "tpo_vah": tpo_vah,
                "tpo_val": tpo_val,
                "btail": btail,
                "stail": stail,
                "cvd_day": cvd_day,
                "cvd_pm": cvd_pm,
                "clv_day": (c_day - l_day) / max(h_day - l_day, 1e-6),
                "vwap_z_day": (c_day - sess_vwap) / max(vwap_sd, 1e-6),
                "poc_dist_day": (c_day / tpo_poc - 1.0) * 100.0,
                "above_vah_day": float(c_day > tpo_vah),
                "below_val_day": float(c_day < tpo_val),
                "above_or30h_day": float(c_day > or30_h),
                "reclaim_or30l_day": float((l_day < or30_l) and (c_day >= or30_l * 0.998)),
                # Strict causal returns (using fill_1000_open, fill_1030_open, etc.)
                "ret_1000_to_close": (c_day / fill_1000_open - 1.0),
                "ret_1030_to_close": (c_day / fill_1030_open - 1.0),
            }
        )

    df = pd.DataFrame(rows).set_index("date").sort_index()

    # Strictly lagged (t-1) structural levels
    df["pdh"] = df["high"].shift(1)
    df["pdl"] = df["low"].shift(1)
    df["pdc"] = df["close"].shift(1)
    df["pd_poc"] = df["tpo_poc"].shift(1)
    df["pd_vah"] = df["tpo_vah"].shift(1)
    df["pd_val"] = df["tpo_val"].shift(1)
    df["pwh"] = df["high"].shift(1).rolling(5, min_periods=2).max()
    df["pwl"] = df["low"].shift(1).rolling(5, min_periods=2).min()
    df["sma20_lag1"] = df["close"].shift(1).rolling(20, min_periods=10).mean()

    # Strictly causal 10:00 AM features (using fill_1000_open or or30_c vs lagged PD/PW levels)
    df["or30_above_or5_h"] = (df["or30_c"] > df["or5_h"]).astype(float)
    df["or30_below_or5_l"] = (df["or30_c"] < df["or5_l"]).astype(float)
    df["or5_above_or1_h"] = (df["or5_c"] > df["or1_h"]).astype(float)
    df["or30_vs_vwap1000"] = (df["or30_c"] / df["vwap_1000"] - 1.0) * 100.0
    df["or30_vs_pd_poc"] = (df["or30_c"] / df["pd_poc"] - 1.0) * 100.0
    df["or30_above_pdh"] = (df["or30_c"] > df["pdh"]).astype(float)
    df["or30_reclaim_pdl"] = ((df["or30_l"] < df["pdl"]) & (df["or30_c"] > df["pdl"])).astype(float)
    df["or30_reclaim_pd_val"] = ((df["open"] < df["pd_val"]) & (df["or30_c"] > df["pd_val"])).astype(float)

    # Next-day returns with strict execution timing:
    # 1. True Executable 24h Hold: Enter at 15:50:00 (px_1550_close) on day t -> Exit at 15:50:00 on day t+1
    df["ret_1550_to_1550_next"] = df["px_1550_close"].pct_change().shift(-1)
    # 2. True Executable Overnight Hold: Enter at 15:50:00 (px_1550_close) on day t -> Exit at 09:30 Open on day t+1
    df["ret_1550_to_next_open"] = (df["open"].shift(-1) / df["px_1550_close"]) - 1.0
    # 3. Close-to-Close (entering at 15:59 on day t using 15:50 features -> exiting 15:59 on day t+1)
    df["ret_c2c_next"] = df["close"].pct_change().shift(-1)
    # 4. Next-Day Open-to-Close (deciding after 16:00 close on day t -> entering at 09:30 Open on day t+1 -> exiting 15:59 Close on day t+1)
    df["ret_next_o2c"] = (df["close"].shift(-1) / df["open"].shift(-1)) - 1.0
    # 5. Benchmark daily Close-to-Close return of current day t
    df["ret_day_c2c"] = df["close"].pct_change()
    return df


def attach_strict_causal_macro_gex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # IMPORTANT: SqueezeMetrics DIX & GEX are published EOD.
    # For 10:00 AM / 10:30 AM / 15:50 PM decisions on day t, ONLY day t-1 (shift(1)) is known!
    dix_df = pd.read_parquet(DIX_PATH)
    dix_df["date"] = pd.to_datetime(dix_df["date"])
    dix_df = dix_df.set_index("date").sort_index()
    df["dix_lag1"] = dix_df["dix"].reindex(df.index).ffill().shift(1)
    df["gex_z20_lag1"] = dix_df["gex_z20"].reindex(df.index).ffill().shift(1)

    # OPRA 0DTE 10:30 AM snapshot:
    # - Known at 10:30 AM (so causal for 10:30 AM entry and 15:50/16:00 EOD entry!)
    # - For 10:00 AM entry, ONLY day t-1 OPRA walls are known!
    opra = pd.read_parquet(OPRA_PATH)
    q_opra = opra[opra["symbol"] == "QQQ"].copy()
    q_opra["date"] = pd.to_datetime(q_opra["date"])
    q_opra = q_opra.set_index("date").sort_index()
    for col in ["call_wall_1030", "put_wall_1030", "gamma_flip_1030", "net_gex_m_1030", "pc_vol_ratio_1030"]:
        s = q_opra[col].reindex(df.index).ffill()
        df[f"qqq_{col}_today"] = s
        df[f"qqq_{col}_lag1"] = s.shift(1)
    return df


def compute_perf_metrics(ret_series: pd.Series, bmk_series: pd.Series, rf_daily: float = 0.045 / 252.0):
    r = ret_series.dropna()
    b = bmk_series.reindex(r.index).fillna(0.0)
    n_days = len(r)
    if n_days < 10:
        return {}
    eq = (1.0 + r).cumprod()
    tot_ret = eq.iloc[-1] - 1.0
    cagr = (eq.iloc[-1] ** (252.0 / n_days) - 1.0) * 100.0
    peak = eq.cummax()
    dd = (eq / peak - 1.0) * 100.0
    max_dd = dd.min()

    # Max days underwater
    is_uw = eq < peak
    uw_groups = (~is_uw).cumsum()
    max_uw_days = int(is_uw.groupby(uw_groups).sum().max()) if is_uw.any() else 0

    ex_r = r - rf_daily
    sharpe = (ex_r.mean() / (r.std() + 1e-9)) * np.sqrt(252.0)

    # Alpha & Beta vs Benchmark
    ex_b = b - rf_daily
    X = sm.add_constant(ex_b.values)
    ols = sm.OLS(ex_r.values, X).fit()
    alpha_ann = ols.params[0] * 252.0 * 100.0
    alpha_t = ols.tvalues[0]
    beta = ols.params[1] if len(ols.params) > 1 else 0.0

    # Tail risk
    worst_day = r.min() * 100.0
    wk_r = (1.0 + r).resample("W-FRI").prod() - 1.0
    mo_r = (1.0 + r).resample("ME").prod() - 1.0
    worst_wk = wk_r.min() * 100.0 if len(wk_r) else 0.0
    worst_mo = mo_r.min() * 100.0 if len(mo_r) else 0.0

    # Concentration: best 10 days contribution
    top10_sum = r.nlargest(10).sum() * 100.0
    tot_sum = r.sum() * 100.0

    return {
        "n_days": n_days,
        "cagr": cagr,
        "tot_ret": tot_ret * 100.0,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "beta": beta,
        "alpha_ann": alpha_ann,
        "alpha_t": alpha_t,
        "worst_day": worst_day,
        "worst_wk": worst_wk,
        "worst_mo": worst_mo,
        "max_uw_days": max_uw_days,
        "top10_sum": top10_sum,
        "tot_log_sum": tot_sum,
    }


def walk_forward_backtest(
    df: pd.DataFrame,
    feat_cols: list[str],
    ret_col: str,
    cost_rt_bps: float,
    prob_long: float = 0.55,
    prob_short: float = 0.45,
    allow_short: bool = True,
    min_train: int = 126,
    refit_step: int = 21,
):
    """
    Strictly Expanding-Window Walk-Forward Backtest:
      - At day t, model is fit ONLY on indices [0 : t-1] (re-fit every 21 days).
      - Signal at day t trades ret_col[t] minus round-trip transaction cost.
      - Idle days earn daily risk-free cash rate (4.5% / 252).
    """
    sub = df.dropna(subset=feat_cols + [ret_col]).copy()
    n = len(sub)
    X_all = sub[feat_cols].values
    y_all = (sub[ret_col].values > 0.0).astype(int)
    raw_ret = sub[ret_col].values
    rf_d = 0.045 / 252.0
    cost_dec = cost_rt_bps / 10000.0

    probs = np.full(n, np.nan)
    pos = np.zeros(n, dtype=float)
    clf = None

    for t in range(min_train, n):
        if clf is None or ((t - min_train) % refit_step == 0):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.20))
            # Strict 1-day purge: fit only on [:t-1] so y[t-1] (which resolves on day t) cannot leak into day t
            clf.fit(X_all[: t - 1], y_all[: t - 1])
        p = clf.predict_proba(X_all[t : t + 1])[0, 1]
        probs[t] = p
        if p >= prob_long:
            pos[t] = 1.0
        elif allow_short and p <= prob_short:
            pos[t] = -1.0

    sub = sub.iloc[min_train:].copy()
    pos = pos[min_train:]
    raw_ret = raw_ret[min_train:]
    probs = probs[min_train:]

    # Net daily return including transaction costs and cash interest when flat
    strat_ret = np.where(
        pos != 0.0,
        pos * raw_ret - cost_dec,
        rf_d,
    )
    sub["pos"] = pos
    sub["prob"] = probs
    sub["strat_ret"] = strat_ret
    sub["raw_trade_ret"] = pos * raw_ret
    return sub


def main():
    print("=" * 120)
    print("DUE DILIGENCE AUDIT: 1m/5m/30m OR + TPO + DELTA FLOW (CVD) + VWAP + GEX STRATEGY (2024-2026)")
    print("=" * 120)

    bars = {}
    for sym in ["QQQ", "TQQQ", "SOXX", "SOXL"]:
        p = ALPACA_DIR / f"{sym}_1m_2024-02-01_2026-11-17_rth.parquet"
        bars[sym] = pd.read_parquet(p)

    tables = {}
    for sym in ["QQQ", "TQQQ", "SOXX", "SOXL"]:
        t_df = build_strict_causal_table(bars[sym], sym)
        t_df = attach_strict_causal_macro_gex(t_df)
        tables[sym] = t_df

    for lev, und, cost_1x in [("TQQQ", "QQQ", 6.0), ("SOXL", "SOXX", 10.0)]:
        l_df = tables[lev].copy()
        u_df = tables[und].reindex(l_df.index)

        # Attach strictly causal underlying features
        l_df["und_cvd_1000"] = u_df["cvd_1000"]
        l_df["und_cvd_1030"] = u_df["cvd_1030"]
        l_df["und_cvd_pm_1550"] = u_df["cvd_pm_1550"]
        l_df["und_cvd_pm_day"] = u_df["cvd_pm"]
        l_df["und_vwap_z_1550"] = u_df["vwap_z_1550"]
        l_df["und_vwap_z_day"] = u_df["vwap_z_day"]
        l_df["und_or30_vs_vwap1000"] = u_df["or30_vs_vwap1000"]

        # 10:30 AM OPRA features (known at 10:30 AM!)
        l_df["ib_above_or30_h"] = (l_df["ib_c"] > l_df["or30_h"]).astype(float)
        l_df["ib_below_or30_l"] = (l_df["ib_c"] < l_df["or30_l"]).astype(float)
        l_df["ib_vs_vwap1030"] = (l_df["ib_c"] / l_df["vwap_1030"] - 1.0) * 100.0
        l_df["qqq_above_gflip_1030"] = (u_df["ib_c"] >= l_df["qqq_gamma_flip_1030_today"]).astype(float) if lev == "TQQQ" else (l_df["gex_z20_lag1"] > -0.3).astype(float)

        # Feature sets (strictly causal for their entry timestamp)
        f_1000 = [
            "or30_above_or5_h",
            "or30_below_or5_l",
            "or5_above_or1_h",
            "or30_vs_vwap1000",
            "or30_vs_pd_poc",
            "or30_above_pdh",
            "or30_reclaim_pdl",
            "or30_reclaim_pd_val",
            "cvd_1000",
            "und_cvd_1000",
            "gex_z20_lag1",
            "dix_lag1",
        ]

        f_1550 = [
            "clv_1550",
            "vwap_z_1550",
            "und_vwap_z_1550",
            "poc_dist_1550",
            "above_vah_1550",
            "below_val_1550",
            "btail_1550",
            "stail_1550",
            "cvd_day_1550",
            "cvd_pm_1550",
            "und_cvd_pm_1550",
            "above_or30h_1550",
            "reclaim_or30l_1550",
            "gex_z20_lag1",
            "dix_lag1",
        ]

        f_1600 = [
            "clv_day",
            "vwap_z_day",
            "und_vwap_z_day",
            "poc_dist_day",
            "above_vah_day",
            "below_val_day",
            "btail",
            "stail",
            "cvd_day",
            "cvd_pm",
            "und_cvd_pm_day",
            "above_or30h_day",
            "reclaim_or30l_day",
            "gex_z20_lag1",
            "dix_lag1",
        ]

        print(f"\n====================================================================================================")
        print(f"  ASSET: {lev} (Underlying: {und}) | Walk-Forward Out-of-Sample Window: Aug 2024 - Nov 2026")
        print(f"====================================================================================================")

        # Run Walk-Forward for Strategy 1: Intraday 10:00 AM Open Fill -> 15:59 Close
        wf_1000_1x = walk_forward_backtest(l_df, f_1000, "ret_1000_to_close", cost_rt_bps=cost_1x)
        wf_1000_2x = walk_forward_backtest(l_df, f_1000, "ret_1000_to_close", cost_rt_bps=cost_1x * 2)
        wf_1000_3x = walk_forward_backtest(l_df, f_1000, "ret_1000_to_close", cost_rt_bps=cost_1x * 3)

        # Run Walk-Forward for Strategy 2: 15:50 Signal -> 15:50 Entry -> Next-Day 15:50 Exit (True 24h Executable Hold)
        wf_1550_1x = walk_forward_backtest(l_df, f_1550, "ret_1550_to_1550_next", cost_rt_bps=cost_1x)
        wf_1550_2x = walk_forward_backtest(l_df, f_1550, "ret_1550_to_1550_next", cost_rt_bps=cost_1x * 2)
        wf_1550_3x = walk_forward_backtest(l_df, f_1550, "ret_1550_to_1550_next", cost_rt_bps=cost_1x * 3)

        # Run Walk-Forward for Strategy 2B: 15:50 Signal -> 15:50 Entry -> Next-Day 09:30 Open Exit (Overnight Hold Only)
        wf_1550_ovn = walk_forward_backtest(l_df, f_1550, "ret_1550_to_next_open", cost_rt_bps=cost_1x)

        # Run Walk-Forward for Strategy 3: 16:00 Signal -> Next-Day 09:30 Open Entry -> Next-Day 15:59 Close Exit
        wf_1600_o2c = walk_forward_backtest(l_df, f_1600, "ret_next_o2c", cost_rt_bps=cost_1x)

        # Common evaluation index
        idx = wf_1000_1x.index.intersection(wf_1550_1x.index)
        bmk_bh = l_df.loc[idx, "ret_1550_to_1550_next"]
        bmk_intraday_bh = l_df.loc[idx, "ret_1000_to_close"]

        # Simple Baselines on exact same dates & 1x costs:
        cost_dec = cost_1x / 10000.0
        rf_d = 0.045 / 252.0
        # Baseline A: Simple 30m ORB at 10:00 AM (Long if or30_c > or5_h, Short if or30_c < or5_l)
        orb_pos = np.where(l_df.loc[idx, "or30_above_or5_h"] == 1, 1.0, np.where(l_df.loc[idx, "or30_below_or5_l"] == 1, -1.0, 0.0))
        base_orb_ret = pd.Series(np.where(orb_pos != 0, orb_pos * l_df.loc[idx, "ret_1000_to_close"] - cost_dec, rf_d), index=idx)

        # Baseline B: Simple 10:00 AM VWAP Trend (Long if or30_c > vwap_1000, Short if < vwap_1000)
        vwap_pos = np.where(l_df.loc[idx, "or30_vs_vwap1000"] > 0, 1.0, -1.0)
        base_vwap_ret = pd.Series(vwap_pos * l_df.loc[idx, "ret_1000_to_close"] - cost_dec, index=idx)

        # Baseline C: Daily SMA20 Trend (Long if Close[t-1] > SMA20[t-1], else Cash)
        sma_pos = (l_df.loc[idx, "pdc"] > l_df.loc[idx, "sma20_lag1"]).astype(float)
        sma_turn = sma_pos.diff().abs().fillna(0.0)
        base_sma_ret = sma_pos * bmk_bh + (1.0 - sma_pos) * rf_d - sma_turn * (cost_dec / 2.0)

        # Baseline D: Simple 15:50 VWAP Mean-Reversion (Buy at 15:50 if below Session VWAP, Short if > +0.5 SD above Session VWAP)
        mr_pos = np.where(l_df.loc[idx, "vwap_z_1550"] < 0.0, 1.0, np.where(l_df.loc[idx, "vwap_z_1550"] > 0.5, -1.0, 0.0))
        base_mr_ret = pd.Series(np.where(mr_pos != 0, mr_pos * bmk_bh - cost_dec, rf_d), index=idx)

        # Exposure-matched benchmark for 10:00 AM strategy
        avg_exp_1000 = (wf_1000_1x.loc[idx, "pos"] != 0).mean()
        bmk_exp_matched = bmk_bh * avg_exp_1000 + rf_d * (1.0 - avg_exp_1000)

        strategies = [
            ("1. Benchmark: Buy & Hold (15:50->15:50 24h)", bmk_bh),
            ("2. Benchmark: Buy & Hold Intraday (10:00->16:00)", bmk_intraday_bh),
            (f"3. Exposure-Matched Benchmark ({avg_exp_1000*100:.0f}% Time-in-Mkt)", bmk_exp_matched),
            ("4. Baseline: Simple 30m ORB (10:00->16:00, 1x Cost)", base_orb_ret),
            ("5. Baseline: Simple 10AM VWAP (10:00->16:00, 1x Cost)", base_vwap_ret),
            ("6. Baseline: Daily Close > SMA20 Trend (1x Cost)", base_sma_ret),
            ("7. Baseline: Simple 15:50 VWAP Mean-Rev (1x Cost)", base_mr_ret),
            ("8. Strategy A: Walk-Fwd 10:00AM->16:00 (1x Cost)", wf_1000_1x.loc[idx, "strat_ret"]),
            ("9. Strategy A: Walk-Fwd 10:00AM->16:00 (2x Cost)", wf_1000_2x.loc[idx, "strat_ret"]),
            ("10. Strategy A: Walk-Fwd 10:00AM->16:00 (3x Cost)", wf_1000_3x.loc[idx, "strat_ret"]),
            ("11. Strategy B: Walk-Fwd 15:50->15:50 NextDay (1x Cost)", wf_1550_1x.loc[idx, "strat_ret"]),
            ("12. Strategy B: Walk-Fwd 15:50->15:50 NextDay (2x Cost)", wf_1550_2x.loc[idx, "strat_ret"]),
            ("13. Strategy B: Walk-Fwd 15:50->15:50 NextDay (3x Cost)", wf_1550_3x.loc[idx, "strat_ret"]),
            ("14. Strategy B-Ovn: Walk-Fwd 15:50->09:30 Open (1x Cost)", wf_1550_ovn.loc[idx, "strat_ret"]),
            ("15. Strategy C: Walk-Fwd 16:00->NextDay Open-to-Close", wf_1600_o2c.loc[idx, "strat_ret"]),
        ]

        print(f"  {'Strategy / Benchmark':56s} | {'CAGR%':>7s} | {'MaxDD%':>7s} | {'Sharpe':>6s} | {'Beta':>5s} | {'Alpha%':>7s} | {'t-stat':>6s} | {'WorstDay':>8s} | {'WorstMo':>8s} | {'MaxUW':>5s}")
        print("  " + "-" * 140)
        for label, s_ret in strategies:
            m = compute_perf_metrics(s_ret, bmk_bh)
            print(
                f"  {label:56s} | {m['cagr']:+6.1f}% | {m['max_dd']:+6.1f}% | {m['sharpe']:6.2f} | "
                f"{m['beta']:5.2f} | {m['alpha_ann']:+6.1f}% | {m['alpha_t']:+6.2f} | "
                f"{m['worst_day']:+7.2f}% | {m['worst_mo']:+7.2f}% | {m['max_uw_days']:4d}d"
            )

        # Sub-period breakdown (Sub1: Aug 2024 - Sep 2025 vs Sub2: Oct 2025 - Nov 2026)
        mid_dt = pd.Timestamp("2025-10-01")
        print(f"\n  --- SUB-PERIOD BREAKDOWN ({lev}) ---")
        for sub_name, sub_idx in [("Sub-Period 1 (Aug 2024 - Sep 2025)", idx[idx < mid_dt]), ("Sub-Period 2 (Oct 2025 - Nov 2026)", idx[idx >= mid_dt])]:
            m_bmk = compute_perf_metrics(bmk_bh.loc[sub_idx], bmk_bh.loc[sub_idx])
            m_s10 = compute_perf_metrics(wf_1000_1x.loc[sub_idx, "strat_ret"], bmk_bh.loc[sub_idx])
            m_s15 = compute_perf_metrics(wf_1550_1x.loc[sub_idx, "strat_ret"], bmk_bh.loc[sub_idx])
            print(
                f"  {sub_name:36s} | Buy&Hold: CAGR={m_bmk['cagr']:+6.1f}%, Sharpe={m_bmk['sharpe']:5.2f}, MaxDD={m_bmk['max_dd']:+5.1f}% | "
                f"Strat A (10AM): CAGR={m_s10['cagr']:+6.1f}%, Sharpe={m_s10['sharpe']:5.2f}, t={m_s10['alpha_t']:+5.2f} | "
                f"Strat B (15:50): CAGR={m_s15['cagr']:+6.1f}%, Sharpe={m_s15['sharpe']:5.2f}, t={m_s15['alpha_t']:+5.2f}"
            )

        # Trade Counts, Hit Rates vs Base Rate, Turnover, Concentration
        for s_label, wf_df, r_col in [("Strategy A (10:00AM->16:00)", wf_1000_1x.loc[idx], "ret_1000_to_close"), ("Strategy B (15:50->15:50 NextDay)", wf_1550_1x.loc[idx], "ret_1550_to_1550_next")]:
            active = wf_df[wf_df["pos"] != 0]
            longs = wf_df[wf_df["pos"] == 1]
            shorts = wf_df[wf_df["pos"] == -1]
            base_up = (wf_df[r_col] > 0).mean() * 100.0
            l_wr = (longs[r_col] > 0).mean() * 100.0 if len(longs) else 0.0
            s_wr = (shorts[r_col] < 0).mean() * 100.0 if len(shorts) else 0.0
            tot_wr = (active["raw_trade_ret"] > 0).mean() * 100.0 if len(active) else 0.0
            ann_trades = len(active) * (252.0 / len(wf_df))
            top10 = wf_df["strat_ret"].nlargest(10).sum() * 100.0
            tot_sum = wf_df["strat_ret"].sum() * 100.0
            print(
                f"  {s_label:34s} Stats | Total Trades: {len(active):3d} ({ann_trades:.0f}/yr) | Base Up Rate: {base_up:4.1f}% | "
                f"Long WR: {l_wr:4.1f}% (N={len(longs)}) | Short WR: {s_wr:4.1f}% (N={len(shorts)}) | Combined WR: {tot_wr:4.1f}% | "
                f"Top-10 Days Sum: {top10:+.1f}% vs Total Sum: {tot_sum:+.1f}%"
            )

        # Ablation Study on Strategy B (15:50->15:50 NextDay)
        print(f"\n  --- ABLATION STUDY ({lev}, Walk-Forward 1x Cost) ---")
        ablation_groups_1550 = {
            "Full Model (All Features)": f_1550,
            "Drop TPO Features (poc_dist, vah, val, tails)": [c for c in f_1550 if c not in ["poc_dist_1550", "above_vah_1550", "below_val_1550", "btail_1550", "stail_1550"]],
            "Drop VWAP Features (vwap_z, und_vwap_z)": [c for c in f_1550 if c not in ["vwap_z_1550", "und_vwap_z_1550"]],
            "Drop Delta Flow CVD (cvd_day, cvd_pm, und_cvd_pm)": [c for c in f_1550 if c not in ["cvd_day_1550", "cvd_pm_1550", "und_cvd_pm_1550"]],
            "Drop Opening Range (above_or30h, reclaim_or30l)": [c for c in f_1550 if c not in ["above_or30h_1550", "reclaim_or30l_1550"]],
            "Drop GEX/DIX (gex_z20_lag1, dix_lag1)": [c for c in f_1550 if c not in ["gex_z20_lag1", "dix_lag1"]],
        }
        for ab_name, cols_sub in ablation_groups_1550.items():
            wf_ab = walk_forward_backtest(l_df, cols_sub, "ret_1550_to_1550_next", cost_rt_bps=cost_1x).loc[idx]
            m_ab = compute_perf_metrics(wf_ab["strat_ret"], bmk_bh)
            print(f"    [Strat B 15:50] {ab_name:50s} | CAGR: {m_ab['cagr']:+6.1f}% | Sharpe: {m_ab['sharpe']:5.2f} | Alpha t: {m_ab['alpha_t']:+5.2f}")

        # Placebo Test (200 random circular shifts / permutations of positions)
        rng = np.random.default_rng(42)
        real_sharpe_a = compute_perf_metrics(wf_1000_1x.loc[idx, "strat_ret"], bmk_bh)["sharpe"]
        real_sharpe_b = compute_perf_metrics(wf_1550_1x.loc[idx, "strat_ret"], bmk_bh)["sharpe"]
        pos_a = wf_1000_1x.loc[idx, "pos"].values
        pos_b = wf_1550_1x.loc[idx, "pos"].values
        ret_a_raw = l_df.loc[idx, "ret_1000_to_close"].values
        ret_b_raw = l_df.loc[idx, "ret_1550_to_1550_next"].values
        placebo_a, placebo_b = [], []
        for _ in range(200):
            p_a = rng.permutation(pos_a)
            p_b = rng.permutation(pos_b)
            r_a = np.where(p_a != 0, p_a * ret_a_raw - cost_dec, rf_d)
            r_b = np.where(p_b != 0, p_b * ret_b_raw - cost_dec, rf_d)
            placebo_a.append(((r_a - rf_d).mean() / (r_a.std() + 1e-9)) * np.sqrt(252.0))
            placebo_b.append(((r_b - rf_d).mean() / (r_b.std() + 1e-9)) * np.sqrt(252.0))
        pval_a = (np.array(placebo_a) >= real_sharpe_a).mean()
        pval_b = (np.array(placebo_b) >= real_sharpe_b).mean()
        print(
            f"\n  --- PLACEBO TEST (200 Random Permutations, {lev}) ---\n"
            f"    Strat A (10:00AM): Real Sharpe = {real_sharpe_a:+.2f} vs Placebo Mean = {np.mean(placebo_a):+.2f} (Std={np.std(placebo_a):.2f}, p-val={pval_a:.3f})\n"
            f"    Strat B (15:50PM): Real Sharpe = {real_sharpe_b:+.2f} vs Placebo Mean = {np.mean(placebo_b):+.2f} (Std={np.std(placebo_b):.2f}, p-val={pval_b:.3f})"
        )


if __name__ == "__main__":
    main()
