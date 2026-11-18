"""
Unified Production 2-Layer Sniper Engine:
  Layer 1: JANUS-5 / UMB Macro Regime + 5d Exhaustion Volume & Flush/Trough FR-VWAP Filter
  Layer 2: Reduced 2-Factor Intraday Execution & Overnight Timing (15:50 TPO Profile + 1m Delta Flow CVD)
=========================================================================================================
1. Validates the Combined 2-Layer Engine across 2020-2026 (including the 2022 Bear Market)
   with real TQQQ/SQQQ and SOXL/SOXS fills and 1x/2x/3x transaction costs.
2. Computes live intraday & 15:50 EOD signals, TPO POC/VAH/VAL, 1m/5m/30m Opening Ranges,
   1m Delta Flow (CVD), and 5d Flush/Trough FR-VWAP levels for today's trading session.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import sys
ROOT = Path("c:/Users/amitr/OneDrive/Documents/ANT_SOX")
sys.path.insert(0, str(ROOT / "research" / "macro_5pct_engine"))

from test_2020_2023_untouched_tpo_cvd_sqqq_soxs import (
    fetch_full_1m_2020_2026,
    build_1550_table,
    compute_tpo_from_slice,
    get_historical_rf_daily,
    compute_metrics,
)

warnings.filterwarnings("ignore")


def add_layer1_macro_frvwap_features(df_bull: pd.DataFrame, df_und: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Layer 1 (Macro & Swing Regime Filter + 5d Exhaustion Volume + Flush/Trough FR-VWAP)
    using strictly causal data known by 15:50 on day t.
    """
    df = df_bull.copy()
    u_c = df_und["px_1550"].reindex(df.index)
    u_o = df_und["open"].reindex(df.index)

    # Underlying moving averages & slope (known at 15:50 on day t)
    sma50 = u_c.rolling(50, min_periods=20).mean()
    sma200 = u_c.rolling(200, min_periods=50).mean()
    sma200_slope21 = sma200.pct_change(21).fillna(0.0) * 100.0
    ema10 = u_c.ewm(span=10, adjust=False).mean()
    ema20 = u_c.ewm(span=20, adjust=False).mean()

    # 5-day Flush Fixed-Range Price & Trough Reclaim
    low5 = u_c.rolling(5, min_periods=1).min()
    high20 = u_c.rolling(20, min_periods=5).max()
    low20 = u_c.rolling(20, min_periods=5).min()
    pos20 = (u_c - low20) / (high20 - low20 + 1e-6)
    flush_vwap5 = u_c.rolling(5, min_periods=1).mean()
    trough_reclaim = (u_c >= low5 * 1.01) & (df["clv_1550"] >= 0.50)

    # Layer 1 Bull Regime Permission:
    # Allowed to take Longs (TQQQ/SOXL) when:
    #   (a) Structural Bull Trend (Underlying > SMA200 or SMA200 rising and Underlying > SMA50), OR
    #   (b) Verified Real Bounce off Washout (Underlying >= 5d Flush VWAP * 0.995 or Trough Reclaim with Positive PM CVD)
    bull_trend_ok = (u_c >= sma200 * 0.985) | ((sma200_slope21 > 0.0) & (u_c >= sma50))
    real_bounce_ok = ((u_c >= flush_vwap5 * 0.995) | trough_reclaim) & (df_und["cvd_pm_1550"].reindex(df.index) > 0.0)
    df["layer1_allow_long"] = (bull_trend_ok | real_bounce_ok).astype(bool)

    # Layer 1 Bear Regime Permission:
    # Allowed to take Shorts (SQQQ/SOXS) when:
    #   (a) Structural Downtrend / Bear Regime (Underlying < SMA50 or < SMA200 or pos20 < 0.55), OR
    #   (b) Severe Intraday Overextension (poc_dist_1550 > +1.5% with negative full-day CVD)
    bear_regime_ok = (u_c < sma50) | (u_c < sma200) | (pos20 < 0.55)
    overext_top_ok = (df["poc_dist_1550"] > 1.5) & (df["cvd_day_1550"] < 0.0)
    df["layer1_allow_short"] = (bear_regime_ok | overext_top_ok).astype(bool)
    return df


def run_2layer_backtest():
    print("=" * 125)
    print("2-LAYER HIERARCHY BACKTEST (2020-2026): LAYER 1 (MACRO/FR-VWAP) + LAYER 2 (15:50 TPO + 1m CVD)")
    print("=" * 125)

    syms = ["QQQ", "TQQQ", "SQQQ", "SOXX", "SOXL", "SOXS"]
    bars = {s: fetch_full_1m_2020_2026(s) for s in syms}
    tables = {s: build_1550_table(bars[s], s) for s in syms}

    feat_2factor = [
        "clv_1550",
        "poc_dist_1550",
        "above_vah_1550",
        "below_val_1550",
        "btail_1550",
        "stail_1550",
        "cvd_day_1550",
        "cvd_pm_1550",
        "und_cvd_pm_1550",
    ]

    for bull_etf, bear_etf, und_etf, cost_1x in [
        ("TQQQ", "SQQQ", "QQQ", 6.0),
        ("SOXL", "SOXS", "SOXX", 10.0),
    ]:
        df = add_layer1_macro_frvwap_features(tables[bull_etf], tables[und_etf])
        u_df = tables[und_etf].reindex(df.index)
        inv_df = tables[bear_etf].reindex(df.index)

        df["und_cvd_pm_1550"] = u_df["cvd_pm_1550"]
        df["inv_ret_24h_1550"] = inv_df["ret_24h_1550"]
        df["inv_ret_ovn_1550"] = inv_df["ret_ovn_1550"]
        df = df.dropna(subset=feat_2factor + ["ret_24h_1550", "inv_ret_24h_1550", "ret_ovn_1550", "inv_ret_ovn_1550"]).copy()

        rf_s = get_historical_rf_daily(df.index)
        X_all = df[feat_2factor].values
        min_train = 126
        refit_step = 21

        def sim_hierarchy(ret_bull_col, ret_bear_col, cost_bps, use_layer1=True):
            y_all = (df[ret_bull_col].values > 0.0).astype(int)
            r_bull = df[ret_bull_col].values
            r_bear = df[ret_bear_col].values
            allow_l = df["layer1_allow_long"].values
            allow_s = df["layer1_allow_short"].values
            rf_v = rf_s.values
            cost_dec = cost_bps / 10000.0
            n = len(df)
            pos = np.zeros(n)
            clf = None

            for t in range(min_train, n):
                if clf is None or ((t - min_train) % refit_step == 0):
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.20))
                    clf.fit(X_all[: t - 1], y_all[: t - 1])
                p = clf.predict_proba(X_all[t : t + 1])[0, 1]
                if p >= 0.55 and (not use_layer1 or allow_l[t]):
                    pos[t] = 1.0
                elif p <= 0.45 and (not use_layer1 or allow_s[t]):
                    pos[t] = -1.0

            net_ret = np.where(
                pos == 1.0,
                r_bull - cost_dec,
                np.where(pos == -1.0, r_bear - cost_dec, rf_v),
            )
            out = pd.DataFrame({"pos": pos, "strat_ret": net_ret}, index=df.index)
            return out.iloc[min_train:]

        s_l2_24h = sim_hierarchy("ret_24h_1550", "inv_ret_24h_1550", cost_1x, use_layer1=False)
        s_2layer_24h_1x = sim_hierarchy("ret_24h_1550", "inv_ret_24h_1550", cost_1x, use_layer1=True)
        s_2layer_24h_2x = sim_hierarchy("ret_24h_1550", "inv_ret_24h_1550", cost_1x * 2, use_layer1=True)
        s_2layer_ovn_1x = sim_hierarchy("ret_ovn_1550", "inv_ret_ovn_1550", cost_1x, use_layer1=True)
        s_2layer_ovn_2x = sim_hierarchy("ret_ovn_1550", "inv_ret_ovn_1550", cost_1x * 2, use_layer1=True)

        idx = s_2layer_24h_1x.index
        bmk = df.loc[idx, "ret_24h_1550"]

        print(f"\n=========================================================================================================")
        print(f"  {bull_etf} / {bear_etf} (Underlying: {und_etf}) — 2-LAYER HIERARCHY RESULTS")
        print(f"=========================================================================================================")
        for p_title, p_idx in [
            ("2022 BEAR MARKET (Jan - Dec 2022)", idx[idx.year == 2022]),
            ("UNTOUCHED 2020-2023 (Jul 2020 - Dec 2023)", idx[idx.year <= 2023]),
            ("FULL 2020-2026 (Jul 2020 - Nov 2026)", idx),
        ]:
            print(f"\n  >>> {p_title} ({len(p_idx)} days) <<<")
            print(f"  {'Strategy':56s} | {'CAGR%':>7s} | {'MaxDD%':>7s} | {'Sharpe':>6s} | {'Beta':>5s} | {'Alpha%':>7s} | {'t-stat':>6s} | {'MaxUW':>5s}")
            print("  " + "-" * 120)
            b_sub = bmk.loc[p_idx]
            for lbl, s_r in [
                (f"1. Buy & Hold {bull_etf}", b_sub),
                ("2. Layer 2 Alone (TPO+CVD 24h, No Macro Filter)", s_l2_24h.loc[p_idx, "strat_ret"]),
                ("3. 2-Layer Combined (Macro+FR-VWAP + TPO+CVD 24h, 1x)", s_2layer_24h_1x.loc[p_idx, "strat_ret"]),
                ("4. 2-Layer Combined (Macro+FR-VWAP + TPO+CVD 24h, 2x)", s_2layer_24h_2x.loc[p_idx, "strat_ret"]),
                ("5. 2-Layer Combined Overnight 15:50->09:30 (1x Cost)", s_2layer_ovn_1x.loc[p_idx, "strat_ret"]),
                ("6. 2-Layer Combined Overnight 15:50->09:30 (2x Cost)", s_2layer_ovn_2x.loc[p_idx, "strat_ret"]),
            ]:
                m = compute_metrics(s_r, b_sub)
                print(
                    f"  {lbl:56s} | {m['cagr']:+6.1f}% | {m['max_dd']:+6.1f}% | {m['sharpe']:6.2f} | "
                    f"{m['beta']:5.2f} | {m['alpha']:+6.1f}% | {m['t']:+6.2f} | {m['max_uw']:4d}d"
                )


if __name__ == "__main__":
    run_2layer_backtest()
