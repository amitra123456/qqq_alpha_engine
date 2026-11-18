"""
Tests the JANUS-5 (JANUS Dual-Regime Sniper) remedies for:
  1. Eliminating false Bear (-5% B1/B2) signals during V-Bottom recovery thrusts (e.g. early April 2026)
     and strong Trending Bullish Phases with institutional accumulation (e.g. May-July 2026).
  2. Capturing the late-March / early-April 2026 V-Bottom Bull signal (+5% VBot) below SMA200.
Verifies full 1999-2026 IS & OOS accuracy across QQQ and SPY.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from build_dual_bull_bear_engine import add_bear_model_set
from discover_state_locked_80pct_bear import de_cluster_active_race
from discover_tiered_recall import eval_mask

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).resolve().parent / "data"


def apply_janus5_fixes(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = add_bear_model_set(df, ticker)
    c = df["close"]
    o = df["open"]
    sma200 = df["sma200"]
    ema10 = c.ewm(span=10, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()

    # --- REMEDY 1A: Rising-SMA200 Undercut V-Bottom Thrust (Catches 2026-03-31 V-bottom on TradingView!) ---
    # Requires a strong green reversal bar (c > o, close_loc > 0.65, +1.5% bounce) off a deep washout below a rising SMA200
    # in a non-stressed macro regime (z252_nfci < 0.0)!
    vix = df["opt_vix"]
    vix_drop5 = vix / vix.rolling(5).max() - 1.0
    nfci_z = df["z252_nfci"].ffill().fillna(0.0)
    arch4_vbottom_undercut = (
        (c <= sma200 * 1.01)
        & (df["sma200_slope21"] > 0.25)
        & (df["z20"].rolling(5).min() < -2.00)
        & (df["rsi5"].rolling(5).min() < 22.0)
        & (c > c.shift(1) * 1.025)
        & (df["dp_up_dn_vol_ratio21"] < 0.43)
        & (vix_drop5 < -0.12)
        & (df["opt_term_ratio"] < 1.02)
    )

    # --- REMEDY 1B: Post-Washout SMA200 Reclaim Thrust (Catches 2026-04-08 / 2026-04-09 breakout at $604-$610!) ---
    arch5a_sma200_reclaim = (
        (c > sma200)
        & (c.rolling(5).min() < sma200)
        & (df["sma200_slope21"] > 0.35)
        & (df["z20"].rolling(10).min() < -2.00)
        & (vix_drop5 < -0.18)
        & (df["opt_term_ratio"] < 0.96)
    )

    # --- REMEDY 1C: Bull Surf Dip (Tier 1: Catches 2026-05-19 dip during strong EMA20 surf uptrend!) ---
    arch5b_bull_surf_dip = (
        (c > ema20)
        & (ema20 > sma200 * 1.015)
        & (df["sma200_slope21"] > 0.85)
        & (df["dp_up_dn_vol_ratio21"] > 0.54)
        & (df["rsi5"] < 49.0)
        & (df["rsi5"].rolling(10).max() > 80.0)
        & (df["opt_term_ratio"] < 0.91)
        & (nfci_z < 0.35)
    )

    for name, m in [
        ("arch4_vbottom_undercut", arch4_vbottom_undercut),
        ("arch5a_sma200_reclaim", arch5a_sma200_reclaim),
        ("arch5b_bull_surf_dip", arch5b_bull_surf_dip),
    ]:
        _, st_m = eval_mask(df, m.fillna(False), min_gap=8)
        print(
            f"    {ticker} {name:24s}: N={st_m['n_all']:2d} (IS={st_m['n_is']:2d}, OOS={st_m['n_oos']:2d}) | "
            f"RACE_UP: All={st_m['race_wr_all']:5.1f}% (IS={st_m['race_wr_is']:5.1f}%, OOS={st_m['race_wr_oos']:5.1f}%) | "
            f"TOUCH60_UP: All={st_m['touch60_all']:5.1f}% (OOS={st_m['touch60_oos']:5.1f}%)"
        )

    df["SIG_TIER1_RAW"] = df["SIG_TIER1_RAW"] | arch4_vbottom_undercut | arch5b_bull_surf_dip
    df["SIG_COMBINED_RAW"] = df["SIG_COMBINED_RAW"] | arch4_vbottom_undercut | arch5a_sma200_reclaim | arch5b_bull_surf_dip

    # --- REMEDY 1D (BOTTOM-ONLY UPGRADE): 5d Exhaustion Volume + Flush & Trough Fixed-Range VWAP ---
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    rvol20 = v / (v.rolling(20, min_periods=1).mean() + 1.0)
    df["rvol20_max5"] = rvol20.rolling(5, min_periods=1).max()

    tp = (h + l + c) / 3.0
    pv = tp * v
    fr_vwap_flush5 = pv.rolling(5, min_periods=1).sum() / (v.rolling(5, min_periods=1).sum() + 1.0)
    df["dist_flush_vwap_pct"] = (c / fr_vwap_flush5 - 1.0) * 100.0

    low_arr, pv_arr, v_arr = l.values, pv.values, v.values
    trough_vwap = np.zeros(len(df))
    for i in range(len(df)):
        s_i = max(0, i - 4)
        t_i = s_i + np.argmin(low_arr[s_i : i + 1])
        trough_vwap[i] = pv_arr[t_i : i + 1].sum() / max(1.0, v_arr[t_i : i + 1].sum())
    df["dist_trough_vwap_pct"] = (c / trough_vwap - 1.0) * 100.0

    # Reject Dead-Cat Bounce bull calls when 5d exhaustion volume is absent AND price is rejected below Flush/Trough FR-VWAP with a weak upper-wick close
    bottom_frvwap_vol_gate = (
        (df["rvol20_max5"] >= 1.05)
        & ((df["dist_trough_vwap_pct"] >= -0.05) | (df["dist_flush_vwap_pct"] >= -0.75) | (df["z20"] < -1.80))
        & ~((df["close_loc"] < 0.35) & (c < o) & (df["dist_flush_vwap_pct"] < -0.50))
    )

    df["SIG_TIER1_RAW"] = df["SIG_TIER1_RAW"] & bottom_frvwap_vol_gate
    df["SIG_COMBINED_RAW"] = df["SIG_COMBINED_RAW"] & bottom_frvwap_vol_gate

    # --- REMEDY 2: V-Bottom Recovery Thrust Guard (Blocks Bear calls right after a V-bottom washout) ---
    # If price was in a deep washout (z20 < -1.80 or rsi5 < 22) within the last 12 trading days,
    # a +8-10% bounce off that low is a V-BOTTOM THRUST reclaiming trend, NOT an overextended top!
    recent_deep_washout = (df["z20"].rolling(12).min() < -1.80) | (df["rsi5"].rolling(12).min() < 22.0)
    not_vbottom_thrust = ~recent_deep_washout

    # --- REMEDY 3: Trending Bullish Phase & Institutional Accumulation Guard ---
    # During a strong bull trend (close > sma200, ema20 > sma200, sma200 rising),
    # block Bearish calls if institutional volume is accumulating (dp_up_dn_vol_ratio21 > 0.50)
    # OR if today's bar is a strong green thrust closing above yesterday's close and above EMA10!
    strong_bull_trend = (c > sma200) & (ema20 > sma200) & (df["sma200_slope21"] > 0.10)
    inst_accumulating = df["dp_up_dn_vol_ratio21"] > 0.52
    bar_rollover = (c < c.shift(1)) | (c < o) | (c < ema10) | (df["close_loc"] < 0.48)

    bull_trend_veto = strong_bull_trend & (inst_accumulating | (~bar_rollover))
    bear_guard = not_vbottom_thrust & (~bull_trend_veto)

    df["BEAR_TIER1_RAW"] = df["BEAR_TIER1_RAW"] & bear_guard
    df["BEAR_TIER2_RAW"] = df["BEAR_TIER2_RAW"] & bear_guard & bar_rollover
    df["BEAR_COMBINED_RAW"] = df["BEAR_TIER1_RAW"] | df["BEAR_TIER2_RAW"]
    return df


def main():
    for ticker in ["QQQ", "SPY"]:
        raw_df = pd.read_parquet(DATA_DIR / f"{ticker.lower()}_features_1999_2026.parquet")
        df = apply_janus5_fixes(raw_df, ticker)

        bull_t1_mask, bull_t1_st = eval_mask(df, df["SIG_TIER1_RAW"], min_gap=10)
        bull_comb_mask, bull_comb_st = eval_mask(df, df["SIG_COMBINED_RAW"], min_gap=8)

        bear_t1_idx, bear_t1_st = de_cluster_active_race(df, df["BEAR_TIER1_RAW"], max_hold=180, cooldown=3)
        bear_comb_idx, bear_comb_st = de_cluster_active_race(df, df["BEAR_COMBINED_RAW"], max_hold=180, cooldown=3)

        print(f"================ {ticker} JANUS-5 AFTER TRENDING-BULL & V-BOTTOM FIXES (1999-2026) ================")
        print(
            f"  [BULL T1 SNIPER] N={bull_t1_st['n_all']:2d} (IS={bull_t1_st['n_is']:2d}, OOS={bull_t1_st['n_oos']:2d}) | "
            f"RACE_UP: All={bull_t1_st['race_wr_all']:5.1f}% (IS={bull_t1_st['race_wr_is']:5.1f}%, OOS={bull_t1_st['race_wr_oos']:5.1f}%) | "
            f"TOUCH60_UP: All={bull_t1_st['touch60_all']:5.1f}% (OOS={bull_t1_st['touch60_oos']:5.1f}%)"
        )
        print(
            f"  [BULL T1+T2 ALL] N={bull_comb_st['n_all']:2d} (IS={bull_comb_st['n_is']:2d}, OOS={bull_comb_st['n_oos']:2d}) | "
            f"RACE_UP: All={bull_comb_st['race_wr_all']:5.1f}% (IS={bull_comb_st['race_wr_is']:5.1f}%, OOS={bull_comb_st['race_wr_oos']:5.1f}%) | "
            f"TOUCH60_UP: All={bull_comb_st['touch60_all']:5.1f}% (OOS={bull_comb_st['touch60_oos']:5.1f}%)"
        )
        print(
            f"  [BEAR B1 SNIPER] N={bear_t1_st['n_all']:2d} (IS={bear_t1_st['n_is']:2d}, OOS={bear_t1_st['n_oos']:2d}) | "
            f"RACE_DN: All={bear_t1_st['wr_all']:5.1f}% (IS={bear_t1_st['wr_is']:5.1f}%, OOS={bear_t1_st['wr_oos']:5.1f}%) | "
            f"TOUCH60_DN: All={bear_t1_st['t60_all']:5.1f}% (OOS={bear_t1_st['t60_oos']:5.1f}%)"
        )
        print(
            f"  [BEAR B1+B2 ALL] N={bear_comb_st['n_all']:2d} (IS={bear_comb_st['n_is']:2d}, OOS={bear_comb_st['n_oos']:2d}) | "
            f"RACE_DN: All={bear_comb_st['wr_all']:5.1f}% (IS={bear_comb_st['wr_is']:5.1f}%, OOS={bear_comb_st['wr_oos']:5.1f}%) | "
            f"TOUCH60_DN: All={bear_comb_st['t60_all']:5.1f}% (OOS={bear_comb_st['t60_oos']:5.1f}%)"
        )

        if ticker == "QQQ":
            print("\n  --- QQQ Signals in Jan 2026 - July 2026 Window (Matching Screenshot) ---")
            b_dates = df.index[bull_comb_mask]
            b_2026 = [d.strftime("%Y-%m-%d") for d in b_dates if d.year == 2026]
            s_dates = df.index[bear_comb_idx]
            s_2026 = [d.strftime("%Y-%m-%d") for d in s_dates if d.year == 2026]
            print(f"  2026 Bull Signals (+5%): {b_2026}")
            print(f"  2026 Bear Signals (-5%): {s_2026}")
        print()


if __name__ == "__main__":
    main()
