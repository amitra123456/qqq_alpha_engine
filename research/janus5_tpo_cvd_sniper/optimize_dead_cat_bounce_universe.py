"""
Autonomous Optimization Loop: Dead-Cat Bounce vs. Genuine Bottom Discovery
==========================================================================
Universe: QQQ, NVDA, AMZN, GOOGL, AAPL, META
Data: 2015-2026 (Common 2,978 sessions) + Full Inception histories
Performs exhaustive grid search across 8 micro-structural and macro dimensions
to find the optimal combination that maximizes Genuine Bottom win rate
and eliminates > 85% of Dead-Cat Bounces.
"""

import itertools
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path("c:/Users/amitr/OneDrive/Documents/ANT_SOX")
DATA_DIR = ROOT / "data" / "historical"
P_DROP_PATH = ROOT / "qqq_alpha_engine" / "data" / "macro" / "final_9tier_daily_probabilities_1999_2026.parquet"
P_BOT_PATH = ROOT / "qqq_alpha_engine" / "data" / "macro" / "final_bottom_daily_scores_1999_2026.parquet"

TICKERS = ["QQQ", "NVDA", "AMZN", "GOOGL", "AAPL", "META"]

def load_data():
    p_drop = pd.read_parquet(P_DROP_PATH)
    p_bot = pd.read_parquet(P_BOT_PATH)
    
    ticker_dfs = {}
    for t in TICKERS:
        path = DATA_DIR / f"{t}_1d.parquet"
        df = pd.read_parquet(path)
        common_idx = df.index.intersection(p_drop.index).intersection(p_bot.index)
        df = df.loc[common_idx].copy()
        
        # Attach macro features
        df["QQQ_close"] = p_drop.loc[common_idx, "QQQ_close"]
        df["M_qqq"] = p_drop.loc[common_idx, "prob_tier_5pct"] * 100.0
        df["B_qqq"] = np.clip(p_bot.loc[common_idx, "TOTAL_BOTTOM_SCORE"] / 25.5 * 100.0, 0.0, 100.0)
        df["dM_qqq_dt1"] = df["M_qqq"].diff(1).fillna(0.0)
        
        # Compute stock technical features
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]
        
        # RSI 2
        delta = close.diff(1)
        gain = delta.clip(lower=0.0).ewm(span=2, adjust=False).mean()
        loss = (-delta).clip(lower=0.0).ewm(span=2, adjust=False).mean()
        df["rsi2"] = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))
        
        # ATR 14
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(14, min_periods=1).mean()
        
        # Moving averages
        df["ema10"] = close.ewm(span=10, adjust=False).mean()
        df["ema20"] = close.ewm(span=20, adjust=False).mean()
        df["dist_ema10_pct"] = (close / df["ema10"] - 1.0) * 100.0
        df["dist_ema20_atr"] = (close - df["ema20"]) / (df["atr14"] + 1e-9)
        
        # Volume features (including 5d Multi-Day Exhaustion Volume & Fixed-Range VWAPs)
        vol_sma20 = vol.rolling(20, min_periods=1).mean()
        df["rvol20"] = vol / (vol_sma20 + 1.0)
        df["rvol20_max5"] = df["rvol20"].rolling(5, min_periods=1).max()
        df["vol_fade_ratio"] = vol / (vol.shift(1) + 1.0)
        
        # Fixed-Range VWAPs: 5d Flush FR-VWAP, 5d Trough-Anchored FR-VWAP, and 20d FR-VWAP
        tp = (high + low + close) / 3.0
        pv = tp * vol
        df["fr_vwap_flush5"] = pv.rolling(5, min_periods=1).sum() / (vol.rolling(5, min_periods=1).sum() + 1.0)
        df["fr_vwap20"] = pv.rolling(20, min_periods=1).sum() / (vol.rolling(20, min_periods=1).sum() + 1.0)
        df["dist_flush_vwap_pct"] = (close / df["fr_vwap_flush5"] - 1.0) * 100.0
        df["dist_vwap20_pct"] = (close / df["fr_vwap20"] - 1.0) * 100.0
        
        # Trough-Anchored FR-VWAP (anchored at lowest low of past 5 bars to current bar)
        low_arr = low.values
        pv_arr = pv.values
        v_arr = vol.values
        trough_vwap = np.zeros(len(df))
        for i in range(len(df)):
            s_idx = max(0, i - 4)
            min_rel = np.argmin(low_arr[s_idx : i + 1])
            t_idx = s_idx + min_rel
            trough_vwap[i] = pv_arr[t_idx : i + 1].sum() / max(1.0, v_arr[t_idx : i + 1].sum())
        df["fr_vwap_trough"] = trough_vwap
        df["dist_trough_vwap_pct"] = (close / df["fr_vwap_trough"] - 1.0) * 100.0
        
        # Candlestick wicks
        rng = (high - low).clip(lower=1e-4)
        df["clv"] = (close - low) / rng  # Close Location (0 to 1)
        df["upper_wick"] = (high - close) / rng  # Upper shadow (0 to 1)
        
        # Relative strength vs QQQ
        ratio = close / df["QQQ_close"]
        df["rs_3d"] = ratio.pct_change(3) * 100.0
        df["rs_10d"] = ratio.pct_change(10) * 100.0
        
        # Model B (Stock-level capitulation baseline with 5d Exhaustion Volume)
        b1_rsi = np.where(df["rsi2"] <= 5.0, 30.0, np.where(df["rsi2"] <= 10.0, 18.0, np.where(df["rsi2"] <= 18.0, 8.0, 0.0)))
        b1_dist = np.where(df["dist_ema20_atr"] <= -2.5, 25.0, np.where(df["dist_ema20_atr"] <= -1.8, 15.0, np.where(df["dist_ema20_atr"] <= -1.2, 5.0, 0.0)))
        b1_wick = np.where(df["clv"] >= 0.40, 10.0, np.where(df["clv"] >= 0.25, 5.0, 0.0))
        b2_vol = np.where(df["rvol20_max5"] >= 1.5, 15.0, np.where(df["rvol20_max5"] >= 1.25, 8.0, 0.0))
        b3_rs = np.where(df["rs_3d"] > 0.0, 10.0, 0.0)
        b4_mac = np.where((df["B_qqq"] >= 65.0) | ((df["M_qqq"] >= 70.0) & (df["dM_qqq_dt1"] <= 0.0)), 10.0, 0.0)
        df["B_stock"] = np.clip(b1_rsi + b1_dist + b1_wick + b2_vol + b3_rs + b4_mac, 0.0, 100.0)
        
        # PEAD shock detection (gap down <= -5.0% on rvol >= 2.0x in past 4 days)
        gap = (df["open"] / close.shift(1) - 1.0)
        df["pead_shock"] = ((gap <= -0.05) & (df["rvol20"] >= 2.0)).rolling(4, min_periods=1).max().astype(bool)
        
        ticker_dfs[t] = df
    return ticker_dfs

def extract_bounce_episodes(df, is_index=False):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    n = len(df)
    
    # Thresholds: index is less volatile than single stocks
    min_pullback = 0.04 if is_index else 0.06
    min_bounce = 0.020 if is_index else 0.025
    target_gain = 0.05 if is_index else 0.06
    
    roll_max20 = close.rolling(20, min_periods=1).max()
    in_pullback = (close <= roll_max20 * (1.0 - min_pullback))
    
    roll_min3 = low.rolling(3, min_periods=1).min()
    bounced = ((close / roll_min3 - 1.0) >= min_bounce) & (close > close.shift(1))
    
    candidate_dates = df[in_pullback & bounced].index
    
    # Deduplicate 5 trading days
    episodes = []
    last_d = None
    for d in candidate_dates:
        if last_d is None or (d - last_d).days > 5:
            episodes.append(d)
            last_d = d
            
    rows = []
    for d in episodes:
        pos = df.index.get_loc(d)
        if pos + 15 >= n:
            continue
            
        cur_px = close.iloc[pos]
        cur_low = low.iloc[pos]
        fwd_15_low = low.iloc[pos+1 : pos+16].min()
        fwd_15_high = high.iloc[pos+1 : pos+16].max()
        fwd_15_close = close.iloc[pos+15]
        
        # Dead-Cat Bounce: breaks below the bounce low
        is_dcb = bool(fwd_15_low < cur_low * 0.995)
        # Genuine Bottom: defends the low and gains >= target_gain
        is_gb = bool((not is_dcb) and (fwd_15_high >= cur_px * (1.0 + target_gain)))
        
        fwd_ret15 = (fwd_15_close / cur_px - 1.0) * 100.0
        fwd_mae = (fwd_15_low / cur_px - 1.0) * 100.0
        fwd_mfe = (fwd_15_high / cur_px - 1.0) * 100.0
        
        rows.append({
            "date": d,
            "is_dcb": 1 if is_dcb else 0,
            "is_gb": 1 if is_gb else 0,
            "fwd_ret15": fwd_ret15,
            "fwd_mae": fwd_mae,
            "fwd_mfe": fwd_mfe,
            "B_stock": df.loc[d, "B_stock"],
            "B_qqq": df.loc[d, "B_qqq"],
            "dM_qqq_dt1": df.loc[d, "dM_qqq_dt1"],
            "M_qqq": df.loc[d, "M_qqq"],
            "rvol20": df.loc[d, "rvol20"],
            "rvol20_max5": df.loc[d, "rvol20_max5"],
            "vol_fade_ratio": df.loc[d, "vol_fade_ratio"],
            "clv": df.loc[d, "clv"],
            "upper_wick": df.loc[d, "upper_wick"],
            "dist_ema10_pct": df.loc[d, "dist_ema10_pct"],
            "dist_flush_vwap_pct": df.loc[d, "dist_flush_vwap_pct"],
            "dist_trough_vwap_pct": df.loc[d, "dist_trough_vwap_pct"],
            "dist_vwap20_pct": df.loc[d, "dist_vwap20_pct"],
            "dist_ema20_atr": df.loc[d, "dist_ema20_atr"],
            "rs_3d": df.loc[d, "rs_3d"],
            "rs_10d": df.loc[d, "rs_10d"],
            "pead_shock": 1 if df.loc[d, "pead_shock"] else 0
        })
        
    return pd.DataFrame(rows)

def run_grid_search(ep_df, ticker_name):
    # Parameter grid for candidate rules to reject DCB and select GB
    # Rules:
    # 1. Macro hazard condition: dM_qqq <= Thresh
    # 2. Volume condition: rvol >= Thresh
    # 3. Candlestick wick condition: upper_wick <= Thresh
    # 4. Moving average condition: dist_ema10 >= Thresh
    # 5. Relative strength coherence: rs_10d >= Thresh
    # 6. Capitulation score: B_stock >= Thresh
    
    dM_opts = [0.0, 1.0, 999.0]
    rvol_opts = [0.0, 1.0, 1.3]
    wick_opts = [0.35, 0.45, 1.0]
    ema10_opts = [-1.0, 0.0, -999.0]
    rs10_opts = [-3.0, 0.0, -999.0]
    b_opts = [0.0, 60.0, 75.0]
    
    best_results = []
    
    total_combos = len(dM_opts) * len(rvol_opts) * len(wick_opts) * len(ema10_opts) * len(rs10_opts) * len(b_opts)
    
    base_n = len(ep_df)
    base_dcb_rate = ep_df["is_dcb"].mean() * 100.0
    base_gb_rate = ep_df["is_gb"].mean() * 100.0
    
    for dM_max, rvol_min, wick_max, ema10_min, rs10_min, b_min in itertools.product(
        dM_opts, rvol_opts, wick_opts, ema10_opts, rs10_opts, b_opts
    ):
        # Apply filter rule: Must pass ALL conditions to be accepted as a "Genuine Bottom"
        mask = (
            (ep_df["dM_qqq_dt1"] <= dM_max) &
            (ep_df["rvol20"] >= rvol_min) &
            (ep_df["upper_wick"] <= wick_max) &
            (ep_df["dist_ema10_pct"] >= ema10_min) &
            (ep_df["rs_10d"] >= rs10_min) &
            (ep_df["B_stock"] >= b_min) &
            (ep_df["pead_shock"] == 0)
        )
        
        accepted = ep_df[mask]
        n_acc = len(accepted)
        if n_acc < 8:  # Require statistical minimum sample size
            continue
            
        gb_prec = accepted["is_gb"].mean() * 100.0
        dcb_trap_rate = accepted["is_dcb"].mean() * 100.0
        avg_ret15 = accepted["fwd_ret15"].mean()
        avg_mae = accepted["fwd_mae"].median()
        
        # Profit factor
        pos_ret = accepted.loc[accepted["fwd_ret15"] > 0, "fwd_ret15"].sum()
        neg_ret = abs(accepted.loc[accepted["fwd_ret15"] < 0, "fwd_ret15"].sum())
        pf = pos_ret / max(0.01, neg_ret)
        
        # Traps eliminated (% of total DCBs in dataset successfully avoided)
        rejected = ep_df[~mask]
        traps_avoided_pct = (rejected["is_dcb"].sum() / max(1, ep_df["is_dcb"].sum())) * 100.0
        
        # Score metric balancing high precision, high traps avoided, and reasonable sample size
        score = gb_prec * 0.40 + traps_avoided_pct * 0.35 + min(10.0, pf * 2.0) * 0.15 + (n_acc / base_n) * 100.0 * 0.10
        
        best_results.append({
            "ticker": ticker_name,
            "dM_max": dM_max,
            "rvol_min": rvol_min,
            "wick_max": wick_max,
            "ema10_min": ema10_min,
            "rs10_min": rs10_min,
            "b_min": b_min,
            "n_acc": n_acc,
            "gb_prec": gb_prec,
            "dcb_trap_rate": dcb_trap_rate,
            "traps_avoided_pct": traps_avoided_pct,
            "avg_ret15": avg_ret15,
            "avg_mae": avg_mae,
            "pf": pf,
            "score": score
        })
        
    res_df = pd.DataFrame(best_results)
    if len(res_df) == 0:
        return None
    res_df = res_df.sort_values(by="score", ascending=False)
    return res_df

def main():
    print("1. Loading datasets for QQQ, NVDA, AMZN, GOOGL, AAPL, META...")
    data_dict = load_data()
    
    summary_table = []
    
    for t in TICKERS:
        df = data_dict[t]
        is_idx = (t == "QQQ")
        ep_df = extract_bounce_episodes(df, is_index=is_idx)
        print(f"\n[{t}] Total Bounces: {len(ep_df)} | Base DCB Rate: {ep_df['is_dcb'].mean()*100:.1f}% | Base GB Rate: {ep_df['is_gb'].mean()*100:.1f}%")
        
        grid_res = run_grid_search(ep_df, t)
        if grid_res is not None and len(grid_res) > 0:
            top = grid_res.iloc[0]
            print(f"  -> TOP COMBINATION FOR {t}:")
            print(f"     Rule: dM<={top['dM_max']} | RVol>={top['rvol_min']} | Wick<={top['wick_max']} | EMA10>={top['ema10_min']}% | RS10>={top['rs10_min']}% | B>={top['b_min']}")
            print(f"     Results: N={int(top['n_acc'])} | GB Precision={top['gb_prec']:.1f}% | DCB Traps={top['dcb_trap_rate']:.1f}% | Traps Avoided={top['traps_avoided_pct']:.1f}% | Avg15d={top['avg_ret15']:+.2f}% | MAE={top['avg_mae']:.2f}% | PF={top['pf']:.2f}")
            summary_table.append(top.to_dict())
            
    summary_df = pd.DataFrame(summary_table)
    summary_df.to_csv(ROOT / "research" / "qqq_drops_first_principles" / "dead_cat_bounce_optimization_results.csv", index=False)
    print("\nSaved full optimization summary to dead_cat_bounce_optimization_results.csv")

if __name__ == "__main__":
    main()
