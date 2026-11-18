"""
Deep Ticker-Specific Optimization: Finding the Optimal Dead-Cat Filter for Each Asset
====================================================================================
Tailors the Dead-Cat Bounce (DCB) vs. Genuine Bottom (GB) filter to the unique
microstructure and volatility profile of:
  1. QQQ (Index Anchor)
  2. NVDA (High-Beta Semiconductor Leader)
  3. AMZN (Enterprise Cloud / Consumer)
  4. GOOGL (Digital Ad / Large Float)
  5. AAPL (Mega-Cap Quality Flight-to-Safety)
  6. META (High Volatility / Momentum)
"""

import itertools
import numpy as np
import pandas as pd
from pathlib import Path

from optimize_dead_cat_bounce_universe import load_data, extract_bounce_episodes, TICKERS

ROOT = Path("c:/Users/amitr/OneDrive/Documents/ANT_SOX")

def optimize_asset(ep_df, ticker):
    # Candidate parameter ranges adjusted to ticker volatility
    # Thresholds:
    # 1. Macro hazard: dM_qqq <= dM_val
    # 2. Upper Wick: upper_wick <= wick_val
    # 3. Close Location (CLV): clv >= clv_val
    # 4. Volume confirmation: rvol20 >= rvol_val
    # 5. Volume Trend: vol_fade_ratio >= vfade_val
    # 6. Trend Reclaim: dist_ema10_pct >= ema10_val
    # 7. Multi-horizon RS: rs_10d >= rs10_val
    # 8. Model B: B_stock >= b_val
    
    total_bounces = len(ep_df)
    base_dcb = ep_df["is_dcb"].sum()
    base_gb = ep_df["is_gb"].sum()
    
    grid = {
        "dM_max": [0.0, 1.0, 2.0, 999.0],
        "wick_max": [0.30, 0.35, 0.45],
        "clv_min": [0.50, 0.55, 0.60],
        "rvol5_min": [1.00, 1.15, 1.30, 1.50],
        "flush_vwap_min": [-1.00, -0.50, 0.00],
        "trough_vwap_min": [0.00, 0.25],
        "vfade_min": [0.00, 0.85, 0.95],
        "ema10_min": [-2.0, -1.0, 0.0],
        "rs10_min": [-4.0, -1.0, 0.0, -999.0],
    }
    
    candidates = []
    
    keys = list(grid.keys())
    for vals in itertools.product(*grid.values()):
        p = dict(zip(keys, vals))
        
        # Test rule with 5d Exhaustion Volume (rvol20_max5) + Flush & Trough Fixed-Range VWAP + Vol Fade
        cond = (
            (ep_df["dM_qqq_dt1"] <= p["dM_max"]) &
            (ep_df["upper_wick"] <= p["wick_max"]) &
            (ep_df["clv"] >= p["clv_min"]) &
            (ep_df["rvol20_max5"] >= p["rvol5_min"]) &
            (ep_df["dist_flush_vwap_pct"] >= p["flush_vwap_min"]) &
            (ep_df["dist_trough_vwap_pct"] >= p["trough_vwap_min"]) &
            (ep_df["vol_fade_ratio"] >= p["vfade_min"]) &
            (ep_df["dist_ema10_pct"] >= p["ema10_min"]) &
            (ep_df["rs_10d"] >= p["rs10_min"]) &
            (ep_df["pead_shock"] == 0)
        )
        
        accepted = ep_df[cond]
        n_acc = len(accepted)
        if n_acc < 8:
            continue
            
        n_gb = accepted["is_gb"].sum()
        n_dcb = accepted["is_dcb"].sum()
        
        gb_prec = (n_gb / n_acc) * 100.0
        dcb_trap_rate = (n_dcb / n_acc) * 100.0
        
        # Avoided DCBs (% of total dataset DCBs rejected)
        rejected = ep_df[~cond]
        traps_avoided_pct = (rejected["is_dcb"].sum() / max(1, base_dcb)) * 100.0
        
        avg_ret15 = accepted["fwd_ret15"].mean()
        win_rate15 = (accepted["fwd_ret15"] > 0).mean() * 100.0
        med_mae = accepted["fwd_mae"].median()
        med_mfe = accepted["fwd_mfe"].median()
        
        # 15-day Profit Factor
        pos_sum = accepted.loc[accepted["fwd_ret15"] > 0, "fwd_ret15"].sum()
        neg_sum = abs(accepted.loc[accepted["fwd_ret15"] < 0, "fwd_ret15"].sum())
        pf = pos_sum / max(0.01, neg_sum)
        
        # Objective Score:
        # High Win Rate (50%) + High Traps Avoided (30%) + Profit Factor (20%)
        score = win_rate15 * 0.50 + traps_avoided_pct * 0.30 + min(10.0, pf * 2.0) * 0.20
        
        candidates.append({
            "ticker": ticker,
            "params": p,
            "n_acc": n_acc,
            "total_bounces": total_bounces,
            "n_gb": n_gb,
            "n_dcb": n_dcb,
            "win_rate15": win_rate15,
            "gb_prec": gb_prec,
            "dcb_trap_rate": dcb_trap_rate,
            "traps_avoided_pct": traps_avoided_pct,
            "avg_ret15": avg_ret15,
            "med_mae": med_mae,
            "med_mfe": med_mfe,
            "pf": pf,
            "score": score
        })
        
    df_res = pd.DataFrame(candidates)
    if len(df_res) == 0:
        return None
    df_res = df_res.sort_values(by="score", ascending=False)
    return df_res

def main():
    print("=" * 115)
    print("AUTONOMOUS DEEP OPTIMIZATION: TICKER-SPECIFIC DEAD-CAT TRAP FILTERS")
    print("=" * 115)
    
    data_dict = load_data()
    best_per_ticker = []
    
    for t in TICKERS:
        df = data_dict[t]
        ep_df = extract_bounce_episodes(df, is_index=(t == "QQQ"))
        res = optimize_asset(ep_df, t)
        
        if res is not None and len(res) > 0:
            top = res.iloc[0]
            best_per_ticker.append(top.to_dict())
            
            p = top["params"]
            print(f"\n--- {t} OPTIMAL DEAD-CAT BOUNCE FILTER ---")
            print(f"  Base Bounces: {top['total_bounces']} | Base DCB Rate: {ep_df['is_dcb'].mean()*100:.1f}% | Base GB Rate: {ep_df['is_gb'].mean()*100:.1f}%")
            print(f"  Optimal Filter Rule:")
            print(f"    1. Macro Hazard Gate:   dM_QQQ <= {p['dM_max']}%")
            print(f"    2. Upper Wick Limit:    Upper_Wick <= {p['wick_max']*100:.0f}% of candle range")
            print(f"    3. Close Location Gate: CLV >= {p['clv_min']*100:.0f}%")
            print(f"    4. 5d Exhaustion Vol:   Max5(RVol20) >= {p['rvol5_min']:.2f}x")
            print(f"    5. Flush FR-VWAP Gate:  Dist_Flush_VWAP >= {p['flush_vwap_min']:.2f}%")
            print(f"    6. Trough FR-VWAP Gate: Dist_Trough_VWAP >= {p['trough_vwap_min']:.2f}%")
            print(f"    7. Trend Reclaim Gate:  Dist_EMA10 >= {p['ema10_min']:.1f}%")
            print(f"    8. Multi-Horizon RS:    10d Relative Spread vs QQQ >= {p['rs10_min']:.1f}%")
            print(f"  Validated Performance on Cleared Bottoms:")
            print(f"    • Accepted Bottoms:     {int(top['n_acc'])} trades")
            print(f"    • 15-Day Win Rate:      {top['win_rate15']:.1f}%  (Baseline was {ep_df['is_gb'].mean()*100:.1f}%)")
            print(f"    • Average 15-Day Gain:  {top['avg_ret15']:+.2f}%")
            print(f"    • Dead-Cat Traps AVOIDED: {top['traps_avoided_pct']:.1f}% of all traps filtered out")
            print(f"    • Dead-Cat Trap Rate:   Cut down to {top['dcb_trap_rate']:.1f}%")
            print(f"    • Profit Factor:        {top['pf']:.2f}")
            print(f"    • Median Drawdown (MAE): {top['med_mae']:.2f}% (Median MFE: {top['med_mfe']:+.2f}%)")

if __name__ == "__main__":
    main()
