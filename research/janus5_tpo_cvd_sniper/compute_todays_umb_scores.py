"""
Calculate Live UMB Scores for Today (2026-11-09)
================================================
Evaluates the Tri-Engine Architecture:
  1. Model U: Structural Bull Trend Resilience (0% to 100%)
  2. Model M: Downside Cascade Hazard Probability (0% to 100%)
  3. Model B: Bottom Capitulation Reversal Score (0% to 100%)

Evaluates across:
  - 2026-11-04 (Wednesday)
  - 2026-11-05 (Thursday)
  - 2026-11-06 (Friday Dip)
  - 2026-11-09 (Today Rebound)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("C:/Users/amitr/OneDrive/Documents/ANT_SOX")

def compute_umb_scores():
    # 1. Load historical disjoint features
    d_df = pd.read_parquet(PROJECT_ROOT / "research" / "qqq_drops_first_principles" / "data" / "qqq_disjoint_6domain_dataset_1999_2026.parquet")
    p_df = pd.read_parquet(PROJECT_ROOT / "qqq_alpha_engine" / "data" / "macro" / "final_9tier_daily_probabilities_1999_2026.parquet")
    b_df = pd.read_parquet(PROJECT_ROOT / "qqq_alpha_engine" / "data" / "macro" / "final_bottom_daily_scores_1999_2026.parquet")
    
    # Slice QQQ OHLCV
    qqq = d_df[["QQQ_open", "QQQ_high", "QQQ_low", "QQQ_close", "QQQ_volume"]].copy()
    
    # 2. Append Nov 6 and Nov 9 (Today) Real Alpaca Bars
    # Nov 6: QQQ dropped to 656.31 (Dip)
    # Nov 9: QQQ surged to 679.46 (Violent Bull Rebound)
    nov6 = pd.Series({
        "QQQ_open": 658.42, "QQQ_high": 664.28, "QQQ_low": 654.02, "QQQ_close": 656.31, "QQQ_volume": 45700000.0
    }, name=pd.Timestamp("2026-11-06"))
    
    nov9 = pd.Series({
        "QQQ_open": 662.65, "QQQ_high": 681.69, "QQQ_low": 662.19, "QQQ_close": 679.46, "QQQ_volume": 41200000.0
    }, name=pd.Timestamp("2026-11-09"))
    
    qqq = pd.concat([qqq, pd.DataFrame([nov6, nov9])])
    
    c = qqq["QQQ_close"]
    h = qqq["QQQ_high"]
    l = qqq["QQQ_low"]
    v = qqq["QQQ_volume"]
    
    # ---------------------------------------------------------
    # 3. Model U: Structural Bull Trend Resilience (0% to 100%)
    # ---------------------------------------------------------
    sma50 = c.rolling(50, min_periods=20).mean()
    sma200 = c.rolling(200, min_periods=50).mean()
    ema10 = c.ewm(span=10, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    
    golden_spread = (sma50 / sma200 - 1.0) * 100.0
    dist_200 = (c / sma200 - 1.0) * 100.0
    slope_200 = sma200.pct_change(21).fillna(0.0) * 100.0
    
    # 20-day position
    high20 = h.rolling(20).max()
    low20 = l.rolling(20).min()
    pos_20d = ((c - low20) / (high20 - low20).replace(0, np.nan)).fillna(0.5)
    
    # Sub-scores
    s_altitude = np.clip((golden_spread + dist_200 + 10.0) / 25.0 * 100.0, 0.0, 100.0)
    s_slope = np.clip((slope_200 + 2.0) / 4.0 * 100.0, 0.0, 100.0)
    s_breadth = np.clip(pos_20d * 100.0, 0.0, 100.0)
    
    # Volatility term buffer (VIX term structure in backwardation on dip, contango today)
    # Nov 6 term ratio spiked to ~0.98, today relaxed to ~0.88
    term_ratio_series = pd.Series(index=qqq.index, dtype=float)
    term_ratio_series.loc[:"2026-11-05"] = d_df["m4_term_structure_ratio"]
    term_ratio_series.loc["2026-11-06"] = 0.98
    term_ratio_series.loc["2026-11-09"] = 0.88
    
    vol_buffer = (1.0 - term_ratio_series) * 100.0
    s_vol = np.clip((vol_buffer + 15.0) / 30.0 * 100.0, 0.0, 100.0)
    
    u_raw = 0.35 * s_altitude + 0.20 * s_slope + 0.25 * s_breadth + 0.20 * s_vol
    model_u = u_raw.rolling(3, min_periods=1).mean()
    
    # ---------------------------------------------------------
    # 4. Model M: Downside Cascade Hazard Probability (0% to 100%)
    # ---------------------------------------------------------
    # ATR14
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    
    # VIX level: Nov 6 spiked to 21.8, Nov 9 collapsed to 16.2
    vix_series = pd.Series(index=qqq.index, dtype=float)
    vix_series.loc[:"2026-11-05"] = 19.5
    vix_series.loc["2026-11-06"] = 21.8
    vix_series.loc["2026-11-09"] = 16.2
    
    vix_z = (vix_series - vix_series.rolling(50, min_periods=10).mean()) / vix_series.rolling(50, min_periods=10).std().replace(0, 1.0)
    
    # Credit HYG / IEF: Nov 6 = 77.23 / 91.84 = 0.841, Nov 9 = 77.51 / 91.79 = 0.844
    hyg_ief = pd.Series(index=qqq.index, dtype=float)
    hyg_ief.loc[:"2026-11-05"] = 0.842
    hyg_ief.loc["2026-11-06"] = 77.23 / 91.84
    hyg_ief.loc["2026-11-09"] = 77.51 / 91.79
    hyg_ief_z = (hyg_ief - hyg_ief.rolling(50, min_periods=10).mean()) / hyg_ief.rolling(50, min_periods=10).std().replace(0, 1.0)
    
    # Canary XLY / XLP: Nov 6 = 117.57 / 90.09 = 1.305, Nov 9 = 121.06 / 90.90 = 1.332
    canary = pd.Series(index=qqq.index, dtype=float)
    canary.loc[:"2026-11-05"] = 1.31
    canary.loc["2026-11-06"] = 117.57 / 90.09
    canary.loc["2026-11-09"] = 121.06 / 90.90
    canary_z = (canary - canary.rolling(50, min_periods=10).mean()) / canary.rolling(50, min_periods=10).std().replace(0, 1.0)
    
    # CLV
    clv_daily = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    clv_ema5 = clv_daily.ewm(span=5, adjust=False).mean().fillna(0.0)
    
    # Compute Model M factors
    m_term_shock = np.maximum(0.0, (term_ratio_series - 0.95) / 0.15) * 2.0
    m_vix_surge = np.maximum(0.0, (vix_series - 20.0) / 10.0) * 1.5
    m_hyg_spread = np.maximum(0.0, -hyg_ief_z) * 1.5
    m_dark_outflow = np.maximum(0.0, -clv_ema5) * 1.5
    m_roll_20d = np.maximum(0.0, (0.50 - pos_20d) * 2.0) * 1.5
    m_ma_break = (c < sma50).astype(float) * 1.0 + (c < sma200).astype(float) * 1.5
    m_canary_crush = np.maximum(0.0, -canary_z) * 1.5
    
    m_raw_sum = m_term_shock + m_vix_surge + m_hyg_spread + m_dark_outflow + m_roll_20d + m_ma_break + m_canary_crush
    model_m = (1.0 / (1.0 + np.exp(-(m_raw_sum - 6.5) / 2.0))) * 100.0
    
    # ---------------------------------------------------------
    # 5. Model B: Bottom Capitulation Reversal Score (0% to 100%)
    # ---------------------------------------------------------
    # RSI 5
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(5).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(5).mean()
    rs5 = gain / loss.replace(0, np.nan)
    rsi5 = (100.0 - (100.0 / (1.0 + rs5))).fillna(50.0)
    
    z20_str = (c - ema20) / atr14.replace(0, np.nan)
    
    # B components
    b_rsi5_ext = np.maximum(0.0, (20.0 - rsi5) / 15.0) * 2.5
    b_z20_wash = np.maximum(0.0, (-z20_str - 1.5) / 1.5) * 2.0
    b_pos_wash = np.maximum(0.0, (0.15 - pos_20d) / 0.15) * 2.5
    b_dist_ext = np.maximum(0.0, (-dist_200 - 5.0) / 10.0) * 2.0
    
    # B2 Volume (5d Exhaustion Volume), Fixed-Range VWAP (Flush & Trough) & CLV
    rvol20 = v / (v.rolling(20, min_periods=1).mean() + 1.0)
    rvol20_max5 = rvol20.rolling(5, min_periods=1).max()
    tp = (h + l + c) / 3.0
    pv = tp * v
    fr_vwap_flush5 = pv.rolling(5, min_periods=1).sum() / (v.rolling(5, min_periods=1).sum() + 1.0)
    low_arr, pv_arr, v_arr = l.values, pv.values, v.values
    trough_vwap = np.zeros(len(c))
    for i in range(len(c)):
        s_i = max(0, i - 4)
        t_i = s_i + np.argmin(low_arr[s_i : i + 1])
        trough_vwap[i] = pv_arr[t_i : i + 1].sum() / max(1.0, v_arr[t_i : i + 1].sum())
    fr_vwap_trough = pd.Series(trough_vwap, index=c.index)

    b_exhaustion_vol5 = np.where(rvol20_max5 >= 1.50, 2.5, np.where(rvol20_max5 >= 1.25, 1.5, 0.0))
    b_fr_vwap = np.where((c >= fr_vwap_trough) & (c >= fr_vwap_flush5 * 0.995), 2.0, 0.0)
    b_clv_flow = np.where((clv_daily > 0.0) | (clv_ema5 > 0.0), 1.5, 0.0)
    b_canary_div = np.where(canary_z > -0.50, 2.0, 0.0)
    
    # B4 Volatility Crest
    vix_crest = (vix_series < vix_series.shift(1)) & (vix_series.shift(1) > 20.0)
    b_vix_crest = np.where(vix_crest, 2.5, 0.0)
    b_term_inv = np.maximum(0.0, (term_ratio_series - 1.05) / 0.15) * 3.0
    
    b_raw_sum = b_rsi5_ext + b_z20_wash + b_pos_wash + b_dist_ext + b_exhaustion_vol5 + b_fr_vwap + b_clv_flow + b_canary_div + b_vix_crest + b_term_inv
    model_b = np.clip((b_raw_sum / 25.5) * 100.0, 0.0, 100.0)
    
    # ---------------------------------------------------------
    # 6. Tri-Engine Status & Telemetry
    # ---------------------------------------------------------
    dates_to_show = ["2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"]
    rows = []
    for d_str in dates_to_show:
        d = pd.Timestamp(d_str)
        q_px = c.loc[d]
        u_val = model_u.loc[d]
        m_val = model_m.loc[d]
        b_val = model_b.loc[d]
        rsi_val = rsi5.loc[d]
        pos_val = pos_20d.loc[d] * 100.0
        vix_val = vix_series.loc[d]
        
        # State machine classification
        raw_hazard = (m_val >= 70.0) and (q_px < ema10.loc[d])
        emergency = (m_val >= 85.0) or (term_ratio_series.loc[d] >= 1.0)
        
        if b_val >= 40.0:
            status = "GREEN REVERSAL: Model B Capitulation Bottom Fired"
            posture = "AGGRESSIVE LONG: TQQQ Booster Activated (+5% Target)"
        elif raw_hazard and (u_val >= 52.0) and not emergency:
            status = "ORANGE: Shallow Bull Pullback (Vetoed Cash Exit)"
            posture = "BULL DEFENSIVE: Stay Invested (U Resilience Intact)"
        elif raw_hazard and ((u_val < 52.0) or emergency):
            status = "RED CASCADE: Severe Downside Cascade Confirmed"
            posture = "DEFENSIVE: Rotate to 100% Cash / SGOV"
        elif u_val >= 60.0 and m_val < 50.0:
            status = "CYAN BULL: Confirmed Structural Golden Uptrend"
            posture = "BULL CONVICTION: 16 Equities + 15% TQQQ Active"
        else:
            status = "NEUTRAL CONSOLIDATION: Absorption in Progress"
            posture = "BALANCED CORE: Full 16-Stock Portfolio"
            
        rows.append({
            "Date": d_str,
            "QQQ Close": q_px,
            "Model U (%)": u_val,
            "Model M (%)": m_val,
            "Model B (%)": b_val,
            "RSI-5": rsi_val,
            "20d Pos (%)": pos_val,
            "VIX": vix_val,
            "Tri-Engine Regime Status": status,
            "Operational Posture": posture
        })
        
    df_out = pd.DataFrame(rows)
    print("\n" + "="*115)
    print("  FP-DCTE UMB TRI-ENGINE TELEMETRY (November 3 - November 9, 2026)")
    print("="*115)
    print(df_out.to_string(index=False, formatters={
        "QQQ Close": lambda x: f"${x:6.2f}",
        "Model U (%)": lambda x: f"{x:5.1f}%",
        "Model M (%)": lambda x: f"{x:5.1f}%",
        "Model B (%)": lambda x: f"{x:5.1f}%",
        "RSI-5": lambda x: f"{x:4.1f}",
        "20d Pos (%)": lambda x: f"{x:5.1f}%",
        "VIX": lambda x: f"{x:4.1f}"
    }))
    
    return df_out

if __name__ == "__main__":
    compute_umb_scores()
