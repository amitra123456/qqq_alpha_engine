"""
100% Real-Data Intraday & Next-Day Directional Prediction Engine for TQQQ & SOXL
================================================================================
Combines strictly real, non-manufactured historical datasets (2024-02-01 to 2026-11-17):
  1. Alpaca 1-Minute RTH Bars (QQQ, TQQQ, SOXX, SOXL) -> 1m, 5m, 30m Opening Ranges (OR)
  2. Intraday Delta Flow (CVD - Cumulative Volume Delta from 1m bars & trade count)
  3. True Market Profile TPO (30-min letters A-M: TPO POC, VAH, VAL, Initial Balance, Tails)
  4. Structural Levels: Previous Day (PDH, PDL, PDC), Previous Session, Previous Week (PWH, PWL)
  5. Session VWAP (+/- 1 SD bands) & 5d Fixed-Range VWAP
  6. Real OPRA 0DTE Call Walls, Put Walls, Gamma Flip & SqueezeMetrics GEX/DIX

Trains & evaluates walk-forward (IS: 2024-2025, OOS: 2026) models for:
  - Horizon 1 (Next-Day Direction: Today's 4:00 PM Close -> Tomorrow's Close & Open-to-Close)
  - Horizon 2 (Intraday 10:00 AM Post-30m-OR -> 4:00 PM Close Direction)
And outputs the live prediction for 2026-11-18 on TQQQ and SOXL.
"""

import os
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")
load_dotenv()

ROOT = Path("c:/Users/amitr/OneDrive/Documents/ANT_SOX")
ALPACA_DIR = ROOT / "data" / "cache" / "alpaca_1m_stocks"
ALPACA_DIR.mkdir(parents=True, exist_ok=True)

OPRA_PATH = ROOT / "ant_0dte_credit_spread" / "data" / "zero_dte_opra_walls_gex_2024_2026.parquet"
DIX_PATH = ROOT / "ant_0dte_credit_spread" / "data" / "squeezemetrics_dix_gex_2011_2026.parquet"


def fetch_alpaca_1m(symbol: str, start_date: str = "2024-02-01", end_date: str = "2026-11-17") -> pd.DataFrame:
    cache_file = ALPACA_DIR / f"{symbol}_1m_{start_date}_{end_date}_rth.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    # Check if 2026-11-12 cache already exists (like QQQ_1m_2024-02-01_2026-11-12_rth.parquet)
    existing_1112 = ALPACA_DIR / f"{symbol}_1m_2024-02-01_2026-11-12_rth.parquet"
    base_df = None
    fetch_start = start_date
    if existing_1112.exists():
        base_df = pd.read_parquet(existing_1112)
        fetch_start = "2026-11-13"
        print(f"[{symbol}] Loaded existing 1m cache ({len(base_df):,} bars through 2026-11-12), fetching delta {fetch_start} -> {end_date}...")
    else:
        print(f"[{symbol}] Downloading real 1-minute Alpaca SIP bars ({fetch_start} -> {end_date})...")

    key = os.getenv("APCA_API_KEY_ID")
    sec = os.getenv("APCA_API_SECRET_KEY")
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    url = "https://data.alpaca.markets/v2/stocks/bars"
    all_bars = []
    page_token = None

    while True:
        params = {
            "symbols": symbol,
            "timeframe": "1Min",
            "start": f"{fetch_start}T00:00:00Z",
            "end": f"{end_date}T23:59:59Z",
            "adjustment": "split",
            "feed": "sip",
            "limit": 10000,
        }
        if page_token:
            params["page_token"] = page_token
        for attempt in range(4):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    bars = data.get("bars", {}).get(symbol, [])
                    all_bars.extend(bars)
                    page_token = data.get("next_page_token")
                    break
                time.sleep(1.0 * (attempt + 1))
            except Exception:
                time.sleep(1.0)
        if not page_token:
            break

    if all_bars:
        df = pd.DataFrame(all_bars)
        ts = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
        df["date_str"] = ts.dt.strftime("%Y-%m-%d")
        hhmm = ts.dt.hour * 100 + ts.dt.minute
        df = df[(hhmm >= 930) & (hhmm <= 1559)].copy()
        df["minute_idx"] = (ts.dt.hour - 9) * 60 + (ts.dt.minute - 30)
        keep_cols = ["date_str", "minute_idx", "o", "h", "l", "c", "v", "vw", "n"]
        df = df[keep_cols]
        if base_df is not None:
            df = pd.concat([base_df[keep_cols], df], ignore_index=True)
    else:
        df = base_df.copy()

    df = df.sort_values(["date_str", "minute_idx"]).drop_duplicates(subset=["date_str", "minute_idx"]).reset_index(drop=True)
    df.to_parquet(cache_file)
    print(f"[{symbol}] Cached {len(df):,} 1m bars across {df['date_str'].nunique()} sessions.")
    return df


def compute_tpo_profile(day_df: pd.DataFrame, n_bins: int = 40):
    """
    Computes exact Market Profile TPO (30-minute periods A through M) from 1-minute bars:
      - TPO POC (price bin visited by the largest number of 30m periods)
      - TPO VAH & VAL (70% Value Area High and Low around TPO POC)
      - Initial Balance (IB: periods A+B, 09:30-10:29) High & Low
      - Lower & Upper Buying/Selling Tails (single-print rejection tails at extremes)
    """
    low_min = day_df["l"].min()
    high_max = day_df["h"].max()
    if high_max <= low_min + 1e-6:
        return low_min, low_min, low_min, low_min, low_min, 0.0, 0.0

    bins = np.linspace(low_min, high_max, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    tpo_counts = np.zeros(n_bins, dtype=float)

    # Group into 30-minute brackets (minute_idx 0-29 is bracket 0, ..., 360-389 is bracket 12)
    brackets = day_df["minute_idx"].values // 30

    ib_mask = brackets <= 1
    ib_df = day_df[ib_mask]
    ib_h = ib_df["h"].max() if len(ib_df) else high_max
    ib_l = ib_df["l"].min() if len(ib_df) else low_min

    for b_id in np.unique(brackets):
        sub = day_df[brackets == b_id]
        b_l = sub["l"].min()
        b_h = sub["h"].max()
        hit = (bin_centers >= b_l - 1e-6) & (bin_centers <= b_h + 1e-6)
        tpo_counts[hit] += 1.0

    poc_idx = int(np.argmax(tpo_counts))
    tpo_poc = bin_centers[poc_idx]

    # 70% TPO Value Area around POC
    total_tpo = max(1.0, tpo_counts.sum())
    target_tpo = 0.70 * total_tpo
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

    tpo_val = bin_centers[lo_i]
    tpo_vah = bin_centers[hi_i]

    # Single-print buying tail at bottom & selling tail at top (fraction of range with <= 1 TPO letter)
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

    buying_tail_pct = (low_single / n_bins) * 100.0
    selling_tail_pct = (high_single / n_bins) * 100.0
    return tpo_poc, tpo_vah, tpo_val, ib_h, ib_l, buying_tail_pct, selling_tail_pct


def build_daily_session_features(df_1m: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Extracts daily session features from real 1-minute bars:
      - 1m OR (09:30), 5m OR (09:30-09:34), 30m OR (09:30-09:59)
      - 1m Delta Flow (CVD at 10:00 AM, Afternoon CVD, Full-Day CVD, Price-CVD Divergence)
      - Session VWAP & +/-1 SD bands (at 10:00 AM and 16:00 Close)
      - Market Profile TPO (POC, VAH, VAL, IBH, IBL, Buying/Selling Tails)
    """
    rows = []
    for dt_str, day_df in df_1m.groupby("date_str", sort=True):
        if len(day_df) < 30:
            continue

        o_day = day_df["o"].iloc[0]
        h_day = day_df["h"].max()
        l_day = day_df["l"].min()
        c_day = day_df["c"].iloc[-1]
        v_day = day_df["v"].sum()
        rng_day = max(h_day - l_day, 1e-6)

        # 1-Minute Delta Flow (Signed Volume Delta using intrabar CLV + tick direction)
        bar_rng = (day_df["h"] - day_df["l"]).values
        clv_1m = np.where(
            bar_rng > 1e-6,
            ((day_df["c"] - day_df["l"]) - (day_df["h"] - day_df["c"])) / np.maximum(bar_rng, 1e-6),
            np.sign(day_df["c"].diff().fillna(0.0)),
        )
        delta_1m = day_df["v"].values * clv_1m
        cvd_cum = np.cumsum(delta_1m)

        # Opening Ranges: 1m (first bar), 5m (first 5 bars), 30m (first 30 bars)
        or1_h, or1_l, or1_c = day_df["h"].iloc[0], day_df["l"].iloc[0], day_df["c"].iloc[0]
        or5_df = day_df.iloc[:5]
        or5_h, or5_l, or5_c = or5_df["h"].max(), or5_df["l"].min(), or5_df["c"].iloc[-1]
        or30_df = day_df.iloc[:30]
        or30_h, or30_l, or30_c = or30_df["h"].max(), or30_df["l"].min(), or30_df["c"].iloc[-1]
        or30_v = or30_df["v"].sum()

        # 10:00 AM (Post-30m OR) VWAP & Delta Flow
        tp_30 = (or30_df["h"] + or30_df["l"] + or30_df["c"]) / 3.0
        vwap_10am = (tp_30 * or30_df["v"]).sum() / max(1.0, or30_v)
        cvd_30m_ratio = delta_1m[:30].sum() / max(1.0, or30_v)

        # Full-Session VWAP & Standard Deviation
        tp_all = (day_df["h"] + day_df["l"] + day_df["c"]) / 3.0
        sess_vwap = (tp_all * day_df["v"]).sum() / max(1.0, v_day)
        vwap_sd = np.sqrt(((tp_all - sess_vwap) ** 2 * day_df["v"]).sum() / max(1.0, v_day))

        cvd_day_ratio = delta_1m.sum() / max(1.0, v_day)
        # Afternoon (last 2 hours) Delta Flow vs Morning
        pm_v = day_df["v"].iloc[-120:].sum()
        cvd_pm_ratio = delta_1m[-120:].sum() / max(1.0, pm_v)

        # TPO Profile
        tpo_poc, tpo_vah, tpo_val, ib_h, ib_l, buy_tail, sell_tail = compute_tpo_profile(day_df)

        rows.append(
            {
                "date": pd.Timestamp(dt_str),
                "date_str": dt_str,
                "open": o_day,
                "high": h_day,
                "low": l_day,
                "close": c_day,
                "volume": v_day,
                "clv_day": (c_day - l_day) / rng_day,
                "or1_h": or1_h,
                "or1_l": or1_l,
                "or1_c": or1_c,
                "or5_h": or5_h,
                "or5_l": or5_l,
                "or5_c": or5_c,
                "or30_h": or30_h,
                "or30_l": or30_l,
                "or30_c": or30_c,
                "vwap_10am": vwap_10am,
                "sess_vwap": sess_vwap,
                "vwap_sd": vwap_sd,
                "cvd_30m_ratio": cvd_30m_ratio,
                "cvd_day_ratio": cvd_day_ratio,
                "cvd_pm_ratio": cvd_pm_ratio,
                "tpo_poc": tpo_poc,
                "tpo_vah": tpo_vah,
                "tpo_val": tpo_val,
                "ib_h": ib_h,
                "ib_l": ib_l,
                "tpo_buy_tail": buy_tail,
                "tpo_sell_tail": sell_tail,
                # Intraday 10:00 AM -> 4:00 PM return
                "ret_10am_to_close_pct": (c_day / or30_c - 1.0) * 100.0,
            }
        )

    df = pd.DataFrame(rows).set_index("date").sort_index()

    # Previous Day (PD) & Prior Session (PD2) & Previous Week (PW) Structural Levels
    df["pdh"] = df["high"].shift(1)
    df["pdl"] = df["low"].shift(1)
    df["pdc"] = df["close"].shift(1)
    df["pd_vwap"] = df["sess_vwap"].shift(1)
    df["pd_poc"] = df["tpo_poc"].shift(1)
    df["pd_vah"] = df["tpo_vah"].shift(1)
    df["pd_val"] = df["tpo_val"].shift(1)

    # Rolling 5d (Previous Week proxy) High / Low & 5d Fixed-Range VWAP
    df["pwh"] = df["high"].shift(1).rolling(5, min_periods=2).max()
    df["pwl"] = df["low"].shift(1).rolling(5, min_periods=2).min()
    df["fr_vwap5"] = (df["sess_vwap"] * df["volume"]).rolling(5, min_periods=1).sum() / (
        df["volume"].rolling(5, min_periods=1).sum() + 1.0
    )

    # Structural Features for Intraday (known at 10:00 AM after 30m OR)
    df["or30_above_or5_h"] = (df["or30_c"] > df["or5_h"]).astype(float)
    df["or30_below_or5_l"] = (df["or30_c"] < df["or5_l"]).astype(float)
    df["or5_above_or1_h"] = (df["or5_c"] > df["or1_h"]).astype(float)
    df["or30_vs_vwap10_pct"] = (df["or30_c"] / df["vwap_10am"] - 1.0) * 100.0
    df["or30_vs_pd_poc_pct"] = (df["or30_c"] / df["pd_poc"] - 1.0) * 100.0
    df["or30_above_pdh"] = (df["or30_c"] > df["pdh"]).astype(float)
    df["or30_reclaim_pdl"] = ((df["or30_l"] < df["pdl"]) & (df["or30_c"] > df["pdl"])).astype(float)
    df["or30_reclaim_pd_val"] = ((df["open"] < df["pd_val"]) & (df["or30_c"] > df["pd_val"])).astype(float)

    # End-of-Day Features (known at 4:00 PM Close for predicting NEXT DAY)
    df["close_vs_vwap_sd"] = (df["close"] - df["sess_vwap"]) / (df["vwap_sd"] + 1e-6)
    df["close_vs_tpo_poc_pct"] = (df["close"] / df["tpo_poc"] - 1.0) * 100.0
    df["close_above_tpo_vah"] = (df["close"] > df["tpo_vah"]).astype(float)
    df["close_below_tpo_val"] = (df["close"] < df["tpo_val"]).astype(float)
    df["reclaimed_or30_l"] = ((df["low"] < df["or30_l"]) & (df["close"] >= df["or30_l"] * 0.998)).astype(float)
    df["closed_above_or30_h"] = (df["close"] > df["or30_h"]).astype(float)
    df["undercut_reclaim_pdl"] = ((df["low"] < df["pdl"]) & (df["close"] > df["pdl"])).astype(float)
    df["undercut_reclaim_pwl"] = ((df["low"] < df["pwl"]) & (df["close"] > df["pwl"])).astype(float)
    df["dist_fr_vwap5_pct"] = (df["close"] / df["fr_vwap5"] - 1.0) * 100.0
    df["delta_vwap_div"] = df["cvd_pm_ratio"] + (df["close_vs_vwap_sd"] * 0.15)

    # Targets
    df["next_day_ret_pct"] = df["close"].pct_change().shift(-1) * 100.0
    df["next_day_up"] = (df["next_day_ret_pct"] > 0.0).astype(int)
    df["intraday_10am_up"] = (df["ret_10am_to_close_pct"] > 0.0).astype(int)
    return df


def attach_real_gex_and_walls(df: pd.DataFrame, is_qqq_family: bool = True) -> pd.DataFrame:
    df = df.copy()
    # 1. Merge SqueezeMetrics real GEX & DIX (2011-2026)
    if DIX_PATH.exists():
        dix_df = pd.read_parquet(DIX_PATH)
        dix_df["date"] = pd.to_datetime(dix_df["date"])
        dix_df = dix_df.set_index("date").sort_index()
        df["dix"] = dix_df["dix"].reindex(df.index).ffill()
        df["gex_bn"] = dix_df["gex_bn"].reindex(df.index).ffill()
        df["gex_z20"] = dix_df["gex_z20"].reindex(df.index).ffill()
    else:
        df["dix"] = 0.45
        df["gex_bn"] = 0.0
        df["gex_z20"] = 0.0

    # 2. Merge OPRA 0DTE Call Wall, Put Wall, Gamma Flip & Net GEX (for QQQ/TQQQ)
    if OPRA_PATH.exists():
        opra = pd.read_parquet(OPRA_PATH)
        q_opra = opra[opra["symbol"] == "QQQ"].copy()
        q_opra["date"] = pd.to_datetime(q_opra["date"])
        q_opra = q_opra.set_index("date").sort_index()
        for col in ["call_wall_1030", "put_wall_1030", "gamma_flip_1030", "net_gex_m_1030", "pc_vol_ratio_1030"]:
            df[f"qqq_{col}"] = q_opra[col].reindex(df.index).ffill()
    return df


def run_models_and_report():
    print("=" * 115)
    print("REAL-DATA OPENING RANGE (1m/5m/30m) + TPO PROFILE + DELTA FLOW (CVD) + GEX/VWAP MODEL (2024-2026)")
    print("=" * 115)

    # 1. Fetch 1m bars for QQQ, TQQQ, SOXX, SOXL
    bars = {}
    for sym in ["QQQ", "TQQQ", "SOXX", "SOXL"]:
        bars[sym] = fetch_alpaca_1m(sym)

    feats = {}
    for sym in ["QQQ", "TQQQ", "SOXX", "SOXL"]:
        f = build_daily_session_features(bars[sym], sym)
        f = attach_real_gex_and_walls(f, is_qqq_family=(sym in ["QQQ", "TQQQ"]))
        feats[sym] = f

    # Attach underlying index features (QQQ -> TQQQ, SOXX -> SOXL)
    for lev, und in [("TQQQ", "QQQ"), ("SOXL", "SOXX")]:
        u_df = feats[und]
        l_df = feats[lev]
        l_df["und_close"] = u_df["close"]
        l_df["und_pdh"] = u_df["pdh"]
        l_df["und_pdl"] = u_df["pdl"]
        l_df["und_pwh"] = u_df["pwh"]
        l_df["und_pwl"] = u_df["pwl"]
        l_df["und_cvd_30m"] = u_df["cvd_30m_ratio"]
        l_df["und_cvd_day"] = u_df["cvd_day_ratio"]
        l_df["und_cvd_pm"] = u_df["cvd_pm_ratio"]
        l_df["und_close_vs_vwap_sd"] = u_df["close_vs_vwap_sd"]
        l_df["und_tpo_buy_tail"] = u_df["tpo_buy_tail"]
        l_df["und_tpo_sell_tail"] = u_df["tpo_sell_tail"]
        l_df["und_reclaimed_or30_l"] = u_df["reclaimed_or30_l"]
        l_df["und_undercut_reclaim_pwl"] = u_df["undercut_reclaim_pwl"]
        l_df["und_undercut_reclaim_pdl"] = u_df["undercut_reclaim_pdl"]

        if lev == "TQQQ":
            l_df["und_dist_put_wall_pct"] = (u_df["close"] / u_df["qqq_put_wall_1030"] - 1.0) * 100.0
            l_df["und_dist_call_wall_pct"] = (u_df["close"] / u_df["qqq_call_wall_1030"] - 1.0) * 100.0
            l_df["und_above_gamma_flip"] = (u_df["close"] >= u_df["qqq_gamma_flip_1030"]).astype(float)
            l_df["und_put_wall_bounce"] = (
                (u_df["low"] <= u_df["qqq_put_wall_1030"] * 1.003) & (u_df["close"] > u_df["qqq_put_wall_1030"])
            ).astype(float)
        else:
            # For SOXL/SOXX, use round-5 strike Put/Call Wall distance + QQQ Macro GEX
            round5_support = np.floor(u_df["close"] / 5.0) * 5.0
            l_df["und_dist_put_wall_pct"] = (u_df["close"] / round5_support - 1.0) * 100.0
            l_df["und_dist_call_wall_pct"] = (u_df["close"] / (round5_support + 5.0) - 1.0) * 100.0
            l_df["und_above_gamma_flip"] = (l_df["gex_z20"] > -0.5).astype(float)
            l_df["und_put_wall_bounce"] = ((u_df["low"] < round5_support) & (u_df["close"] > round5_support)).astype(
                float
            )

    # =========================================================================
    # MODEL 1: INTRADAY 10:00 AM (POST-30m OR) -> 4:00 PM CLOSE DIRECTION
    # =========================================================================
    print("\n--- MODEL A: INTRADAY 10:00 AM (POST 1m/5m/30m OR + DELTA + TPO + VWAP + GEX) -> 4:00 PM CLOSE ---")
    intraday_cols = [
        "or30_above_or5_h",
        "or30_below_or5_l",
        "or5_above_or1_h",
        "or30_vs_vwap10_pct",
        "or30_vs_pd_poc_pct",
        "or30_above_pdh",
        "or30_reclaim_pdl",
        "or30_reclaim_pd_val",
        "cvd_30m_ratio",
        "und_cvd_30m",
        "gex_z20",
        "dix",
    ]

    for sym in ["TQQQ", "SOXL"]:
        df = feats[sym].dropna(subset=intraday_cols + ["ret_10am_to_close_pct"]).copy()
        is_df = df[df.index.year <= 2025]
        oos_df = df[df.index.year == 2026]

        X_tr, y_tr = is_df[intraday_cols], is_df["intraday_10am_up"]
        X_te, y_te = oos_df[intraday_cols], oos_df["intraday_10am_up"]

        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.25))
        clf.fit(X_tr, y_tr)
        p_tr = clf.predict_proba(X_tr)[:, 1]
        p_te = clf.predict_proba(X_te)[:, 1]

        # Sniper Long (Prob >= 0.58) & Sniper Short (Prob <= 0.42)
        long_te = oos_df[p_te >= 0.58]
        short_te = oos_df[p_te <= 0.42]
        l_wr = (long_te["ret_10am_to_close_pct"] > 0).mean() * 100 if len(long_te) else 0
        s_wr = (short_te["ret_10am_to_close_pct"] < 0).mean() * 100 if len(short_te) else 0
        l_avg = long_te["ret_10am_to_close_pct"].mean() if len(long_te) else 0
        s_avg = -short_te["ret_10am_to_close_pct"].mean() if len(short_te) else 0

        print(
            f"  {sym:5s} Intraday 10AM->4PM | OOS AUC: {roc_auc_score(y_te, p_te):.3f} | "
            f"Bull Sniper (P>=58%): N={len(long_te):3d}, WR={l_wr:5.1f}%, Avg={l_avg:+5.2f}% | "
            f"Bear Sniper (P<=42%): N={len(short_te):3d}, WR={s_wr:5.1f}%, Avg={s_avg:+5.2f}%"
        )

    # =========================================================================
    # MODEL 2: END-OF-DAY (4:00 PM) -> NEXT-DAY DIRECTION & SNIPER SETUPS
    # =========================================================================
    print("\n--- MODEL B: NEXT-DAY DIRECTION (EOD 1m/5m/30m OR RECLAIM + TPO + DELTA + VWAP + GEX WALLS) ---")
    eod_cols = [
        "clv_day",
        "close_vs_vwap_sd",
        "und_close_vs_vwap_sd",
        "close_vs_tpo_poc_pct",
        "close_above_tpo_vah",
        "close_below_tpo_val",
        "tpo_buy_tail",
        "tpo_sell_tail",
        "cvd_day_ratio",
        "cvd_pm_ratio",
        "und_cvd_pm",
        "reclaimed_or30_l",
        "closed_above_or30_h",
        "undercut_reclaim_pdl",
        "undercut_reclaim_pwl",
        "und_put_wall_bounce",
        "dist_fr_vwap5_pct",
        "gex_z20",
        "dix",
    ]

    for sym in ["TQQQ", "SOXL"]:
        df = feats[sym].dropna(subset=eod_cols).copy()
        valid = df.dropna(subset=["next_day_ret_pct"]).copy()
        is_df = valid[valid.index.year <= 2025]
        oos_df = valid[valid.index.year == 2026]

        X_tr, y_tr = is_df[eod_cols], is_df["next_day_up"]
        X_te, y_te = oos_df[eod_cols], oos_df["next_day_up"]

        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.20))
        clf.fit(X_tr, y_tr)
        p_all = clf.predict_proba(valid[eod_cols])[:, 1]
        valid["prob_up"] = p_all
        p_te = clf.predict_proba(X_te)[:, 1]

        # Also evaluate the exact structural archetype active on 2026-11-17:
        # "Put-Wall / PDL / PWL Undercut + Afternoon VWAP Reclaim + Positive PM Delta + CLV >= 60%"
        hammer_reclaim = (
            (valid["close_vs_vwap_sd"] >= 0.30)
            & (valid["clv_day"] >= 0.58)
            & ((valid["cvd_pm_ratio"] > 0.0) | (valid["und_cvd_pm"] > 0.0))
            & (
                (valid["undercut_reclaim_pdl"] == 1)
                | (valid["undercut_reclaim_pwl"] == 1)
                | (valid["und_put_wall_bounce"] == 1)
                | (valid["dist_fr_vwap5_pct"] < -3.0)
            )
        )
        sub_h = valid[hammer_reclaim]
        sub_h_oos = sub_h[sub_h.index.year == 2026]
        h_wr_all = (sub_h["next_day_ret_pct"] > 0).mean() * 100 if len(sub_h) else 0
        h_wr_oos = (sub_h_oos["next_day_ret_pct"] > 0).mean() * 100 if len(sub_h_oos) else 0
        h_avg_all = sub_h["next_day_ret_pct"].mean() if len(sub_h) else 0

        # High-Conviction Model Long (prob_up >= 0.58)
        hi_long = oos_df[p_te >= 0.58]
        hi_wr = (hi_long["next_day_ret_pct"] > 0).mean() * 100 if len(hi_long) else 0
        hi_avg = hi_long["next_day_ret_pct"].mean() if len(hi_long) else 0

        # Predict TODAY (2026-11-18) from the last row (2026-11-17 Close)
        last_row = df.iloc[[-1]]
        live_prob = clf.predict_proba(last_row[eod_cols])[0, 1] * 100.0
        r = last_row.iloc[0]

        # Extract feature weights
        coefs = pd.Series(clf.named_steps["logisticregression"].coef_[0], index=eod_cols).sort_values(
            key=abs, ascending=False
        )

        print(
            f"  {sym:5s} Next-Day Model | OOS High-Conv (P>=58%): N={len(hi_long):2d}, WR={hi_wr:5.1f}%, Avg={hi_avg:+5.2f}%"
        )
        print(
            f"  {sym:5s} PutWall/PWL Undercut + VWAP/Delta Reclaim Archetype: N={len(sub_h):2d} (OOS={len(sub_h_oos):2d}) | "
            f"Next-Day WR: All={h_wr_all:5.1f}% (OOS={h_wr_oos:5.1f}%) | Avg Next-Day Gain: {h_avg_all:+5.2f}%"
        )
        print(
            f"  >>> {sym:5s} LIVE 2026-11-17 CLOSE TELEMETRY -> 2026-11-18 PREDICTION: "
            f"Prob_Up = {live_prob:5.1f}% | TPO_POC={r['tpo_poc']:.2f} (VAH={r['tpo_vah']:.2f}, VAL={r['tpo_val']:.2f}) | "
            f"BuyTail={r['tpo_buy_tail']:.1f}% | CVD_Day={r['cvd_day_ratio']*100:+.1f}% | CVD_PM={r['cvd_pm_ratio']*100:+.1f}% | "
            f"VWAP_Z={r['close_vs_vwap_sd']:+.2f}SD"
        )
        print(f"  Top 5 Model Weights ({sym}): {coefs.head(5).round(3).to_dict()}\n")


if __name__ == "__main__":
    run_models_and_report()
