"""
Untouched 2020-2023 (including 2022 Bear Market) Due-Diligence Backtest
of the Reduced 2-Factor Model (15:50 TPO Profile + 1m Delta Flow CVD Only)
with Real SQQQ / SOXS Fills on Bearish Signals
=======================================================================
1. Downloads & caches real 1-minute Alpaca SIP RTH bars (2020-01-02 to 2026-11-17)
   for QQQ, TQQQ, SQQQ, SOXX, SOXL, SOXS.
2. Uses ONLY the reduced 2-factor feature set at 15:50:
   - TPO Profile (09:30-15:50): clv_1550, poc_dist_1550, above_vah_1550, below_val_1550, btail_1550, stail_1550
   - 1m Delta Flow (CVD 09:30-15:50): cvd_day_1550, cvd_pm_1550, und_cvd_pm_1550
   (VWAP, 30m OR, and DIX/GEX dropped per prior ablation).
3. Executes bearish signals (prob <= 0.45) using REAL long positions in SQQQ (for TQQQ)
   and SOXS (for SOXL) at 15:50 -> 15:50 next day (and 15:50 -> 09:30 next open),
   charging 1x, 2x, and 3x round-trip transaction costs.
4. Evaluates both:
   - Test A: Chronological Purged Expanding Walk-Forward from 2020 through 2026
     (reporting 2020-2023 Untouched, 2022 Bear Market Alone, 2024-2026, and Full 2020-2026).
   - Test B: Frozen-Weight Out-of-Sample Test (fit strictly on 2024-2026, tested untouched on 2020-2023).
"""

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
ROOT = Path("c:/Users/amitr/OneDrive/Documents/ANT_SOX")
load_dotenv(ROOT / ".env", override=True)

ALPACA_DIR = ROOT / "data" / "cache" / "alpaca_1m_stocks"
ALPACA_DIR.mkdir(parents=True, exist_ok=True)

KEY = (os.getenv("APCA_API_KEY_ID") or "").strip()
SEC = (os.getenv("APCA_API_SECRET_KEY") or "").strip()
HEADERS = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC}


def fetch_chunk(symbol: str, s_date: str, e_date: str) -> list[dict]:
    url = "https://data.alpaca.markets/v2/stocks/bars"
    bars_out = []
    page_token = None
    while True:
        params = {
            "symbols": symbol,
            "timeframe": "1Min",
            "start": f"{s_date}T00:00:00Z",
            "end": f"{e_date}T23:59:59Z",
            "adjustment": "split",
            "feed": "sip",
            "limit": 10000,
        }
        if page_token:
            params["page_token"] = page_token
        for attempt in range(4):
            try:
                r = requests.get(url, headers=HEADERS, params=params, timeout=25)
                if r.status_code == 200:
                    data = r.json()
                    b = data.get("bars", {}).get(symbol, [])
                    bars_out.extend(b)
                    page_token = data.get("next_page_token")
                    break
                time.sleep(1.0 * (attempt + 1))
            except Exception:
                time.sleep(1.0)
        if not page_token:
            break
    return bars_out


def fetch_full_1m_2020_2026(symbol: str) -> pd.DataFrame:
    full_cache = ALPACA_DIR / f"{symbol}_1m_2020-01-02_2026-11-17_rth.parquet"
    if full_cache.exists():
        df = pd.read_parquet(full_cache)
        print(f"[{symbol}] Loaded 2020-2026 cache: {len(df):,} bars ({df['date_str'].nunique()} sessions)")
        return df

    # Check if 2024-02-01_2026-11-17 cache already exists
    cache_2024 = ALPACA_DIR / f"{symbol}_1m_2024-02-01_2026-11-17_rth.parquet"
    end_fetch = "2024-01-31" if cache_2024.exists() else "2026-11-17"
    print(f"[{symbol}] Downloading 1m SIP bars 2020-01-02 -> {end_fetch}...")

    cur = datetime.strptime("2020-01-02", "%Y-%m-%d")
    end_dt = datetime.strptime(end_fetch, "%Y-%m-%d")
    windows = []
    while cur <= end_dt:
        c_end = min(cur + timedelta(days=180), end_dt)
        windows.append((cur.strftime("%Y-%m-%d"), c_end.strftime("%Y-%m-%d")))
        cur = c_end + timedelta(days=1)

    all_bars = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(fetch_chunk, symbol, w[0], w[1]) for w in windows]
        for f in as_completed(futs):
            all_bars.extend(f.result())

    if not all_bars:
        raise RuntimeError(f"[{symbol}] Zero bars returned from Alpaca!")

    df = pd.DataFrame(all_bars)
    ts = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
    df["date_str"] = ts.dt.strftime("%Y-%m-%d")
    hhmm = ts.dt.hour * 100 + ts.dt.minute
    df = df[(hhmm >= 930) & (hhmm <= 1559)].copy()
    df["minute_idx"] = (ts.dt.hour - 9) * 60 + (ts.dt.minute - 30)
    keep_cols = ["date_str", "minute_idx", "o", "h", "l", "c", "v", "vw", "n"]
    df = df[keep_cols]

    if cache_2024.exists():
        df_2024 = pd.read_parquet(cache_2024)[keep_cols]
        df = pd.concat([df, df_2024], ignore_index=True)

    df = df.sort_values(["date_str", "minute_idx"]).drop_duplicates(subset=["date_str", "minute_idx"]).reset_index(drop=True)
    df.to_parquet(full_cache)
    print(f"[{symbol}] Saved {len(df):,} RTH 1m bars across {df['date_str'].nunique()} sessions (2020-2026).")
    return df


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


def build_1550_table(df_1m: pd.DataFrame, sym: str) -> pd.DataFrame:
    feat_cache = ALPACA_DIR / f"{sym}_1550_tpo_cvd_2020_2026.parquet"
    if feat_cache.exists():
        return pd.read_parquet(feat_cache)

    rows = []
    for dt_str, day_df in df_1m.groupby("date_str", sort=True):
        if len(day_df) < 100:
            continue
        day_df = day_df.sort_values("minute_idx")
        pre_1550 = day_df[day_df["minute_idx"] <= 380]
        if len(pre_1550) < 60:
            continue

        o_day = day_df["o"].iloc[0]
        c_day = day_df["c"].iloc[-1]
        px_1550 = pre_1550["c"].iloc[-1]
        h_1550 = pre_1550["h"].max()
        l_1550 = pre_1550["l"].min()
        clv_1550 = (px_1550 - l_1550) / max(h_1550 - l_1550, 1e-6)

        bar_rng = (pre_1550["h"] - pre_1550["l"]).values
        clv_1m = np.where(
            bar_rng > 1e-6,
            ((pre_1550["c"] - pre_1550["l"]) - (pre_1550["h"] - pre_1550["c"])) / np.maximum(bar_rng, 1e-6),
            np.sign(pre_1550["c"].diff().fillna(0.0)),
        )
        delta_1m = pre_1550["v"].values * clv_1m
        cvd_day_1550 = delta_1m.sum() / max(1.0, pre_1550["v"].sum())
        cvd_pm_1550 = delta_1m[-110:].sum() / max(1.0, pre_1550["v"].iloc[-110:].sum())

        poc_1550, vah_1550, val_1550, btail_1550, stail_1550 = compute_tpo_from_slice(pre_1550)

        rows.append(
            {
                "date": pd.Timestamp(dt_str),
                "open": o_day,
                "close": c_day,
                "px_1550": px_1550,
                "clv_1550": clv_1550,
                "poc_dist_1550": (px_1550 / poc_1550 - 1.0) * 100.0,
                "above_vah_1550": float(px_1550 > vah_1550),
                "below_val_1550": float(px_1550 < val_1550),
                "btail_1550": btail_1550,
                "stail_1550": stail_1550,
                "cvd_day_1550": cvd_day_1550,
                "cvd_pm_1550": cvd_pm_1550,
            }
        )

    df = pd.DataFrame(rows).set_index("date").sort_index()
    df["ret_24h_1550"] = df["px_1550"].pct_change().shift(-1)
    df["ret_ovn_1550"] = (df["open"].shift(-1) / df["px_1550"]) - 1.0
    df["sma20_lag1"] = df["close"].shift(1).rolling(20, min_periods=10).mean()
    df["pdc"] = df["close"].shift(1)
    df.to_parquet(feat_cache)
    return df


def get_historical_rf_daily(idx: pd.DatetimeIndex) -> pd.Series:
    # Actual historical Fed Funds / T-bill regime by year (2020-2026)
    yr = idx.year
    ann_rate = np.where(
        yr == 2020,
        0.0025,
        np.where(yr == 2021, 0.0008, np.where(yr == 2022, 0.0180, np.where(yr == 2023, 0.0500, 0.0475))),
    )
    return pd.Series(ann_rate / 252.0, index=idx)


def compute_metrics(ret_s: pd.Series, bmk_s: pd.Series):
    r = ret_s.dropna()
    b = bmk_s.reindex(r.index).fillna(0.0)
    rf = get_historical_rf_daily(r.index)
    n = len(r)
    if n < 10:
        return {}
    eq = (1.0 + r).cumprod()
    cagr = (eq.iloc[-1] ** (252.0 / n) - 1.0) * 100.0
    peak = eq.cummax()
    max_dd = ((eq / peak - 1.0) * 100.0).min()
    is_uw = eq < peak
    max_uw = int(is_uw.groupby((~is_uw).cumsum()).sum().max()) if is_uw.any() else 0

    ex_r = r - rf
    ex_b = b - rf
    sharpe = (ex_r.mean() / (r.std() + 1e-9)) * np.sqrt(252.0)

    X = sm.add_constant(ex_b.values)
    ols = sm.OLS(ex_r.values, X).fit()
    alpha_ann = ols.params[0] * 252.0 * 100.0
    alpha_t = ols.tvalues[0]
    beta = ols.params[1] if len(ols.params) > 1 else 0.0

    worst_day = r.min() * 100.0
    worst_mo = ((1.0 + r).resample("ME").prod() - 1.0).min() * 100.0
    return {
        "n": n,
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "beta": beta,
        "alpha": alpha_ann,
        "t": alpha_t,
        "worst_day": worst_day,
        "worst_mo": worst_mo,
        "max_uw": max_uw,
    }


def run_simulation():
    print("=" * 125)
    print("UNTOUCHED 2020-2023 (+ 2022 BEAR MARKET) BACKTEST: 2-FACTOR (TPO + 1m CVD) WITH REAL SQQQ / SOXS FILLS")
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
        df = tables[bull_etf].copy()
        u_df = tables[und_etf].reindex(df.index)
        inv_df = tables[bear_etf].reindex(df.index)

        df["und_cvd_pm_1550"] = u_df["cvd_pm_1550"]
        df["inv_ret_24h_1550"] = inv_df["ret_24h_1550"]
        df["inv_ret_ovn_1550"] = inv_df["ret_ovn_1550"]

        df = df.dropna(subset=feat_2factor + ["ret_24h_1550", "inv_ret_24h_1550", "ret_ovn_1550", "inv_ret_ovn_1550"]).copy()
        rf_s = get_historical_rf_daily(df.index)
        X_all = df[feat_2factor].values

        # ---------------------------------------------------------------------
        # TEST 1: Chronological Purged Expanding Walk-Forward (Jul 2020 - Nov 2026)
        # ---------------------------------------------------------------------
        min_train = 126
        refit_step = 21

        def sim_wf(ret_bull_col, ret_bear_col, cost_bps, mode="expanding"):
            y_all = (df[ret_bull_col].values > 0.0).astype(int)
            r_bull = df[ret_bull_col].values
            r_bear = df[ret_bear_col].values
            rf_v = rf_s.values
            cost_dec = cost_bps / 10000.0
            n = len(df)
            pos = np.zeros(n)

            if mode == "expanding":
                clf = None
                for t in range(min_train, n):
                    if clf is None or ((t - min_train) % refit_step == 0):
                        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.20))
                        clf.fit(X_all[: t - 1], y_all[: t - 1])
                    p = clf.predict_proba(X_all[t : t + 1])[0, 1]
                    if p >= 0.55:
                        pos[t] = 1.0
                    elif p <= 0.45:
                        pos[t] = -1.0
            elif mode == "frozen_2024_2026_on_2020_2023":
                # Fit strictly on 2024-2026, predict untouched 2020-2023!
                m_24 = df.index.year >= 2024
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.20))
                clf.fit(X_all[m_24], y_all[m_24])
                probs = clf.predict_proba(X_all)[:, 1]
                pos = np.where(probs >= 0.55, 1.0, np.where(probs <= 0.45, -1.0, 0.0))

            # Real ETF execution:
            # pos == +1 -> Long TQQQ/SOXL (r_bull - cost)
            # pos == -1 -> Long REAL SQQQ/SOXS (r_bear - cost)
            # pos ==  0 -> Cash (rf_v)
            net_ret = np.where(
                pos == 1.0,
                r_bull - cost_dec,
                np.where(pos == -1.0, r_bear - cost_dec, rf_v),
            )
            out = pd.DataFrame({"pos": pos, "strat_ret": net_ret, "r_bull": r_bull, "r_bear": r_bear}, index=df.index)
            return out.iloc[min_train:] if mode == "expanding" else out[out.index.year <= 2023]

        wf_24h_1x = sim_wf("ret_24h_1550", "inv_ret_24h_1550", cost_1x, "expanding")
        wf_24h_2x = sim_wf("ret_24h_1550", "inv_ret_24h_1550", cost_1x * 2, "expanding")
        wf_24h_3x = sim_wf("ret_24h_1550", "inv_ret_24h_1550", cost_1x * 3, "expanding")
        wf_ovn_1x = sim_wf("ret_ovn_1550", "inv_ret_ovn_1550", cost_1x, "expanding")
        wf_ovn_2x = sim_wf("ret_ovn_1550", "inv_ret_ovn_1550", cost_1x * 2, "expanding")
        wf_ovn_3x = sim_wf("ret_ovn_1550", "inv_ret_ovn_1550", cost_1x * 3, "expanding")
        frozen_20_23 = sim_wf("ret_24h_1550", "inv_ret_24h_1550", cost_1x, "frozen_2024_2026_on_2020_2023")
        frozen_ovn_20_23 = sim_wf("ret_ovn_1550", "inv_ret_ovn_1550", cost_1x, "frozen_2024_2026_on_2020_2023")

        idx = wf_24h_1x.index
        bmk = df.loc[idx, "ret_24h_1550"]
        # Simple SMA20 Baseline (Long Bull ETF when PDC > SMA20, Long Bear ETF when PDC < SMA20)
        sma_long = df.loc[idx, "pdc"] > df.loc[idx, "sma20_lag1"]
        sma_ret = pd.Series(
            np.where(sma_long, df.loc[idx, "ret_24h_1550"], df.loc[idx, "inv_ret_24h_1550"]) - (cost_1x / 20000.0),
            index=idx,
        )

        print(f"\n=========================================================================================================")
        print(f"  PAIR: {bull_etf} (Long) / {bear_etf} (Short) | Underlying: {und_etf} | Jul 2020 - Nov 2026 ({len(idx)} Sessions)")
        print(f"=========================================================================================================")

        # 1. Period Breakdown Table
        periods = [
            ("UNTOUCHED 2020-2023 (Jul 2020 - Dec 2023)", idx[idx.year <= 2023]),
            ("2022 BEAR MARKET ALONE (Jan 2022 - Dec 2022)", idx[idx.year == 2022]),
            ("RECENT PERIOD (Jan 2024 - Nov 2026)", idx[idx.year >= 2024]),
            ("FULL WALK-FORWARD (Jul 2020 - Nov 2026)", idx),
        ]

        for p_title, p_idx in periods:
            print(f"\n  >>> {p_title} ({len(p_idx)} trading days) <<<")
            print(f"  {'Strategy / Benchmark':56s} | {'CAGR%':>7s} | {'MaxDD%':>7s} | {'Sharpe':>6s} | {'Beta':>5s} | {'Alpha%':>7s} | {'t-stat':>6s} | {'WorstDay':>8s} | {'WorstMo':>8s} | {'MaxUW':>5s}")
            print("  " + "-" * 140)
            b_sub = bmk.loc[p_idx]
            rows_to_print = [
                (f"1. Buy & Hold {bull_etf} (24h)", b_sub),
                (f"2. Simple SMA20 ({bull_etf} vs {bear_etf}, 1x Cost)", sma_ret.loc[p_idx]),
                ("3. 2-Factor (TPO+CVD) 24h Hold (1x Cost)", wf_24h_1x.loc[p_idx, "strat_ret"]),
                ("4. 2-Factor (TPO+CVD) 24h Hold (2x Cost)", wf_24h_2x.loc[p_idx, "strat_ret"]),
                ("5. 2-Factor (TPO+CVD) 24h Hold (3x Cost)", wf_24h_3x.loc[p_idx, "strat_ret"]),
                ("6. 2-Factor (TPO+CVD) Overnight 15:50->09:30 (1x Cost)", wf_ovn_1x.loc[p_idx, "strat_ret"]),
                ("7. 2-Factor (TPO+CVD) Overnight 15:50->09:30 (2x Cost)", wf_ovn_2x.loc[p_idx, "strat_ret"]),
                ("8. 2-Factor (TPO+CVD) Overnight 15:50->09:30 (3x Cost)", wf_ovn_3x.loc[p_idx, "strat_ret"]),
            ]
            if p_title.startswith("UNTOUCHED 2020-2023") or p_title.startswith("2022 BEAR"):
                rows_to_print.append(("9. FROZEN 2024-26 Weights -> Untouched 24h (1x Cost)", frozen_20_23.loc[p_idx, "strat_ret"]))
                rows_to_print.append(("10. FROZEN 2024-26 Weights -> Untouched Ovn (1x Cost)", frozen_ovn_20_23.loc[p_idx, "strat_ret"]))

            for lbl, s_r in rows_to_print:
                m = compute_metrics(s_r, b_sub)
                print(
                    f"  {lbl:56s} | {m['cagr']:+6.1f}% | {m['max_dd']:+6.1f}% | {m['sharpe']:6.2f} | "
                    f"{m['beta']:5.2f} | {m['alpha']:+6.1f}% | {m['t']:+6.2f} | "
                    f"{m['worst_day']:+7.2f}% | {m['worst_mo']:+7.2f}% | {m['max_uw']:4d}d"
                )

        # Print Long (TQQQ/SOXL) vs Short (SQQQ/SOXS) trade breakdown on Untouched 2020-2023 and Full 2020-2026
        for tag, sub_w in [("UNTOUCHED 2020-2023", wf_24h_1x.loc[idx[idx.year <= 2023]]), ("FULL 2020-2026", wf_24h_1x)]:
            l_tr = sub_w[sub_w["pos"] == 1.0]
            s_tr = sub_w[sub_w["pos"] == -1.0]
            l_wr = (l_tr["r_bull"] > 0).mean() * 100.0 if len(l_tr) else 0.0
            s_wr = (s_tr["r_bear"] > 0).mean() * 100.0 if len(s_tr) else 0.0
            print(
                f"\n  [{tag} TRADE AUDIT ({bull_etf}/{bear_etf})]: "
                f"Long {bull_etf}: N={len(l_tr):3d}, WR={l_wr:5.1f}%, Avg={l_tr['r_bull'].mean()*100:+.2f}% | "
                f"Long {bear_etf} (Short Signal): N={len(s_tr):3d}, WR={s_wr:5.1f}%, Avg={s_tr['r_bear'].mean()*100:+.2f}%"
            )


if __name__ == "__main__":
    run_simulation()
