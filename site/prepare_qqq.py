"""Builds site/build/qqq_data.json for the QQQ Weather Plan page from the qqq_alpha_engine repo (v6.0.0).
Re-runs the repo's own simulation, recording every rebalance, and packs daily growth series compactly."""
import json, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent / "build"; HERE.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("bt", str(REPO / "scripts" / "run_voltarget_backtest.py")); bt = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt)

df = bt.load_data(); close = df["close"]
REG = [(0.40, "Stormy"), (0.25, "Choppy"), (0.15, "Normal"), (0.0, "Calm")]
def regime(vol):
    for thr, name in REG:
        if vol >= thr: return name
vol21 = close.pct_change().rolling(21).std() * np.sqrt(252)

def run_with_log(cap):
    """Same logic as bt.simulate (QLD leverage, 10 bps, 10-point deadband) with a trade log."""
    tgt = bt.target_exposure(close, 0.20, cap, 21)
    q_on, q_in, l_on, l_in, rf = bt.leg_returns(df)
    t = tgt.values; i0 = int(np.searchsorted(df.index.values, bt.START))
    v_q = v_l = 0.0; cash = bt.INITIAL; navs = []; log = []
    for i in range(i0, len(df)):
        v_q *= 1 + q_on[i]; v_l *= 1 + l_on[i]
        nav_open = v_q + v_l + cash; held = (v_q + 2 * v_l) / nav_open
        w = float(t[i - 1]); crossed = (held <= 1 < w) or (w <= 1 < held)
        if i == i0 or abs(w - held) >= 0.10 or crossed:
            a = max(w - 1, 0)
            nq, nl = nav_open * (min(w, 1) - a), nav_open * a
            trade = abs(nq - v_q) + abs(nl - v_l); net = nav_open - trade * 0.001
            nq, nl = net * (min(w, 1) - a), net * a
            cash = net - nq - nl; v_q, v_l = nq, nl
            log.append({"d": df.index[i], "from": round(held, 3), "to": round(w, 3), "vol": round(float(vol21.iloc[i - 1]) * 100, 1),
                        "reg": regime(float(vol21.iloc[i - 1])), "qqq": round(float(df["open"].iloc[i]), 2),
                        "w": [round((min(w, 1) - a) * 100, 1), round(a * 100, 1), round(max(1 - w, 0) * 100, 1)]})
        v_q *= 1 + q_in[i]; v_l *= 1 + l_in[i]; cash *= 1 + rf[i]
        navs.append(v_q + v_l + cash)
    run_with_log.held = (v_q + 2 * v_l) / (v_q + v_l + cash)
    return pd.Series(navs, pd.to_datetime(df.index[i0:])), log

nav15, log15 = run_with_log(1.5); held15 = run_with_log.held
nav10, log10 = run_with_log(1.0); held10 = run_with_log.held
bh, _, _ = bt.simulate(df, pd.Series(1.0, df.index))
check = pd.read_csv(REPO / "data" / "outputs" / "voltarget_daily_nav.csv", index_col=0, parse_dates=True)["STRATEGY: Vol target 20%, max 1.5x"]
assert abs(nav15.iloc[-1] / check.iloc[-1] - 1) < 1e-9, "trade-log run does not match the repo backtest"

dates = nav15.index
d0 = dates[0]
series = {"day": [(d - d0).days for d in dates], "start": str(d0.date()),
          "qqq": [round(x / bh.iloc[0], 5) for x in bh.values],
          "s15": [round(x / nav15.iloc[0], 5) for x in nav15.values],
          "s10": [round(x / nav10.iloc[0], 5) for x in nav10.values]}

per = pd.read_csv(REPO / "data" / "outputs" / "voltarget_period_stats.csv")
yrs = pd.read_csv(REPO / "data" / "outputs" / "voltarget_calendar_years.csv", index_col=0)
sig = json.loads((REPO / "data" / "outputs" / "production_voltarget_signal.json").read_text())
lat_vol = float(vol21.iloc[-1])
prices = {"QQQ": sig["holdings"][0]["spot_price"], "QLD": sig["holdings"][1]["spot_price"], "SGOV": sig["holdings"][2]["spot_price"]}
def weights(cap):
    w = min(cap, 0.20 / lat_vol); a = max(w - 1, 0)
    return {"exposure": round(w, 3), "QQQ": round((min(w, 1) - a) * 100, 1), "QLD": round(a * 100, 1), "SGOV": round(max(1 - w, 0) * 100, 1)}
# last 12 months of daily volatility for the weather strip
last = vol21.iloc[-260:]
out = {
    "asOf": str(dates[-1].date()), "vol": round(lat_vol * 100, 1), "regime": regime(lat_vol),
    "qqqClose": float(close.iloc[-1]), "prices": prices, "now": {"s15": weights(1.5), "s10": weights(1.0)}, "held": {"s15": round(held15, 3), "s10": round(held10, 3)}, "lastTrade": {"s15": str(pd.Timestamp(log15[-1]["d"]).date()), "s10": str(pd.Timestamp(log10[-1]["d"]).date())},
    "volStrip": [[str(d), round(float(v) * 100, 1)] for d, v in last.items()],
    "series": series,
    "years": [{"y": int(y), "qqq": round(float(r["QQQ buy & hold (total return)"]), 1), "s15": round(float(r["STRATEGY: Vol target 20%, max 1.5x"]), 1),
               "s10": round(float(r["Vol target 20%, max 1.0x"]), 1)} for y, r in yrs.iterrows()],
    "periods": [{"p": p, **{k: round(float(per[(per.period == p) & (per.strategy == s)][m].iloc[0]), 2)
                            for k, s in (("qqq", "QQQ buy & hold (total return)"), ("s15", "STRATEGY: Vol target 20%, max 1.5x"), ("s10", "Vol target 20%, max 1.0x"))
                            for m in ["cagr_pct"]},
                 **{k + "dd": round(float(per[(per.period == p) & (per.strategy == s)]["max_dd_pct"].iloc[0]), 1)
                    for k, s in (("qqq", "QQQ buy & hold (total return)"), ("s15", "STRATEGY: Vol target 20%, max 1.5x"), ("s10", "Vol target 20%, max 1.0x"))}}
                for p in per.period.unique()],
    "log15": [{**e, "d": str(pd.Timestamp(e["d"]).date())} for e in log15],
    "log10": [{**e, "d": str(pd.Timestamp(e["d"]).date())} for e in log10],
}
(HERE / "qqq_data.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print("asOf", out["asOf"], "vol", out["vol"], out["regime"], out["now"], "| trades 1.5x:", len(log15), "1.0x:", len(log10),
      "| per year:", round(len(log15) / 27.6, 1), "| size KB:", round(len(json.dumps(out, separators=(',', ':'))) / 1024))
print("regime share of days:", pd.Series([regime(v) for v in vol21.dropna()]).value_counts(normalize=True).round(2).to_dict())
