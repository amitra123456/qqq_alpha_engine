"""Builds site/build/big8_data.json for the Big Eight Weather Plan page from the repo's own backtest (scripts/run_big8_backtest.py)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent / "build"; HERE.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))
from scripts.big8_data import load_prices
from scripts.run_big8_backtest import regime, run

r = run()
nav, bh, vol, cal = r["nav"], r["qqq_bh"], r["vol"], r["calendar"]
asof = cal[-1]
NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon", "GOOGL": "Alphabet (Google)", "META": "Meta (Facebook)", "TSLA": "Tesla",
         "AVGO": "Broadcom", "INTC": "Intel", "CSCO": "Cisco", "ORCL": "Oracle", "QCOM": "Qualcomm", "CMCSA": "Comcast", "GILD": "Gilead Sciences",
         "PEP": "PepsiCo", "COST": "Costco", "NFLX": "Netflix", "ADBE": "Adobe", "PYPL": "PayPal", "ASML": "ASML", "WMT": "Walmart", "MU": "Micron",
         "STX": "Seagate", "SPCX": "SpaceX", "AMD": "AMD"}

lineups, prev = [], set()
for qd, g in r["lineups"].items():
    nxt = [d for d in cal if d > qd]
    if not nxt:
        continue
    cur = list(g.symbol)
    lineups.append({"q": qd, "trade": nxt[0], "stocks": [[s, round(float(w) * 100, 1)] for s, w in zip(g.symbol, g.weight)],
                    "added": sorted(set(cur) - prev) if prev else [], "removed": sorted(prev - set(cur)) if prev else []})
    prev = set(cur)

units, trc = r["units"], r["trc"]
val = {s: u * trc[s][-1] for s, u in units.items()}
tot = sum(val.values())
px = load_prices(units)
stocks = sorted([{"t": s, "n": NAMES.get(s, s), "basket": round(v / tot * 100, 2), "px": round(float(px[s].loc[:asof, "close"].iloc[-1]), 2)}
                 for s, v in val.items()], key=lambda x: -x["basket"])
sgov = pd.read_csv(REPO / "data" / "cache" / "alpaca_sip_daily" / "SGOV.csv").set_index("date_str")["close"]
v_now = float(vol.loc[asof])
log = [{"d": str(pd.Timestamp(e["date"]).date()), "from": round(e["held_before"], 3), "to": round(e["target"], 3), "vol": round(e["qqq_vol_pct"], 1), "reg": e["regime"]} for e in r["log"]]
d0 = nav.index[0]
bh0 = bh.loc[d0:]
out = {
    "asOf": asof, "vol": round(v_now * 100, 1), "regime": regime(v_now), "invested": round(min(1.0, 0.20 / v_now), 4), "held": round(float(r["exposure"].iloc[-1]), 4),
    "lastTrade": log[-1]["d"], "lineupQuarter": lineups[-1]["q"], "lineupTrade": lineups[-1]["trade"],
    "stocks": stocks, "sgov": round(float(sgov.loc[:asof].iloc[-1]), 2),
    "volStrip": [[d, round(float(v) * 100, 1)] for d, v in vol.loc[:asof].iloc[-260:].items()],
    "series": {"start": str(d0.date()), "day": [(d - d0).days for d in nav.index], "plan": [round(x / nav.iloc[0], 5) for x in nav.values],
               "qqq": [round(x / bh0.iloc[0], 5) for x in bh0.values]},
    "log": log, "lineups": lineups, "names": NAMES,
    "regShare": {k: round(v * 100) for k, v in pd.Series([regime(x) for x in vol.loc["2011-02-01":asof].dropna()]).value_counts(normalize=True).items()},
}
yr = pd.DataFrame({"plan": nav.resample("YE").last(), "qqq": bh0.resample("YE").last()})
ann = yr.pct_change(); ann.iloc[0] = yr.iloc[0] / pd.Series({"plan": nav.iloc[0], "qqq": bh0.iloc[0]}) - 1
out["years"] = [{"y": int(d.year), "plan": round(float(x.plan) * 100, 1), "qqq": round(float(x.qqq) * 100, 1)} for d, x in ann.iterrows()]


def st(n, a, b):
    n = n.loc[a:b]; n = n / n.iloc[0]; y = (n.index[-1] - n.index[0]).days / 365.25
    return round(((n.iloc[-1]) ** (1 / y) - 1) * 100, 1), round((n / n.cummax() - 1).min() * 100, 1)


out["periods"] = []
for lab, a, b in (("All, Feb 2011 to today", None, None), ("2011 to mid-2016", None, "2016-06-30"), ("Mid-2016 to 2019", "2016-07-01", "2019-12-31"),
                  ("2020 to 2022 (Covid, inflation bear)", "2020-01-01", "2022-12-31"), ("2023 to today", "2023-01-01", None)):
    p1, p2 = st(nav, a, b), st(bh0, a, b)
    out["periods"].append({"p": lab, "plan": p1[0], "planDD": p1[1], "qqq": p2[0], "qqqDD": p2[1]})
(HERE / "big8_data.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print("asOf", asof, out["regime"], "invested", out["invested"], "held", out["held"], "| line-up", out["lineupQuarter"], [(s["t"], s["basket"]) for s in stocks])
print("periods", out["periods"][0], "| years beating QQQ", sum(y["plan"] > y["qqq"] for y in out["years"]), "of", len(out["years"]))
