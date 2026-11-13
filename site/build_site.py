"""Builds the Weather Plans site into site/dist/: index.html (front page comparing both plans) + qqq.html + big8.html.
Usage (from the repo root):  python site/prepare_qqq.py && python site/prepare_big8.py && python site/build_site.py"""
import json
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"; TPL = HERE / "templates"
OUT = HERE / "dist"; OUT.mkdir(exist_ok=True)
Q = json.loads((BUILD / "qqq_data.json").read_text(encoding="utf-8"))
E = json.loads((BUILD / "big8_data.json").read_text(encoding="utf-8"))

def ser(S, keys):
    d0 = pd.Timestamp(S["start"]); idx = d0 + pd.to_timedelta(S["day"], unit="D")
    return pd.DataFrame({k: S[k] for k in keys}, index=idx)
q = ser(Q["series"], ["qqq", "s15", "s10"]); e = ser(E["series"], ["plan"])
df = q.join(e, how="inner").dropna(); df = df / df.iloc[0]

def st(x):
    y = (x.index[-1] - x.index[0]).days / 365.25
    return {"cagr": round((x.iloc[-1] ** (1 / y) - 1) * 100, 1), "dd": round((x / x.cummax() - 1).min() * 100, 1), "end": round(10000 * x.iloc[-1])}
common = {k: st(df[k]) for k in df}
yr = df.resample("YE").last(); ann = yr.pct_change(); ann.iloc[0] = yr.iloc[0] - 1
beat = {k: int((ann[k] > ann["qqq"]).sum()) for k in ["s15", "s10", "plan"]}
m = df.resample("ME").last(); m.iloc[-1] = df.iloc[-1]
front = {
    "asOf": E["asOf"], "vol": E["vol"], "regime": E["regime"], "volStrip": E["volStrip"],
    "qqqNow": Q["now"], "big8Inv": E["invested"], "big8Top": [s["t"] for s in E["stocks"]],
    "start": str(df.index[0].date()), "nYears": len(ann), "common": common, "beat": beat,
    "chart": {"d": [str(d.date()) for d in m.index], **{k: [round(v, 4) for v in m[k]] for k in df}},
    "qqqFull": Q["periods"][0], "big8Trades": round(len(E["log"]) / ((df.index[-1] - df.index[0]).days / 365.25)),
    "qqqTrades": {"s15": round(len(Q["log15"]) / 27.6), "s10": round(len(Q["log10"]) / 27.6)},
}
print(json.dumps({k: front[k] for k in ["common", "beat", "start", "big8Trades", "qqqTrades"]}))

HEAD = '<!doctype html>\n<html lang="en" data-theme="dark">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
BACK = '<a href="index.html" class="home-link" aria-label="All plans">&larr; All plans</a>'
BACK_CSS = '<style>.home-link{color:var(--muted);text-decoration:none;font-weight:600;font-size:.9rem;padding:6px 10px;border:1px solid var(--line);border-radius:999px;margin-right:6px}.home-link:hover{color:var(--fg)}</style>\n'
def page(template, data, brand_tag):
    t = (TPL / template).read_text(encoding="utf-8")
    t = t.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    t = t.replace(brand_tag, BACK + brand_tag, 1)
    head_end = t.index("<header>")
    return HEAD + BACK_CSS + t[:head_end] + "</head>\n<body>\n" + t[head_end:] + "\n</body>\n</html>\n"
(OUT / "qqq.html").write_text(page("qqq.html", Q, '<div class="brand">'), encoding="utf-8")
(OUT / "big8.html").write_text(page("big8.html", E, '<div class="brand">'), encoding="utf-8")
fp = (TPL / "front.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(front, separators=(",", ":")).replace("</", "<\\/"))
(OUT / "index.html").write_text(fp, encoding="utf-8")
print("built", [(p.name, round(p.stat().st_size / 1024)) for p in OUT.iterdir()])
