"""Tests for the volatility-target QQQ strategy (v6.0.0)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_voltarget_backtest import load_data, simulate, target_exposure
from scripts.run_production_voltarget import run_production, split_exposure, target_exposure_from_closes


def test_target_uses_only_past_data_and_respects_cap() -> None:
    df = load_data()
    t = target_exposure(df["close"], 0.20, 1.5, 21)
    assert t.max() <= 1.5 + 1e-12 and t.min() > 0
    # changing prices after a date must not change any target up to that date
    cut = "2020-02-14"
    shocked = df["close"].copy()
    shocked.loc[shocked.index > cut] *= np.linspace(0.5, 2.0, (shocked.index > cut).sum())
    t2 = target_exposure(shocked, 0.20, 1.5, 21)
    assert np.allclose(t.loc[:cut].values, t2.loc[:cut].values)


def test_split_exposure_weights() -> None:
    for w in (0.3, 1.0, 1.25, 1.5):
        s = split_exposure(w)
        assert abs(sum(s.values()) - 1.0) < 1e-12
        assert abs(s["QQQ"] + 2.0 * s["QLD"] - w) < 1e-12
        assert min(s.values()) >= -1e-12


def test_backtest_trades_next_open_and_cost_monotonic() -> None:
    df = load_data()
    t = target_exposure(df["close"], 0.20, 1.5, 21)
    nav10, expo, turn = simulate(df, t, cost_bps=10.0)
    nav30, _, _ = simulate(df, t, cost_bps=30.0)
    assert nav10.iloc[-1] > nav30.iloc[-1]
    assert expo.max() <= 1.5 * 1.25          # drift between rebalances stays bounded
    assert turn > 0
    # a constant 1.0x target must reproduce QQQ total return from the first open to the last close (minus one entry cost)
    nav_bh, _, _ = simulate(df, pd.Series(1.0, df.index))
    i0 = int(np.searchsorted(df.index.values, "1999-04-12"))
    qqq = df["close"].iloc[-1] / df["open"].iloc[i0]
    assert abs(nav_bh.iloc[-1] / 100000.0 / qqq - (1 - 0.001)) < 1e-6


def test_saved_outputs_match_current_code() -> None:
    summ = json.loads((PROJECT_ROOT / "data" / "outputs" / "voltarget_summary.json").read_text(encoding="utf-8"))
    df = load_data()
    nav, _, _ = simulate(df, target_exposure(df["close"], 0.20, 1.5, 21))
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100
    assert abs(cagr - summ["full_period"]["STRATEGY: Vol target 20%, max 1.5x"]["cagr_pct"]) < 1e-6


def test_production_replay(tmp_path: Path) -> None:
    closes = pd.Series(np.linspace(100, 110, 40))
    sig = target_exposure_from_closes(closes, 0.20, 1.5, 21)
    assert sig["target_exposure"] <= 1.5
    p = run_production(capital=100000.0, current_exposure=1.0, output_dir=tmp_path, db_path=tmp_path / "t.db", as_of_override="2026-11-05")
    assert p["bar_as_of_date"] == "2026-11-05"
    assert abs(sum(h["target_weight_pct"] for h in p["holdings"]) - 100.0) < 0.05
    assert p["exposure_after_orders"] <= 1.5
    # a held exposure within the deadband of the target must not trade
    held = p["target_exposure"] + 0.05 if p["target_exposure"] + 0.05 <= 1.0 else p["target_exposure"] - 0.05
    p2 = run_production(capital=100000.0, current_exposure=held, output_dir=tmp_path / "b", db_path=tmp_path / "b.db", as_of_override="2026-11-05")
    assert p2["action"] == "HOLD_NO_TRADE"
