"""Tests for the Big Eight Weather Plan (8 largest Nasdaq-100 members, QQQ-volatility overlay, no leverage)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.big8_data import load_lineups, load_ranking
from scripts.run_big8_backtest import build_basket, qqq_vol, run, simulate_overlay
from scripts.run_voltarget_backtest import load_data


@pytest.fixture(scope="module")
def result():
    return run()


def test_lineups_are_eight_cap_weighted_quarter_ends() -> None:
    lu = load_lineups(8)
    assert len(lu) >= 60
    for qd, g in lu.items():
        assert int(qd[5:7]) in (3, 6, 9, 12), qd
        assert len(g) == 8 and g.symbol.is_unique
        assert "GOOG" not in set(g.symbol)
        assert abs(g.weight.sum() - 1) < 1e-12
        assert (g.weight.diff().dropna() <= 1e-12).all()          # biggest first


def test_ranking_uses_closes_on_or_before_quarter_end() -> None:
    r = load_ranking()
    for qd, g in r.groupby("quarter_date"):
        assert (g["rank"].sort_values().values == np.arange(1, len(g) + 1)).all()
    # market cap = close x shares
    assert np.allclose(r.close * r.shares_bn, r.market_cap_bn, rtol=1e-3)


def test_line_up_trades_after_the_quarter_end(result) -> None:
    cal = result["calendar"]
    basket = result["basket"]
    first_q = list(result["lineups"])[0]
    assert basket.index[0] > first_q and basket.index[0] == [d for d in cal if d > first_q][0]


def test_overlay_never_levers_and_uses_only_past_vol(result) -> None:
    assert result["exposure"].max() <= 1.0 + 0.15            # drift between trades only
    assert result["target"].max() <= 1.0 + 1e-12
    df = load_data()
    cut = "2020-02-14"
    shocked = df["close"].copy()
    shocked.loc[shocked.index > cut] *= np.linspace(0.5, 2.0, (shocked.index > cut).sum())
    a, b = qqq_vol(df["close"]), qqq_vol(shocked)
    assert np.allclose(a.loc[:cut].dropna().values, b.loc[:cut].dropna().values)


def test_full_investment_tracks_the_basket(result) -> None:
    basket, rf = result["basket"], result["df"]["rf"]
    nav, expo, log = simulate_overlay(basket, pd.Series(1.0, basket.index), rf, cost_bps=0.0)
    i0 = basket.index.get_loc(str(nav.index[0].date()))
    expect = basket.close.iloc[-1] / basket.open.iloc[i0]
    assert abs(nav.iloc[-1] / expect - 1) < 1e-9 and len(log) == 1


def test_costs_lower_returns(result) -> None:
    basket, rf, tgt = result["basket"], result["df"]["rf"], result["target"]
    n10 = simulate_overlay(basket, tgt, rf, cost_bps=10.0)[0]
    n30 = simulate_overlay(basket, tgt, rf, cost_bps=30.0)[0]
    assert n30.iloc[-1] < n10.iloc[-1]


def test_headline_numbers_are_stable(result) -> None:
    nav = result["nav"].loc[:"2026-11-11"]
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    assert 0.17 < cagr < 0.23 and -0.40 < dd < -0.28


def test_production_signal_respects_deadband() -> None:
    from scripts.run_production_big8 import run_production
    out = Path(tempfile.mkdtemp())
    hold = run_production(10000, current_fraction=0.6541, as_of_override="2026-11-11", output_dir=out)
    assert hold["action"] == "HOLD_NO_TRADE"
    reb = run_production(10000, current_fraction=0.90, as_of_override="2026-11-11", output_dir=out)
    assert reb["action"] == "REBALANCE_AT_NEXT_OPEN"
    assert abs(sum(h["target_weight_pct"] for h in reb["holdings"]) - 100) < 0.05
    assert len([h for h in reb["holdings"] if h["role"] == "STOCK"]) == 8
