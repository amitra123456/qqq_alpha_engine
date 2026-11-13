"""
Verification Suite for Production Strategy:
Quarterly Point-in-Time Cap-Weight (Top-8 / Top-16) + QQQ 200-Day Moving Average Cash Rule
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rebuild_pit_rosters import is_constituent_on_date
from scripts.run_backtest import load_total_return_qqq_and_sip
from scripts.run_production_daily import run_production


def test_pit_rosters_integrity() -> None:
    roster_path = PROJECT_ROOT / "data" / "point_in_time_rosters.csv"
    df = pd.read_csv(roster_path)
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    assert len(df) == 672, f"Expected 672 rows (42 quarters x 16), got {len(df)}"
    assert df["quarter"].nunique() == 42
    assert "GOOG" not in set(df["ticker"]), "Dual-class duplicate GOOG leaked into rosters"

    for _, row in df.iterrows():
        assert is_constituent_on_date(str(row["ticker"]), str(row["date_str"])), (
            f"PIT interval check failed for {row['ticker']} on {row['date_str']}"
        )

    for q_lbl, grp in df.groupby("quarter"):
        top8 = grp[grp["rank"] <= 8]
        assert len(top8) == 8, f"{q_lbl} missing Top-8 rows"
        assert abs(top8["weight_top8_pct"].sum() - 100.0) <= 0.05, f"{q_lbl} weight_top8_pct sum != 100"
        assert abs(grp["weight_pct"].sum() - 100.0) <= 0.05, f"{q_lbl} weight_pct sum != 100"
        assert int(grp["pit_members_with_data"].iloc[0]) >= 85, f"{q_lbl} low constituent data coverage"


def test_backtest_ledgers_sma200_and_dtb3_rate() -> None:
    q_tr, _ = load_total_return_qqq_and_sip()
    assert q_tr.loc["2016-06-30", "sma200"] > 0

    daily_path = PROJECT_ROOT / "data" / "outputs" / "daily_nav_ledger.csv"
    q_path = PROJECT_ROOT / "data" / "outputs" / "quarterly_performance_ledger.csv"
    df = pd.read_csv(daily_path)
    q_df = pd.read_csv(q_path)

    assert df["QQQ_SMA200"].isna().sum() == 0, "QQQ_SMA200 contains NaNs (warmup missing)"
    assert df["Date"].iloc[0] == "2016-06-30"
    assert len(q_df) == 42

    # Confirm 2023-2025 annualized risk-free rate matches FRED DTB3 (> 4.0%)
    rf_23_25 = df[(df["Date"] >= "2023-01-01") & (df["Date"] <= "2025-12-31")]["RF_Daily_Rate"].mean() * 252.0
    assert rf_23_25 > 0.04, f"Expected DTB3 annualized yield > 4.0% in 2023-2025, got {rf_23_25:.4f}"

    # Verify cost monotonicity (10 bps > 20 bps > 30 bps)
    assert df["PIT_Top8_200DMA_10bps_NAV_100k"].iloc[-1] > df["PIT_Top8_200DMA_20bps_NAV_100k"].iloc[-1]
    assert df["PIT_Top8_200DMA_20bps_NAV_100k"].iloc[-1] > df["PIT_Top8_200DMA_30bps_NAV_100k"].iloc[-1]
    assert df["PIT_Top16_200DMA_10bps_NAV_100k"].iloc[-1] > df["PIT_Top16_200DMA_20bps_NAV_100k"].iloc[-1]


def test_production_signal_replay_causality(tmp_path: Path) -> None:
    # 2026-11-05 close dropped below SMA200 -> exit equities to T-bills at next open
    p_exit = run_production(
        capital=100000.0,
        top_k=8,
        output_dir=tmp_path / "exit",
        db_path=tmp_path / "exit.db",
        as_of_override="2026-11-05",
    )
    assert p_exit["bar_as_of_date"] == "2026-11-05"
    assert p_exit["qqq_macro"]["above_sma200"] is False
    assert p_exit["transition_action"] == "EXIT_EQUITIES_TO_TBILLS_AT_NEXT_OPEN"
    assert len(p_exit["holdings"]) == 1
    assert p_exit["holdings"][0]["symbol"] in {"SGOV", "BIL"}
    assert abs(p_exit["holdings"][0]["target_weight_pct"] - 100.0) < 0.01

    # 2026-11-11 close crossed back above SMA200 -> enter Top-8 equities at next open
    enter8_db = tmp_path / "enter8.db"
    p_enter_8 = run_production(
        capital=100000.0,
        top_k=8,
        output_dir=tmp_path / "enter8",
        db_path=enter8_db,
        as_of_override="2026-11-11",
    )
    assert p_enter_8["bar_as_of_date"] == "2026-11-11"
    assert p_enter_8["qqq_macro"]["above_sma200"] is True
    assert p_enter_8["transition_action"] == "ENTER_EQUITIES_AT_NEXT_OPEN"
    assert len(p_enter_8["holdings"]) == 8
    syms8 = [h["symbol"] for h in p_enter_8["holdings"]]
    assert "GOOG" not in syms8
    assert abs(sum(h["target_weight_pct"] for h in p_enter_8["holdings"]) - 100.0) <= 0.1

    conn = sqlite3.connect(enter8_db)
    row_cnt = conn.execute("SELECT COUNT(*) FROM live_target_portfolio").fetchone()[0]
    conn.close()
    assert row_cnt == 8

    # Top-16 supported variant on 2026-11-11
    p_enter_16 = run_production(
        capital=100000.0,
        top_k=16,
        output_dir=tmp_path / "enter16",
        db_path=tmp_path / "enter16.db",
        as_of_override="2026-11-11",
    )
    assert len(p_enter_16["holdings"]) == 16
    assert abs(sum(h["target_weight_pct"] for h in p_enter_16["holdings"]) - 100.0) <= 0.15
