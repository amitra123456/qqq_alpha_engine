"""
Nasdaq-100 Point-in-Time Cap-Weight + QQQ 200-Day Moving Average Terminal
==========================================================================
Shared Streamlit terminal renderer used by standalone `app.py` and the parent
workspace terminal wrapper. Computes all summary statistics, HAC CAPM regressions,
drawdowns, and order-ticket tables dynamically from audited CSV/JSON outputs in
`data/outputs/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import streamlit as st


def _calc_window_metrics(
    nav_series: pd.Series,
    bench_series: pd.Series,
    rf_daily: np.ndarray,
    dates_series: pd.Series,
) -> Dict[str, float]:
    """Dynamically computes CAGR, MaxDD, Excess Sharpe, HAC Alpha/t-stat, and Beta over a window."""
    s = nav_series / nav_series.iloc[0] * 100000.0
    b = bench_series / bench_series.iloc[0] * 100000.0

    dt_idx = pd.to_datetime(dates_series)
    yrs = max((dt_idx.iloc[-1] - dt_idx.iloc[0]).days / 365.25, 1e-6)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1.0 / yrs) - 1.0
    b_cagr = (b.iloc[-1] / b.iloc[0]) ** (1.0 / yrs) - 1.0

    dd_curve = (s - s.cummax()) / s.cummax()
    b_dd_curve = (b - b.cummax()) / b.cummax()
    max_dd = float(dd_curve.min())
    b_max_dd = float(b_dd_curve.min())

    r_s = s.pct_change().fillna(0.0).values[1:]
    r_b = b.pct_change().fillna(0.0).values[1:]
    rf_sub = rf_daily[1:]
    ex_s = r_s - rf_sub
    ex_b = r_b - rf_sub

    sharpe = float((np.mean(ex_s) / np.std(ex_s, ddof=1)) * np.sqrt(252.0)) if np.std(ex_s, ddof=1) > 0 else 0.0
    b_sharpe = float((np.mean(ex_b) / np.std(ex_b, ddof=1)) * np.sqrt(252.0)) if np.std(ex_b, ddof=1) > 0 else 0.0

    mod = sm.OLS(ex_s, sm.add_constant(ex_b)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    alpha_ann = float(mod.params[0]) * 252.0
    beta = float(mod.params[1])
    t_stat = float(mod.tvalues[0])
    p_val = float(mod.pvalues[0])

    return {
        "ending_nav": float(s.iloc[-1]),
        "cagr": float(cagr),
        "bench_cagr": float(b_cagr),
        "max_dd": max_dd,
        "bench_max_dd": b_max_dd,
        "sharpe": sharpe,
        "bench_sharpe": b_sharpe,
        "alpha_ann": alpha_ann,
        "beta": beta,
        "t_stat": t_stat,
        "p_val": p_val,
    }


def render_pit_ma200_terminal(
    data_outputs_dir: Optional[Path] = None,
    key_prefix: str = "pit_term",
) -> None:
    """Renders the interactive audit ledger, live signal monitor, and drawdown analysis."""
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = data_outputs_dir or (repo_root / "data" / "outputs")
    daily_path = out_dir / "daily_nav_ledger.csv"
    quarters_path = out_dir / "quarterly_performance_ledger.csv"
    summary_path = out_dir / "production_audit_summary.json"
    signal_path = out_dir / "production_daily_signals.json"

    if not daily_path.exists() or not quarters_path.exists():
        st.warning(f"Audit ledgers not found in {out_dir}. Run `python scripts/run_backtest.py` first.")
        return

    daily_df = pd.read_csv(daily_path)
    q_df = pd.read_csv(quarters_path)
    summary_data = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    signal_data = json.loads(signal_path.read_text(encoding="utf-8")) if signal_path.exists() else {}

    start_dt = str(daily_df["Date"].iloc[0])
    end_dt = str(daily_df["Date"].iloc[-1])

    st.markdown(
        f"""
        <div style="background: #111827; border: 1px solid #1F2937; border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h2 style="font-size: 21px; font-weight: 700; margin: 0; color: #F8FAFC;">
                        Nasdaq-100 Point-in-Time Cap-Weight + QQQ 200-Day Moving Average Terminal
                    </h2>
                    <p style="color: #94A3B8; font-size: 13px; margin: 4px 0 0 0;">
                        Quarterly PIT Cap-Weighted Equity Basket (Top-8 Default / Top-16 Broad) | Close<sub>t-1</sub> Signal -&gt; Open<sub>t</sub> Execution | 0% TQQQ | 0% Collars | Lagged FRED DTB3 Cash
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 12px; color: #94A3B8;">Sample: {start_dt} to {end_dt} ({len(daily_df)} sessions)</div>
                    <div style="font-size: 13px; font-weight: 600; color: #10B981;">Initial Capital: $100,000.00</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # First-Paragraph Skeptical Limitations & Weaknesses Disclosure
    st.warning(
        "**Caveats & Known Limitations (Read First):** "
        "(1) **Post-selection sample:** Strategy selection compared historical rules on 2016–2026, so 2016–2026 is not untouched out-of-sample "
        "(justification is structural simplicity: zero fitted thresholds — cap weights + standard 200-day SMA + 3-month T-bills). "
        "(2) **Flat-MA whipsaw drag:** Binary 200-day MA exits suffer whipsaws when QQQ oscillates near its 200-day average (e.g., Nov 5–12, 2026); "
        "over 1999–2026 index-only data, QQQ 200d MA (7.85% CAGR) trailed 20% Volatility-Target (10.51% CAGR) on V-bottom re-entry lag. "
        "(3) **Constituent history & dividends:** Constituent PIT data begins 2016-Q2 (untested in 2000–2002); cached constituent bars are price-only "
        "(omitting ~0.5%/yr dividends), evaluated against a Total-Return QQQ benchmark and 1-day-lagged FRED DTB3 cash yield."
    )

    c_var, c_cost, c_per, c_bmk, c_ref = st.columns([1.4, 1.2, 1.3, 1.4, 0.7])
    with c_var:
        var_sel = st.radio(
            "Strategy Variant",
            ["PIT Top-8 + QQQ 200d MA (Default)", "PIT Top-16 + QQQ 200d MA"],
            index=0,
            key=f"{key_prefix}_var",
        ) or "PIT Top-8 + QQQ 200d MA (Default)"
    with c_cost:
        cost_sel = st.radio(
            "Cost Assumption",
            ["1x Base (10 bps)", "2x Stress (20 bps)", "3x Stress (30 bps)"],
            index=0,
            key=f"{key_prefix}_cost",
        ) or "1x Base (10 bps)"
    with c_per:
        period_sel = st.radio(
            "Sample Window",
            ["Full Sample (2016-2026)", "First Half (2016-2021)", "Second Half (2022-2026)"],
            index=0,
            key=f"{key_prefix}_period",
        ) or "Full Sample (2016-2026)"
    with c_bmk:
        bench_sel = st.radio(
            "Benchmark",
            ["QQQ Buy & Hold (Total Return)", "Beta-Matched QQQ + T-Bills"],
            index=0,
            key=f"{key_prefix}_bench",
        ) or "QQQ Buy & Hold (Total Return)"
    with c_ref:
        st.write("")
        st.write("")
        if st.button("Refresh", key=f"{key_prefix}_refresh", width="stretch"):
            st.rerun()

    is_top8 = "Top-8" in var_sel
    bps = 10 if "10 bps" in cost_sel else (20 if "20 bps" in cost_sel else 30)
    strat_col = f"PIT_Top{8 if is_top8 else 16}_200DMA_{bps}bps_NAV_100k"
    passive_col = f"Passive_PIT_Top{8 if is_top8 else 16}_NAV_100k"
    q_beat_col = "Top8_Beat_QQQ" if is_top8 else "Top16_Beat_QQQ"
    bmk_col = "QQQ_TotalReturn_NAV_100k" if "Total Return" in bench_sel else "QQQ_BetaMatched_NAV_100k"

    if "First Half" in period_sel:
        sub_daily = daily_df[daily_df["Date"] <= "2021-12-31"].copy()
        sub_q = q_df[q_df["Subperiod"] == "FIRST_HALF_2016_2021"].copy()
        p_label = "First Half (2016–2021)"
        sp_key = "FIRST_HALF_2016_2021"
    elif "Second Half" in period_sel:
        sub_daily = daily_df[daily_df["Date"] >= "2022-01-03"].copy()
        sub_q = q_df[q_df["Subperiod"] == "SECOND_HALF_2022_2026"].copy()
        p_label = "Second Half (2022–2026)"
        sp_key = "SECOND_HALF_2022_2026"
    else:
        sub_daily = daily_df.copy()
        sub_q = q_df.copy()
        p_label = f"Full Sample ({start_dt} to {end_dt})"
        sp_key = "FULL_2016_2026"

    rf_arr = sub_daily["RF_Daily_Rate"].values
    metrics = _calc_window_metrics(sub_daily[strat_col], sub_daily[bmk_col], rf_arr, sub_daily["Date"])
    qqq_metrics = _calc_window_metrics(sub_daily["QQQ_TotalReturn_NAV_100k"], sub_daily["QQQ_TotalReturn_NAV_100k"], rf_arr, sub_daily["Date"])

    wins_q = int(sub_q[q_beat_col].sum()) if q_beat_col in sub_q.columns else 0
    tot_q = max(len(sub_q), 1)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric(
            f"Ending $100k ({bps} bps)",
            f"${metrics['ending_nav']:,.0f}",
            f"{(metrics['ending_nav']/100000.0 - 1.0)*100:+.1f}% total",
        )
    with k2:
        st.metric(
            "Net CAGR",
            f"{metrics['cagr']*100:.2f}%",
            f"{(metrics['cagr'] - metrics['bench_cagr'])*100:+.2f}% vs Bmk",
        )
    with k3:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_dd']*100:.2f}%",
            f"{(metrics['max_dd'] - qqq_metrics['max_dd'])*100:+.2f}% vs QQQ",
            delta_color="normal",
        )
    with k4:
        st.metric(
            "Excess Sharpe (DTB3)",
            f"{metrics['sharpe']:.2f}",
            f"Bmk: {metrics['bench_sharpe']:.2f}",
        )
    with k5:
        st.metric(
            "HAC Alpha (Ann)",
            f"{metrics['alpha_ann']*100:+.2f}%",
            f"Beta {metrics['beta']:.2f} | t = {metrics['t_stat']:+.2f} (p={metrics['p_val']:.3f})",
        )
    with k6:
        st.metric(
            "Quarter Win Rate",
            f"{wins_q}/{tot_q} ({wins_q/tot_q*100:.1f}%)",
            "vs QQQ Total Return",
        )

    s_strat = sub_daily[strat_col] / sub_daily[strat_col].iloc[0] * 100000.0
    s_pass = sub_daily[passive_col] / sub_daily[passive_col].iloc[0] * 100000.0
    s_qma = sub_daily["QQQ_Simple_200DMA_NAV_100k"] / sub_daily["QQQ_Simple_200DMA_NAV_100k"].iloc[0] * 100000.0
    s_qvt = sub_daily["QQQ_VolTarget20_NAV_100k"] / sub_daily["QQQ_VolTarget20_NAV_100k"].iloc[0] * 100000.0
    s_qqq = sub_daily["QQQ_TotalReturn_NAV_100k"] / sub_daily["QQQ_TotalReturn_NAV_100k"].iloc[0] * 100000.0

    dd_strat = ((s_strat - s_strat.cummax()) / s_strat.cummax()) * 100.0
    dd_qqq = ((s_qqq - s_qqq.cummax()) / s_qqq.cummax()) * 100.0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.70, 0.30],
        subplot_titles=(f"Cumulative $100k Growth — {p_label}", "Drawdown Profile (%)"),
    )
    fig.add_trace(go.Scatter(x=sub_daily["Date"], y=s_qqq, name="QQQ Buy & Hold (Total Return)", line=dict(color="#94A3B8", width=1.6, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub_daily["Date"], y=s_qma, name="Baseline: Simple QQQ 200d MA", line=dict(color="#38BDF8", width=1.6, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub_daily["Date"], y=s_qvt, name="Baseline: QQQ 20% Vol-Target", line=dict(color="#A78BFA", width=1.6, dash="dashdot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub_daily["Date"], y=s_pass, name=f"Ablation: Passive PIT Top-{8 if is_top8 else 16} (No MA)", line=dict(color="#F59E0B", width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub_daily["Date"], y=s_strat, name=f"PIT Top-{8 if is_top8 else 16} + 200d MA ({bps} bps)", line=dict(color="#10B981", width=2.8)), row=1, col=1)

    fig.add_trace(go.Scatter(x=sub_daily["Date"], y=dd_qqq, name="QQQ Drawdown", line=dict(color="#94A3B8", width=1.2, dash="dot"), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=sub_daily["Date"], y=dd_strat, name="Strategy Drawdown", line=dict(color="#10B981", width=1.8), fill="tozeroy", fillcolor="rgba(16,185,129,0.12)", showlegend=False), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0E14",
        plot_bgcolor="#111827",
        height=540,
        margin=dict(l=40, r=20, t=40, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="NAV ($)", tickprefix="$", tickformat=",", gridcolor="#1F2937", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)", ticksuffix="%", gridcolor="#1F2937", row=2, col=1)
    fig.update_xaxes(gridcolor="#1F2937")
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_equity_dd")

    st.subheader("Live Daily Production Signal & Next-Open Target Allocation")
    if signal_data:
        qm = signal_data.get("qqq_macro", {})
        ip = signal_data.get("intraday_preview")
        if ip:
            st.info(
                f"Intraday Session Notice ({ip.get('et_timestamp', '')}): Live QQQ print is \\${ip.get('qqq_last_price', 0):.2f} vs "
                f"provisional 200d SMA \\${ip.get('provisional_sma200', 0):.2f} (Above={ip.get('provisional_above_sma200')}). "
                "Official target uses the last completed daily close below."
            )

        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            st.metric("Completed Close Date", str(signal_data.get("bar_as_of_date", "N/A")), str(signal_data.get("data_source_mode", "")))
        with s2:
            st.metric("QQQ Close vs SMA200", f"${qm.get('spot', 0):.2f} / ${qm.get('sma200', 0):.2f}", f"{qm.get('dist_sma200_pct', 0):+.2f}%")
        with s3:
            st.metric("Active Quarter Roster", str(signal_data.get("active_quarter", "N/A")), f"QE: {signal_data.get('active_quarter_end_date', '')}")
        with s4:
            st.metric("Regime State", str(signal_data.get("regime_state", "N/A")), f"Prev: Above={qm.get('prev_above_sma200')}")
        with s5:
            st.metric("Next-Open Action", str(signal_data.get("transition_action", "N/A")), str(signal_data.get("execution_order_type", "")))

        disp_holdings = signal_data.get("holdings", [])
        if qm.get("above_sma200") and ((is_top8 and signal_data.get("top_k") != 8) or ((not is_top8) and signal_data.get("top_k") != 16)):
            disp_holdings = signal_data.get("companion_variant_holdings", disp_holdings)

        if disp_holdings:
            df_hold = pd.DataFrame(disp_holdings)[
                ["symbol", "name", "sector", "qe_base_weight_pct", "target_weight_pct", "target_dollars", "target_shares", "spot_price", "status"]
            ]
            st.dataframe(df_hold, width="stretch", hide_index=True)
    else:
        st.info("Run `python scripts/run_production_daily.py` to generate live order tickets.")

    c_comp, c_log = st.columns([1.25, 1.0])
    with c_comp:
        st.subheader(f"Baselines, Ablation & Cost Sensitivity ({p_label})")
        rows_sp = summary_data.get("subperiod_stats", {}).get(sp_key, [])
        if rows_sp:
            df_sp = pd.DataFrame(rows_sp)[
                ["strategy", "ending_nav", "cagr_pct", "max_dd_pct", "sharpe", "beta_qqq", "alpha_qqq_pct", "t_stat_qqq", "turnover_x", "best10_removed_cagr_pct"]
            ]
            df_sp.columns = ["Strategy", "End $100k", "CAGR %", "MaxDD %", "Sharpe", "Beta", "Alpha %", "t-stat", "Turn x/yr", "Ex-Best10d %"]
            st.dataframe(df_sp, width="stretch", height=360, hide_index=True)
    with c_log:
        st.subheader("Recent Daily Signal & Whipsaw Log (Late 2026)")
        rec_rows = summary_data.get("recent_daily_log", [])
        if rec_rows:
            df_rec = pd.DataFrame(rec_rows)[
                ["date", "qqq_close", "qqq_sma200", "qqq_ret_pct", "top8_ret_pct", "top16_ret_pct", "held_at_open", "close_signal"]
            ]
            df_rec.columns = ["Date", "QQQ", "SMA200", "QQQ %", "Top8 %", "Top16 %", "Open Pos", "Close Signal"]
            st.dataframe(df_rec, width="stretch", height=360, hide_index=True)

    st.subheader("42-Quarter Point-in-Time Audit Ledger")
    q_cols = [
        "Quarter", "Start_Date", "End_Date", "Subperiod",
        "QQQ_Ret_Pct", "QQQ_200DMA_Ret_Pct", "Passive_Top8_Ret_Pct",
        "PIT_Top8_200DMA_Ret_Pct", "PIT_Top16_200DMA_Ret_Pct",
        "Top8_Edge_vs_QQQ_Pct", "Top8_Beat_QQQ",
        "PIT_True_Members", "PIT_Members_With_Data",
    ]
    avail_q_cols = [c for c in q_cols if c in sub_q.columns]
    st.dataframe(sub_q[avail_q_cols], width="stretch", height=320, hide_index=True)
