# Strategy Specification: Point-in-Time Nasdaq-100 Cap-Weight + QQQ 200-Day Moving Average Cash Rule

**Configuration File:** [`config/strategy_config.yaml`](../config/strategy_config.yaml) (`version: 5.0.0-pit-capweight-ma200`)  
**Primary Universe:** Point-in-Time Nasdaq-100 Constituents (`Top-8` Cap-Weighted default; `Top-16` Cap-Weighted supported)  
**Benchmarks:** Total-Return `QQQ` Buy & Hold, Beta-Matched `QQQ` (`63%` `QQQ` / `37%` `DTB3`), Simple `QQQ` 200-Day SMA, `QQQ` 20% Volatility-Target  
**Audited Backtest Window:** `2016-06-30` to `2026-11-11` (`2,607` trading sessions, `42` quarters)

---

## 1. Verdict, Weaknesses, and Caveats

This specification documents a two-part rule: quarterly point-in-time market-cap weighting of the largest Nasdaq-100 constituents paired with a binary `QQQ` 200-day simple moving average (`SMA200`) cash filter. All multi-factor scoring overlays, `15% TQQQ` leverage sleeves, synthetic option collars, and intraday buy/sell flow filters from prior iterations have been removed.

Four weaknesses apply directly to deployment:

1. **Post-selection sample caveat:** `2016-06-30` to `2026-11-11` was inspected while comparing candidate architectures. Accordingly, the backtest is not untouched out-of-sample evidence. Resistance to overfitting rests on structural simplicity: zero fitted numerical thresholds.
2. **Whipsaw risk around a flat moving average:** Binary transitions around `SMA200` generate whipsaw losses during sharp range-bound reversals (for example, `2026-11-05` to `2026-11-12`) and lag V-shaped recoveries. On index-only `QQQ` over `1999–2026`, `QQQ > 200d SMA` (`7.85%` CAGR) underperformed `QQQ 20% Volatility-Target` (`10.51%` CAGR).
3. **Sample coverage (`2016-Q2` start) and unadjusted dividends:** SEC EDGAR XBRL share-count and constituent SIP daily bars start in `2016-Q2`, leaving the `2000–2002` dot-com bear market untested for Top-8/Top-16 constituent picking. Constituent equity bars (`data/cache/alpaca_sip_daily/*.csv`) omit cash dividends (`~0.5%/yr`), whereas the `QQQ` benchmark includes reinvested dividends.
4. **Statistical significance:** Full-period Newey-West (`maxlags=5`) HAC alpha vs Total-Return `QQQ` is `+7.43%` (`t = +1.88`, `p = 0.0606`) for `Top-8` and `+5.51%` (`t = +1.51`, `p = 0.1322`) for `Top-16`. Neither achieves `p < 0.05` over the 10.3-year span.

---

## 2. Point-in-Time Universe & Cap-Weight Construction

Implemented in [`scripts/rebuild_pit_rosters.py`](../scripts/rebuild_pit_rosters.py):

1. **Multi-Interval Membership Filtering:** Each ticker `i` is eligible on calendar quarter-end `Q_E` only if [`config/nasdaq100_pit_membership.json`](../config/nasdaq100_pit_membership.json) contains an interval `[inclusion_date, removal_date]` covering `Q_E`.
2. **Dual-Class Deduplication:** `GOOG` is excluded so Alphabet enters once via `GOOGL`.
3. **Price Completeness & Reuse Guard:** Candidate series must contain at least `60` daily bars on or before `Q_E`, have an active bar within `7` calendar days of `Q_E`, and have `60` bars spanning at most `105` calendar days (rejecting delisted tickers and sparse symbol-reuse artifacts). Across the `42` quarters (`2016-Q2` to `2026-Q3`), deduplicated true PIT membership ranges from `103` to `107`, with `99–101` (`102` in `2016-Q2/Q3`) having complete price history.
4. **Causal SEC EDGAR Share Counts:** For each eligible ticker, `SecEdgarClient._extract_merged_quarterly_series()` queries SEC XBRL shares tags strictly filtered by `filed <= Q_E`, taking the earliest filing per period end to avoid restatement look-ahead, and multiplies by `get_split_factor(sym, share_period_date)`.
5. **Market-Cap Weighting & Intra-Quarter Drift:** Sort descending by `MCAP(Q_E) = Close(Q_E) * SplitAdjShares(Q_E)` and normalize the top `K` (`K = 8` or `K = 16`):
   $$w_i(Q_E) = \frac{\text{MCAP}_i(Q_E)}{\sum_{j=1}^{K} \text{MCAP}_j(Q_E)}$$
   Reconstitution occurs at the first official open strictly after `Q_E`. Between quarter-ends and regime flips, exact share quantities stay constant so weights drift naturally with market prices.

---

## 3. Trend Signal, Execution Timing, and Accounting

Implemented in [`scripts/run_backtest.py`](../scripts/run_backtest.py) and [`scripts/run_production_daily.py`](../scripts/run_production_daily.py):

1. **Pre-Warmed 200-Day SMA:** `SMA200` is computed over the stitched `1999–2026` Total-Return `QQQ` series (`data/cache/qqq_total_return_1999_2026.csv`) before slicing to `2016-06-30`, preventing initial warmup gaps or midday snapshot distortion.
2. **No Same-Bar Look-Ahead:** Daily target regime for trading day `t` depends strictly on completed close `t - 1` (`QQQ Close_{t-1} > QQQ SMA200_{t-1}`). Any position or weight change executes at official open `Open_t` (`09:30 ET`).
3. **Completed-Close Live Guard (`16:05 ET`):** In live execution (`run_production_daily.py`), if run before `16:05 ET`, today's partial daily bar from Alpaca SIP is removed from official target generation and logged separately under `intraday_preview`.
4. **Treasury Bill Yield & Cash Constraints:**
   - **Backtest:** Idle/defensive cash compounds daily at 1-day-lagged `FRED DTB3` (`DTB3_{t-1} / 100 / 252`), starting from `data/cache/fred_macro/DTB3.csv`.
   - **Live Execution:** Defensive target allocates `100%` to `SGOV` (`BIL` fallback).
   - **Leverage / Options / Margin:** Strictly prohibited (`cash >= 0`, `allow_tqqq = false`, `allow_option_collars = false`, `allow_margin = false`).
5. **Transaction Cost Stress Testing:** Every traded dollar (`buys + sells`) incurs `10 bps` base (`1x`), `20 bps` (`2x`), and `30 bps` (`3x`) slippage/commission in `daily_nav_ledger.csv`.

---

## 4. Production Runbook

| Cadence | Action | Command | Output |
| :--- | :--- | :--- | :--- |
| Daily (after `16:05 ET`) | Generate next-open allocation tickets | `python scripts/run_production_daily.py --capital 100000 --top-k 8` | `data/outputs/production_daily_signals.json` & SQLite `live_target_portfolio` |
| Quarterly (after quarter-end close) | Refresh PIT Nasdaq-100 cap-weight rosters | `python scripts/rebuild_pit_rosters.py` | `data/point_in_time_rosters.csv` & `config/universe_qqq.json` |
| Audit / Verification | Recompute full ledgers & run test suite | `python scripts/run_backtest.py && python -m pytest tests/test_production_pit_ma200.py -v` | `data/outputs/daily_nav_ledger.csv`, `quarterly_performance_ledger.csv`, `production_audit_summary.json` |
| Monitoring UI | Inspect curves, drawdowns, and tickets | `python -m streamlit run app.py` | Interactive local dashboard |
