# Empirical Audit, Component Ablation, and Rejected Overlays

This note summarizes why complex overlays were stripped from the repository and documents the ablation and recent-episode behavior of the surviving rule (`PIT Cap-Weight + QQQ 200-Day SMA`).

---

## 1. Summary Verdict & Caveats

Combining quarterly PIT Top-8 cap-weighting with a `QQQ` 200-day moving average cash exit cut maximum drawdown from `-44.15%` (`Passive PIT Top-8`) to `-23.50%` (`PIT Top-8 + QQQ 200d MA`, `10 bps`) while retaining `21.81%` CAGR (`0.98` Sharpe) vs Total-Return `QQQ` (`20.49%` CAGR, `-35.12%` MaxDD, `0.84` Sharpe). Key caveats remain: post-selection bias across `2016–2026`, marginal HAC significance (`t = +1.88`, `p = 0.0606`), price-only constituent CSVs vs Total-Return `QQQ`, and `2016-Q2` constituent start date.

---

## 2. Component Ablation (`2016-06-30` to `2026-11-11`, `10 bps`)

Both components perform distinct roles (`Passive PIT Top-K` supplies concentration beta above `1.0` without drawdown control; `QQQ > 200d SMA` caps drawdowns at ~`-23%` while sacrificing bull upside when applied to the broad index):

| Variant | End `$100k` | CAGR | MaxDD | Sharpe | Beta | HAC Alpha vs QQQ | `t`-stat | 2-Way Turn |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `QQQ` Total Return (Buy & Hold) | \$690,297 | 20.49% | -35.12% | 0.84 | 1.00 | 0.00% | — | 0.10x |
| Simple `QQQ > 200d SMA` (No Stock Selection) | \$481,810 | 16.38% | -23.35% | 0.84 | 0.58 | +3.33% | +1.05 | 4.05x |
| Passive PIT Top-8 (No 200d MA Filter) | \$978,381 | 24.61% | -44.15% | 0.89 | 1.10 | +2.28% | +1.03 | 0.24x |
| Passive PIT Top-16 (No 200d MA Filter) | \$839,292 | 22.78% | -42.00% | 0.86 | 1.08 | +0.94% | +0.61 | 0.35x |
| **Combined: `PIT Top-8 + QQQ 200d MA`** | **\$772,665** | **21.81%** | **-23.50%** | **0.98** | **0.63** | **+7.43%** | **+1.88** | **4.20x** |
| Combined: `PIT Top-16 + QQQ 200d MA` | \$633,798 | 19.50% | -23.13% | 0.91 | 0.62 | +5.51% | +1.51 | 4.28x |

---

## 3. Why Legacy Components Were Removed

Each complex component from V2/V3/V4 failed causal, placebo, or realistic instrument checks:

1. **Multi-Domain Factor Scoring, Hilbert Combiners, and Mid-Tier Selection:** Ranked constituent sleeves showed weak skill out-of-sample; random 4-of-8 constituent picks in placebo simulations matched or outperformed score-weighted selectors.
2. **Tactical `15% TQQQ` Overlay:** Elevated crash fragility and daily compounding drag during volatile choppy regimes without improving net risk-adjusted Sharpe over simple baselines.
3. **Synthetic "Zero-Cost Crash Collar":** Earlier backtests approximated a 1-month collar using a daily return clamp (`-7%`/`+7.5%`), which understated option decay and overstated protection. Replacing the daily clamp with Black-Scholes mark-to-market option valuation (`1.15x` put skew / `0.95x` call skew, fixed strike and expiry) removed the artificial edge.
4. **FINRA RegSHO, CatBoost Ranker, and Intraday SIP Buy/Sell Volume Ratios:** Adding cross-sectional or time-series Alpaca SIP aggressive buy/sell volume ratios and average trade-size features to `PIT Top-8 + QQQ 200d MA` did not beat shuffled-signal placebos (`p = 0.275`; paired daily return difference `t = -1.67` to `+0.78`).

---

## 4. Late-2026 Whipsaw Episode (`2026-10-28` to `2026-11-11`)

From [`data/outputs/production_audit_summary.json`](../data/outputs/production_audit_summary.json), illustrating mechanical next-open exits and re-entries when `QQQ` crosses a rising `SMA200`:

| Date | `QQQ` Close | `QQQ` SMA200 | `QQQ` Ret | Top-8 Ret | Top-16 Ret | Open Position | Close Signal |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| `2026-10-28` | \$725.61 | \$677.55 | +0.60% | +0.17% | +0.16% | `EQUITIES` | `ABOVE_SMA200` |
| `2026-10-29` | \$715.34 | \$678.01 | -1.42% | -2.53% | -2.14% | `EQUITIES` | `ABOVE_SMA200` |
| `2026-10-30` | \$690.49 | \$678.38 | -3.47% | -2.43% | -3.17% | `EQUITIES` | `ABOVE_SMA200` |
| `2026-11-02` | \$681.09 | \$678.68 | -1.36% | -1.32% | -1.33% | `EQUITIES` | `ABOVE_SMA200` |
| `2026-11-03` | \$685.28 | \$679.01 | +0.62% | +1.24% | +1.10% | `EQUITIES` | `ABOVE_SMA200` |
| `2026-11-04` | \$691.95 | \$679.44 | +0.97% | +2.31% | +1.62% | `EQUITIES` | `ABOVE_SMA200` |
| `2026-11-05` | \$647.05 | \$679.61 | -6.49% | -4.88% | -5.68% | `EQUITIES` | `BELOW_SMA200` |
| `2026-11-06` | \$656.10 | \$679.79 | +1.40% | +1.00% | +1.36% | `TBILLS` | `BELOW_SMA200` |
| `2026-11-09` | \$679.35 | \$680.09 | +3.54% | +0.02% | +0.02% | `TBILLS` | `BELOW_SMA200` |
| `2026-11-10` | \$674.76 | \$680.35 | -0.68% | +0.02% | +0.02% | `TBILLS` | `BELOW_SMA200` |
| `2026-11-11` | \$692.35 | \$680.66 | +2.61% | +0.02% | +0.02% | `TBILLS` | `ABOVE_SMA200` |

Note that on `2026-11-06` (when exiting to T-bills at `Open_t` from equity exposure held at `Close_{t-1}`), the `2026-11-06` strategy return captures the overnight-to-open transition gap between `2026-11-05` close and `2026-11-06` open plus `10 bps` exit cost before earning `DTB3` from `2026-11-06` onward.

---

## 5. Why v5 was retired and v6 adopted (November 2026)

**v5 out-of-period test.** Rebuilding point-in-time Nasdaq-100 membership and SEC share counts back to 2010 and re-running the v5 rule (Top-8 cap-weight + QQQ 200-day rule, same simulator) on 2011–2016, a period v5 never covered:

| 2011-01 to 2016-06 | CAGR | Max drawdown |
| :--- | :---: | :---: |
| QQQ total return | 14.1% | -16.1% |
| QQQ 200-day rule | 3.7% | -26.9% |
| Top-8 + 200-day rule (v5) | 3.5% | -27.8% |
| Top-8, no exit rule | 12.7% | -24.1% |

Over 2011–2026 v5 returned 15.0% vs QQQ's 18.2% (Sharpe 0.76 vs 0.83). On its own window its result was sensitive to the moving-average length (100-day 13.9%, 150-day 24.2%, 200-day 21.8% CAGR), and removing Apple or Nvidia from the rosters dropped it below QQQ.

**Other approaches rejected along the way** (all tested on point-in-time data, next-day execution): stock-level predictors within the Nasdaq-100 (momentum, reversal, growth, margins, volume, FINRA off-exchange short volume, insider buying) showed no reliable edge; macro, options-implied, dark-pool, acceleration and chart-pattern drop predictors did not beat simple price signals out of sample; faster moving-average and breakout exits lost more to whipsaws than they saved.

**v6 adopted.** A continuous volatility target was the only approach that beat QQQ over both 2002–2014 and 2014–2026 and through the 1999–2002 crash, and 45 of 48 parameter settings tried beat QQQ on both return and drawdown (README, sections 3–4). A fixed "exit when volatility exceeds 30%" rule scored higher (about 17–18% CAGR) but was rejected: its dot-com result depended on that exact threshold (at 35% the crash loss was -39%, at 40% -64%).
