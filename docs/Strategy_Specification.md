# Strategy Specification: Volatility-Target QQQ (v6.0.0)

**Configuration:** [`config/strategy_config.yaml`](../config/strategy_config.yaml), section `volatility_target` (`version: 6.0.0-qqq-voltarget-1.5x`)
**Instruments:** QQQ, QLD (2x daily Nasdaq-100 ETF), SGOV (T-bill ETF; BIL fallback)
**Backtest window:** 1999-04-12 to 2026-11-11, [`scripts/run_voltarget_backtest.py`](../scripts/run_voltarget_backtest.py)
**Live signal:** [`scripts/run_production_voltarget.py`](../scripts/run_production_voltarget.py)

The previous strategy (v5.0.0, point-in-time Top-8 cap-weight + QQQ 200-day rule) is retired; see the README, section 5.

---

## 1. Verdict and weaknesses

12.8% a year with a -54.9% maximum drawdown over 1999–2026, against 10.3% and -83.0% for total-return QQQ. The edge comes mainly from smaller losses in the 2000–2002 and 2008 crashes; in calm bull periods the strategy roughly matches QQQ, and it lags fast recoveries (1999, 2009, 2020). Parameters were chosen after seeing 2002–2026 results; only 1999–2002 was not used in the design. Turnover is about 5.7x a year and taxes are not modelled.

---

## 2. Signal

For each completed daily close `t`:

- `r_k` = QQQ daily close-to-close return (total-return series in the backtest; price returns live, which differ only on dividend days)
- `vol_t` = standard deviation of the last 21 returns `r_{t-20..t}`, times `sqrt(252)`
- target exposure `w_t = min(1.5, 0.20 / vol_t)`

Only data up to and including close `t` is used. The live script drops an incomplete current-day bar if run before 16:05 ET.

---

## 3. Portfolio and trading

| Target exposure | QQQ | QLD | T-bills |
| :--- | :---: | :---: | :---: |
| `w <= 1.0` | `w` | 0 | `1 - w` |
| `w > 1.0` | `2 - w` | `w - 1` | 0 |

Holding `a` in QLD and `1 - a` in QQQ gives exposure `1 + a`, so the maximum 1.5x is 50% QQQ + 50% QLD. No margin is used.

Trades are placed for the next open (`Open_{t+1}`) only when `|w_t - current exposure| >= 0.10` or the target crosses 1.0x. Current exposure counts QLD twice: `(QQQ value + 2 x QLD value) / equity`.

---

## 4. Backtest accounting

- Share-level: holdings drift with prices between rebalances; the overnight move applies to yesterday's holdings, the intraday move to holdings after the open trade.
- Costs: 10 bps on every dollar traded (20 and 30 bps reported), deducted before sizing new positions.
- Cash earns the FRED 3-month T-bill rate lagged one day (`DTB3_{t-1} / 100 / 252`).
- QLD: real adjusted prices from 2006-06-21; before that a synthetic 2x fund `2 x r_QQQ - (T-bill + 0.5%)/252 - 0.95%/252`, which matched real QLD's 2006–2026 annual return within 0.1 point (daily tracking error about 25 bps).
- Benchmarks and baselines: total-return QQQ buy and hold, QQQ 200-day rule, volatility target capped at 1.0x and 1.25x.

---

## 5. Production runbook

| When | Command | Output |
| :--- | :--- | :--- |
| Daily, after 16:05 ET | `python scripts/run_production_voltarget.py --capital <equity> --current-exposure <held>` | `data/outputs/production_voltarget_signal.json`, SQLite table `live_target_portfolio_voltarget` |
| After changing code or data | `python scripts/run_voltarget_backtest.py && python -m pytest tests -v` | `data/outputs/voltarget_*` and test results |

Alternatively pass current holdings with `--qqq-value`, `--qld-value`, `--cash-value`. Without either, the script assumes the previous run's orders were filled.
