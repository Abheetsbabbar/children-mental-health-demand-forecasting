# Methodology

## 1. Pipeline

```
NHS time series zip (Apr 2016–Aug 2026, 700 MB CSV) ─┐
29 monthly four-week-wait files (Apr 2024–Aug 2026)  ├─► download.py ─► data/raw/ (git-ignored)
gov.uk bank holidays JSON                            ─┘
                                                         │
                     build.py ── DuckDB ── sql/01_staging.sql  (date macro, typed England extracts)
                                      └── sql/02_marts.sql     (monthly referrals by age; list stock & flows)
                                                         │
                       data/processed/*.csv (committed) + working-day calendar
                                                         │
                 analysis.py ── models.py   (5 forecasters, rolling backtest, reconciliation)
                             └─ simulate.py (calibrate → benchmark check → Monte Carlo scenarios)
                                                         │
                                 reports/figures/*.png + reports/results.json
```

## 2. Data engineering

| Issue | Handling |
|---|---|
| **Five date formats** across the monthly files: `31/08/2026`, `01/04/24`, `01-02-2025`, `2025-03-01`, and US-style `04/30/2016` in the 2016–23 time series | One DuckDB macro, `parse_nhs_date`, tries each format in order with `try_strptime` |
| Values written as both `44183` and `44183.0` | `try_cast(... as double)` |
| Monthly publications sometimes have both *Performance* and *Final* versions, and re-issues (`_v2`) | `pick_four_week_wait_link()` prefers Final, then the highest version (unit-tested) |
| Some four-week-wait files are `.csv` and others `.zip` | Both are handled at extraction |
| Waiting-list **flows** (MRS11 spells started, MRS16 first contacts) are **rolling 3-month** totals; the **stock** (MRS10) is an end-of-month snapshot | Flows are divided by 3 to give a monthly average. The unpublished "exits without a first contact" come from the stock–flow identity below |

**Stock–flow identity.** Over the same 3-month window:

$$ W_t - W_{t-3} = A_{t-2:t} - C_{t-2:t} - X_{t-2:t} \quad\Rightarrow\quad X = A - C - \Delta W $$

where W is referral-spells waiting, A is referral-spells started, C is first contacts and X
is referrals closed or withdrawn before a first contact. `tests/test_data_contracts.py`
asserts that this identity holds exactly in the processed data.

## 3. Forecasting

**Target:** monthly referrals of 0–17 year olds starting in the month (MHS32, sum of
age bands), Apr 2022 – Aug 2026 (53 months). All models run on **log(referrals)**,
because seasonality is multiplicative.

| Model | Specification |
|---|---|
| Seasonal naive | ŷ(t+h) = y(t+h−12). The benchmark every model must beat |
| ETS | Additive error, **damped** additive trend, additive seasonality (period 12), log scale |
| SARIMAX | (p,d,q)(P,D,Q)₁₂ + **working days** as an exogenous regressor. Order chosen by AIC over 48 candidates on the **first training window only**: (0,0,0)(1,1,0)₁₂ |
| Harmonic regression | OLS: intercept + linear trend + 11 month dummies + working days. Prediction intervals from the OLS predictive distribution |
| Ensemble | Geometric mean of ETS, SARIMAX and harmonic regression (mean on the log scale) |

**Evaluation:** rolling-origin, expanding-window cross-validation. The first origin
trains on 30 months; every model is refitted at each of 18 origins and forecasts 6
months ahead (108 forecasts per model).

- **MASE**: MAE ÷ in-sample MAE of the seasonal naive. Below 1 means better than that benchmark.
- **MAPE**, **MAE**.
- **80% PI coverage**: should be close to 80%. ETS's 59% shows over-confident intervals; harmonic regression's 81% is well calibrated.

| Model | MAPE | MASE | 80% PI coverage |
|---|---|---|---|
| Harmonic regression | 3.8% | 0.56 | 81% |
| Ensemble | 4.0% | 0.57 | 77% |
| SARIMAX + working days | 4.4% | 0.65 | 75% |
| ETS (damped, log) | 5.4% | 0.78 | 59% |
| Seasonal naive | 5.4% | 0.83 | – |

**Hierarchical reconciliation.** Total referrals = sum of four age bands. Base ETS
forecasts for all five series are made coherent with the optimal-combination
projection

$$ \tilde y = S (S^\top W^{-1} S)^{-1} S^\top W^{-1} \hat y $$

with W = I (OLS) or diag(seasonal-naive error variances) (WLS), and compared with
bottom-up. In the backtest, bottom-up (5.19% MAPE) and WLS (5.23%) both improved the
total over base ETS (5.39%). This is a modest gain, consistent with the literature
for short series.

## 4. Waiting-list simulator

### State and dynamics
The state is a vector n(a): the number of referral-spells that have waited a days
(a = 0 … 4,000). Each simulated day:

1. **Arrivals:** A ~ Poisson(λ) children join at a = 0.
2. **First contacts:** C ~ Poisson(μ) are removed, allocated across ages in
   proportion to n(a)·exp(−k_serve·a), using iterative "water-filling" so that no age
   bin goes negative.
3. **Exits without contact:** θ_daily × list size, allocated by the same rule with k_exit.
4. **Ageing:** everyone moves to a + 1. Ages above 4,000 days are pooled.

Policies: k_serve > 0 means newest-first, k_serve = 0 means random order. **FIFO**
gives capacity to the longest waiters first. **Ring-fence s** gives a share s of
capacity FIFO and the rest by k_serve.

### Initial state
The published median (MRS10a) and 90th percentile (MRS10b) define a log-normal
distribution of waiting times: σ = ln(p90/median)/1.2816, μ = ln(median). This is
discretised and scaled to MRS10.

### Calibration and benchmark check
- **Inputs:** the observed monthly arrivals, first contacts and exit rate. The exit
  rate is derived from the observed change in the list, so the **list-size** match is
  partly circular and is not evidence of predictive skill.
- **Fitted:** (k_serve, k_exit), by grid search (31 × 16 = 496 points). The objective
  is the mean squared relative error of the simulated median **and** 90th-percentile
  waits over **Sep 2024 – Aug 2025**.
- The loss surface is **flat for k_serve ≥ 0.1/day**: the data say "strongly newest
  first", but can't tell strong from extreme. A **parsimony rule** picks the smallest
  k_serve within 5% of the best loss: **k_serve = 0.15/day (half-life 4.6 days),
  k_exit = 0**.
- **Later year, Sep 2025 – Aug 2026,** using that year's observed flows (MAPE):

  | Measure | Simulator | No change since Aug 2025 |
  |---|---|---|
  | List size (partly circular) | 4.2% | 13.0% |
  | Median wait | 3.4% | 3.5% |
  | 90th-percentile wait | 12.1% | 5.4% |

  The simulator is no better than the benchmark on median wait and worse on the 90th
  percentile. Its structure was also revised after an early look at this year (a separate
  exit priority, then the parsimony rule), so this is **not a clean holdout test**.
- In the calibration year, random order (loss 0.77) and FIFO (loss 1.24) fit far worse
  than the calibrated rule (0.016), so the data do favour a newest-first pattern.

### Scenarios
- **Horizon:** 24 months from Aug 2026, starting from the observed list (279,604;
  median 290; p90 1,036 days).
- **Demand:** the ensemble referral forecast × the observed spell-to-referral ratio
  (0.647). Year 2 repeats year 1.
- **Capacity:** the last 12-month mean of first contacts (52,556 a month) × {1.0,
  1.1, 1.2, 1.3}. θ is the last 12-month mean (5.05% a month).
- **Policy:** current (calibrated), 25% ring-fenced, or strict FIFO.
- 40 Monte Carlo runs per scenario; the median and 5th–95th percentiles are reported.

**Fluid steady state:** dW/dt = λ − μ − θW = 0 ⇒ W* = (λ − μ)/θ ≈ 353,000 at
current capacity. Capacity needed to hold today's list level: λ − θW₀ ≈ 56,300 a
month.

## References
- Hyndman R.J. & Athanasopoulos G. (2021). *Forecasting: Principles and Practice*, 3rd ed. OTexts.
- Hyndman R.J. & Koehler A.B. (2006). Another look at measures of forecast accuracy. *IJF* 22(4).
- Wickramasuriya S.L., Athanasopoulos G. & Hyndman R.J. (2019). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization. *JASA* 114(526).
- Cleveland R.B. et al. (1990). STL: A seasonal-trend decomposition procedure based on loess. *J. Official Statistics* 6(1).
- NHS England. *Mental Health Services Monthly Statistics: four-week-wait measures – metadata.*
