# Data dictionary

All files are in `data/processed/`. All figures are for **England** and for **children and young people aged under 18**.

## `referrals_monthly.csv`. Grain: one row per month (Apr 2022 – Aug 2026)

| Column | Definition | Source |
|---|---|---|
| `month_end` | Last day of the reporting month | MHSDS time series |
| `referrals_0_17` | Referrals starting in the month, ages 0–17 (sum of the bands below) | MHS32, `England; Age` |
| `age_0_5`, `age_6_10`, `age_11_15` | Same, by age band | MHS32 |
| `age_16_17` | Ages 16 + 17 (published separately, summed here) | MHS32 |

## `waiting_list_monthly.csv`. Grain: one row per month (Apr 2024 – Aug 2026)

| Column | Definition | Source |
|---|---|---|
| `waiting` | Open CYP referral-spells waiting for a first contact at month end | MRS10 |
| `median_wait_days`, `p90_wait_days` | Median and 90th percentile of days waited so far by those spells | MRS10a, MRS10b |
| `arrivals_per_month` | CYP referral-spells started, rolling 3 months ÷ 3 | MRS11 |
| `first_contacts_per_month` | Referral-spells receiving a first contact, rolling 3 months ÷ 3 | MRS16 |
| `closures_per_month` | Referral-spells closed, rolling 3 months ÷ 3 | MRS12 |
| `other_exits_per_month` | **Derived:** spells leaving the list without a first contact, (MRS11 − MRS16 − ΔMRS10 over 3 months) ÷ 3. Empty for the first 3 months | derived |

A *referral-spell* groups overlapping referrals for the same child. Spells are about
0.65× the number of referrals.

## `calendar_monthly.csv`
| Column | Definition |
|---|---|
| `working_days` | Mon–Fri minus England & Wales bank holidays (gov.uk API) |
| `days_in_month`, `is_august` | Calendar helpers |

## `forecast_referrals_0_17.csv`
One row per future month (Sep 2026 – Aug 2027). There are `<model> | forecast`,
`| lo80` and `| hi80` columns for each model (80% prediction interval).

## `backtest_errors.csv`
One row per (origin, model, horizon). Columns: actual, forecast, absolute error,
APE, scaled error (for MASE) and `covered_80`.

## `simulation_calibration_grid.csv`
Calibration loss for each (k_serve, k_exit) grid point.

## `scenario_results.csv`
One row per (capacity uplift, policy, month). Columns: Monte Carlo median list
size with its 5th/95th percentiles, median wait, and the share waiting over 4 and
52 weeks.

## `reports/results.json`
This file holds every number in the README: data coverage, the SARIMA order,
backtest metrics, the forecast, hierarchy MAPE, simulator calibration and
benchmark check, and scenario inputs and outputs.
