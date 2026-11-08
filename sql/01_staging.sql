-- =====================================================================
-- STAGING: England-level monthly series from two NHS England products.
-- Placeholders in curly braces are filled in by build.py (str.format).
-- =====================================================================

-- Robust date parser for the five date formats found across 29 monthly files:
--   31/08/2026, 01/04/24 (2-digit year), 01-02-2025, 2025-03-01 and US-style 04/30/2016
create or replace macro parse_nhs_date(s) as coalesce(
    try_strptime(trim(s), '%Y-%m-%d'),
    try_strptime(trim(s), '%d/%m/%Y'),
    try_strptime(trim(s), '%d-%m-%Y'),
    try_strptime(trim(s), '%d/%m/%y'),
    try_strptime(trim(s), '%m/%d/%Y')
)::date;


-- (1) MHSDS time series: monthly referrals starting in the month (MHS32) by age band.
--     The Apr-2016 to Mar-2023 file writes dates US-style (04/30/2016), so the
--     reporting month is taken from REPORTING_PERIOD_END and checked to be a month end.
create or replace table stg_referrals_by_age as
select
    parse_nhs_date(REPORTING_PERIOD_END)           as month_end,
    trim(STATUS)                                   as status,
    trim(SECONDARY_LEVEL_DESCRIPTION)              as age_band,
    try_cast(trim(MEASURE_VALUE) as double)        as referrals
from read_csv('{time_series_glob}', all_varchar = true, union_by_name = true)
where trim(MEASURE_ID) = 'MHS32'
  and trim(BREAKDOWN) = 'England; Age'
  and trim(SECONDARY_LEVEL_DESCRIPTION) in ({cyp_age_bands});


-- (2) Four-week-wait (4WW) files: England children's waiting-list measures, one file per month.
--     Flows (MRS11 spells started, MRS16 first contacts, MRS12 closed) are ROLLING 3-MONTH totals;
--     MRS10 / MRS10a / MRS10b are END-OF-MONTH snapshots of the open waiting list.
create or replace table stg_four_week_wait as
with raw as (
    select
        regexp_extract(filename, '4WW_(\d{{4}}-\d{{2}})', 1)   as file_month,
        parse_nhs_date(REPORTING_PERIOD_START)                as period_start,
        parse_nhs_date(REPORTING_PERIOD_END)                  as period_end,
        trim(MEASURE_ID)                                      as measure_id,
        try_cast(trim(MEASURE_VALUE) as double)               as value       -- '44183.0' and '44183' both parse
    from read_csv('{four_week_wait_glob}', all_varchar = true, union_by_name = true, filename = true)
    where trim(BREAKDOWN) = 'England'
      and trim(MEASURE_ID) in ('MRS10', 'MRS10a', 'MRS10b', 'MRS11', 'MRS12', 'MRS16')
)
select *
from raw
-- MRS08-style double windows: keep the shortest (latest-starting) window per file and measure
qualify period_start = max(period_start) over (partition by file_month, measure_id);
