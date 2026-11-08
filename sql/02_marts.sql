-- =====================================================================
-- MARTS: one row per month, analysis-ready.
-- =====================================================================

-- Monthly CYP referrals (aged 0-17) - total and by age band (wide)
create or replace table mart_referrals_monthly as
pivot (
    select month_end, age_band, referrals from stg_referrals_by_age
) on age_band in ({cyp_age_bands})
using sum(referrals)
order by month_end;

create or replace table mart_referrals_monthly as
select
    month_end,
    "0 to 5" + "6 to 10" + "11 to 15" + "16" + "17" as referrals_0_17,
    "0 to 5"   as age_0_5,
    "6 to 10"  as age_6_10,
    "11 to 15" as age_11_15,
    "16" + "17" as age_16_17
from mart_referrals_monthly
order by month_end;


-- Monthly waiting-list dynamics. Rolling 3-month flows are converted to a monthly
-- average (/3); the implied "other exits" (referrals closed or withdrawn before a
-- first contact) come from the stock-flow identity over the same 3 months:
--     W(t) - W(t-3) = arrivals - first_contacts - other_exits
create or replace table mart_waiting_list_monthly as
with w as (
    pivot (select period_end as month_end, measure_id, value from stg_four_week_wait)
    on measure_id in ('MRS10', 'MRS10a', 'MRS10b', 'MRS11', 'MRS12', 'MRS16')
    using first(value)
)
select
    month_end,
    MRS10                                   as waiting,
    MRS10a                                  as median_wait_days,
    MRS10b                                  as p90_wait_days,
    MRS11 / 3                               as arrivals_per_month,
    MRS16 / 3                               as first_contacts_per_month,
    MRS12 / 3                               as closures_per_month,
    (MRS11 - MRS16 - (MRS10 - lag(MRS10, 3) over (order by month_end))) / 3 as other_exits_per_month
from w
order by month_end;
