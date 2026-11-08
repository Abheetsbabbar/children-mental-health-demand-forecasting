"""Data contracts on the committed processed series (grain, continuity, ranges, identities)."""

import json
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest
from pandera import Check, Column

from demand_forecasting.download import pick_four_week_wait_link

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

REFERRALS = pa.DataFrameSchema(
    {
        "month_end": Column("datetime64[ns]", unique=True, checks=Check(lambda s: s.dt.is_month_end.all())),
        "referrals_0_17": Column(float, Check.in_range(30_000, 250_000)),
        "age_0_5": Column(float, Check.gt(0)),
        "age_6_10": Column(float, Check.gt(0)),
        "age_11_15": Column(float, Check.gt(0)),
        "age_16_17": Column(float, Check.gt(0)),
    },
    checks=Check(lambda d: (d.age_0_5 + d.age_6_10 + d.age_11_15 + d.age_16_17 - d.referrals_0_17).abs().max() < 1,
                 name="bands_sum_to_total"),
    coerce=True,
)

WAITING = pa.DataFrameSchema(
    {
        "month_end": Column("datetime64[ns]", unique=True),
        "waiting": Column(float, Check.in_range(100_000, 1_000_000)),
        "median_wait_days": Column(float, Check.in_range(0, 2000)),
        "p90_wait_days": Column(float, Check.in_range(0, 5000)),
        "arrivals_per_month": Column(float, Check.gt(0)),
        "first_contacts_per_month": Column(float, Check.gt(0)),
    },
    checks=Check(lambda d: (d.median_wait_days <= d.p90_wait_days).all(), name="median_le_p90"),
    coerce=True,
)


def _contiguous(months: pd.Series) -> bool:
    expected = pd.date_range(months.min(), months.max(), freq="ME")
    return len(expected) == len(months) and (expected == pd.DatetimeIndex(months.sort_values())).all()


@pytest.fixture(scope="module")
def ref():
    return pd.read_csv(PROCESSED / "referrals_monthly.csv", parse_dates=["month_end"])


@pytest.fixture(scope="module")
def wl():
    return pd.read_csv(PROCESSED / "waiting_list_monthly.csv", parse_dates=["month_end"])


def test_referrals_schema_and_continuity(ref):
    REFERRALS.validate(ref, lazy=True)
    assert _contiguous(ref.month_end), "gap in monthly referral series"
    assert len(ref) >= 53


def test_waiting_list_schema_and_continuity(wl):
    WAITING.validate(wl, lazy=True)
    assert _contiguous(wl.month_end)


def test_other_exits_identity(wl):
    """other_exits = (arrivals - first contacts) - change in list over 3 months / 3"""
    w = wl.dropna().copy()
    lhs = w.other_exits_per_month * 3
    rhs = w.arrivals_per_month * 3 - w.first_contacts_per_month * 3 - (w.waiting - wl.waiting.shift(3).loc[w.index])
    assert (lhs - rhs).abs().max() < 1e-6


def test_calendar_working_days():
    cal = pd.read_csv(PROCESSED / "calendar_monthly.csv", parse_dates=["month_end"])
    assert cal.working_days.between(18, 23).all()
    # December 2025: 23 weekdays minus Christmas Day and Boxing Day = 21
    assert cal.set_index("month_end").loc["2025-12-31", "working_days"] == 21


def test_results_backtest_beats_naive_benchmark():
    r = json.loads((ROOT / "reports" / "results.json").read_text())
    assert r["backtest"]["Ensemble"]["MASE"] < r["backtest"]["Seasonal naive"]["MASE"]


@pytest.mark.parametrize("links,expected", [
    (["x/MHSDS%20Data_4WW_JanPerf_2025.csv", "x/MHSDS%20Data_4WW_JanFinal_2025.zip"], "Final"),
    (["x/MHSDS%20Data_4WW_MarPerf_2026.zip", "x/MHSDS%20Data_4WW_MarPerf_2026_v2.zip"], "_v2"),
    (["x/MHSDS%20Data_AugPerf_2026.zip"], None),
])
def test_pick_four_week_wait_link(links, expected):
    got = pick_four_week_wait_link(links)
    assert (got is None) if expected is None else (expected in got)
