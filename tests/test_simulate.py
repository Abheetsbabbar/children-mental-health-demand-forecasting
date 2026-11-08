"""Waiting-list simulator: conservation laws and priority behaviour."""

import numpy as np
import pytest

from demand_forecasting import simulate as sim


def flat_state(total=1000.0, max_age=400):
    s = np.zeros(sim.MAX_AGE)
    s[:max_age] = total / max_age
    return s


def test_initial_state_matches_published_percentiles():
    s = sim.initial_state(250_000, median=290, p90=1036)
    assert s.sum() == pytest.approx(250_000)
    assert sim.percentile_wait(s, 0.5) == pytest.approx(290, abs=3)
    # within 3%: the log-normal tail beyond MAX_AGE (~0.4% of mass) is truncated and renormalised
    assert sim.percentile_wait(s, 0.9) == pytest.approx(1036, rel=0.03)


def test_no_flows_conserves_list_and_ages_it():
    s = flat_state()
    out = sim.simulate(s, [sim.MonthPlan(days=30, arrivals=0, capacity=0, theta=0)], k=0.0)
    assert out.waiting.iloc[0] == pytest.approx(1000)
    assert out.median_wait.iloc[0] == pytest.approx(sim.percentile_wait(s, 0.5) + 30, abs=1)


def test_balance_identity():
    """End list = start + arrivals - first contacts - exits (deterministic run)."""
    s = flat_state(10_000)
    plan = [sim.MonthPlan(days=30, arrivals=3000, capacity=2000, theta=0.05)]
    out = sim.simulate(s, plan, k=0.01)
    exits_upper = 0.06 * 10_000  # theta applies to a list of ~10-11k
    assert 10_000 + 3000 - 2000 - exits_upper < out.waiting.iloc[0] < 10_000 + 3000 - 2000
    assert out.seen.iloc[0] == pytest.approx(2000, rel=1e-6)


def test_fifo_serves_longest_waiters_first():
    n = np.zeros(10)
    n[[1, 5, 9]] = 10
    served = sim._allocate(n, 15, "fifo")
    assert served[9] == 10 and served[5] == 5 and served[1] == 0


def test_strong_priority_serves_newest_first():
    n = np.zeros(400)
    n[[1, 300]] = 10
    served = sim._allocate(n, 10, 0.5)
    assert served[1] > 9.9 and served[300] < 0.1


def test_allocation_never_exceeds_list():
    n = np.array([1.0, 2.0, 3.0])
    for k in [0.0, 0.3, "fifo", ("ringfence", 0.25, 0.3)]:
        served = sim._allocate(n, 100, k)
        assert np.all(served <= n + 1e-9)
        assert served.sum() == pytest.approx(6.0)


def test_ringfence_splits_capacity():
    n = np.zeros(400)
    n[[1, 300]] = 100
    served = sim._allocate(n, 40, ("ringfence", 0.25, 1.0))
    assert served[300] == pytest.approx(10)  # 25% of 40 to the longest waiters
    assert served[1] == pytest.approx(30)


def test_steady_state_formula():
    assert sim.steady_state_list_size(70_000, 52_000, 0.05) == pytest.approx(360_000)
    assert sim.steady_state_list_size(50_000, 52_000, 0.05) == 0


def test_stochastic_run_is_reproducible():
    s = flat_state(5000)
    plan = [sim.MonthPlan(30, 1000, 900, 0.03)] * 2
    a = sim.simulate(s, plan, 0.05, 0.0, np.random.default_rng(1))
    b = sim.simulate(s, plan, 0.05, 0.0, np.random.default_rng(1))
    assert a.equals(b)
