"""Waiting-list simulation for children's mental health services (England).

A discrete-time (daily) stock-and-flow model of the list of children waiting for a
first contact, tracked by how long each child has waited so far ("age" in days):

    each day:  new referrals join at age 0             ~ Poisson(arrival rate)
               first contacts remove C children        ~ Poisson(capacity)
               some referrals leave without contact    (closure/withdrawal hazard theta)
               everyone remaining ages by one day

Which children receive the C first contacts is set by a PRIORITY parameter k_serve:
the share of capacity going to children who have waited a days is proportional to
n(a) * exp(-k_serve * a). k_serve > 0 favours recent referrals (urgent triage / "fast-track the new"),
k_serve = 0 is random order, and FIFO ("treat in turn") gives capacity to the longest waiters first.

Referrals closed without a first contact are distributed the same way with their own
parameter k_exit (most closures happen soon after referral - signposting, redirection,
declined appointments - rather than uniformly across the list).

Both parameters are calibrated so the model reproduces the observed median and
90th-percentile waits, then validated on a later, unseen year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_AGE = 4000  # days (~11 years); ages beyond are pooled in the last bin


@dataclass
class MonthPlan:
    """Inputs for one simulated month (all per month)."""
    days: int
    arrivals: float
    capacity: float
    theta: float  # share of the waiting list leaving without a first contact per month


def initial_state(total: float, median: float, p90: float) -> np.ndarray:
    """Waiting-time profile of the current list, from a log-normal fitted to the published
    median and 90th percentile (MRS10a, MRS10b)."""
    sigma = np.log(p90 / median) / 1.2816
    mu = np.log(median)
    ages = np.arange(MAX_AGE) + 0.5
    dens = np.exp(-((np.log(ages) - mu) ** 2) / (2 * sigma**2)) / ages
    return total * dens / dens.sum()


def _allocate(n: np.ndarray, capacity: float, k) -> np.ndarray:
    """Distribute `capacity` first contacts across age bins without exceeding any bin.

    k: float (priority weight exp(-k*age)), "fifo", or ("ringfence", share, k) where `share`
    of capacity is reserved for the longest waiters and the rest follows priority k.
    """
    if capacity <= 0 or n.sum() <= 0:
        return np.zeros_like(n)
    if isinstance(k, tuple) and k[0] == "ringfence":
        _, share, k_rest = k
        fifo_part = _allocate(n, capacity * share, "fifo")
        return fifo_part + _allocate(n - fifo_part, capacity * (1 - share), k_rest)
    if k == "fifo":
        cum = np.cumsum(n[::-1])
        served_rev = np.clip(capacity - (cum - n[::-1]), 0, n[::-1])
        return served_rev[::-1]
    w = np.exp(-k * np.arange(len(n)))
    served = np.zeros_like(n)
    remaining = min(capacity, n.sum())
    for _ in range(50):  # water-filling: re-share capacity that would overflow small bins
        open_ = n - served > 1e-9
        weight = w * (n - served) * open_
        if remaining <= 1e-6 or weight.sum() <= 0:
            break
        give = remaining * weight / weight.sum()
        give = np.minimum(give, n - served)
        served += give
        remaining -= give.sum()
    return served


def percentile_wait(n: np.ndarray, q: float) -> float:
    cum = np.cumsum(n)
    return float(np.searchsorted(cum, q * cum[-1]))


def simulate(state: np.ndarray, plan: list[MonthPlan], k, k_exit: float = 0.0,
             rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Run month by month (daily steps). rng=None -> deterministic expected-value run.

    k: service priority (float or "fifo"); k_exit: closure-without-contact priority.
    """
    n = state.copy()
    rows = []
    for month, mp in enumerate(plan, start=1):
        lam, cap = mp.arrivals / mp.days, mp.capacity / mp.days
        hazard = 1 - (1 - mp.theta) ** (1 / mp.days)
        seen_month = 0.0
        for _ in range(mp.days):
            arrivals = rng.poisson(lam) if rng is not None else lam
            capacity = rng.poisson(cap) if rng is not None else cap
            served = _allocate(n, capacity, k)
            n -= served
            seen_month += served.sum()
            exits = n.sum() * hazard
            if rng is not None:
                exits = rng.poisson(exits)
            n -= _allocate(n, exits, k_exit)
            n = np.clip(n, 0, None)
            n[-1] += n[-2]                    # pool the oldest
            n[1:-1] = n[:-2]                  # age everyone by one day
            n[0] = arrivals
        rows.append({"month": month, "waiting": n.sum(), "median_wait": percentile_wait(n, 0.5),
                     "p90_wait": percentile_wait(n, 0.9), "share_over_4w": n[28:].sum() / n.sum(),
                     "share_over_52w": n[364:].sum() / n.sum(), "seen": seen_month})
    return pd.DataFrame(rows)


def steady_state_list_size(arrivals: float, capacity: float, theta: float) -> float:
    """Fluid approximation: dW/dt = arrivals - capacity - theta*W = 0  ->  W* = (arrivals - capacity) / theta."""
    return max(0.0, (arrivals - capacity) / theta)
