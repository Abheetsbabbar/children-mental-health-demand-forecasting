"""Forecasting models and reconciliation: known-answer tests on synthetic series."""

import numpy as np
import pandas as pd
import pytest

from cypmh_forecast import models as M


@pytest.fixture
def seasonal_series():
    rng = np.random.default_rng(0)
    t = np.arange(60)
    season = np.tile([0.1, 0.05, 0.2, -0.08, 0.1, 0.08, 0.02, -0.35, -0.08, 0.05, 0.12, -0.1], 5)
    y = 100_000 * np.exp(0.002 * t + season + rng.normal(0, 0.01, 60))
    return pd.Series(y)


def test_seasonal_naive_repeats_last_year(seasonal_series):
    f = M.seasonal_naive(seasonal_series, 14)
    assert np.allclose(f.mean[:12], seasonal_series.iloc[-12:])
    assert f.mean[12] == pytest.approx(seasonal_series.iloc[-12])


def test_ets_tracks_strong_seasonality(seasonal_series):
    train, test = seasonal_series.iloc[:48], seasonal_series.iloc[48:]
    f = M.ets(train, 12)
    mape = np.mean(np.abs(f.mean - test.to_numpy()) / test.to_numpy())
    assert mape < 0.05
    assert np.all(f.lower < f.mean) and np.all(f.mean < f.upper)


def test_harmonic_regression_recovers_august_dip(seasonal_series):
    n = len(seasonal_series)
    months = (np.arange(n) % 12) + 1
    exog = (20 + np.random.default_rng(1).integers(0, 3, n)).reshape(-1, 1).astype(float)
    f = M.harmonic_regression(seasonal_series.iloc[:48], 12, exog[:48], exog[48:], months[:48], months[48:])
    # August (index 7) should be the lowest forecast month, ~35% below its neighbours
    assert np.argmin(f.mean) == 7
    assert f.mean[7] / f.mean[6] == pytest.approx(np.exp(-0.37), rel=0.05)


def test_mase_scale_is_seasonal_naive_mae():
    y = np.arange(30, dtype=float)
    assert M.mase_scale(y, m=12) == pytest.approx(12.0)


def test_ensemble_is_geometric_mean():
    a, b = M.Forecast(np.array([100.0])), M.Forecast(np.array([400.0]))
    assert M.ensemble([a, b]).mean[0] == pytest.approx(200.0)


def test_reconciliation_produces_coherent_forecasts():
    S = M.summing_matrix(3)
    base = np.array([[100.0], [30.0], [40.0], [20.0]])  # incoherent: 30+40+20 = 90 != 100
    for w in [None, np.array([4.0, 1.0, 1.0, 1.0])]:
        rec = M.reconcile(base, S, w)
        assert rec[0, 0] == pytest.approx(rec[1:, 0].sum())


def test_reconciliation_leaves_coherent_forecasts_unchanged():
    S = M.summing_matrix(3)
    base = np.array([[90.0], [30.0], [40.0], [20.0]])
    assert np.allclose(M.reconcile(base, S), base)


def test_wls_trusts_low_variance_series_more():
    S = M.summing_matrix(2)
    base = np.array([[110.0], [50.0], [50.0]])
    trust_total = M.reconcile(base, S, np.array([0.01, 10, 10]))
    trust_bottom = M.reconcile(base, S, np.array([100, 0.01, 0.01]))
    assert abs(trust_total[0, 0] - 110) < abs(trust_bottom[0, 0] - 110)
