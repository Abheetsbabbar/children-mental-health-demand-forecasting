"""Forecasting models with a common interface + rolling-origin evaluation + hierarchical reconciliation.

Every model works on log(referrals): seasonal effects in referral data are multiplicative
(August is ~45% lower whatever the level), and logs keep forecasts and intervals positive.
Forecasts are back-transformed with exp(), i.e. they are medians of the forecast distribution.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

Z80 = 1.2816


@dataclass
class Forecast:
    mean: np.ndarray  # point forecast (original scale)
    lower: np.ndarray | None = None  # 80% prediction interval
    upper: np.ndarray | None = None


# --------------------------------------------------------------------------- models


def seasonal_naive(y: pd.Series, h: int, **_) -> Forecast:
    """Benchmark: each month equals the same month last year."""
    last_year = y.iloc[-12:].to_numpy()
    return Forecast(np.resize(last_year, h))


def ets(y: pd.Series, h: int, **_) -> Forecast:
    """Additive-error ETS with damped additive trend and additive seasonality on log scale
    (equivalent to multiplicative seasonality on the original scale)."""
    log_y = pd.Series(np.log(y.to_numpy()), index=pd.RangeIndex(len(y)))
    model = ETSModel(log_y, error="add", trend="add", damped_trend=True,
                     seasonal="add", seasonal_periods=12).fit(disp=False, maxiter=2000)
    pred = model.get_prediction(start=len(y), end=len(y) + h - 1)
    frame = pred.summary_frame(alpha=0.2)
    return Forecast(np.exp(frame["mean"].to_numpy()), np.exp(frame["pi_lower"].to_numpy()),
                    np.exp(frame["pi_upper"].to_numpy()))


def select_sarima_order(y: pd.Series, exog: np.ndarray) -> tuple[tuple, tuple]:
    """Choose (p,d,q)(P,D,Q)12 by AIC on the given (training) window only."""
    best, best_aic = None, np.inf
    for p, d, q, P, D, Q in itertools.product([0, 1, 2], [0, 1], [0, 1], [0, 1], [1], [0, 1]):
        try:
            res = SARIMAX(np.log(y.to_numpy()), exog=exog, order=(p, d, q), seasonal_order=(P, D, Q, 12),
                          trend="c" if d == 0 else "n").fit(disp=False)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(res.aic) and res.aic < best_aic:
            best, best_aic = ((p, d, q), (P, D, Q, 12)), res.aic
    return best


def sarimax(y: pd.Series, h: int, exog_train: np.ndarray, exog_future: np.ndarray, order: tuple,
            **_) -> Forecast:
    (o, so) = order
    res = SARIMAX(np.log(y.to_numpy()), exog=exog_train, order=o, seasonal_order=so,
                  trend="c" if o[1] == 0 else "n").fit(disp=False)
    fc = res.get_forecast(h, exog=exog_future)
    ci = fc.conf_int(alpha=0.2)
    return Forecast(np.exp(fc.predicted_mean), np.exp(ci[:, 0]), np.exp(ci[:, 1]))


def harmonic_regression(y: pd.Series, h: int, exog_train: np.ndarray, exog_future: np.ndarray,
                        months_train: np.ndarray, months_future: np.ndarray, **_) -> Forecast:
    """OLS on log scale: linear trend + month-of-year effects + working days.
    A transparent 'explainable' baseline - every coefficient has a plain-English meaning."""
    def design(t, months, exog):
        dummies = pd.get_dummies(pd.Categorical(months, categories=range(1, 13)), drop_first=True).to_numpy(float)
        return np.column_stack([np.ones_like(t, dtype=float), t, dummies, exog])

    t_train = np.arange(len(y), dtype=float)
    t_future = np.arange(len(y), len(y) + h, dtype=float)
    X, Xf = design(t_train, months_train, exog_train), design(t_future, months_future, exog_future)
    res = sm.OLS(np.log(y.to_numpy()), X).fit()
    pred = res.get_prediction(Xf).summary_frame(alpha=0.2)
    return Forecast(np.exp(pred["mean"].to_numpy()), np.exp(pred["obs_ci_lower"].to_numpy()),
                    np.exp(pred["obs_ci_upper"].to_numpy()))


def ensemble(forecasts: list[Forecast]) -> Forecast:
    """Geometric mean of member forecasts (mean on the log scale); intervals averaged likewise."""
    mean = np.exp(np.mean([np.log(f.mean) for f in forecasts], axis=0))
    if all(f.lower is not None for f in forecasts):
        lower = np.exp(np.mean([np.log(f.lower) for f in forecasts], axis=0))
        upper = np.exp(np.mean([np.log(f.upper) for f in forecasts], axis=0))
        return Forecast(mean, lower, upper)
    return Forecast(mean)


def run_all(y: pd.Series, h: int, exog_train, exog_future, months_train, months_future, order) -> dict[str, Forecast]:
    kw = dict(exog_train=exog_train, exog_future=exog_future, months_train=months_train,
              months_future=months_future, order=order)
    out = {
        "Seasonal naive": seasonal_naive(y, h),
        "ETS (damped, log)": ets(y, h),
        "SARIMAX + working days": sarimax(y, h, **kw),
        "Harmonic regression": harmonic_regression(y, h, **kw),
    }
    out["Ensemble"] = ensemble([out["ETS (damped, log)"], out["SARIMAX + working days"], out["Harmonic regression"]])
    return out


# --------------------------------------------------------------------------- evaluation


def mase_scale(y_train: np.ndarray, m: int = 12) -> float:
    """In-sample MAE of the seasonal naive forecast - the MASE denominator (Hyndman & Koehler 2006)."""
    return float(np.mean(np.abs(y_train[m:] - y_train[:-m])))


def rolling_origin_backtest(df: pd.DataFrame, target: str, min_train: int, horizon: int, order) -> pd.DataFrame:
    """Expanding-window cross-validation: refit every model at each origin, forecast `horizon` months."""
    rows = []
    exog = df[["working_days"]].to_numpy(float)
    months = df["month_end"].dt.month.to_numpy()
    y_all = df[target]
    for origin in range(min_train, len(df) - horizon + 1):
        y_tr = y_all.iloc[:origin]
        fcs = run_all(y_tr, horizon, exog[:origin], exog[origin:origin + horizon],
                      months[:origin], months[origin:origin + horizon], order)
        actual = y_all.iloc[origin:origin + horizon].to_numpy()
        scale = mase_scale(y_tr.to_numpy())
        for name, f in fcs.items():
            for k in range(horizon):
                rows.append({
                    "origin": df.month_end.iloc[origin - 1], "model": name, "h": k + 1,
                    "target_month": df.month_end.iloc[origin + k], "actual": actual[k], "forecast": f.mean[k],
                    "abs_error": abs(actual[k] - f.mean[k]), "ape": abs(actual[k] - f.mean[k]) / actual[k],
                    "scaled_error": abs(actual[k] - f.mean[k]) / scale,
                    "covered_80": (None if f.lower is None else bool(f.lower[k] <= actual[k] <= f.upper[k])),
                })
    return pd.DataFrame(rows)


def summarise_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    s = bt.groupby("model").agg(MAE=("abs_error", "mean"), MAPE=("ape", "mean"), MASE=("scaled_error", "mean"),
                                PI80_coverage=("covered_80", lambda c: c.dropna().astype(float).mean()
                                               if c.notna().any() else np.nan),
                                n=("abs_error", "size"))
    s["MAPE"] *= 100
    s["PI80_coverage"] *= 100
    return s.sort_values("MASE")


# --------------------------------------------------------------------------- hierarchical reconciliation


def summing_matrix(n_bottom: int) -> np.ndarray:
    """S maps bottom-level series (age bands) to all levels: [total; band_1..band_n]."""
    return np.vstack([np.ones((1, n_bottom)), np.eye(n_bottom)])


def reconcile(base: np.ndarray, S: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Optimal-combination reconciliation (Hyndman et al. 2011; Wickramasuriya et al. 2019).

    base: (n_series, h) base forecasts ordered as rows of S.
    weights: per-series variances (WLS); None -> OLS (identity).
    Returns coherent forecasts S @ P @ base where P = (S' W^-1 S)^-1 S' W^-1.
    """
    W_inv = np.eye(S.shape[0]) if weights is None else np.diag(1.0 / np.asarray(weights))
    P = np.linalg.solve(S.T @ W_inv @ S, S.T @ W_inv)
    return S @ P @ base
