"""Forecasting + simulation pipeline: backtests, final forecast, scenarios, figures, results.json.

Run:  python -m cypmh_forecast.analysis
"""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from . import config, style
from . import models as M
from . import simulate as sim

log = logging.getLogger(__name__)
SOURCE = "Source: NHS England, Mental Health Services Monthly Statistics (time series and four-week-wait files)."
BANDS = ["age_0_5", "age_6_10", "age_11_15", "age_16_17"]
BAND_LABELS = {"age_0_5": "0-5", "age_6_10": "6-10", "age_11_15": "11-15", "age_16_17": "16-17"}


def load():
    ref = pd.read_csv(config.PROCESSED / "referrals_monthly.csv", parse_dates=["month_end"])
    cal = pd.read_csv(config.PROCESSED / "calendar_monthly.csv", parse_dates=["month_end"])
    wl = pd.read_csv(config.PROCESSED / "waiting_list_monthly.csv", parse_dates=["month_end"])
    return ref.merge(cal, on="month_end", how="left"), cal, wl


# =========================================================================== forecasting


def forecasting(ref: pd.DataFrame, cal: pd.DataFrame, results: dict) -> pd.DataFrame:
    y = ref["referrals_0_17"]
    exog = ref[["working_days"]].to_numpy(float)

    # Order selection on the FIRST training window only (no peeking at the test periods)
    order = M.select_sarima_order(y.iloc[: config.BACKTEST_MIN_TRAIN], exog[: config.BACKTEST_MIN_TRAIN])
    results["sarima_order"] = {"order": order[0], "seasonal_order": order[1]}
    log.info("SARIMA order %s", order)

    bt = M.rolling_origin_backtest(ref, "referrals_0_17", config.BACKTEST_MIN_TRAIN, config.BACKTEST_HORIZON, order)
    bt.to_csv(config.PROCESSED / "backtest_errors.csv", index=False)
    summary = M.summarise_backtest(bt)
    results["backtest"] = summary.round(3).to_dict(orient="index")
    results["backtest_origins"] = int(bt.origin.nunique())
    fig_backtest(bt, summary)

    # Final 12-month forecast on all data
    fut = cal[cal.month_end > ref.month_end.max()].head(config.HORIZON)
    fcs = M.run_all(y, config.HORIZON, exog, fut[["working_days"]].to_numpy(float),
                    ref.month_end.dt.month.to_numpy(), fut.month_end.dt.month.to_numpy(), order)
    out = pd.DataFrame({"month_end": fut.month_end.to_numpy()})
    for name, f in fcs.items():
        out[f"{name} | forecast"] = f.mean
        if f.lower is not None:
            out[f"{name} | lo80"], out[f"{name} | hi80"] = f.lower, f.upper
    out.to_csv(config.PROCESSED / "forecast_referrals_0_17.csv", index=False)

    best = summary.index[0]
    ens = fcs["Ensemble"]
    last12 = y.iloc[-12:].sum()
    results["forecast"] = {
        "best_model_by_MASE": best,
        "next_12m_total_ensemble": float(ens.mean.sum()),
        "last_12m_actual_total": float(last12),
        "growth_pct": float((ens.mean.sum() / last12 - 1) * 100),
        "monthly": {str(d.date()): round(float(v)) for d, v in zip(fut.month_end, ens.mean, strict=True)},
    }
    fig_forecast(ref, fut, fcs)
    return out


def hierarchical(ref: pd.DataFrame, results: dict) -> None:
    """Age-band hierarchy: total = sum of 4 bands. Compare base vs reconciled forecasts in backtest."""
    S = M.summing_matrix(len(BANDS))
    cols = ["referrals_0_17", *BANDS]
    h = config.BACKTEST_HORIZON
    errs = {k: [] for k in ["base", "bottom_up", "ols", "wls"]}
    band_errs = {k: [] for k in ["base", "wls"]}
    for origin in range(config.BACKTEST_MIN_TRAIN, len(ref) - h + 1):
        train = ref.iloc[:origin]
        base = np.vstack([M.ets(train[c], h).mean for c in cols])
        # WLS weights: variance of each series' seasonal-naive errors in the training window
        w = np.array([np.var(train[c].to_numpy()[12:] - train[c].to_numpy()[:-12]) for c in cols])
        rec = {"base": base, "bottom_up": S @ base[1:], "ols": M.reconcile(base, S), "wls": M.reconcile(base, S, w)}
        actual = ref[cols].iloc[origin:origin + h].to_numpy().T
        for k, f in rec.items():
            errs[k].append(np.abs(f[0] - actual[0]) / actual[0])
        for k in band_errs:
            band_errs[k].append(np.abs(rec[k][1:] - actual[1:]) / actual[1:])
    results["hierarchy_total_MAPE"] = {k: float(np.mean(v) * 100) for k, v in errs.items()}
    results["hierarchy_band_MAPE"] = {
        k: {BAND_LABELS[b]: float(np.mean([e[i] for e in v]) * 100) for i, b in enumerate(BANDS)}
        for k, v in band_errs.items()}
    fig_age_bands(ref)


# =========================================================================== simulation


def plan_from_observed(wl: pd.DataFrame) -> list[sim.MonthPlan]:
    return [sim.MonthPlan(days=r.month_end.days_in_month, arrivals=r.arrivals_per_month,
                          capacity=r.first_contacts_per_month, theta=r.other_exits_per_month / r.waiting)
            for r in wl.itertuples()]


def calibrate_and_validate(wl: pd.DataFrame, results: dict) -> tuple[float, float]:
    """Calibrate (k_serve, k_exit) on one year by grid search, then validate on the following year."""
    wl = wl.dropna(subset=["other_exits_per_month"]).reset_index(drop=True)
    start = wl.iloc[1]  # simulation starts from this month-end snapshot
    calib = wl.iloc[2:14]
    valid = wl.iloc[14:]
    state0 = sim.initial_state(start.waiting, start.median_wait_days, start.p90_wait_days)
    plan_calib = plan_from_observed(calib)
    med, p90 = calib.median_wait_days.to_numpy(), calib.p90_wait_days.to_numpy()

    def loss(k_serve, k_exit) -> float:
        """Mean squared relative error of simulated median and p90 waits over the calibration year."""
        run = sim.simulate(state0, plan_calib, k_serve, k_exit)
        return float(np.mean(((run.median_wait.to_numpy() - med) / med) ** 2
                             + ((run.p90_wait.to_numpy() - p90) / p90) ** 2))

    grid_serve = np.round(np.concatenate([[0.0], np.geomspace(0.002, 0.3, 30)]), 4)
    grid_exit = np.round(np.linspace(0, 0.03, 16), 4)
    losses = {(float(a), float(b)): loss(a, b) for a in grid_serve for b in grid_exit}
    grid = pd.Series(losses).rename_axis(["k_serve", "k_exit"]).rename("loss").reset_index()
    grid.to_csv(config.PROCESSED / "simulation_calibration_grid.csv", index=False)
    # The loss surface is flat for large k_serve (any strong "newest first" priority fits equally
    # well), so pick the most parsimonious point: the smallest k_serve within 5% of the best loss.
    best_loss = grid.loss.min()
    chosen = grid[grid.loss <= best_loss * 1.05].sort_values(["k_serve", "loss"]).iloc[0]
    k_serve, k_exit = float(chosen.k_serve), float(chosen.k_exit)
    results["simulation_calibration"] = {
        "start_month": str(start.month_end.date()), "calibration": f"{calib.month_end.iloc[0].date()} to "
        f"{calib.month_end.iloc[-1].date()}", "validation": f"{valid.month_end.iloc[0].date()} to "
        f"{valid.month_end.iloc[-1].date()}", "k_serve_per_day": k_serve, "k_exit_per_day": k_exit,
        "priority_half_life_days": float(np.log(2) / k_serve) if k_serve > 0 else None,
        "chosen_loss": float(chosen.loss), "best_loss_on_grid": float(best_loss),
        "random_order_loss": losses[(0.0, 0.0)], "fifo_loss": loss("fifo", k_exit), "grid_size": len(losses)}
    log.info("calibrated k_serve=%.4f k_exit=%.4f", k_serve, k_exit)

    full = sim.simulate(state0, plan_from_observed(wl.iloc[2:]), k_serve, k_exit)
    full["month_end"] = wl.month_end.iloc[2:].to_numpy()
    obs = wl.iloc[2:].reset_index(drop=True)
    in_valid = full.month_end > calib.month_end.iloc[-1]
    mape = lambda a, b: float(np.mean(np.abs(a - b) / b) * 100)  # noqa: E731
    results["simulation_validation_MAPE"] = {
        "waiting_list_size": mape(full.waiting[in_valid].to_numpy(), obs.waiting[in_valid].to_numpy()),
        "median_wait": mape(full.median_wait[in_valid].to_numpy(), obs.median_wait_days[in_valid].to_numpy()),
        "p90_wait": mape(full.p90_wait[in_valid].to_numpy(), obs.p90_wait_days[in_valid].to_numpy()),
    }
    fig_sim_validation(full, obs, calib.month_end.iloc[-1])
    return k_serve, k_exit


def scenarios(wl: pd.DataFrame, fc: pd.DataFrame, ref: pd.DataFrame, k: float, k_exit: float,
              results: dict) -> None:
    """24-month what-if: forecast demand x capacity uplift x prioritisation policy (Monte Carlo)."""
    last = wl.iloc[-1]
    recent = wl.dropna().tail(12)
    theta = float((recent.other_exits_per_month / recent.waiting).mean())
    capacity0 = float(recent.first_contacts_per_month.mean())

    # Convert forecast *referrals* to waiting-list *spells* using the observed ratio (3-month aligned)
    ref3 = ref.set_index("month_end")["referrals_0_17"].rolling(3).mean()
    ratio = float((recent.set_index("month_end").arrivals_per_month / ref3.reindex(recent.month_end)).mean())
    demand = fc["Ensemble | forecast"].to_numpy() * ratio
    demand = np.concatenate([demand, demand])  # year 2: repeat year-1 seasonal profile (flat growth)
    months = pd.date_range(last.month_end + pd.offsets.MonthEnd(1), periods=24, freq="ME")

    state0 = sim.initial_state(last.waiting, last.median_wait_days, last.p90_wait_days)
    rng = np.random.default_rng(config.RANDOM_SEED)
    runs = []
    policies = {"Current priorities": k, "25% ring-fenced for longest waiters": ("ringfence", 0.25, k),
                "Strict FIFO": "fifo"}
    for uplift in [0, 10, 20, 30]:
        for policy_name, policy in policies.items():
            plan = [sim.MonthPlan(m.days_in_month, demand[i], capacity0 * (1 + uplift / 100), theta)
                    for i, m in enumerate(months)]
            n_runs = 40
            for r in range(n_runs):
                out = sim.simulate(state0, plan, policy, k_exit, rng)
                out["run"], out["uplift"], out["policy"] = r, uplift, policy_name
                out["month_end"] = months
                runs.append(out)
    allr = pd.concat(runs)
    agg = (allr.groupby(["uplift", "policy", "month_end"])
           .agg(waiting=("waiting", "median"), waiting_lo=("waiting", lambda s: s.quantile(0.05)),
                waiting_hi=("waiting", lambda s: s.quantile(0.95)), median_wait=("median_wait", "median"),
                share_over_52w=("share_over_52w", "median"), share_over_4w=("share_over_4w", "median"))
           .reset_index())
    agg.to_csv(config.PROCESSED / "scenario_results.csv", index=False)
    end = agg[agg.month_end == months[-1]].set_index(["policy", "uplift"])
    avg_demand = float(demand[:12].mean())
    results["scenarios"] = {
        "inputs": {"demand_spells_per_month_next12": avg_demand, "capacity_first_contacts_per_month": capacity0,
                   "theta_monthly_exit_without_contact": theta, "referral_to_spell_ratio": ratio,
                   "start_waiting": float(last.waiting), "start_median_wait": float(last.median_wait_days)},
        "steady_state_list_size_status_quo": sim.steady_state_list_size(avg_demand, capacity0, theta),
        "capacity_to_stop_list_growing": float(avg_demand - theta * last.waiting),
        "at_24_months": {f"{p} +{u}%": {"waiting": round(float(r.waiting)), "median_wait_days": float(r.median_wait),
                                         "pct_over_52_weeks": round(float(r.share_over_52w) * 100, 1),
                                         "pct_over_4_weeks": round(float(r.share_over_4w) * 100, 1)}
                         for (p, u), r in end.iterrows()},
    }
    fig_scenarios(agg, wl)


# =========================================================================== figures


def fig_forecast(ref, fut, fcs):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(ref.month_end, ref.referrals_0_17 / 1000, color=style.INK_2, lw=1.6, label="Actual")
    ens = fcs["Ensemble"]
    ax.fill_between(fut.month_end, ens.lower / 1000, ens.upper / 1000, color=style.BLUE, alpha=0.18, lw=0,
                    label="80% prediction interval")
    ax.plot(fut.month_end, ens.mean / 1000, color=style.BLUE, lw=2, label="Ensemble forecast")
    ax.plot(fut.month_end, fcs["Seasonal naive"].mean / 1000, color=style.MUTED, lw=1, ls="--",
            label="Seasonal naive benchmark")
    ax.axvline(ref.month_end.max(), color=style.AXIS, lw=1)
    ax.set_ylabel("New referrals per month (thousands)")
    ax.set_ylim(bottom=0)
    ax.set_title("Children's mental health referrals: 12-month forecast")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    style.source_note(fig, SOURCE + " Referrals starting in month (MHS32), ages 0-17, England.")
    fig.savefig(config.FIGURES / "01_forecast.png")
    plt.close(fig)


def fig_backtest(bt, summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    s = summary.sort_values("MASE", ascending=False)
    colors = [style.BLUE if m == summary.index[0] else style.NEUTRAL for m in s.index]
    ax.barh(s.index, s.MASE, color=colors, height=0.6)
    for i, v in enumerate(s.MASE):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color=style.INK_2)
    ax.axvline(1, color=style.CRITICAL, lw=1, ls="--")
    ax.set_xlabel("MASE (lower is better; <1 beats seasonal naive in-sample)")
    ax.set_title("Accuracy across all backtests", fontsize=11)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    by_h = bt.groupby(["model", "h"]).ape.mean().unstack(0) * 100
    palette = {"Harmonic regression": style.BLUE, "Ensemble": style.ORANGE, "SARIMAX + working days": style.AQUA}
    for m in [*palette, *[c for c in by_h.columns if c not in palette]]:
        ax.plot(by_h.index, by_h[m], color=palette.get(m, style.NEUTRAL), lw=2 if m in palette else 1.2,
                ls="-" if m != "Seasonal naive" else "--", marker="o", ms=4 if m in palette else 3, label=m)
    ax.set_xlabel("Months ahead")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Error by forecast horizon", fontsize=11)
    ax.legend(fontsize=7.5)
    fig.suptitle(f"Rolling-origin cross-validation ({bt.origin.nunique()} origins x 6-month horizon)",
                 x=0.01, ha="left", fontweight="bold")
    style.source_note(fig, SOURCE)
    fig.savefig(config.FIGURES / "02_backtest_accuracy.png")
    plt.close(fig)


def fig_age_bands(ref):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    stl = STL(np.log(ref.set_index("month_end").referrals_0_17), period=12, robust=True).fit()
    month_effect = (np.exp(stl.seasonal.groupby(stl.seasonal.index.month).mean()) - 1) * 100
    ax = axes[0]
    colors = [style.ORANGE if v < 0 else style.BLUE for v in month_effect]
    ax.bar(range(1, 13), month_effect, color=colors, width=0.7)
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.axhline(0, color=style.AXIS, lw=1)
    ax.set_ylabel("% vs trend (STL seasonal component)")
    ax.set_title("School holidays drive the seasonal pattern", fontsize=11)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    for b, c in zip(BANDS, [style.MUTED, style.AQUA, style.BLUE, style.ORANGE], strict=True):
        ax.plot(ref.month_end, ref[b] / 1000, color=c, lw=1.6, label=f"Age {BAND_LABELS[b]}")
    ax.set_ylabel("Referrals per month (thousands)")
    ax.set_title("11-15 year olds are half of all referrals", fontsize=11)
    ax.set_ylim(0, ref[BANDS].max().max() / 1000 * 1.25)
    ax.legend(fontsize=8, ncol=4, loc="upper center", handlelength=1.2)
    style.date_axis(ax)
    style.source_note(fig, SOURCE)
    fig.savefig(config.FIGURES / "03_seasonality_and_age.png")
    plt.close(fig)


def fig_sim_validation(full, obs, calib_end):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, sim_col, obs_col, title, div in [
        (axes[0], "waiting", "waiting", "Children waiting for first contact (thousands)", 1000),
        (axes[1], "median_wait", "median_wait_days", "Median days waited so far", 1),
    ]:
        ax.plot(obs.month_end, obs[obs_col] / div, color=style.INK_2, lw=1.6, marker="o", ms=3, label="Observed")
        ax.plot(full.month_end, full[sim_col] / div, color=style.BLUE, lw=2, label="Simulated")
        ax.axvspan(full.month_end.min(), calib_end, color=style.GRID, alpha=0.6, lw=0)
        ax.text(full.month_end.min(), ax.get_ylim()[1], " calibration", va="top", fontsize=8, color=style.INK_2)
        ax.text(calib_end, ax.get_ylim()[1], "  out-of-sample validation", va="top", fontsize=8, color=style.INK_2)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        style.date_axis(ax)
    fig.suptitle("The waiting-list simulator reproduces the real list it was not fitted to",
                 x=0.01, ha="left", fontweight="bold")
    style.source_note(fig, SOURCE + " Four-week-wait measures MRS10/10a/10b/11/16, England.")
    fig.savefig(config.FIGURES / "04_simulator_validation.png")
    plt.close(fig)


def fig_scenarios(agg, wl):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(wspace=0.28)

    # Left: list size by capacity uplift (policy barely changes the size, only who waits)
    ax = axes[0]
    cur = agg[agg.policy == "Current priorities"]
    shades = {0: style.SEQUENTIAL[2], 10: style.SEQUENTIAL[3], 20: style.SEQUENTIAL[4], 30: style.SEQUENTIAL[6]}
    hist = wl.tail(12)
    ax.plot(hist.month_end, hist.waiting / 1000, color=style.INK_2, lw=1.6, label="Observed")
    for u, g in cur.groupby("uplift"):
        ax.fill_between(g.month_end, g.waiting_lo / 1000, g.waiting_hi / 1000, color=shades[u], alpha=0.2, lw=0)
        ax.plot(g.month_end, g.waiting / 1000, color=shades[u], lw=2)
        ax.text(g.month_end.iloc[-1], g.waiting.iloc[-1] / 1000, f"  +{u}%", va="center", fontsize=8,
                color=style.INK_2)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Children waiting for first contact (thousands)")
    ax.set_title("List size by first-contact capacity uplift", fontsize=11)
    style.date_axis(ax)
    ax.legend(fontsize=8, loc="lower left")

    # Right: share waiting over a year after 24 months, by policy and uplift
    ax = axes[1]
    end = agg[agg.month_end == agg.month_end.max()]
    policies = ["Current priorities", "25% ring-fenced for longest waiters", "Strict FIFO"]
    colors = [style.ORANGE, style.BLUE, style.AQUA]
    x = np.arange(4)
    width = 0.26
    for i, (p, c) in enumerate(zip(policies, colors, strict=True)):
        vals = end[end.policy == p].sort_values("uplift").share_over_52w.to_numpy() * 100
        ax.bar(x + (i - 1) * width, vals, width=width - 0.02, color=c, label=p)
        for xi, v in zip(x + (i - 1) * width, vals, strict=True):
            ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=7.5, color=style.INK_2)
    ax.set_xticks(x, ["+0%", "+10%", "+20%", "+30%"])
    ax.set_xlabel("First-contact capacity uplift")
    ax.set_ylabel("% of children still waiting who have waited > 52 weeks")
    ax.set_title("Who is left waiting after 24 months?", fontsize=11)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Capacity shrinks the list; prioritisation decides who is left behind",
                 x=0.01, ha="left", fontweight="bold")
    style.source_note(fig, SOURCE + " Monte Carlo: 40 stochastic runs per scenario; bands = 5th-95th percentile. "
                           "Demand = ensemble forecast.")
    fig.savefig(config.FIGURES / "05_capacity_scenarios.png")
    plt.close(fig)


# =========================================================================== main


def run() -> dict:
    style.apply()
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    ref, cal, wl = load()
    results: dict = {"data": {"referral_months": len(ref), "first_month": str(ref.month_end.min().date()),
                              "last_month": str(ref.month_end.max().date()), "waiting_list_months": len(wl)}}
    fc = forecasting(ref, cal, results)
    hierarchical(ref, results)
    k, k_exit = calibrate_and_validate(wl, results)
    scenarios(wl, fc, ref, k, k_exit, results)
    (config.ROOT / "reports" / "results.json").write_text(json.dumps(_json_safe(results), indent=2))
    log.info("done")
    return results


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
