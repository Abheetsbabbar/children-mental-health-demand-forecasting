"""Extract, stage and model the monthly series with DuckDB.

Run:  python -m cypmh_forecast.build
Outputs (small, committed):
    data/processed/referrals_monthly.csv
    data/processed/waiting_list_monthly.csv
    data/processed/calendar_monthly.csv
"""

from __future__ import annotations

import json
import logging
import zipfile

import duckdb
import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def _extract_all() -> tuple[str, str]:
    ts_dir = config.RAW / "extracted" / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(config.TIME_SERIES_ZIP) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv") and not (ts_dir / name.split("/")[-1]).exists():
                log.info("extract %s", name)
                (ts_dir / name.split("/")[-1]).write_bytes(zf.read(name))

    fww_dir = config.RAW / "extracted" / "four_week_wait"
    fww_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(config.FOUR_WEEK_WAIT_DIR.iterdir()):
        target = fww_dir / f"{f.stem}.csv"
        if target.exists():
            continue
        if f.suffix == ".csv":
            target.write_bytes(f.read_bytes())
        else:
            with zipfile.ZipFile(f) as zf:
                name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                target.write_bytes(zf.read(name))
    return str(ts_dir / "*.csv"), str(fww_dir / "4WW_*.csv")


def working_days_calendar(months: pd.DatetimeIndex) -> pd.DataFrame:
    """Working days (Mon-Fri excluding England & Wales bank holidays) per calendar month."""
    events = json.loads(config.BANK_HOLIDAYS_FILE.read_text())["england-and-wales"]["events"]
    holidays = np.array([e["date"] for e in events], dtype="datetime64[D]")
    rows = []
    for m in months:
        start = np.datetime64(m.to_period("M").start_time.date(), "D")
        end = np.datetime64((m.to_period("M").end_time + pd.Timedelta(days=1)).date(), "D")
        rows.append({
            "month_end": m,
            "working_days": int(np.busday_count(start, end, holidays=holidays)),
            "days_in_month": m.days_in_month,
            # English school summer holidays fall mainly in August -> fewer school-led referrals
            "is_august": int(m.month == 8),
        })
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    ts_glob, fww_glob = _extract_all()
    con = duckdb.connect()
    params = dict(time_series_glob=ts_glob, four_week_wait_glob=fww_glob,
                  cyp_age_bands=", ".join(f"'{b}'" for b in config.CYP_AGE_BANDS))
    for sql_file in sorted(config.SQL_DIR.glob("*.sql")):
        log.info("run %s", sql_file.name)
        con.execute(sql_file.read_text().format(**params)).fetchall()

    ref = con.execute("select * from mart_referrals_monthly").df()
    wl = con.execute("select * from mart_waiting_list_monthly").df()
    ref.to_csv(config.PROCESSED / "referrals_monthly.csv", index=False)
    wl.to_csv(config.PROCESSED / "waiting_list_monthly.csv", index=False)

    months = pd.date_range(ref.month_end.min(), pd.Timestamp(ref.month_end.max()) + pd.offsets.MonthEnd(config.HORIZON),
                           freq="ME")
    working_days_calendar(months).to_csv(config.PROCESSED / "calendar_monthly.csv", index=False)
    log.info("referrals: %d months (%s to %s); waiting list: %d months",
             len(ref), ref.month_end.min(), ref.month_end.max(), len(wl))


if __name__ == "__main__":
    main()
