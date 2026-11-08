"""Download raw open data (idempotent).

Run:  python -m cypmh_forecast.download
"""

from __future__ import annotations

import calendar
import logging
import re

import requests

from . import config

log = logging.getLogger(__name__)
TIMEOUT = 300


def _fetch(url: str, dest) -> None:
    if dest.exists():
        log.info("cached   %s", dest.name)
        return
    log.info("download %s", url)
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.rename(dest)


def _months(first: tuple[int, int], last: tuple[int, int]):
    y, m = first
    while (y, m) <= last:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def pick_four_week_wait_link(links: list[str]) -> str | None:
    """Prefer the 'Final' file over 'Performance' (more complete submissions), and the
    highest re-issue version (_v2, _v3) within each."""
    cands = [u for u in links if "4WW" in u and u.lower().endswith((".zip", ".csv"))]
    if not cands:
        return None

    def rank(u: str):
        version = int(m.group(1)) if (m := re.search(r"_v(\d+)\.", u)) else 1
        return ("Final" in u, version)

    return max(cands, key=rank)


def download_four_week_wait() -> None:
    config.FOUR_WEEK_WAIT_DIR.mkdir(parents=True, exist_ok=True)
    for year, month in _months(config.FOUR_WEEK_WAIT_FIRST, config.FOUR_WEEK_WAIT_LAST):
        stem = f"4WW_{year}-{month:02d}"
        if any(config.FOUR_WEEK_WAIT_DIR.glob(stem + ".*")):
            continue
        page = config.PUBLICATION_PAGE.format(month=calendar.month_name[month].lower(), year=year)
        r = requests.get(page, timeout=60)
        if r.status_code != 200:
            log.info("no page  %s-%02d", year, month)
            continue
        url = pick_four_week_wait_link(re.findall(r'href="([^"]+)"', r.text))
        if url is None:
            log.info("no 4WW   %s-%02d", year, month)
            continue
        _fetch(url, config.FOUR_WEEK_WAIT_DIR / f"{stem}{'.zip' if url.lower().endswith('.zip') else '.csv'}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.RAW.mkdir(parents=True, exist_ok=True)
    _fetch(config.TIME_SERIES_URL, config.TIME_SERIES_ZIP)
    _fetch(config.BANK_HOLIDAYS_URL, config.BANK_HOLIDAYS_FILE)
    download_four_week_wait()


if __name__ == "__main__":
    main()
