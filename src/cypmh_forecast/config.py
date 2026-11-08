"""Configuration: data sources, paths and modelling settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
SQL_DIR = ROOT / "sql"

# NHS England - Mental Health Services Monthly Statistics: time series file (Apr 2016 - Aug 2026)
# https://digital.nhs.uk/data-and-information/publications/statistical/mental-health-services-monthly-statistics/performance-august-2026
TIME_SERIES_URL = "https://files.digital.nhs.uk/AC/0A475D/MHSDS%20Time_Series_data_Apr_2016_Aug_Perf_2026.zip"
TIME_SERIES_ZIP = RAW / "MHSDS_Time_Series_Apr2016_Aug2026.zip"

# Monthly publication pages; each links a children's four-week-wait (4WW) file
PUBLICATION_PAGE = (
    "https://digital.nhs.uk/data-and-information/publications/statistical/"
    "mental-health-services-monthly-statistics/performance-{month}-{year}"
)
FOUR_WEEK_WAIT_FIRST = (2023, 4)   # scan from here; months without a 4WW file are skipped
FOUR_WEEK_WAIT_LAST = (2026, 8)
FOUR_WEEK_WAIT_DIR = RAW / "four_week_wait"

# UK Government bank holidays API (England and Wales) - used for working-day calendar features
BANK_HOLIDAYS_URL = "https://www.gov.uk/bank-holidays.json"
BANK_HOLIDAYS_FILE = RAW / "bank_holidays.json"

# Age bands that make up "children and young people" (under 18) in the MHS32 age breakdown
CYP_AGE_BANDS = ["0 to 5", "6 to 10", "11 to 15", "16", "17"]

# Modelling
HORIZON = 12               # months ahead for the published forecast
BACKTEST_HORIZON = 6       # months ahead evaluated in rolling-origin cross-validation
BACKTEST_MIN_TRAIN = 30    # first training window (months)
SEASONAL_PERIOD = 12
RANDOM_SEED = 2026
