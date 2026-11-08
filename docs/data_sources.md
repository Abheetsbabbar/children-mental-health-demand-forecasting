# Data sources and licensing

All inputs are official, openly licensed and **aggregate** (England totals). No patient-level data is used.

| Dataset | Publisher | Used for | Period | Licence |
|---|---|---|---|---|
| [Mental Health Services Monthly Statistics: time series file](https://digital.nhs.uk/data-and-information/publications/statistical/mental-health-services-monthly-statistics/performance-august-2026) (`MHSDS Time_Series_data_Apr_2016_Aug_Perf_2026.zip`) | NHS England | Monthly referrals by age (MHS32) | Apr 2016 – Aug 2026 (age breakdown from Apr 2022) | OGL v3.0 |
| [Mental Health Services Monthly Statistics: four-week-wait files](https://digital.nhs.uk/data-and-information/publications/statistical/mental-health-services-monthly-statistics), 29 monthly releases | NHS England | Waiting-list stock, flows and wait percentiles (MRS10–MRS16) | Apr 2024 – Aug 2026 | OGL v3.0 |
| [UK bank holidays](https://www.gov.uk/bank-holidays.json) | Government Digital Service | Working-day calendar | 2019–2028 | OGL v3.0 |

`python -m cypmh_forecast.download` retrieves everything: it scrapes each monthly
publication page for its four-week-wait link. Raw files go to `data/raw/` (git-ignored).

**Status:** months up to Mar 2026 use **Final** data. Apr–Aug 2026 are
**Performance** (provisional) and may be revised.

**Attribution:** Copyright © 2026, NHS England. Licensed under the Open Government
Licence v3.0. Contains public sector information licensed under the Open Government
Licence v3.0.

**Independence:** This is an independent portfolio project, not produced or
endorsed by NHS England.
