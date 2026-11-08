.PHONY: install data build analysis test lint all

install:            ## install package + dev tools
	pip install -e ".[dev]"

data:               ## download NHS time series, 29 monthly four-week-wait files, bank holidays
	python -m cypmh_forecast.download

build:              ## DuckDB staging -> monthly marts + working-day calendar
	python -m cypmh_forecast.build

analysis:           ## backtests, forecasts, simulator calibration/validation, scenarios, figures
	python -m cypmh_forecast.analysis

test:               ## models, simulator, data contracts
	pytest -q

lint:
	ruff check src tests

all: data build analysis test
