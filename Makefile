PY := .venv/bin/python

.PHONY: setup data eda preprocess features baselines gap test lint all

setup:            ## create the environment (Python 3.11 via uv)
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -r requirements.txt
	@echo "macOS: XGBoost also needs 'brew install libomp'"

data:             ## download METR-LA + sensor graph into data/raw
	./scripts/download_data.sh

eda:              ## Phase 1: execute the EDA notebook, regenerating figures + findings
	cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb

preprocess:       ## Phase 2: leak-free windows -> data/processed/<protocol>/
	$(PY) -m src.data.preprocess

features:         ## Phase 3: per-sensor feature tables (memory-mapped .npy)
	$(PY) -m src.features.build

baselines:        ## Phase 3: baseline ladder -> results/baselines.csv + figures 6-7
	$(PY) -m src.train.baselines persistence hist_avg ridge xgb rf lstm --peak
	$(PY) -m src.viz.baseline_plots

gap:              ## Phase 4: gap evidence -> results/gap_results.csv, results/gap/, figures 8-11
	$(PY) -m src.train.phase4 neighbours onset residuals table
	$(PY) -m src.viz.gap_plots

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests

all: data eda preprocess features baselines gap test lint
