PY := .venv/bin/python

.PHONY: setup data eda preprocess features baselines gap accuracy report dashboard test lint all

setup:            ## create the environment (Python 3.11 via uv) and install pinned requirements
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -r requirements-pipeline.txt
	@echo "macOS: XGBoost also needs 'brew install libomp'; the report needs 'brew install tectonic'"

data:             ## download METR-LA and the sensor graph into data/raw
	./scripts/download_data.sh

eda:              ## execute the EDA notebook: figures 1-5 and results/phase1_findings.json
	cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb

preprocess:       ## leak-free windows -> data/processed/<protocol>/
	$(PY) -m src.data.preprocess

features:         ## per-sensor feature tables (memory-mapped .npy)
	$(PY) -m src.features.build

baselines:        ## six baselines -> results/baselines*.csv and figures 6-7
	$(PY) -m src.train.baselines persistence hist_avg ridge xgb rf lstm --peak
	$(PY) -m src.viz.baseline_plots

gap:              ## gap experiments -> results/gap_results.csv, results/gap/ and figures 8-11
	$(PY) -m src.train.phase4 neighbours onset residuals table
	$(PY) -m src.viz.gap_plots

accuracy:         ## percent accuracy and within-5-mph scores -> results/accuracy.csv
	$(PY) -m src.eval.accuracy

report:           ## regenerate the report tables from results/ and compile docs/report/report.pdf
	$(PY) scripts/make_report_tables.py
	cd docs/report && tectonic report.tex

dashboard:        ## interactive Streamlit dashboard over results/ (app.py)
	$(PY) -m streamlit run app.py

test:             ## run the automated tests
	$(PY) -m pytest -q

lint:             ## static checks
	.venv/bin/ruff check src tests scripts app.py

all: data eda preprocess features baselines gap accuracy report test lint
