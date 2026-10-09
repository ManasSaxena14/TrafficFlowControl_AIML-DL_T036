PY := .venv/bin/python

.PHONY: setup data eda test lint all

setup:            ## create the environment (Python 3.11 via uv)
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -r requirements.txt
	@echo "macOS: XGBoost also needs 'brew install libomp'"

data:             ## download METR-LA + sensor graph into data/raw
	./scripts/download_data.sh

eda:              ## Phase 1: execute the EDA notebook, regenerating figures + findings
	cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests

all: data eda test lint
