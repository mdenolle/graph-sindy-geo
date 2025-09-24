.PHONY: venv install test run format

PYTHON := python3
VENV := .venv
ACT := . $(VENV)/bin/activate

venv:
$(PYTHON) -m venv $(VENV)

install: venv
$(ACT); pip install -U pip wheel
# Install torch first
$(ACT); pip install torch
# Install PyG deps from the official CPU wheel index
$(ACT); pip install --index-url https://data.pyg.org/whl/torch-2.2.0+cpu torch-scatter torch-sparse torch-cluster
$(ACT); pip install torch-geometric
# Remaining project deps
$(ACT); pip install -r requirements.txt
$(ACT); pip install -e .

test:
$(ACT); pytest -q

run:
$(ACT); python -m pyg_sindy_demo.demo

format:
$(ACT); python -m pip install ruff
$(ACT); ruff format .
