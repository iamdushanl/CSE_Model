# Shortcuts for Linux/macOS. Windows users: python run.py <task>
PY ?= python
setup:
	$(PY) -m pip install -r requirements.txt
data:
	$(PY) run.py data
models:
	$(PY) run.py models
metrics:
	$(PY) run.py metrics
test:
	$(PY) run.py test
all:
	$(PY) run.py all
.PHONY: setup data models metrics test all
