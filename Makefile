PYTHON ?= .venv/bin/python

.PHONY: check test doctor plan status
check:
	$(PYTHON) scripts/project.py validate-config
test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
doctor:
	$(PYTHON) scripts/project.py doctor --gpu
plan:
	$(PYTHON) scripts/project.py plan train
status:
	$(PYTHON) scripts/project.py status
