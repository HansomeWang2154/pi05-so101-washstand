PYTHON ?= .venv/bin/python

.PHONY: check test doctor plan status sim-test sim-generate privacy-check privacy-history-check install-hooks
check:
	$(PYTHON) scripts/project.py validate-config
test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
privacy-check:
	python3 scripts/check_public.py --index
privacy-history-check:
	python3 scripts/check_public.py --history HEAD
install-hooks:
	git config --local core.hooksPath .githooks
doctor:
	$(PYTHON) scripts/project.py doctor --gpu
plan:
	$(PYTHON) scripts/project.py plan train
status:
	$(PYTHON) scripts/project.py status

SIM_RUN ?= data/sim/raw/manual_run
SIM_ATTEMPTS ?= 4
sim-test:
	MUJOCO_GL=$${MUJOCO_GL:-egl} WASHSTAND_RENDER_TESTS=1 $(PYTHON) -m unittest discover -s sim/washstand/tests -v
sim-generate:
	MUJOCO_GL=$${MUJOCO_GL:-egl} $(PYTHON) -m sim.washstand.generate --output "$(SIM_RUN)" --attempts $(SIM_ATTEMPTS)
