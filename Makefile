IMAGE ?= outreach-churn-effect
PYTHON ?= python3

.PHONY: build solve verify-fixtures clean

build:
	docker build -t $(IMAGE) task/environment

solve:
	$(PYTHON) task/solution/solve.py

verify-fixtures:
	$(PYTHON) scripts/verify_fixtures.py

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache
