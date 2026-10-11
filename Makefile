IMAGE ?= outreach-churn-effect

.PHONY: build solve clean

build:
	docker build -t $(IMAGE) task/environment

solve:
	python3 task/solution/solve.py

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache
