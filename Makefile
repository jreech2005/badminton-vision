.PHONY: check test test-all extract p4a p4b

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy

test:
	uv run pytest

test-all:
	uv run pytest -m "unit or integration"

extract:
	sh scripts/extract_raw.sh

p4a:
	uv run bviz experiment p4a

p4b:
	uv run bviz experiment p4b --config configs/experiment_p4b_v1.yaml
