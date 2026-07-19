# badminton-vision

Badminton match ingestion, walk-forward evaluation harness, and prediction
experiments. Increment 1 (no camera): dataset loaders with a leak firewall,
coin-flip-verified harness, Elo/log5 baselines, records-only P4a machinery,
and the ShuttleSet P4b rehearsal. See `badminton-vision-system.md` (master
plan) and `CLAUDE.md` (engineering contract).

## Setup

```sh
sh scripts/bootstrap.sh      # installs uv, pins Python 3.12, syncs deps, installs pre-commit
sh scripts/extract_raw.sh    # unzips dataset archives from ~/Downloads (override: BVIZ_ZIP_DIR)
```

## Commands

```sh
make check      # ruff format --check + ruff lint + mypy strict
make test       # unit tests (synthetic fixtures only)
make test-all   # unit + integration (reads data/raw)
uv run bviz extract-check
```
