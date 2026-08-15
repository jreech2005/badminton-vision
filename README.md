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
uv run bviz timeline show --tail 5
uv run bviz run --predictors coinflip,elo,log5
uv run bviz report --run latest
uv run bviz experiment p4a
uv run bviz experiment p4b
uv run bviz records init && uv run bviz records add-player "Name"
```

## Increment-1 results (2026-07-19, elite timeline)

The global timeline holds 103 matches (ShuttleSet 44 + ShuttleSet22 50 after
8 cross-dataset duplicates + BadmintonDB 9), 2018-06-29 to 2022-10-30. All
numbers are walk-forward; the harness is verified by the coin-flip null
scoring per-match Brier exactly 0.2500 (and log loss ln 2) on all 103.

Mean Brier, full window / scored window (both players with 3+ prior matches):

| predictor | full (n=103) | scored (n=32) |
|---|---|---|
| coin flip | 0.2500 | 0.2500 |
| Elo (adaptive K) | 0.2341 | 0.2304 |
| log5 (Beta, decayed) | 0.2340 | 0.2261 |
| LR records-only | 0.2762 | 0.2776 |
| XGB records-only | 0.2581 | 0.2633 |

P4a verdict: the rating baselines keep the crown. Records-only models do not
beat Elo/log5 at this sample size; XGB is significantly worse than Elo on the
full window (paired bootstrap +0.024 Brier, 95% CI [+0.005, +0.042]). Elo and
log5 beat the coin flip on point estimate but the CIs still brush zero
(p about 0.07) — decisive H1-over-H0 rejection needs more matches.

P4b verdict (94-match elite subset, preregistered contrasts): B <= A. Adding
the 13 perfect-label tactical features to records features adds no detectable
lift (lr_b minus lr_a: +0.020 Brier full window, -0.017 scored, CIs straddle
zero); tactical-only is worst. Per the master plan this routes tactical and
future video features to the explanation/coaching track, and the prediction
product remains ratings-first. Caveats that bound these claims: lifts smaller
than ~0.04 Brier are undetectable at this N; the population is elite MS/WS
singles, not club play; and the preregistered XGB regularization
(min_child_weight=10) leaves it effectively a base-rate predictor at N~94
(150 trees, 5 split nodes total — see DECISIONS.md).

Club records: `db/club.sqlite` starts empty; the same harness runs unchanged
against it (proven on a synthetic fixture). Club-level P4a claims wait for a
defensible volume of real matches.
