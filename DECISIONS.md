# Decisions

Dated one-paragraph entries for every deviation from the increment plan or
CLAUDE.md, and for every judgment call a future session must not silently
reverse. Newest first.

## 2026-07-19 — Pre-commit uses local `uv run` hooks, not remote mirrors

The ruff/mypy pre-commit hooks are `repo: local` entries running `uv run ruff`
/ `uv run mypy` so the hook versions are exactly the project's locked
versions. Remote mirror repos (e.g. ruff-pre-commit) pin their own tool
version and drift from `uv.lock`; two ruff versions disagreeing on formatting
is the failure mode this avoids. `end-of-file-fixer` and
`check-added-large-files` stay remote (pre-commit-hooks) — they have no
project-local counterpart.

## 2026-07-19 — BadmintonDB dates: week precision until overrides are sourced

BadmintonDB filenames encode only `YYYY-WW` (ISO week). The curated
`match_date_overrides.csv` will promote all 9 matches to day precision, each
row carrying a provenance URL. Until a row is sourced, the loader keeps
`date_precision="week"` with the ISO week's Monday as the date, and dedup
treats week-precision dates as non-exact (cannot collide on the exact-date
key). Do not invent day-precision dates without a URL.

## 2026-07-19 — Elo constants: K=40 provisional, K=20 after 10 matches

The master doc says "K=40 under 30 games"; the SFU platform (out of scope but
the doc's actual referent) uses 40→24 at 8 matches. This repo standardizes on
K=40 for a player's first 10 matches per format, K=20 after, scale 400,
initial 1500 — all in `configs/harness_v1.yaml`. Chosen as round, defensible
values; the config is the tuning surface, not code.

## 2026-07-19 — Logistic regression carries increment-1 claims, not XGBoost

At N≈100 matches, XGBoost cannot be defended as the headline model. The P4a/
P4b experiments report both, but conclusions are worded against LR (L2,
C=1.0). XGB is machinery rehearsed for club-scale data. Reversing this
requires a bootstrap-CI win for XGB over LR, not a point estimate.

## 2026-07-19 — Unknown-shot policy diverges from reference code on purpose

CoachAI's `preprocess_data.py` drops entire rallies containing a flawed or
unknown (未知球種) stroke, and contains a winner-leak bug in its
`getpoint_player` replacement — it is reference material only, never ported.
This repo's policy: unknown strokes are excluded from class-conditional rate
features but their rallies still count for rally-level features. Documented in
`configs/shot_labels_v1.yaml`, enforced in `features/tactical.py`, unit-tested.

## 2026-07-19 — Club ground truth is a fresh local SQLite; SFU platform out of scope

User decision during planning: the SFU badminton-platform (Supabase) holds
only test data and stays untouched; this repo owns a fresh `db/club.sqlite`
populated going forward. No real club matches exist yet, so all increment-1
model claims are made on the elite (ShuttleSet-family) timeline; club-side
P4a waits for a defensible volume (~150–200 matches).
