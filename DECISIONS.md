# Decisions

Dated one-paragraph entries for every deviation from the increment plan or
CLAUDE.md, and for every judgment call a future session must not silently
reverse. Newest first.

## 2026-07-19 — Feature state updates follow t_idx order, not strictly-earlier days

The increment plan says feature builders use "date < target.date — strictly
earlier calendar day, never same-day". The implementation instead updates all
accumulators (Elo, log5, form, h2h, tactical) match-by-match in t_idx order,
where t_idx sorts by (date, round_rank, match_uid) — so a same-day semi-final
result IS visible to the same-day final's features. This is causally sound
(the semi finished before the final started) and matches how the Elo/log5
baseline predictors accumulate inside the harness, keeping model and baseline
information sets identical. The residual approximation: same-day same-round
matches are ordered by match_uid, which is arbitrary. Reversing this (freezing
features at the day boundary) would also have to freeze the baselines to stay
comparable; do both together or neither.

## 2026-07-19 — M6 judgment calls and findings

(1) P4b headline (preregistered contrasts, elite 94-match timeline): B <= A —
records+tactical does not detectably beat records-only (lr_b - lr_a: +0.020
Brier full window, -0.017 scored, CIs straddle zero), and every model loses
to Elo/log5. Per the master plan this routes tactical/video features to the
explanation/reporting track, not prediction; lifts under ~0.04 Brier are
undetectable at this N, so the wording is "no detectable lift", never "no
lift". (2) The three XGB variants produced bit-identical predictions because
the preregistered min_child_weight=10 yields 150 trees with only 5 split
nodes at N~94 — XGB is effectively a base-rate predictor at this scale
(verified by dumping the booster). Kept as-is: retuning after seeing results
is the forking-paths failure preregistration exists to prevent. (3) Feature
denominators chosen where the plan was silent: backhand/aroundhead shares
use flag-carrying strokes; third-stroke attack rate conditions on own-serve
rallies reaching stroke 3; serve-conditional features gate on weighted serve
support. All pinned by hand-computed fixture tests. (4) Calibration ships as
tables in the report JSON; reliability plots deferred until someone reads
them (viz group stays optional).

## 2026-07-19 — M4 judgment calls

(1) The increment plan's exit example "gates reject 22-20" is wrong as a
matter of badminton law: 22-20 is a legal deuce win (20-20, then two clear).
The gate accepts 22-20/20-22 and the tests pin that; 21-20 and 22-19 remain
illegal. (2) Name similarity uses stdlib `difflib.SequenceMatcher.ratio()` at
0.85 on normalized names instead of a Levenshtein dependency — boring option,
zero deps, same intent. (3) Doubles matches are stored by the DB but
`as_timeline()` emits singles only: doubles prediction arrives via pair
composition in a later increment, and emitting half-designed pair rows would
poison the harness. (4) Same-day rematches log a warning and record anyway —
rematches are normal in a club night; blocking would corrupt ground truth.
(5) CSV import auto-creates unknown players (guard still applies) and
quarantines failing rows with reasons rather than aborting the file.

## 2026-07-19 — M2 judgment calls

(1) No BadmintonDB points/stroke loader was built: increment 1 uses BDB for
match results only and no test or plan requirement needs stroke rows; build it
when a requirement exists. (2) The events converter emits
`MatchOutcome.game_scores = None`: `roundscore_*` semantics (pre- vs
post-rally score) are unverified, and wrong game scores are worse than absent
ones — resolve semantics before populating. (3) The curated override table's
`winner` column is a mandatory cross-check against score-walk extraction (a
disagreement raises), not merely a fallback — stronger than the plan's
original design. (4) Unit tests may read `configs/*.yaml`: versioned config
artifacts are code, not data; the "no filesystem beyond tmp_path" rule is
about `data/` and speed. (5) `DISCIPLINE_SEEDS` carries a third seed
(Michelle LI, WS) because the real opponent graph has an isolated two-player
component {KOSETSKAYA, LI}; discipline-by-connectivity needs exactly one seed
per component. (6) The Sudirman Cup filename week (20) is wrong — the sourced
date 2019-05-25 is ISO week 21; the override row carries `week_mismatch_ok=yes`
and the loader enforces that flag for any out-of-week override.

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
