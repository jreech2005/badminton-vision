# The win/loss prediction pipeline — sources, methods, and tests

Reference document for Increment 1 ("the no-camera foundation"). It describes
where every number comes from: the raw archives, the ingest path that turns
them into a timeline, the exact arithmetic of every feature and model, the
evaluation protocol, and every test with the edge case it exists to catch.

Companion documents: `badminton-vision-system.md` (master plan),
`CLAUDE.md` (engineering contract), `DECISIONS.md` (dated judgment calls),
`FINDINGS.md` (the Increment-1 result report).

Everything below is read off the code at commit `bafa779` and the run
artifacts under `data/derived/runs/`.

---

## 1. Scope

**In scope:** predicting the winner of a singles badminton match from records
and (as a falsifiable experiment) from stroke-derived tactical features.

**Not in scope in this increment:** any computer vision. No frame is read, no
model is trained on video. `data/raw/tracknetv3` and `data/raw/racketdb` are
extracted and locked but unused — they are staged for a later increment. The
`events_v1` schema exists so CV code has a contract to write into, and one
converter proves the contract fits real data.

**Population:** elite men's and women's singles (BWF broadcast matches,
2018–2022). Nothing here is validated on club play.

---

## 2. Data sources

Five archives are extracted into `data/raw` (63 MB total), which is then
`chmod -R a-w`. Three feed the pipeline; two are staged for later.

| # | Source | Archive | Used for | Rows delivered |
|---|---|---|---|---|
| 1 | **ShuttleSet** | `ShuttleSet.zip` | matches + strokes | 44 matches, 104 set CSVs, 36,484 strokes |
| 2 | **ShuttleSet22** (CoachAI Challenge IJCAI-2023) | `ShuttleSet22_CoachAI_Challenge.zip` | matches + strokes | 58 matches, 140 set CSVs, 52,356 strokes |
| 3 | **BadmintonDB** | `BadmintonDB.zip` | match results only | 9 matches (Ginting–Momota) |
| 4 | TrackNetV3 | `TrackNetV3_code.zip` | staged, unused | — |
| 5 | RacketDB | `RacketDB_annotations.zip` | staged, unused | — |

### 2.1 What each source actually contains

**ShuttleSet / ShuttleSet22** share one schema. Per dataset:

- `set/match.csv` — one row per match: `id, video, tournament, round, year,
  month, day, set, duration, winner, loser, downcourt, url`. This file is
  **authoritative for identity and dates**. Video folder names are treated as
  opaque verbatim keys (they contain stray spaces and are not winner-first).
- `set/<video>/set1.csv, set2.csv, …` — one row per stroke, exactly 30
  columns, enforced as a contract (`STROKE_COLUMNS` in
  `ingest/shuttleset.py:20`). The columns are: `rally, ball_round, time,
  frame_num, roundscore_A, roundscore_B, player, server, type, aroundhead,
  backhand, hit_height, hit_area, hit_x, hit_y, landing_height, landing_area,
  landing_x, landing_y, lose_reason, win_reason, getpoint_player, flaw,
  player_location_area, player_location_x, player_location_y,
  opponent_location_area, opponent_location_x, opponent_location_y, db`.
- `set/homography.csv` — a 3×3 court homography per video, parsed and shape-
  validated, stored for the CV increment. Unused by any current feature.

**Critical property:** in these files `player == "A"` means *the match
winner*. Several columns are therefore winner-encoded and leak the outcome by
construction. This is the entire reason the leak firewall (§4) exists.

**BadmintonDB** is per-point JSON, one file per match. The filename encodes
the metadata: `YYYY-WW-<n><ROUND>-MS-<player1>-<player2>-<tournament>-1080p[-50fps].json`,
parsed by a strict regex (`ingest/badminton_db.py:23`). Each `PointN.PointInfo`
records the **pre-point** state for both players: `Score` (games won) and
`Point` (points in the current game).

### 2.2 Curated inputs (git-tracked, provenance-carrying)

**`data/curated/player_aliases.csv`** — 3 rows. ShuttleSet and ShuttleSet22
were verified to share one naming convention (19 shared names, zero variant
spellings), so the only aliases needed strip BadmintonDB's filename country
suffix and fix one casing variant:

```
Anthony Sinisuka GINTING (INA) -> Anthony Sinisuka GINTING
Kento MOMOTA (JPN)             -> Kento MOMOTA
Kento Momota (JPN)             -> Kento MOMOTA
```

The loader rejects duplicate aliases, self-referential aliases, and any table
where a canonical name is also used as an alias.

**`data/curated/match_date_overrides.csv`** — 9 rows, one per BadmintonDB
match. BadmintonDB filenames encode only an ISO week, so each row supplies a
day-precision date **plus a mandatory provenance URL** (BWF news, Kompas,
Xinhua, Wikipedia) and the sourced score. The `winner` column is not a
fallback — it is a **cross-check**: if the score-walk extraction disagrees
with the curated winner, ingest raises rather than picking one. One row
(`bdb:2019-20`, Sudirman Cup) carries `week_mismatch_ok=yes` because the
filename's week (20) is wrong; the sourced date 2019-05-25 is ISO week 21.

---

## 3. Data collection pipeline

```
data/raw (read-only)          data/curated (git-tracked)
  |                             |
  +-- extract_check ------------+
  |     counts + lockdown verified, else raise
  |
  +-- load_match_table (ss)  ---+-- aliases applied
  +-- load_match_table (ss22) --+
  +-- load_bdb_match_table -----+-- date overrides + winner cross-check
  |
  v
  111 match rows (44 + 58 + 9)
  |
  +-- dedup_matches            -> 8 twins dropped, near-miss audit emitted
  |
  v
  103 unique matches
  |
  +-- build_timeline           -> sort, t_idx, round_rank, discipline
  |
  v
  the global timeline (103 matches, 2018-06-29 .. 2022-10-30)
```

### Stage 0 — extraction and lockdown

`scripts/extract_raw.sh` unzips the five archives into `data/raw` and runs
`chmod -R a-w`. It is idempotent: an already-extracted dataset is skipped, and
re-extraction requires deleting the target by hand. `data/raw` is never
written by Python code.

`bviz extract-check` (`ingest/extract_check.py`) then verifies the tree
against counts pinned by direct archive inspection on 2026-07-19:

- ShuttleSet: exactly 44 match dirs, 104 `set*.csv`
- ShuttleSet22: exactly 58 match dirs, 140 `set*.csv`
- BadmintonDB: exactly 9 JSONs
- **zero writable paths** anywhere under `data/raw` (walked with `os.access(…, W_OK)`)

Any mismatch raises `DataContractError` listing every problem at once.

### Stage 1 — match tables

`load_match_table` reads `match.csv` and emits one frozen `MatchMeta` per row.
Gates applied per row:

- **Video directory must exist verbatim.** Names with stray spaces are not
  normalised — if the directory is not found byte-for-byte, that is a defect.
- **Dates are coerced through `_coerce_int`.** ShuttleSet22 stores integers as
  floats (`"2022.0"`); only exactly-integral values are accepted. `"2022.5"`
  raises. An impossible calendar date (month 13) raises.
- **Aliases applied** to winner and loser if a table is supplied.
- `match_uid` is namespaced: `ss:<id>`, `ss22:<id>`, `bdb:<year>-<week>`.

`load_strokes` concatenates a match's set CSVs, enforcing that (a) the column
tuple is exactly the 30-column contract, (b) the set files present are exactly
`set1.csv…setN.csv` for the N declared in `match.csv`. It stamps four
provenance columns (`source, match_uid, video, set_no`) that are never used as
features.

### Stage 2 — BadmintonDB winner extraction

For each JSON, the points are read in strict `Point1…PointN` order (a gap in
the numbering raises). The winner is derived from the **final recorded point's
pre-point state** by asking, for each player: *could this player end the match
by winning this point?* That is true when

```
would_win_game(own, opp):   after = own + 1
                            (after == 30 and opp == 29)  or
                            (after >= 21 and after - opp >= 2 and after <= 30)

and  Score + 1 == 2          # best-of-three, this game would be their second
```

If **exactly one** player satisfies both, that player is the winner. If zero
or two do (e.g. 29–29 in a decider, where either would end it), extraction
returns `None`.

Then:

| extraction | override present | result |
|---|---|---|
| unique winner | yes | override used; **disagreement raises** |
| unique winner | no | extraction used, `date_precision="week"` (ISO week Monday) |
| ambiguous | yes | override used |
| ambiguous | no | **raises** — never guessed |

All 9 rows currently have overrides, so all 9 are day-precision. Winner
counts: Momota 7, Ginting 2.

### Stage 3 — cross-dataset dedup

ShuttleSet and ShuttleSet22 overlap. `dedup_matches` keys on
**(unordered player pair, exact calendar day)** and keeps the row from the
higher-priority source: `shuttleset (0) > shuttleset22 (1) > badminton_db (2)`.

Three deliberate properties:

- **Only day-precision rows can collide.** A week-precision date is not an
  exact date, so it is excluded from the key entirely and always survives.
- **A group whose rows disagree on the winner raises.** A cross-source
  contradiction is a data defect, not something to resolve by priority.
- **Near-miss audit.** Same pair on *different* days within 3 days is emitted
  as a separate audit frame — a legitimate rematch or a dirty date, for a
  human to look at. It is never auto-resolved.

Result on real data: **8 duplicates dropped**, all from `shuttleset22`,
leaving **103 unique matches**.

### Stage 4 — the global timeline

`build_timeline` produces a **strict total order** over matches:

1. Reject null dates and duplicate `match_uid`s.
2. Map `round_name → round_rank` through a fixed table covering both
   vocabularies: `Group-Stage 0, R32 1, R16 2, Quarter-finals/QF 3,
   Semi-finals/SF 4, Finals/F 5`. **A round without a rank raises** — new
   vocabulary is a contract change, not something to default.
3. Sort by `(date, round_rank, match_uid)` and stamp `t_idx = 0…n-1`.
4. **Infer discipline by connected components** of the opponent graph. Singles
   players only ever meet within their own discipline, so the graph must split
   into exactly one component per seed. Seeds: `Kento MOMOTA → MS`,
   `An Se Young → WS`, `Michelle LI → WS`. The third seed exists because the
   real graph contains an isolated two-player component `{KOSETSKAYA, LI}`
   who played only each other (Denmark Open 2020) — connectivity alone cannot
   label them. A component with zero or two seeds raises (the usual cause
   would be an alias failure).
5. Cross-check that winner and loser always share a discipline.

**Output:** 103 matches, 2018-06-29 to 2022-10-30, disciplines `{MS, WS}`,
`t_idx` contiguous `0…102`.

The **elite timeline** (used by P4b) is this minus the 9 BadmintonDB rows,
re-sorted and re-indexed: **94 matches**, every one of which has stroke data.

### Stage 5 — club records (the second, parallel intake)

`db/club.sqlite` is a separate ground-truth store for matches recorded going
forward. It currently holds **no real matches**. It is documented in §9; it
emits the same match-table contract, so the identical harness runs on it
unchanged.

---

## 4. The leak firewall

This is the non-negotiable module boundary. Feature code never touches raw
stroke frames. It consumes exactly two views, both built in
`ingest/preoutcome.py`.

### 4.1 Deterministic side assignment

Every match is symmetrised so no model can learn "the winner is always
listed first":

```python
digest = blake2b(f"{seed}:{match_uid}".encode(), digest_size=8).digest()
if int.from_bytes(digest, "big") & 1:  return winner, loser, 1
else:                                   return loser, winner, 0
```

`y = 1` iff `side0` won. `blake2b` is used rather than Python's built-in
`hash()` because `hash()` is salted per process — using it would silently
break bit-identical reruns. This is a stated non-negotiable in `CLAUDE.md`.

### 4.2 `HistoricalView` — for strictly earlier matches

Full 30 columns, with `player` and `getpoint_player` re-keyed from `A`/`B` to
real player names. Legal **only** for matches strictly earlier than the
prediction target; the caller (the chronological feature loop) is responsible
for that ordering. Any `player` value outside `{A, B}` raises.

### 4.3 `PreOutcomeView` — for the target match

The target match as a camera would see it. `A`/`B` are mapped to
`side0`/`side1` per the assignment, and **nine columns are stripped**:

| stripped column | why |
|---|---|
| `roundscore_A`, `roundscore_B` | winner-encoded column *names* leak the outcome |
| `player` | `A` = match winner by construction; re-keyed to `side`, then dropped |
| `getpoint_player` | literal rally outcome, winner-encoded |
| `win_reason`, `lose_reason` | post-rally outcome annotations |
| `flaw` | post-hoc annotator QA flag, winner-referenced |
| `server` | undocumented domain `{1,2,3}`; serve identity is recoverable as the `ball_round == 1` striker |
| `db` | undocumented annotation artifact |

Adding or removing a column from this list requires updating the
justification table and `tests/test_preoutcome.py` **in the same commit**.

Both views reject any frame that is not a loader-shaped frame (30 contract
columns + 4 provenance columns), so a hand-assembled or partially-processed
frame cannot sneak past.

---

## 5. Features

### 5.1 Records features (Tier 0–2) — 12 columns

Built by `features/match_row.py` in **one chronological pass** over the
timeline. For each match, the feature row is emitted from accumulator state
*before* that match's outcome touches any accumulator; the outcome is revealed
to the accumulators only afterwards. All pairwise features are
`side0 − side1` differences, so they negate under a side swap.

| column | definition |
|---|---|
| `elo_diff` | `R(side0) − R(side1)` in that discipline's rating pool |
| `log5_logit` | `log(p / (1−p))` where `p` = log5 prediction — a logit so it negates on swap |
| `h2h_net_wins` | prior wins of side0 over side1, minus the reverse |
| `n_matches_diff` | difference in career matches seen so far |
| `days_since_last_diff` | difference in days since each player's last match |
| `winrate5_diff` | difference in Beta(2,2)-shrunk win rate over the last ≤5 results |
| `winrate10_diff` | same over the last ≤10 |
| `streak_diff` | difference in signed consecutive-result streak |
| `game_win_share_diff` | difference in Beta(2,2)-shrunk games-won share |
| `opp_quality_diff` | difference in mean Elo of the last 10 opponents (at the time each was played) |
| `round_rank` | 0–5, **swap-invariant** |
| `is_ws` | 1.0 for women's singles, **swap-invariant** |

Shrunk rate: `(wins + 2) / (games + 4)`.

Games bookkeeping: the winner is credited 2 games won, the loser `n_sets − 2`;
both are credited `n_sets` games played.

Streak: `+1` continues a winning run and resets a losing one to `+1`;
symmetrically for losses. A player who lost, then won, has streak `+1`.

**Missingness:** a player with zero prior matches yields `NaN` for
`days_since_last`, `winrate5`, `winrate10`, `streak`, `game_win_share`, and
`opp_quality`. `elo_diff`, `log5_logit`, `h2h_net_wins`, and `n_matches_diff`
are always defined (0.0 at first meeting). NaNs are passed to the models
intact — LR imputes them and adds indicator columns, XGB handles them
natively. Absence is never masked with a default.

### 5.2 Tactical features — 13 columns

Built by `features/tactical.py` from `HistoricalView` frames only. These are
the P4b experiment's subject: **perfect human labels**, used as an upper bound
on what noisy CV-derived features could ever provide.

**Per-match counters** are extracted rally by rally (`match_counters`). Per
rally: the server is the `ball_round == 1` striker; the rally winner is the
last row's `getpoint_player`.

Counters, per player:

- `strokes_total`, `classified_strokes` (strokes with a known label)
- `smash_strokes` (`smash`, `wrist_smash`), `smash_kills` (rally winner struck
  the last stroke and it was a smash class)
- `net_strokes` (`shot_group == net`), `net_last_losses` (lost the rally on a
  net stroke)
- `last_stroke_losses` (struck the final stroke and lost the rally)
- `serves`, `serve_faults` (a 1-stroke rally lost by the server),
  `short_serves`, `long_serves`
- `attack_strokes` (`shot_group == attack`)
- `flagged_strokes` (strokes where both `backhand` and `aroundhead` are
  non-null), `backhand_strokes`, `aroundhead_strokes`
- `serve_rallies` / `serve_rally_wins`, `receive_rallies` / `receive_rally_wins`
- `long_played` / `long_won` (rally length ≥ 9), `short_played` / `short_won`
  (length ≤ 4)
- `third_stroke_chances` / `third_stroke_attacks` (own-serve rallies reaching
  stroke 3 where the server struck it; attack if `shot_group == attack`)
- `rally_lengths` — appended for both players, so this is a match-level
  quantity; it differs *between* players only because they played different
  matches

**Aggregation to a feature.** History entries are exponentially decayed with
half-life 300 days:

```
w_i = 0.5 ** (days_ago_i / 300)
```

Each rate is a Beta-style shrunk ratio toward the **pooled elite prior** for
that feature (pooled numerator / pooled denominator over every player-match
observed so far):

```
rate = ( Σ w_i · num_i  +  s · prior ) / ( Σ w_i · den_i  +  s )
```

with `s = 30` for stroke-based rates and `s = 10` for serve-conditional ones.

**Support gates (NaN policy):**

- Weighted `strokes_total < 150` → **every** tactical feature is `NaN`.
- Weighted `serves < 30` → the four serve-conditional features are `NaN`:
  `serve_fault_rate`, `short_serve_share`, `serve_advantage`,
  `third_stroke_attack_rate`.

Two features are not plain shrunk ratios:

- `median_rally_len` — the **weight-weighted median** rally length: pairs
  `(length, w)` sorted by length, take the first length at which cumulative
  weight reaches half the total.
- `serve_advantage` — `serve_rally_wins/serve_rallies −
  receive_rally_wins/receive_rallies`, computed only under serve support and
  only when both denominators are positive. Not shrunk.

The 13 emitted columns are the `_diff` of each of these between side0 and
side1.

**Unknown-shot policy** (`configs/shot_labels_v1.yaml`, deliberately diverging
from CoachAI's reference `preprocess_data.py`, which drops whole rallies and
contains a winner-leak bug in its `getpoint_player` handling):

> An unknown stroke (`未知球種`) is **excluded from class-conditional rate
> features** but **its rally still counts for rally-level features.**

So an unknown stroke increments `strokes_total` and the rally's length, but
not `classified_strokes`, and an unknown final stroke can never be a smash
kill.

### 5.3 The label map

`configs/shot_labels_v1.yaml` maps every raw Chinese stroke token to exactly
one of **18 canonical classes**, each in a group
(`serve, net, attack, transition, neutral, defense`), plus **1 unknown token**.

The map is versioned and additive-only. A token that is neither mapped nor
listed as unknown raises `LabelMapError` — new vocabulary demands a new
label-map version, never a silent default.

One wart is handled explicitly: passive drop is spelled `過度切球` in the data
(2,443 strokes) but `過渡切球` in the READMEs. Both tokens map to
`passive_drop` so either survives a re-annotation.

Real-data coverage is pinned: **1,407 unknown strokes in ShuttleSet, 1,078 in
ShuttleSet22**, everything else mapped.

---

## 6. Models

Five predictor types. Three baselines, two refit ML models. All share one
interface:

```python
class Predictor(Protocol):
    name: str
    def predict(self, ctx: MatchContext) -> float: ...
    def observe(self, result: MatchResult) -> None: ...
    # predict() is ALWAYS called before observe() for the same match
```

### 6.1 `coinflip` — H₀ and the harness gate

Returns exactly `0.5` for every match. Because `0.5**2 == 0.25` is float-exact
in binary, **every per-match Brier must be exactly 0.25** and every log loss
exactly `ln 2 ≈ 0.6931`. Any deviation means the harness is broken. This is
asserted with `==`, no tolerance.

### 6.2 `elo` — adaptive-K Elo

Rating pools are keyed by `(player, discipline)`, so MS and WS never mix.
Unseen players start at 1500.

```
E(p0 beats p1) = 1 / (1 + 10 ** (-(R0 - R1) / 400))

on a result:  surprise = 1 - E(winner beats loser)
              R_winner += K_winner * surprise
              R_loser  -= K_loser  * surprise
```

Each player's own K is used: `K = 40` while they have fewer than 10 matches in
that pool, `20` after. (`DECISIONS.md`: the master doc says "K=40 under 30
games" and the SFU platform uses 40→24 at 8; this repo standardises on round,
defensible values in config, not code.)

### 6.3 `log5` — Beta-shrunk, time-decayed win rates

Per-player decayed win/game sums. Reading as of date `d`:

```
factor = 0.5 ** (days_since_last_update / 540)
r = (wins*factor + 5) / (games*factor + 10)          # Beta(5,5) shrinkage
```

Combined through the log5 formula:

```
P(p0 beats p1) = r0(1-r1) / [ r0(1-r1) + r1(1-r0) ]
```

Two unseen players give exactly 0.5. Against a 0.5 opponent, log5 returns the
player's own rate.

### 6.4 `lr` — L2 logistic regression (the claim-bearing model)

An sklearn `Pipeline`:

```
SimpleImputer(strategy="median", add_indicator=True)
  -> StandardScaler()
  -> LogisticRegression(C=1.0, max_iter=1000)
```

`add_indicator=True` appends binary missingness columns, so "this player has
no history" is itself a signal rather than being erased by the median.

### 6.5 `xgb` — heavily regularised XGBoost (rehearsal machinery)

```
max_depth=2, n_estimators=150, learning_rate=0.05, min_child_weight=10,
subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
random_state=7, n_jobs=1, tree_method="hist", eval_metric="logloss"
```

NaNs pass through natively (no imputation). **Monotone-increasing constraints**
are applied to `elo_diff`, `log5_logit`, `h2h_net_wins`, `winrate10_diff`: a
larger side0 advantage can never lower `P(side0)`. Naming a constraint on a
feature that is not in the feature set raises — except for the tactical-only
set `c`, which legitimately has no `elo_diff`, where `strict_monotone=False`
drops the absent names.

At N ≈ 94–103 this regularisation collapses XGB to approximately a base-rate
predictor: 150 trees with **5 split nodes in total** (verified by dumping the
booster). It is kept unchanged because retuning after seeing results is
precisely the forking-paths failure preregistration prevents. `DECISIONS.md`
states LR carries all Increment-1 claims; reversing that requires a bootstrap
CI win for XGB over LR, not a point estimate.

### 6.6 How the ML models are trained: refit-per-match

`RefitModelPredictor` is **not** a train/test split. Before *every single
match*, a fresh model is fitted on **all revealed history to that point**:

```
predict(ctx):
    if len(labels) < 10 or fewer than 2 distinct classes:  return 0.5
    fit a NEW model on every observed (features, y) so far
    p = model.predict_proba(this match's feature row)[0, 1]
    return clip(p, 1e-6, 1 - 1e-6)

observe(result):
    append (result.ctx.features, result.y) to the training pool
```

Training rows enter **only** through `observe()`. The predictor never receives
the feature table, so a future row physically cannot reach a fit. Requesting a
prediction with no features raises rather than defaulting.

`min_train = 10` and the two-class requirement mean the first several matches
score exactly 0.5 — the honest answer with no history, and it costs the model
0.25 Brier per match on those.

---

## 7. The evaluation harness

`eval/harness.py` walks the timeline exactly once, in strict `t_idx` order.
Per match:

1. Assign sides deterministically → `(side0, side1, y)`.
2. Build `MatchContext` — everything legally visible *before* the match:
   uid, `t_idx`, date, discipline, both sides, both prior-match counts, and
   the feature row. It carries no outcome field at all.
3. **Every predictor predicts.** Each `p` is validated `0 < p < 1`; a
   violation raises.
4. **Only then** every predictor observes `MatchResult(ctx, y)`.
5. Increment both players' prior-match counts.

Structural guards: duplicate predictor names raise; a timeline whose `t_idx`
is not contiguous from 0 raises.

### Windows

- **Full window** — every match (n=103 for P4a, n=94 for P4b).
- **Scored window** — matches where **both** players have ≥ 3 prior matches
  (`scored_min_history: 3`). This is where models have any right to work.
  n=32 (P4a), n=25 (P4b).

### Metrics

| metric | definition |
|---|---|
| Brier | `(p − y)²` per match |
| log loss | `−[y·ln p + (1−y)·ln(1−p)]`, clipped to `[1e-15, 1−1e-15]` so `p ∈ {0,1}` is catastrophic (≈34.5) but finite |
| accuracy | directional, with **half credit at exactly 0.5** — so the coin flip scores 0.5, not 0 or 1 |
| calibration | 5 buckets (small N wants few), predicted mean vs empirical rate |

### Uncertainty: paired cluster bootstrap

The only uncertainty currency in the design. **The match is the cluster
unit** — not the stroke, not the rally.

```
pivot per-match Brier by predictor
diffs = brier[A] - brier[B]                    # paired, per match
resample diffs with replacement, 10,000 times, rng seed 7
CI    = 2.5th and 97.5th percentiles of the resampled means
p     = 2 * min( P(mean <= 0), P(mean >= 0) ), capped at 1
```

Sign convention: **positive means the first predictor is worse** (higher
Brier). Missing overlap between two predictors raises.

### Provenance: no manifest, no run

Every run writes `manifest.json` containing: git SHA, config version, SHA-256
of the serialised config, seed, SHA-256 of the timeline
(`match_uid|date|winner|loser` per row), match count, predictor names, exact
versions of pandas/numpy/scipy/scikit-learn/xgboost/pydantic, any extra
version strings (feature version, label-map version, experiment version), and
a creation timestamp. Alongside it, `predictions.parquet` holds the full
per-match prediction log. A missing git identity is recorded as `"unknown"`
rather than being fatal — the one deliberate non-fatal path.

---

## 8. The experiments

### 8.1 P4a — do records-only models beat the rating baselines?

Timeline: global, 103 matches. Predictors: `coinflip, elo, log5, lr, xgb`,
the models on the 12 records features.

Preregistered contrasts (`experiments/p4a_records.py:22`), challenger vs
champion: `lr–elo, lr–log5, xgb–elo, xgb–lr, elo–coinflip, log5–coinflip`.

**Results** (`data/derived/runs/20260720-010212-p4a/`):

| predictor | Brier full (n=103) | log loss | acc | Brier scored (n=32) | log loss | acc |
|---|---|---|---|---|---|---|
| coinflip | 0.2500 | 0.6931 | 0.500 | 0.2500 | 0.6931 | 0.500 |
| elo | 0.2341 | 0.6610 | 0.587 | 0.2304 | 0.6546 | 0.625 |
| log5 | 0.2340 | 0.6607 | 0.607 | 0.2261 | 0.6463 | 0.656 |
| lr | 0.2762 | 0.8470 | 0.544 | 0.2776 | 0.8660 | 0.594 |
| xgb | 0.2581 | 0.7094 | 0.461 | 0.2633 | 0.7201 | 0.438 |

Contrasts, full window (positive = first predictor worse):

| contrast | mean ΔBrier | 95% CI | p |
|---|---|---|---|
| xgb − elo | +0.0239 | [+0.0052, +0.0416] | 0.011 |
| lr − elo | +0.0421 | [−0.0045, +0.0928] | 0.078 |
| lr − log5 | +0.0423 | [−0.0053, +0.0941] | 0.081 |
| xgb − lr | −0.0182 | [−0.0747, +0.0337] | 0.519 |
| elo − coinflip | −0.0159 | [−0.0323, +0.0013] | 0.068 |
| log5 − coinflip | −0.0160 | [−0.0331, +0.0016] | 0.077 |

**Verdict:** the rating baselines keep the crown. Records-only models do not
beat Elo/log5 at this sample size, and XGB is significantly worse than Elo on
the full window. Elo and log5 beat the coin flip on point estimate but the CIs
still brush zero (p ≈ 0.07) — decisive H₁-over-H₀ rejection needs more matches.

### 8.2 P4b — do perfect-label tactical features add lift?

Timeline: elite, 94 matches (every row has stroke data — a missing stroke
frame raises before anything runs). Three preregistered feature sets:

- **A** = 12 records features
- **B** = A + 13 tactical features (25 total)
- **C** = 13 tactical features only

Nine predictors: the 3 baselines + `lr_a/b/c` + `xgb_a/b/c`.

Preregistered contrasts (`configs/experiment_p4b_v1.yaml`), headline first:
`lr_b–lr_a` (does tactical add lift over records?), `lr_c–lr_a`, `lr_a–elo`,
`lr_b–elo`, `xgb_b–xgb_a`.

**Results** (`data/derived/runs/20260720-010536-p4b/`):

| predictor | Brier full (n=94) | Brier scored (n=25) |
|---|---|---|
| coinflip | 0.2500 | 0.2500 |
| elo | 0.2359 | 0.2420 |
| log5 | 0.2351 | 0.2372 |
| lr_a (records) | 0.2691 | 0.3191 |
| lr_b (records+tactical) | 0.2887 | 0.3021 |
| lr_c (tactical only) | 0.3070 | 0.3243 |
| xgb_a / xgb_b / xgb_c | 0.2604 | 0.2533 |

Contrasts:

| contrast | full: ΔBrier [CI] p | scored: ΔBrier [CI] p |
|---|---|---|
| **lr_b − lr_a** | **+0.0195 [−0.0190, +0.0587] 0.318** | **−0.0169 [−0.0899, +0.0414] 0.651** |
| lr_c − lr_a | +0.0379 [−0.0172, +0.0931] 0.173 | +0.0052 [−0.0819, +0.0835] 0.869 |
| lr_a − elo | +0.0332 [−0.0156, +0.0851] 0.190 | +0.0771 [−0.0242, +0.1860] 0.143 |
| lr_b − elo | +0.0527 [−0.0045, +0.1139] 0.073 | +0.0601 [−0.0468, +0.1784] 0.298 |
| xgb_b − xgb_a | 0.0000 [0.0000, 0.0000] 1.000 | 0.0000 exactly |

**Verdict: B ≤ A.** Adding 13 perfect-label tactical features to records
features produces **no detectable lift** — the headline CI straddles zero in
both directions. Tactical-only is worst. Every model variant loses to
Elo/log5.

The `xgb_b − xgb_a` contrast being *exactly* zero is not a bug: the three XGB
variants produced bit-identical predictions, because `min_child_weight=10`
leaves the booster with 5 split nodes at N≈94, so it ignores its features
entirely.

Per the master plan's P4b exit rule, this routes tactical — and future video —
features to the **explanation/coaching track**, and the prediction product
remains ratings-first. This is the designed outcome of a falsifiable
experiment, not a failure.

### 8.3 Hypothesis ladder status

| | claim | status |
|---|---|---|
| H₀ | coin flip | verified by construction — the harness gate |
| H₁ | ratings carry signal | favoured on point estimate, not decisive (p ≈ 0.07) |
| H₂ | records-only ML beats ratings | rejected at this N; XGB significantly worse |
| H₃ | tactical features add lift | no detectable lift; routed to explanation |

### 8.4 Caveats that bound every number above

- **Population** — elite MS/WS singles from broadcast annotation. Club play is
  structurally different (unforced-error-dominated rallies). Nothing here
  transfers to club claims.
- **Power** — at 94–103 matches, paired-Brier lifts smaller than ~0.04 are
  undetectable. The wording is always "no detectable lift", never "no lift".
  Scored windows are small (n=32, n=25).
- **XGB is not meaningfully evaluated** — it is near-base-rate at this N. It is
  machinery rehearsed for club-scale data.
- **Ordering approximation** — accumulators update in `t_idx` order, so a
  same-day semifinal result *is* visible to that day's final. Causally sound,
  and identical for models and baselines, so contrasts are unaffected. The
  residual: same-day same-round matches are ordered by `match_uid`, which is
  arbitrary. See `DECISIONS.md`.
- **Effective sample size is the match count**, not the stroke count. All
  uncertainty is cluster-aware.

---

## 9. The club records path

`records/club_db.py` — SQLite ground truth behind rulebook gates. Three
tables (`players`, `matches`, `match_games`) with SQL constraints as a
backstop; the authoritative gates are in Python.

**Quarantine principle: a record that fails a gate is rejected with a reason,
never silently fixed.**

Gates:

- **Legal game score.** `21-x` with `x ≤ 19`; or `22..30` winning by exactly
  2; or `30-29`. Note `22-20` **is legal** (deuce at 20-20, then two clear) —
  the increment plan's example claiming otherwise was wrong as a matter of
  badminton law, and `DECISIONS.md` records the correction. `21-20` and
  `22-19` remain illegal.
- **Match structure.** 2 or 3 games; the winner must take exactly 2 **and**
  must win the final game.
- **Roster.** Singles needs 1 player per side, doubles 2. No player may appear
  twice (checked case-insensitively, so "Viraj" vs "viraj" is caught).
- **Dates.** Not in the future, not before 2020-01-01.
- **Unknown players are rejected**, never auto-created by `add_match`.
- **Name similarity guard.** `difflib.SequenceMatcher.ratio() ≥ 0.85` on
  normalised names rejects near-twins ("Samuel Lee" vs "Samuel Le") unless
  `force_new` is passed. Chosen over a Levenshtein dependency: stdlib, zero
  deps, same intent.
- **Same-day rematch** logs a warning and records anyway — rematches are normal
  on a club night, and blocking would corrupt ground truth.
- **`import_csv`** auto-creates unknown players (the similarity guard still
  applies) and quarantines failing rows with their reason and CSV line number,
  rather than aborting the file.
- **`revalidate()`** re-runs every gate over stored rows, so an external SQL
  edit surfaces as a violation.

`as_timeline()` emits the same match-table contract as the elite loaders, with
`discipline = "singles"`. **Doubles rows are stored but not emitted** —
doubles prediction needs pair-composition modelling, and emitting
half-designed pair rows would poison the harness.

CLI: `bviz records init | add-player | add-match | import-csv | validate`.
Gate failures surface as exit code 1, not a traceback.

---

## 10. The `events_v1` contract

`schemas/events_v1.py` — the interface future CV work writes to and analytics
reads from. Pydantic v2, frozen, additive-only; a breaking change becomes
`events_v2` with a new version string.

Structure: `MatchEventFile` → `RallyEvent[]` → `ShotEvent[]`, plus an optional
heavy `FrameObservation[]` channel (shuttle position, poses, confidence) that
CV increments will populate, and an optional `homography`.

Two design decisions carry the leak discipline into the schema:

- **Sides are symmetrised** (`side0`/`side1` per `assign_sides`), never
  winner-first.
- **Outcome lives only in `MatchEventFile.outcome`**, so the observation
  stream is outcome-free. `RallyEvent` deliberately carries no winner and no
  reason.

Every datum carries a `SourceKind` (`human_annotation`, `cv_tracknet`,
`cv_pose`, `cv_fused`, `manual_entry`) so human and CV provenance never blur.
`shot_class` is nullable — `None` means the label was unknown at annotation
time — and always accompanied by `shot_class_version`.

`ingest/events_convert.py` converts one real ShuttleSet match into this
contract, building the observation stream from the **PreOutcomeView** and the
outcome channel from the match record — never the other way around. It exists
to prove the contract fits real data before any CV code depends on it.

`MatchOutcome.game_scores` is deliberately left `None`: the pre- vs post-rally
semantics of `roundscore_*` are unverified, and wrong game scores are worse
than absent ones.

---

## 11. Determinism and reproducibility

- All randomness flows from config seeds through `numpy.random.Generator`.
  Seed 7 for the harness, 7 for the bootstrap, 7 for XGBoost.
- Side assignment uses `blake2b`, never built-in `hash()` (salted per process).
- Bootstrap uses `np.random.default_rng(seed)`, so identical inputs give
  bit-identical CIs (pinned by a test).
- Every constant a claim depends on lives in `configs/*.yaml`, not in code:
  `harness_v1.yaml` (Elo, log5, scored threshold, buckets, bootstrap),
  `models_v1.yaml` (LR/XGB hyperparameters, monotone constraints),
  `experiment_p4b_v1.yaml` (tactical constants and preregistered contrasts),
  `shot_labels_v1.yaml` (label contract).
- Re-running any experiment with the same config is bit-identical. The two
  P4a runs and two P4b runs on disk (2026-07-19 and 2026-07-20) produce
  identical numbers to every printed digit — this is the reproducibility check.

---

## 12. Tests

**136 tests across 15 files.** `make test` runs 128; `make test-all` runs all
136. Markers: `unit` (no filesystem beyond `tmp_path`, synthetic fixtures
only) and `integration` (reads `data/raw`).

| file | n | what it pins |
|---|---|---|
| `test_club_db.py` | 31 | rulebook gates, name guard, quarantine, CLI |
| `test_loader_shuttleset.py` | 11 | loader contract, real counts |
| `test_dedup_timeline.py` | 11 | twin detection, ordering, discipline inference |
| `test_tactical.py` | 10 | tactical counters, decay, gates, invariances |
| `test_match_row.py` | 10 | prefix invariance, antisymmetry, model fallback |
| `test_ratings.py` | 9 | Elo and log5 arithmetic |
| `test_preoutcome.py` | 9 | the leak firewall |
| `test_badminton_db.py` | 9 | filename contract, winner extraction, overrides |
| `test_labels.py` | 7 | label-map contract |
| `test_harness_coinflip.py` | 7 | the coin-flip gate, predict-before-reveal |
| `test_metrics.py` | 6 | scoring rules |
| `test_bootstrap.py` | 5 | bootstrap behaviour |
| `test_extract_check.py` | 4 | raw counts and lockdown |
| `test_events_schema.py` | 4 | contract round-trip and freshness |
| `test_smoke.py` | 3 | package layout, error taxonomy, CLI wiring |

Unit tests use tiny synthetic fixtures in `tests/conftest.py` — never real
data. The main builders are `make_loader_frame()` (a 30-column stroke frame),
`write_mini_dataset()` (a two-match ShuttleSet-shaped tree), and
`make_synthetic_timeline(n, strong_period)` (four MS players where player S
wins every match except each `strong_period`-th — a deterministic dominant
player, used as a signal-detection floor).

### 12.1 The named policy tests

Per `CLAUDE.md`, every policy has a named test — if a policy has no test, it
is not a policy yet.

**Leak firewall** (`test_preoutcome.py`)

| test | edge case |
|---|---|
| `test_strip_list_is_exact` | output columns are **exactly** `side + kept + provenance` — catches both a forgotten strip and an accidental extra column |
| `test_side_rekey_follows_assignment` | `A` maps to `side0` or `side1` depending on assignment, both directions checked |
| `test_pre_outcome_view_rejects_unknown_side0` | `side0` that is neither winner nor loser raises |
| `test_views_reject_non_ab_player_values` | a `player` value of `"C"` raises in **both** views |
| `test_views_reject_non_loader_frames` | a frame missing `set_no` raises — a hand-assembled frame cannot sneak in |
| `test_historical_view_rekeys_players_to_names` | `A`/`B` become real names; `getpoint_player` too |
| `test_assign_sides_deterministic` | same inputs, same output; `y` agrees with which side is the winner |
| `test_assign_sides_is_balanced_and_seed_sensitive` | over 200 uids the flip is fair (60 < sum < 140) **and** two seeds disagree — catches a constant or seed-ignoring hash |
| `test_assignment_flip_flips_label_and_sides` | flipping assignment flips both the side labels and `y` together |

**Feature correctness** (`test_match_row.py`)

| test | edge case |
|---|---|
| `test_prefix_invariance` | the row for match *k* is **identical** whether the timeline has 8 or 24 matches — this is the test that proves no future information reaches a feature. Frame-level equality, not spot checks. |
| `test_antisymmetry_under_seed_flip` | for every match whose assignment flipped between seed 1 and seed 2, **every** diff feature negates exactly and `y` flips, while `round_rank`/`is_ws` stay put. Also asserts some rows flipped and some did not, so the test can actually bite. NaN maps to NaN. |
| `test_first_meeting_features_are_neutral` | at t=0, `elo_diff`, `log5_logit`, `h2h`, `n_matches_diff` are 0.0 and `winrate10_diff` is NaN — absence is NaN, not 0 |
| `test_h2h_accumulates_with_sign` | by the third S-vs-P1 meeting, h2h is ±2 depending on assignment — sign is tied to side, not to the player |
| `test_streak_resets_on_direction_change` | hand-traced: S lost match 9, won match 10, so a correct reset gives streak +1 (not 0, not −1+1 bookkeeping); P3 is on −3; the diff must be exactly 4 |
| `test_feature_slice_excludes_outcome_and_identity` | the harness-facing slice contains exactly `FEATURE_COLUMNS` — no `y`, no `side0` |
| `test_model_fallback_before_min_train` | the first 10 predictions are exactly 0.5 |
| `test_lr_learns_on_biased_synthetic` | on a deterministic dominant player, LR beats 0.25 on the scored window — a floor: the model must at least detect a signal that is definitely there |
| `test_monotone_constraint_rejects_unknown_feature` | a constraint naming a feature not in the set raises |

**Harness** (`test_harness_coinflip.py`)

| test | edge case |
|---|---|
| `test_coinflip_brier_exact` | per-match, mean, and summary Brier all `== 0.25` — **no tolerance**, since 0.5² is float-exact. This is *the* gate. |
| `test_predict_precedes_reveal` | a probe predictor records how many outcomes it had seen at each `predict()` call; the sequence must be exactly `0,1,2,…,n-1`. Catches off-by-one reveal ordering. |
| `test_out_of_bounds_prediction_raises` | `p = 1.5` raises |
| `test_duplicate_predictor_names_raise` | two `CoinFlip()`s raise before any run |
| `test_run_writes_parquet_and_manifest` | parquet has the right row count; manifest carries the required keys and equals the in-memory manifest |
| `test_elo_and_log5_beat_coinflip_on_biased_synthetic` | signal-detection floor on a 90%-dominant player — a sanity gate, explicitly not a statistical claim |

**Metrics** (`test_metrics.py`)

| test | edge case |
|---|---|
| `test_brier_at_half_is_float_exact_quarter` | `== 0.25` exactly, for both labels |
| `test_brier_hand_computed` | `(0.9−1)² = 0.01` |
| `test_log_loss_is_clipped_finite` | `p ∈ {0,1}` on the wrong label gives a finite score > 30 (`ln 1e-15 ≈ −34.5`), not `inf` |
| `test_accuracy_gives_half_credit_at_exactly_half` | `p = 0.5` scores half a point; hand-computed `5/6` |
| `test_calibration_table_buckets` | bucket edges are half-open (`0.05` and `0.15` share `[0, 0.2)`); 5 rows always emitted even when empty |
| `test_summarize_groups_by_predictor` | grouping and per-group `n` |

**Bootstrap** (`test_bootstrap.py`)

| test | edge case |
|---|---|
| `test_constant_positive_diff_excludes_zero` | a constant +0.05 gives a degenerate CI `[0.05, 0.05]` and `p = 0.0` — the resampling arithmetic is right |
| `test_zero_diffs_have_p_value_one` | an exact null gives `p = 1.0`, not 0 (both tails are 1.0, `2·min` capped at 1) |
| `test_symmetric_diffs_are_not_significant` | `±0.1` alternating straddles zero with `p > 0.2` |
| `test_same_seed_is_bit_identical` | two runs with seed 11 give an identical `BootResult` |
| `test_missing_predictor_raises` | a name absent from the log raises, never silently drops |

**Tactical features** (`test_tactical.py`)

| test | edge case |
|---|---|
| `test_match_counters_hand_computed` | a hand-built 2-rally match; **19 separate counter assertions** across both players — smash kills, serve/receive rally attribution, third-stroke chance, last-stroke loss, net-error attribution, rally lengths |
| `test_unknown_strokes_leave_class_rates_but_keep_rallies` | the unknown-shot policy, exactly: `strokes_total` counts it, `classified_strokes` does not, the rally still counts, and an unknown final stroke is not a smash kill |
| `test_single_stroke_rally_is_a_serve_fault` | a 1-stroke rally lost by the server is a serve fault, and the receiver is credited the rally |
| `test_state_supports_gate_then_decays_below_support` | hand-computed shrinkage against the pooled prior (`(1 + 1·0.2)/(3 + 1)`), the weighted median, **and** the decay gate: two half-lives later the same evidence weighs 0.75 < min support, so every feature goes NaN. Support is a moving target, not a one-time check. |
| `test_unseen_player_is_all_nan` | all 13 features NaN, and the count is asserted so a silently dropped feature fails |
| `test_build_tactical_rows_prefix_invariance` | same frame-equality proof as records features |
| `test_build_tactical_rows_antisymmetry` | every tactical diff negates on a side swap |
| `test_first_row_has_no_tactical_information` | at t=0 all 13 are NaN |
| `test_views_are_the_only_stroke_input` | passing a **raw loader frame** (players still `A`/`B`) to `match_counters` raises — the firewall is the only door in |

**Label map** (`test_labels.py`)

| test | edge case |
|---|---|
| `test_real_yaml_has_18_classes_and_unknown` | the shipped contract's shape |
| `test_both_passive_drop_spellings_map_to_one_class` | `過度切球` (data) and `過渡切球` (README) resolve identically |
| `test_unmapped_token_raises` | unseen vocabulary raises "new vocabulary", never defaults |
| `test_unknown_token_maps_to_none` | the declared unknown token maps to `None`, distinct from unmapped |
| `test_duplicate_token_rejected` | a token in two classes raises at load |
| `test_empty_classes_rejected` | an empty map raises |
| `test_apply_labels_marks_unknown_and_groups` | pandas stores `None` as `NaN`, so null-ness is asserted rather than identity — a real trap this test pins |

**Dedup and timeline** (`test_dedup_timeline.py`)

| test | edge case |
|---|---|
| `test_dedup_drops_lower_priority_source` | ss beats ss22; the dropped row records which uid was kept |
| `test_dedup_conflicting_winner_raises` | two sources disagreeing on the winner raises rather than resolving by priority |
| `test_week_precision_rows_never_join_exact_key` | a week-precision row on the same nominal date does **not** collide |
| `test_near_miss_audit_flags_close_rematches` | a 2-day gap is audited, a 30-day gap is not |
| `test_dedup_unknown_source_raises` | a source with no priority raises |
| `test_infer_discipline_two_components` | components labelled from seeds |
| `test_infer_discipline_missing_seed_raises` | a component with no seed raises — an alias failure surfaces here |
| `test_build_timeline_orders_by_date_then_round` | same-day semifinal sorts before the final |
| `test_build_timeline_unknown_round_raises` | `"Round-of-99"` raises |
| `test_build_timeline_duplicate_uid_raises` | duplicate uid raises |

**Loaders** (`test_loader_shuttleset.py`)

| test | edge case |
|---|---|
| `test_match_table_coerces_float_dates` | `"2022.0"` → 2022; the ShuttleSet22 wart |
| `test_match_table_applies_aliases` | alias applied to winner, non-aliased loser untouched |
| `test_match_table_missing_video_dir_raises` | verbatim directory lookup |
| `test_match_table_invalid_date_raises` | month 13 raises |
| `test_strokes_reject_column_deviation` | dropping one of the 30 columns raises |
| `test_strokes_set_count_mismatch_raises` | declared 2 sets, 1 file present → raises |
| `test_load_all_strokes_stamps_provenance` | provenance columns stamped, set numbers correct |
| `test_load_homography_parses_3x3` | shape validated |

**BadmintonDB** (`test_badminton_db.py`)

| test | edge case |
|---|---|
| `test_filename_regex_parses_both_variants` | with and without `-50fps`; a non-matching filename returns `None` |
| `test_extract_winner_unique_candidate` | 1 game up, 20 points → winner derivable |
| `test_extract_winner_none_when_no_candidate` | mid-game state → `None` |
| `test_extract_winner_none_when_ambiguous` | **29-29 in the decider** — either player would end the match, so extraction must refuse |
| `test_override_outside_week_requires_flag` | a date outside the filename's ISO week raises unless `week_mismatch_ok=yes` |
| `test_override_winner_disagreement_raises` | the cross-check is real |
| `test_no_override_and_ambiguous_raises` | no winner and no override → raises, never guesses |
| `test_overrides_require_provenance` | an override row without a URL raises |

**Extract check** (`test_extract_check.py`) builds a fake raw tree, `chmod`s it
read-only, and asserts: correct counts pass; **deleting one set CSV raises**;
and **a writable tree raises** even with correct counts. The chmod is undone
in a `finally` so a failure does not leave an unwritable tmp dir.

**Events contract** (`test_events_schema.py`)

| test | edge case |
|---|---|
| `test_converter_round_trips_through_json` | model → JSON → model equality, with an unknown stroke surviving as `shot_class = None` |
| `test_converter_outcome_matches_side_assignment` | `outcome.winner_side` agrees with `assign_sides` — the outcome channel and the symmetrisation cannot drift apart |
| `test_checked_in_schema_is_current` | the checked-in `events_v1.schema.json` equals the model's generated schema — a stale artifact fails the build |

**Club DB** (`test_club_db.py`, 31 tests) — the densest edge-case file:

- **Illegal scores rejected** (parametrised): `21-20, 31-5, 22-19, 30-27,
  19-3, 21-21, 20-21`
- **Legal scores accepted** (parametrised): `21-19, 21-0, 30-29, 30-28,
  25-23, 15-21, 22-20, 20-22` — `22-20` is the corrected case from
  `DECISIONS.md`
- Self-play rejected **case-insensitively** ("Viraj" vs "viraj")
- Future date rejected; pre-2020 date rejected
- Winner must take 2 games **and** the final game (two separate tests, because
  `21-15, 21-18, 15-21` passes the first and fails the second)
- Game count must be 2 or 3
- Doubles requires 2 players per side
- Name guard fires at 0.85 similarity and can be overridden with `force_new`;
  an exact duplicate is still rejected even with `force_new`
- Unknown player in a match rejected
- Same-day rematch logs a warning **and still records** (asserts both)
- `import_csv` quarantines the bad row, keeps the good two, and reports the
  correct **CSV line number**
- `revalidate` catches corruption injected by direct SQL — proving the gates
  are not only enforced on the write path
- Doubles stored but absent from `as_timeline()`
- **`test_harness_runs_unchanged_on_club_fixture`** — 30 synthetic club
  matches through the identical harness, coin-flip Brier exactly 0.25. This is
  the proof that the club path and the elite path share one evaluator.
- `test_records_cli_end_to_end` — init, add players, add match, validate; and
  a gate failure returns exit code 1 rather than a traceback

### 12.2 Integration tests (real data)

Ten tests read `data/raw` and pin the verified real numbers:

| test | assertion |
|---|---|
| `test_extract_check_real_tree` | 44/104, 58/140, 9 |
| `test_real_shuttleset_counts` | 44 matches, **36,484 strokes** — the README claims 36,492; the comment records that the extracted CSVs sum to 36,484 |
| `test_real_shuttleset22_counts` | 58 matches, 52,356 strokes |
| `test_real_label_coverage_and_unknown_counts` | full vocabulary coverage; 1,407 / 1,078 unknowns |
| `test_real_bdb_matches` | 9 matches, all day-precision, Momota 7 / Ginting 2, uids unique |
| `test_real_global_timeline_is_103_matches` | 44+58+9 → 8 dropped (all ss22) → 103; date range; contiguous `t_idx`; `{MS, WS}` |
| `test_real_timeline_run` | coin flip is exactly 0.25 on all 103; Elo and log5 beat 0.25 on the scored window |
| `test_converter_fits_one_real_match` | rally and shot counts match the source frame exactly |
| `test_run_p4a_real` | 5 predictors × 103 matches; report and manifest written |
| `test_run_p4b_real` | elite timeline is exactly 94; 9 predictors × 94; calibration present |

### 12.3 Testing rules this repo enforces

- **Exactness where math guarantees it.** The coin-flip Brier is `== 0.25`
  with no tolerance. Elsewhere, explicit `pytest.approx` with a stated reason.
- **Tests must be falsifiable.** A test that asserts a tautology, tests the
  mock, or snapshots current output without independent reasoning is deleted.
  `test_antisymmetry_under_seed_flip` asserts that *some* rows flipped and
  *some* did not, so it cannot pass vacuously.
- **Mock only at true I/O boundaries.** Never to force green.
- **Every bug fix lands with a regression test in the same commit.**
- **Seeded and order-independent.** A flaky test is fixed or deleted the day
  it flakes.
- Unit tests may read `configs/*.yaml` — versioned config artifacts are code,
  not data. The "no filesystem" rule is about `data/` and speed.

### 12.4 One known defect in the test setup

`tests/test_match_row.py` and `tests/test_tactical.py` set a module-level
`pytestmark = pytest.mark.unit`, which is applied **in addition to** the
per-function `@pytest.mark.integration` on `test_run_p4a_real` and
`test_run_p4b_real`. Both tests therefore carry both markers and are selected
by `-m unit`, so `make test` runs two full real-data experiment runs (128
selected, 8 deselected — not the 126/10 the marker intent implies).

Fix: move those two tests to a module without the module-level `unit` mark, or
override with `pytestmark` at function scope.

---

## 13. Mechanical enforcement

- `make check` = `ruff format --check` + `ruff check` + `mypy --strict`
  (Python 3.12, `warn_unreachable`, pydantic plugin, `files = ["src", "tests"]`)
- `make test` = unit pytest; `make test-all` = unit + integration
- ruff lint selects `E,F,W,I,N,UP,B,SIM,RUF,PT,RET,ARG,ERA,T20` — notably
  `ERA` (no commented-out code) and `T20` (no `print` outside `cli.py`)
- Pre-commit runs ruff and mypy as **local `uv run` hooks**, not remote
  mirrors, so hook versions are exactly the versions in `uv.lock` — two ruff
  versions disagreeing on formatting is the failure mode this avoids.
  `end-of-file-fixer` and `check-added-large-files` stay remote; the latter
  blocks accidental data commits.
- Typed exceptions only, from `errors.py`: `BadmintonVisionError` root, with
  `DataContractError`, `LabelMapError`, `LeakGuardError`,
  `ValidationGateError`. The CLI catches the root class and returns exit
  code 1; nothing catches bare `Exception`.

---

## 14. What is not built

Deliberate omissions, each with a reason:

- **Any computer vision.** Increment 1 is the no-camera foundation.
- **A BadmintonDB stroke loader.** No requirement needs stroke rows from it;
  build when one exists.
- **Doubles prediction.** Needs pair-composition Elo plus a shrunk chemistry
  term; `as_timeline()` emits singles only.
- **`events_v1.game_scores`.** Blocked on resolving `roundscore_*` pre- vs
  post-rally semantics.
- **Coordinate/homography tactical features** (`hit_x/y`, `landing_x/y`,
  player positions). Deliberately unused in `tactical_v1`; queued as
  `tactical_v2`.
- **Reliability plots.** Calibration ships as tables in the report JSON; plots
  when someone will read them (the `viz` group stays optional).
- **Real club matches.** `db/club.sqlite` is empty. Club-level P4a waits for
  ~150–200 recorded matches.

The single highest-value next input is **more matches** — it is the only path
to a decisive H₁ and to any club-level claim at all.
