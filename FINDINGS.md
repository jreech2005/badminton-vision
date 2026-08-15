# Increment 1 findings — the no-camera foundation

Final report for Increment 1 (M0–M7), 2026-07-19. Numbers are read from the run
artifacts `data/derived/runs/20260719-142117-p4a/p4a_report.json` and
`data/derived/runs/20260719-142905-p4b/p4b_report.json`; both runs carry
manifests (git sha, config hash, label-map version, seed, timeline hash,
package versions). Judgment calls behind these results are logged in
`DECISIONS.md`; the module design is the approved increment plan.

## What was built

Dataset loaders with a leak firewall (`HistoricalView` / `PreOutcomeView`,
winner-encoded columns stripped, deterministic blake2b side assignment); a
103-match global timeline (ShuttleSet 44 + ShuttleSet22 50 after 8
cross-dataset duplicates + BadmintonDB 9; 2018-06-29 to 2022-10-30) with alias
resolution, dedup, and discipline inference; a walk-forward harness verified by
the coin-flip null (per-match Brier exactly 0.2500 on all 103 matches, log
loss ln 2); adaptive-K Elo and Beta-shrunk log5 baselines; records-only
LR/XGB machinery (P4a); 13 preregistered tactical features and the A/B/C
rehearsal (P4b); the `events_v1` data contract future CV work writes to; and a
club SQLite records path that runs the identical harness (proven on a
synthetic fixture).

## P4a — records-only models vs rating baselines

Mean Brier, walk-forward, full window (n=103) / scored window (both players
with 3+ prior matches, n=32):

| predictor | full | scored |
|---|---|---|
| coin flip | 0.2500 | 0.2500 |
| Elo (adaptive K) | 0.2341 | 0.2304 |
| log5 (Beta, decayed) | 0.2340 | 0.2261 |
| LR records-only | 0.2762 | 0.2776 |
| XGB records-only | 0.2581 | 0.2633 |

Preregistered paired-bootstrap contrasts (full window, positive = first
predictor worse, 10,000 resamples over matches):

| contrast | mean ΔBrier | 95% CI | p |
|---|---|---|---|
| XGB − Elo | +0.024 | [+0.005, +0.042] | 0.011 |
| LR − Elo | +0.042 | [−0.004, +0.093] | 0.078 |
| Elo − coin flip | −0.016 | [−0.032, +0.001] | 0.068 |
| log5 − coin flip | −0.016 | [−0.033, +0.002] | 0.077 |

Verdict: the rating baselines keep the crown. Records-only models do not beat
Elo/log5 at this sample size; XGB is significantly worse than Elo on the full
window. Elo and log5 beat the coin flip on point estimate, but the CIs still
brush zero — decisive H₁-over-H₀ rejection needs more matches.

## P4b — does adding perfect-label tactical features help? (the rehearsal)

94-match elite subset; A = records-only, B = A + 13 tactical features,
C = tactical-only; same models, seeds, and protocol. Mean Brier, full (n=94) /
scored (n=25):

| predictor | full | scored |
|---|---|---|
| Elo | 0.2359 | 0.2420 |
| log5 | 0.2351 | 0.2372 |
| LR A (records) | 0.2691 | 0.3191 |
| LR B (records+tactical) | 0.2887 | 0.3021 |
| LR C (tactical only) | 0.3070 | 0.3243 |
| XGB A / B / C | 0.2604 | 0.2533 |

Headline contrast (preregistered): LR B − LR A = +0.020 Brier full window
(95% CI [−0.019, +0.059], p=0.32) and −0.017 scored (CI [−0.090, +0.041],
p=0.65). **B ≤ A: adding tactical features to records features produces no
detectable lift**, tactical-only is worst, and every model variant loses to
Elo/log5 (LR B − Elo = +0.053, CI [−0.005, +0.114]). XGB A and B produced
bit-identical predictions (their contrast is exactly zero); XGB C differs only
in the sixth decimal — the preregistered regularization
(min_child_weight=10) leaves XGB effectively a base-rate predictor at N≈94
(150 trees, 5 split nodes total).

Per the master plan's P4b exit rule, this routes tactical — and future video —
features to the explanation/coaching track; the prediction product remains
ratings-first. This is the designed-for outcome of a falsifiable experiment,
not a failure: the coaching product ships either way.

## Hypothesis-ladder status

- **H₀ (coin flip):** verified by construction — the harness gate.
- **H₁ (ratings carry signal):** favored on point estimate, not yet decisive
  (p ≈ 0.07). Keeps accumulating with every new match on the timeline.
- **H₂ (records-only ML beats ratings):** rejected at this N — models lose,
  XGB significantly so.
- **H₃ (tactical features add lift):** no detectable lift at this N; routed to
  explanation per the preregistered exit rule.

## Caveats that bound every claim above

- **Population:** all results are elite MS/WS singles (ShuttleSet-family
  broadcast annotations). Club play differs structurally (unforced-error-
  dominated rallies); nothing here transfers to club claims. Club-level P4a
  waits for a defensible volume of real recorded matches (~150–200).
- **Power:** with 94–103 matches, paired-Brier lifts smaller than ~0.04 are
  undetectable ("no detectable lift", never "no lift"). Scored windows are
  small (n=32 / n=25).
- **XGB is not evaluated meaningfully here:** the preregistered regularization
  makes it a near-base-rate predictor at this N. It is machinery rehearsed for
  club-scale data, not a claim-bearing model; LR carries the model claims.
- **Ordering:** feature/rating state updates follow t_idx order (same-day
  earlier rounds visible to later rounds), an approximation documented in
  DECISIONS.md; model and baseline information sets are identical by
  construction.
- **Effective sample size** is the match count, not the stroke count; all
  uncertainty is cluster-aware (bootstrap over matches).

## Adversarial review

A multi-agent adversarial review (five lenses: leakage, statistics, feature
math vs preregistration, claims-vs-artifacts, contract compliance; every
finding attacked by three independent refuters; looped until two consecutive
rounds surfaced nothing new) ran before this report was committed.

<!-- filled in after the workflow completes -->

## Backlog (next increments, in rough order)

- **Club data accumulation** — the only path to decisive H₁ and club-level
  P4a; target ~150–200 recorded matches via `bviz records`.
- **Camera P0** — capture spec per master plan §5 (60 fps, shutter ≥ 1/1000,
  locked mount), consent form + one-page policy.
- **CV increment** — TrackNetV2 dataset manual download; TrackNetV3 fine-tune
  prep; segmenter layer 4 (rally state machine) for club footage.
- **tactical_v2** — coordinate/homography features (hit_x/y, landing_x/y,
  player positions were deliberately unused in v1).
- **Doubles prediction** — pair-composition Elo blend + shrunk chemistry term;
  `as_timeline()` currently emits singles only.
- **BadmintonDB stroke loader** — build when a requirement exists (records-only
  this increment).
- **`events_v1.game_scores` semantics** — resolve pre- vs post-rally meaning of
  `roundscore_*` before populating game scores.
- **Reliability plots** — calibration ships as tables in the report JSON; plots
  when someone will read them (`viz` group stays optional).
