# Badminton Vision — The Complete System Plan

**Status:** consolidated master document. Supersedes all earlier versions. Built solo, part-time; every design decision below reflects that constraint. Companion assets already in hand: ShuttleSet, ShuttleSet22 + CoachAI Challenge, BadmintonDB, TrackNetV3 code, RacketDB annotations (see §7).

---

## 1. Vision — what the system is and why it exists

**In one sentence:** film a badminton match, and the system returns a per-player report — footwork quality, shot statistics with success rates, positional tendencies, and **one prioritized improvement** — then feeds those measurements, alongside recorded match history, into an XGBoost model that outputs a calibrated win probability for any matchup in the club.

It answers the two questions a scoreboard cannot:

1. **Why did a player win or lose?** The measurable habits behind the result — where they are slow to recover, which shots leak points, which zones they lose from.
2. **What are the odds if A plays B?** A calibrated percentage, grounded in real match history, sharpened by what the video measured, honest about its own confidence.

**The governing principle: match results are the only ground truth.** Video features sharpen and explain predictions; they never override a recorded outcome. Any claim about the future must survive walk-forward testing on matches the model never saw.

**The differentiator (the moat):** every existing product analyzes a session in isolation. None connects footwork metrics and shot tendencies to *actual recorded outcomes against specific opponents over time*. This system is a mirror wired to a scoreboard: "68% vs Sam, up from 64% — and here is the measured weakness that cost you seven points against him." That is only possible because the same system owns both the video pipeline **and** the longitudinal match database — plus an integrity/audit layer (§18) that no competitor has. Accuracy sells a demo; auditability sells a contract.

**Commercial trajectory:** prove at club level → sell to clubs, academies, and national training centers. The elite data layer (BWF tournament data to betting operators) is contractually locked by Stats Perform, so "big organizations" realistically means the coaching/development market, not the federation data pipeline (§22).

## 2. Method lineage — where the design comes from

Two tennis-prediction videos supplied the prediction engine and its discipline:
- **The engine:** one row per match, hand-engineered features, gradient-boosted trees (XGBoost). Their strongest features — Elo, surface-specific Elo, head-to-head, recent form — translate directly (surface Elo → format-specific Elo, singles vs doubles).
- **The hard lessons, adopted as law:** their "85% accuracy" collapsed to an honest ~66% once post-match Elo leakage was removed → *pre-match features only, walk-forward evaluation, always*. Their biggest single improvement was an adaptive Elo K-factor (higher for new/returning players) → already implemented in the club platform (K=40 under 30 games). Their cold-start failure on low-data players → the shrinkage/fallback architecture in §13.
- **Their honest ceiling** (~66% vs bookmakers' 72%) sets expectations: do not chase headline accuracy; measure *lift over the rating baseline*.

The CV pipeline descends from the tennis-analysis approach (detect players → court keypoints → homography → top-down court → metrics), rebuilt badminton-native: MediaPipe skeletons, TrackNet shuttle tracking, badminton court geometry, and the published monocular fusion pipeline (Hsu et al. 2024) as the closest reference implementation.

## 3. Scope and operating constraints

- **Builder:** one person, part-time, AI-assisted. Consequences: strict *sequencing* instead of parallel tracks; every human-in-the-loop touchpoint (correction queue, quarantine review) designed as **batched weekly sessions**; scope discipline (below) is survival, not preference.
- **Sequencing:** records-only prediction baseline → dataset rehearsal (ShuttleSet) → club capture → CV pipeline → fusion. One track at a time; the data contract is the interface between work sessions.
- **Singles first.** Doubles perception is the honest hard mode (four bodies, occlusion, re-ID); doubles *prediction* works from day one via pair composition (§14).
- **Near side first.** Club halls rarely allow a high camera mount; the low-camera near-half profile is the validation path (§5).
- **Relationship to Badminton-Data:** separate repo/pipeline. Reads match results from the existing SQLite database; writes derived features back. Does not replace the club platform.

## 4. System architecture — the subsystem map

```
 VIDEO (club or broadcast)
   │
   ▼
 [SEGMENTER]            what am I looking at?  (live rally / dead time / replay / other view)
   │  IN_RALLY frames only, stamped rally_id
   ▼
 [SIDE & IDENTITY MGR]  who am I looking at?   (which player on which side, through switches)
   │  identity-stamped frames
   ▼
 [PERCEPTION]           what are they doing?   (calibration → detect → track → pose → shuttle → impact → shot)
   │  event JSON per data contract
   ▼
 [ANALYTICS]            what does it mean?     (kinematics → features → report with ONE improvement)
   │  per-player feature vector (time-decayed)
   ▼
 [PREDICTION]           who wins?              (feature tiers → XGBoost → fallback chain → calibrated % + confidence)
   ▲
   └── MATCH RECORDS (SQLite; the only ground truth)
```

**The data contract (fixed day one, versioned, additive-only changes):**

```json
// per frame
{"t": 12.433, "player_id": "p1", "kp": [[x, y, conf] x 33], "court_xy": [2.1, 4.7]}

// per shot event
{"t": 12.433, "rally_id": 18, "player_id": "p1", "shot": "drop",
 "conf": 0.91, "hitter_court_xy": [2.1, 4.7],
 "outcome": "in_play|winner|error", "source": "model|human_corrected"}
```

The `source` field is the audit spine: every datum records whether it came from a model (and which version) or a human. TrackNetV3's rectified (inpainted) shuttle points carry it too — a repaired point is not an observed one.

## 5. Capture — two profiles, one pipeline

### Profile C — Club (one low camera, near half is the product)
A high mount is usually impossible in a club, and a low camera behind the baseline sees its near half well but a small, net-occluded far half. Design consequences:
- **Analyze the near half only.** Every model performs best there; validate the whole pipeline where it is strongest.
- **End changes make one camera measure both players.** Players swap ends after every game (and at 11 in a decider), so across a best-of-3 each player spends game(s) on the near side. One low camera yields a report for both players per match — each built from the games that player was nearest — at roughly half-a-match of clean data per player, recovered by aggregating over more matches. This is why the Side & Identity Manager (§10) is core infrastructure: the system must know who is near in each game.
- **Phase B upgrade (optional): a second low camera behind the opposite baseline.** The far player for camera 1 is the *near* player for camera 2; each camera analyzes only its own near half; halves merge in shared court coordinates via their two homographies. Requires time-sync (clap/flash at session start or matched timestamps). This — not a high mount — is the architecture that fits a club hall, and it gives clean data on all four bodies for doubles.

### Profile B — Broadcast (YouTube pro matches, the both-sides testbed)
The main rally camera in pro broadcast is a quasi-fixed elevated behind-baseline view; cuts/zooms/replays happen *between* rallies. Pipeline: shot-boundary detection → keep only rally-view segments → re-estimate court homography **per rally segment** via automatic court-line detection → run the full two-sided analysis within segments. Far player is smaller/noisier — same confidence gradient, same gating. **Scope honesty:** broadcast validates the both-sides pipeline and makes demos; it never feeds the club prediction model (pros are not club members), and training-data volume comes from the released research datasets, not scraping (§21).

### Capture spec (what "lock the camera" means)
The camera caps every downstream model — spend budget here, not on a bigger GPU.
- **Frame rate over resolution:** true **60 fps** is the sweet spot (30 proves concept but misses impacts on smashes/drives; 120 ideal). Prefer 1080p/1440p at true 60 over "4K" locked to 30. (TrackNet's original badminton results came from 30 fps 720p downsized to 640×360 — motion, not resolution, is the bottleneck.)
- **Shutter ≥ 1/1000 s**, manual exposure + white balance locked per session. The shuttle exceeds 300 km/h and occupies ~3–24 pixels; motion blur is the main enemy. Avoid cameras with bad rolling shutter.
- **Framing:** full near court; high enough that the net doesn't dominate, low enough that limbs stay legible; players large enough for pose, shuttle not a permanent dot.
- **Lighting is often the real limiter:** in a dim hall, frame rate and shutter fight each other. Bright uniform low-flicker LEDs beat any model tweak.
- **Never move the camera.** Any bump = recalibrate (drift check in §16 enforces this).

## 6. Tech stack

**Perception (Python):**
- **OpenCV** — video I/O, drawing, homography.
- **YOLO** (player detection) + **ByteTrack or DeepSORT** (tracking) → **cropped** players into pose. Metric: identity-switch rate.
- **MediaPipe Pose** — 33 keypoints per cropped player. *Evaluation candidates to test against it in P2:* badminton-specific improved **YOLOv8-Pose (ELA attention)**; and for shots, **TemPose** (skeleton transformer built for badminton stroke recognition).
- **TrackNet V2/V3** — dedicated shuttle tracker (V2 ≈ 85% acc / 32 FPS baseline; V3 ≈ 97.5% acc, 99% recall / 25 FPS with augmentation + trajectory rectification; rectified points are flagged `source=inpainted`).
- **NumPy/SciPy** — joint angles via `atan2`, speeds from position deltas, Savitzky–Golay smoothing.
- **Shot classifier** — fusion: shuttle trajectory ⊕ pose sequence (impact ± 15 frames) ⊕ hitter court zone; rally-level transition-matrix constraint pass; confidence gate.
- Reference implementation to mine, not rebuild: Hsu et al. 2024 monocular shot-refinement (fusion of shuttle tracking + hit detection).

**Analytics:** court-coordinate kinematics (metres, camera-independent); HTML/PDF report per player per match; feature aggregation with exponential time decay.

**Prediction:** extends Badminton-Data (Python + SQLite). **pandas** (one-row-per-match table), **XGBoost** (native NaN handling — see §13), **scikit-learn** (splits, calibration, Brier/log-loss), **SHAP** (feature attribution; optional at first), **log5 + Beta smoothing** retained as prior, floor, and permanent baseline.

## 7. Datasets in hand — verified contents (read, not just downloaded)

| Dataset | Size (verified) | What it is | Role |
|---|---|---|---|
| **ShuttleSet** | 44 matches, 104 set files, 3,683 rallies, **36,484 strokes**, 27 elite players (2018–21) | Stroke-level CSVs: frame_num, shot `type`, hit_x/y, landing_x/y, both players' positions, win/lose reasons; **per-match homography matrices + court-corner pixels included** | Phase 4b rehearsal (perfect-label features); calibration reference |
| **ShuttleSet22** | 58 matches, 140 set files, 4,944 rallies, **52,356 strokes** (2022) | Same 30-column schema family — one loader serves both | Cross-dataset generalization test; more volume (~89K strokes combined) |
| **CoachAI Challenge (IJCAI'23)** | Track 2 data with train/val/test splits | Stroke forecasting benchmark + baselines; **Track 1 (11-target video benchmark) is *specified* here but its data is not in the repo** — chase separately if pursuing the credibility milestone | External benchmark spec |
| **BadmintonDB** | 9 matches (all Ginting–Momota), **811 points** | Video-timestamped strokes (StrokeBegin/End sec), `Camera` field, player `Location: top/bottom`, `T1P1/T2P1` naming (doubles-ready schema), WonEnd cause; README links the exact YouTube videos; **no coordinates** | End-to-end perception validation against known footage; never training |
| **TrackNetV3 (code)** | full repo | train/predict/eval incl. `predict.py`, dataset prep, error analysis | Shuttle tracker to fine-tune |
| **RacketDB** | 7 annotation-format zips; **22,682 images live on HuggingFace, not local** | racket boxes (16,045 instances) | Parked — scope-creep bait |
| **TrackNetV2 dataset** | *not retrievable from sandbox* — HackMD → Google Drive | 55,563 labeled shuttle frames, 18 matches | Download manually; V3 README documents required folder layout |

**Findings that change how the data is used:**
1. **Leakage landmine:** in ShuttleSet, *"player A" is defined as the match winner.* Any feature touching A/B labels leaks the outcome. **Symmetrize before anything else** (§15).
2. **Undocumented 19th label:** 未知球種 "unknown shot type" on 1,407 strokes (~4%) — the collapsed taxonomy must handle it explicitly.
3. **Label variant:** data spells passive drop 過**度**切球, README spells 過**渡**切球 — exactly the class of dirt the ingestion gates (§16) exist to catch.
4. **Missingness profile:** hit_x/y ~13% null; landing & player positions ~3%; win/lose_reason ~90% null *by design* (only on rally-ending strokes → implies ~10-stroke average rallies, a free consistency check).
5. **Schema landmine:** `getpoint_player`, `win_reason`, `roundscore_*` sit on *every stroke row* — within-rally future. Feature views must physically strip future columns (§17, rule 4).
6. BadmintonDB independently validates two modules designed from first principles: its `Camera` field is the Segmenter problem; its `top/bottom` location is the Side Manager's ground truth.

**Full shot taxonomy (ShuttleSet, 18 + unknown):** net shot 放小球 · return net 擋小球 · smash 殺球 · wrist smash 點扣 · lob 挑球 · defensive return lob 防守回挑 · clear 長球 · drive 平球 · driven flight 小平球 · back-court drive 後場抽平球 · drop 切球 · passive drop 過渡切球(/過度切球) · push 推球 · rush 撲球 · defensive return drive 防守回抽 · cross-court net shot 勾球 · short service 發短球 · long service 發長球 · unknown 未知球種. A **versioned mapping** from this to the collapsed club vocabulary is a day-one ingestion artifact.

## 8. Training-data strategy (the YouTube answer, settled)

Split by component:
- **Recognizers (TrackNet, shot classifier): pretrain on broadcast-domain data — via the released research datasets above — then fine-tune on your own fixed-camera clips.** The broadcast→fixed-camera domain gap is real and large (camera motion, cuts, zoom, compression): broadcast-only models drop several points in your hall; fine-tuning on a small set of your own matches recovers most of it. Always hold out one match *from your own camera* as the real test.
- **Analytics (footwork, recovery, position): own calibrated camera only.** Moving broadcast cameras break the homography; this half cannot use public footage at all.
- **No mass-scraping YouTube.** ToS prohibits it; the AI-training permission is off by default; and the research datasets give the same broadcast-domain volume *with labels* (§21). A handful of videos as a development testbed ≠ a training corpus.
- **Augmentation substitutes for raw volume:** motion blur, brightness/colour shifts, court-colour variation, perspective jitter, scale change, partial occlusion, JPEG artifacts — chosen to mimic the club↔broadcast gap. (TrackNetV3's gains came from augmentation + rectification, not model size.)
- **Cautionary anchor:** a recent mobile app attempting this in the wild hit **37.6%** shuttle detection on BWF videos while velocity-based **hit detection still reached 81.7%** — uncontrolled footage punishes shuttle tracking hard; hit detection is the robust signal. Both conclusions match this design.

## 9. The Segmenter — "what am I looking at?"

Gatekeeper before everything; assigns every frame `{live_rally | dead_time | replay | other_view}` + `rally_id`. Only live-rally frames from the main view enter perception. Four layers, cheap → subtle:

1. **Cut detection** — split into camera segments (frame-difference/histogram; off-the-shelf shot-boundary models exist). Every cut = "the world may have changed" → Side Manager re-checks identity.
2. **View classification per segment** — the court itself is the test: if the automatic court-line detector fits a valid full-court homography, it's the rally view. Close-ups/crowd/side-angles fail geometrically. Backup votes: a tiny court-view CNN (few hundred labeled frames) and score-overlay presence.
3. **Replay & slow-motion detection** (the subtle case: replays from the main angle pass the court test):
   - **Physics** — kinematics in real metres via the homography: a "rally" whose fastest movement is walking pace and whose shuttle drifts below plausible flight speed is impossible as live play. The calibrated speed measurement doubles as a slow-mo detector for free.
   - **Score logic as arbiter** — every real rally must advance the score (OCR overlay in broadcast; rally-outcome inference in club). Rally-looking segment + no score change ⇒ reclassify replay.
   - **Broadcast grammar** — wipe/transition graphics bracket replays; live overlay absent/frozen during them. Weak alone; tiebreaker. (Audio impact-sound desync: optional extra.)
4. **Rally state machine inside live view** — `PRE_SERVE → IN_RALLY → RALLY_OVER → DEAD_TIME`. Start: both players stationary in the service courts the score-parity rule predicts, then serve motion. End: TrackNet loses the shuttle in flight N consecutive frames + player motion energy drops. Only `IN_RALLY` flows downstream.

**Club footage runs layer 4 only** (no cuts/replays) — which also slashes club processing cost, since most session footage is dead time. **Ambiguity is gated, not guessed:** disagreeing layers → human-check queue; a double-counted replayed smash corrupts both players' stats *and* desyncs the score tracking identity depends on. **Reuse:** rally extraction from broadcast is exactly how ShuttleSet/TrackNet datasets were built — adapt CoachAI's clipping tooling, don't reinvent.

## 10. The Side & Identity Manager — "who am I looking at?"

Decomposes "recognize who's on which side, automatically, through switches" into three cross-checking sources:

- **Rules as priors (cheapest, strongest).** The laws schedule every switch: after each game, and at 11 in the decider. Score tracking therefore *predicts* switches before they happen. Serve-court parity (even score → right court, odd → left) verifies identity **every rally** for free. Broadcast adds a fourth signal: **OCR the score overlay** (names, score, switch schedule).
- **Continuous tracking** bridges identity between anchors. Club (no cuts): tracking usually survives the end-change walk-around. Broadcast: every cut kills tracks → re-establish per rally segment.
- **Appearance re-ID** does the re-establishing: enroll each player for a few seconds at match start into a small embedding gallery (kit colour, build, hair, **handedness** — a strong discriminator). After any cut/switch, match detections against the gallery. That is what "instant" realistically means: matching two known templates. It is **not** open-world face recognition — faces at court distance are a few pixels, it's unreliable, and it drags in privacy weight. Pro videos: title names the players, one manual assignment per video. Club: operator assigns at session start; a persistent roster gallery can automate later.

**Fusion & failure:** rules predict, embeddings confirm, tracking bridges. Disagreement (embeddings say no switch when rules demand one) ⇒ segment flagged low-confidence for human check. Side-identity errors are the worst error class — one swap poisons every downstream stat for both players — so they are gated hardest. **Doubles:** the manager operates at *pair* level (which pair on which side); within-pair attribution is a separate, harder problem (§14).

## 11. Perception pipeline — video into numbers

Runs only on Segmenter-approved, identity-stamped frames.

1. **Court calibration.** Homography maps the camera view to a top-down court in real metres. Club: calibrated once per locked setup against known line positions; **re-validated every session** by reprojecting court landmarks (drift gate, §16). Broadcast: re-estimated per rally segment via automatic court-line detection. Achievable error target ≤ 10 cm; a bad homography silently distorts *everything*, which is why drift gets its own alarm.
2. **Detection → tracking → pose.** YOLO boxes players; ByteTrack/DeepSORT keeps identities (metric: identity switches per match); MediaPipe fits 33 keypoints per **cropped** player (crops are why a general pose model works here). The **ankle midpoint** through the homography = ground position in metres. All downstream metrics are metric and camera-independent.
3. **Shuttle tracking.** TrackNet localizes the shuttle per frame; the same homography maps to metres so shot locations compare across matches. V3's inpainted points carry `source=inpainted`.
4. **Impact detection — fused and probabilistic.** Declare a hit when signals agree within a short window: shuttle-trajectory direction change ⊕ wrist-speed spike ⊕ swing cue. Temporal tolerance, not a single exact frame — blur and occlusion make any one signal unreliable. (Hit detection is the robust broadcast signal; lean on it.)
5. **Shot classification.**
   `p(shot) = softmax[ shuttle trajectory ⊕ pose sequence(±15) ⊕ hitter court zone ]`
   then a rally-level constraint pass (transition matrix zeroes impossible sequences — no serve mid-rally; position–shot consistency), then the **confidence gate τ**: auto-accepted predictions must be ≥ 95% accurate; the rest queue for weekly human correction, which feeds the next fine-tune. Gating is what lets per-shot accuracy sit at the realistic ~72–80% while the *aggregate* stats the report is built on stay trustworthy.

## 12. Analytics engine — numbers into meaning

Consumes event JSON only; never touches raw video.
- **Speed:** Δposition/Δt, Savitzky–Golay smoothed. **Recovery time:** impact → re-entering base-position radius. **Coverage:** 2D court-zone histogram. **Lunge depth / split-step timing:** knee-ankle geometry, `atan2` on keypoint triples.
- Rolls into the per-player **video feature vector**: smash success, drop tightness (forced-lift rate), unforced-error rate per shot type, mean recovery, coverage, rear-court/mid/net-zone win rates, serve & serve-receive error — all **exponentially time-decayed** so recent form dominates.
- **The one-improvement rule:** rank weaknesses by **impact on winning × measurement confidence** (a big weakness the pipeline is unsure of loses to a smaller certain one); present the top item in plain language — "you lose too many points after short lifts to the backhand rear corner," never "rear-court defensive transition rate is low." Low-confidence metrics can never be promoted to advice — the report stays trustworthy on top of an imperfect pipeline. Elite-tier expectations run higher (VIRD-style 3D review built with Olympic coaches) — noted for the org product, out of scope now.

## 13. Prediction layer — features, XGBoost, and the never-break chain

**Missing data is the normal state, not an edge case** (new players, unfilmed players, pre-score-era records). Features are organized in availability tiers; every prediction runs on whichever tiers exist.

- **Tier 0 — context (always):** format; days since last match; matches in last 30/90 days; career matches.
- **Tier 1 — records (≥1 logged match):** format-specific Elo per player and the **Elo difference** (strongest single feature); rating-uncertainty proxy (games played / Glicko-style deviation); head-to-head + shrunk H2H rate; time-decayed recent form; streak; strength of schedule.
- **Tier 2 — score-level (where game scores exist):** points for/against per game; close-game (≤3) win rate; deciding-game record; comeback rate — the "clutch" signal W/L hides.
- **Tier 3 — video-derived (analyzed players only):** the §12 vector per player.
- **Tier 4 — matchup interactions (when both sides exist):** A−B differences of the above (explicit diffs matter at small scale) + style-clash terms (A smash rate × B defensive resilience).

**"As many as I can" — collect everything, feed little.** At club scale tree models overfit; a small clean set beats a big noisy one. Deploy with ~10–20 features (Tier 0–1 heavy); SHAP + walk-forward decide what earns entry; Tier 3 enters only after P4b proves lift.

**The five never-break layers:**
1. **XGBoost handles NaN natively** — learned default split directions; missing video features automatically shift weight to record features. Never impute fake zeros; pass NaN.
2. **Shrink the *features***, not just outputs: every small-sample rate pulled toward the club average via pseudo-counts `(n·raw + c·prior)/(n+c)` — a 2-match player reads "slightly above average," never "100% winner."
3. **Missingness as features:** `has_video_A`, `n_matches_A`, `n_h2h`… — the model *learns how much to trust* other columns. "We haven't measured this" is itself information.
4. **The fallback chain:** full XGBoost → XGBoost with video NaN → hard shrink toward log5/Elo → Beta-smoothed log5 → club-prior ~50%. Structurally incapable of failing to output; worst case it says 50%. (Generalizes the existing blend `P_final=(n·P_xgb + c·P_log5)/(n+c)`, c≈20.)
5. **Confidence ships with every number:** "68% — 41 matches, video both sides" vs "55% — low confidence, B has 3 matches, no video." Never present a low-confidence number as a high-confidence one.

**Sober data-size caveat:** with 30–100 players and many under 10–20 matches, **a rating system may keep beating XGBoost until history and features are clean** — which is exactly why P4a/P4b are experiments, not assumptions. "Beats Elo" is only believed when it holds across **multiple time blocks and several dozen future matches**; with small test sets, calibration curves and confidence intervals outrank headline accuracy.

## 14. Doubles — two timelines, deliberately split

- **Prediction: works now, records only.** Pair strength = composition of both players' individual *doubles* Elo (a learned blend, not a plain average — some players punch above their singles level in doubles) + a **pair-chemistry term shrunk toward zero** when the pair has few games together. Enables predicting never-seen pairings. Then the identical machinery: `P = σ(s_A − s_B)`, log5 prior, XGBoost row with pair-level columns (both doubles Elos, pair H2H, pair form).
- **Perception: the deferred hard mode.** Four bodies, matching kit defeating appearance re-ID, net-play occlusion — single-camera doubles identity is still partly open research (two-camera rigs exist precisely because of it). Path of graceful degradation: (a) **pair-level metrics needing no individual ID** — side rally outcomes, rally length, pair coverage, formation detection (side-by-side defense vs front-back attack) — robust immediately; (b) individual attribution only through heavy confidence gating (confident shots get individual stats; the rest pool at pair level — the `source`/confidence discipline carries over); (c) full per-player stability through net play = stretch goal, realistically may want the second camera (§5).
- **Doubles analytics unlocks what singles can't:** rotation quality, coverage overlap and the exploitable *gap* between partners, zone ownership — in doubles these beat individual shot stats anyway.

## 15. Evaluation — the H₀ ladder and its discipline

**H₀: the coin flip. P = 0.5 every match.** True by construction; scores are guaranteed: Brier 0.2500, log loss ln 2 ≈ 0.693, accuracy 50%. Three jobs:
1. **Harness verification:** run the coin flip through the walk-forward harness first; any output ≠ 0.2500 means *the harness is broken*. Every later claim is only as trustworthy as the harness.
2. **Forces symmetrization:** winner-first tables let "predict column one" score 100%. Per row: randomly assign who is "A," flip the label to match, make every feature a symmetric difference that negates on swap. Afterward the base rate is exactly 50% by construction and any accuracy above it is real signal. (Same leakage family as post-match Elo; ShuttleSet's winner-encoded "player A" is this trap in the wild.)
3. **Anchors the ladder** — each champion is the next null:
   - H₁ rejects H₀: **log5/Elo** ("ratings carry signal"). Rejection should be decisive; if Elo can't beat 0.25 Brier, stop and debug — that's a bug or catastrophic data thinness, not a modeling problem.
   - H₂ rejects H₁: **records-only XGBoost** (= Phase 4a).
   - H₃ rejects H₂: **records + video** (= Phase 4b).

**Rejection is statistical, not a headline:** paired per-match Brier differences (same matches, both models), bootstrap CI on the mean difference excluding zero, gap holding across multiple time blocks and dozens of matches. Otherwise the champion keeps the crown. The coin flip is perfectly calibrated with zero resolution — so "beating H₀" means adding discrimination *without wrecking calibration*, which is why Brier/log-loss are primary and accuracy is secondary. **Self-gaming guard:** a final untouched time block; count your peeks; the champion–challenger gap must also hold on genuinely new post-deployment matches (Goodhart applied to oneself).

**The unbeatable-player case (why an apparent 100% winner doesn't break anything):**
- "Guaranteed" is extrapolation: by the rule of three, 50–0 statistically supports ≈ 1 − 3/50 ≈ 94% at 95% confidence — the model pricing in the tail is *correctness*.
- Scoring asymmetry forbids 100%: saying 97% and being right costs pennies; saying 100% and being wrong once makes log loss infinite. Proper scoring rules make honest uncertainty beat false certainty by design.
- Existing machinery already copes: Beta smoothing cannot output 1.0; Elo self-limits (a heavy favorite gains ~nothing per win, so the rating plateaus at "expected score ≈ 0.98 vs this pool").
- The *actual* failure mode is the opposite — saying 60% for him — and the ladder punishes it automatically (Brier ~0.16 vs ~0.0016 per match for 96%). His matches are where a good model gains most over the coin flip. Check: the >90% calibration bucket should empirically win ≥ ~95%.
- The legitimate route to near-certainty is **rally math**: a ~60% rally edge → ~90% per game via the recursion `P(a,b)=r·P(a+1,b)+(1−r)·P(a,b+1)` (win-by-2, cap 30) → ≈97% match via `3p²−2p³`; a 65% rally edge lands beyond 99%. Score margins (Tier 2) carry the evidence W/L hides (21–5 ≠ 21–19).
- Display ">99%", never a rounded "100%". His stabilized matches carry almost no training information — the model learns from the competitive middle, which is expected and fine.

## 16. Data quality — dirty and invalid data

Governing principle: **quarantine, never silently fix or delete.** Validation converts "wrong" into "absent"; the never-break chain (§13) already makes "absent" safe; therefore the system keeps producing output while the quarantine queue waits for the weekly solo review batch. Four gates at every ingestion boundary (club entry, video events, external datasets):

1. **Schema gate.** The data contract enforced as a checkpoint: shape/type/required fields on every match entry, event JSON, dataset row. Malformed → quarantine with reason code.
2. **Rulebook gate.** Badminton's laws as a free validator (the same rules-as-priors asset as §9–10, pointed at integrity): legal scores (21, win-by-2, cap 30); serve court matches score parity; no player-vs-self; plausible, ordered dates (a wrong date silently corrupts walk-forward); shot sequences obey the transition matrix.
3. **Physics & statistics gate.** Bounds on everything measured: no superhuman sprint speeds or teleporting between frames; shuttle obeys flight physics; joints don't bend backwards. Feature-level outlier flags (10% → 90% error rate in one match = flag, not ingest). **Calibration drift alarm:** reproject known court landmarks at every session start; error over threshold blocks that session's positional data until recalibration — the silent killer (a bumped camera makes everything *plausibly* wrong), caught cheaply, and doubling as tamper evidence.
4. **Cross-source agreement gate.** Independent sources must corroborate: video-inferred winner vs recorded winner; OCR score vs rally-counted score; duplicate-match detection (players+date+score); **entity resolution on player names** ("Sam"/"Sam L"/"Samuel" fuzzy-matched and confirmed — duplicate identities split one history into two weak ones). Disagreement → flag, never auto-resolve.

Provenance stamps on every datum (source, model/annotator, version, validation status) + tracked quality metrics (validation pass rate, quarantine size over time) = the reliability story an organization's due diligence actually asks for.

## 17. Multi-dataset limitations — and the ten rules that solve them

**Leakage modes unique to mixing sources:** cross-dataset duplication (BadmintonDB's Ginting–Momota 2018–19 sits inside ShuttleSet's window and roster — the same physical match can appear in both); player overlap making random splits memorize idiosyncrasies (split *grouped* by match, by player when the claim is unseen-player generalization); no global temporal spine (walk-forward must hold across sources on one timeline); **lineage leakage** (if TrackNet pretraining saw match X, evaluating the shot classifier on X's clips is contaminated one level removed — provenance must track which matches touched which *model*); within-rally future columns on every ShuttleSet row (strip in the feature-view code, not by memory); shortcut leakage (court colour/graphics identify tournament→era→players; only cross-dataset evaluation detects it).

**Conceptual limits no cleanliness fixes:** population shift (elite feature→outcome relationships differ from club play — pro rallies end on winners/forced errors, club rallies drown in unforced ones; ShuttleSet rehearsal is an *upper bound* on "can these features carry signal," coefficients always relearned on club data); ontology mismatch is lossy both ways (18 types + backhand flag vs handedness-baked types with **no coordinates** in BadmintonDB → lowest-common-denominator features for cross-dataset work); annotator style as a hidden variable (models can learn *source* rather than badminton — never feed `source` as a feature; test cross-dataset); effective sample size ≪ row count (strokes cluster in rallies in matches — the independent unit is ~the match count; uncertainty must be cluster-aware); selection bias (ShuttleSet = finals/semis of top events; BadmintonDB = one rivalry; club consent refusals won't be random either).

**Physical/operational limits:** license & rights (code is MIT; datasets are research releases derived from BWF broadcast — a model trained on them and *sold* is a rights question to answer before the org conversation; safe pattern: public data pretrains/validates/benchmarks, the commercial model trains on consented club footage — which population shift forces anyway); storage arithmetic (1080p60 ≈ 8–15 GB/hr → a season is terabytes → events and samples forever, raw video on a retention schedule, which consent policy demands anyway); solo annotation drift (no inter-annotator agreement exists → periodic blind self-relabeling of a frozen sample; self-disagreement is the honest error bar); feature-version skew (v1- and v3-extracted features never mix in one training set — re-derive on upgrade; raw-sacred makes it possible, budget the compute); bus-factor-one hardware (one camera, one GPU, one SD card — written capture checklist + same-day copy; the only irreplaceable asset is a match that can't be replayed).

**The ten rules (plain language):**
1. **One timeline for everything** — every match from every source on a single calendar; models only ever see the past.
2. **Check for twins before mixing** — dedup matches across datasets; resolve player-name duplicates.
3. **Split by match and player, never by row.**
4. **Hide the future from every row** — strip future-revealing columns in code, not by memory.
5. **Label where every piece came from, forever** — dataset, version, human/machine; traceable and yankable.
6. **Public data teaches, club data decides** — the sold model trains only on consented club footage.
7. **Suspect anything too good** — train on one dataset, test on another; a big drop means memorized venues, not badminton.
8. **Copy recordings immediately; keep the small stuff forever** — video to a second drive same day; events/results permanent; raw video on schedule.
9. **Re-check yourself** — periodic blind self-relabeling; the disagreement rate is the error bar.
10. **When in doubt, quarantine** — set aside, run on what's clean; the system gets less confident, never breaks.

Do 1–2 and 4 *before any modeling* — those are the silent ruiners.

## 18. Integrity — how the system gets gamed, and the defenses

The system becomes worth gaming the moment outputs carry stakes (seeding, handicaps, selection, money). The deepest vulnerability is not technical: adversaries control the data-generation process itself.

- **Rating manipulation:** sandbagging (lose to deflate, collect handicaps/upsets), win-trading, opponent farming, smurfing (fresh account exploits provisional K=40), decay timing (freeze a peak; cram before cutoffs). Defenses: performance-vs-rating residuals; win clusters after losing streaks; H2H pairs whose frequency/outcomes don't match their rating gap; low opponent diversity capping rating confidence; Glicko-style uncertainty growth in inactivity. **The video layer is a free lie detector:** a "weak" player with elite measured kinematics is a cross-source disagreement — the §16 gate pointed at dishonesty.
- **Camera gaming:** performing below ability on film (near-undetectable if consistent — but ground truth wins: results dominate a player who films weak and keeps winning, and the divergence flags); adversarial appearance vs re-ID (serve-parity verifies identity independent of looks; confusion → quarantine); deliberate camera bumps (session-start drift check = tamper evidence, logged); staged/scripted "rallies" (score-logic arbiter + design rule: **video annotates matches, it never creates them** — a match exists only via the results path).
- **Human-loop gaming (matters at org scale):** results entry can invent matches → dual confirmation by both players; correction-queue nudging → reviewer identity in provenance, no self-correction, periodic audit sampling of corrections against raw video. Boring controls; exactly what due diligence asks.
- **Information gaming:** reports are competitive intelligence → private by default, opt-in sharing, access control as an org feature. **Betting:** calibrated probabilities are gambling infrastructure; every manipulation above acquires cash incentive the moment odds exist — stance decided early (training & matchmaking), publishing probabilities = publishing a line. Feedback loop (predictions driving matchmaking shape the next training data) → monitor calibration by subgroup, occasionally randomize matchups.
- **Self-gaming:** §15's frozen block + peek counting.
- **The ultimate anti-fraud property is architectural:** raw-sacred + derived-disposable means late-discovered fraud is quarantined and *the entire world re-derived cleanly* — ratings, features, predictions — as if it never happened. Most rating systems can't rewrite history; federations that fear sandbagging scandals will value exactly this.

## 19. Scalability — room to grow and contract

For a solo club system, "scalable" means **clean seams and graceful degradation**, not web-scale infrastructure.

**The one property everything hangs on: raw data is sacred, derived data is disposable.** Sources of truth: match results (SQLite), event JSON (data contract), raw video (while retention allows). Everything else — features, Elo, reports, predictions — is re-derivable. Growth: model upgrades retroactively upgrade the whole history by re-running the pipeline. Contraction: any derived table can be deleted/corrupted and rebuilt. Protect this one decision above all.

- **Growth axes & seams:** courts/cameras = per-camera config profiles (homography + capture settings) — adding a court or the Phase-B camera is a config entry, not code (same mechanism as the club/broadcast profiles). Data volume: SQLite is fine at club scale; the Postgres trigger is *concurrent writers or multi-club federation*, not row count; if multi-club ever happens, namespace player IDs and keep per-club Elo pools (trivial to allow for now). Compute: capture already decoupled from processing — make the queue explicit; compute grows (cloud burst) or shrinks (laptop overnight) with latency as the only symptom. Models: every prediction stamped with model+feature-set version; challengers run shadow until they beat the champion walk-forward; rollback = re-point. Features/formats: tiers are the growth path (additive columns; XGBoost tolerates), doubles/broadcast arrive as profiles, not forks.
- **Contraction axes:** consent revocation (legally required): the member's video + Tier-3 features vanish, columns go NaN, the fallback chain absorbs it — predictions quietly degrade to records-only; name it and *test* it. GPU lost a month: queue grows, records-only predictions meanwhile. Club shrinks / data sparse: shrinkage pulls toward priors — less confident, never confidently wrong. **Kill-switch property:** every module is optional except match results + log5; if video fails P4b or time runs out, the CV half switches off and a working predictor remains.
- **Four operating rules:** version everything (schema, models, features, homographies) and stamp outputs; schema changes additive-only (old event files never become unreadable); scale on written triggers, never anticipation (premature scaling is the real solo risk); **rehearse contraction** — run a "delete this player" drill and a "no GPU this month" drill once.

## 20. Compute & deployment reality

Not Raspberry-Pi-first. Full-match YOLO + pose + TrackNet is a GPU workload: desktop or cloud NVIDIA GPU, **offline batch**, reports exported after. Anchors: shuttle tracker alone ≈ real-time-adjacent (V2 ~32 FPS, V3 ~25 FPS) ⇒ ~2–2.5× footage length per hour *before* detection/pose/post-processing. The Pi's role is capture and demos. Club footage costs drop further because the Segmenter discards dead time before heavy models run.

## 21. Legal, licensing, consent

- **YouTube is not a free training pool:** ToS prohibits unauthorized downloading/scraping; the third-party AI-training permission is off by default. Compliant broadcast-domain volume = the released research datasets (+ own recordings). A few videos as a dev testbed ≠ a scraped corpus.
- **Dataset licensing for commerce:** code MIT; datasets are research releases derived from broadcast footage — resolve the rights question *before* selling. Safe pattern (§17 rule 6): public pretrains/validates/benchmarks; the commercial model trains on consented club footage.
- **Filming members is personal data:** opt-in consent before recording; plain statement of what's stored, who sees it, whether faces are retained; a retention schedule; a one-page club policy. Consent revocation is a tested system behaviour (§19), not a promise.

## 22. Competitive landscape & positioning

- **Consumer form-coaching:** SportsReflector (25+ joints, 0–100 form scores, $19.99/mo Pro), AI Sports Trainer (impact point/racket angle/follow-through from uploads, drill suggestions). Session-in-isolation products.
- **Custom-build agencies:** Tezeract — bespoke badminton CV (tracking, shot classification, heatmaps) for academies/federations.
- **The template to study:** BadPro+ — four tiers: consumer app → pay-per-match web analysis → federation SaaS (rosters, scouting, predictive models) → live-stats API. The commercialized version of this roadmap's shape.
- **The locked layer:** Stats Perform is BWF's exclusive live-stream/real-time-data provider to licensed betting operators for elite tournaments — "big organizations" = academies, clubs, national training centers, not the federation data pipeline.
- **Reality checks:** MatchMotion's 37.6% wild-domain shuttle detection (vs 81.7% hit detection) validates the controlled-capture strategy; VIRD (built with Olympic coaches) shows elite expectations exceed flat PDFs — future org-tier consideration.
- **The moat, stated:** longitudinal club database × video features × walk-forward-proven lift × integrity/auditability. Nobody else has the four together.

## 23. Build phases & exit tests (written before code; no advance until pass)

| Phase | Goal | Exit test |
|---|---|---|
| **P0** | Basics work | Camera locked per §5 spec; 3 test matches recorded & same-day backed up; stock MediaPipe runs on one clip |
| **P1** | Trustworthy measurement (near half) | Calibration ≤ 10 cm on the near half; stopwatch-validated footwork report for 3 near-side players; session drift check operational |
| **P2** | Reliable shot reading (near half) | Per-metric on a **frozen** club test set: shuttle tracking ≥ 90% on own camera; hit detection ~90%; shot classification honestly vs the ~72% published bar (beat via collapsed taxonomy); gate tuned so auto-accepted ≥ 95% accurate. Candidate bake-off: MediaPipe vs YOLOv8-Pose-ELA; fusion classifier vs TemPose |
| **P2.5** | Both halves | Broadcast profile validates two-sided analysis (Segmenter + Side Manager operational); optional second club camera merged in court coordinates; **far-half metrics clear the same P1/P2 bars** (the far half is the accuracy floor) |
| **P3** | Real-world usefulness | Three players act on their reports |
| **P4a** | Predictor from records alone | Harness verified on the coin flip (Brier = 0.2500 exactly); leak-free symmetrized time-split; records-only XGBoost beats log5/Elo on walk-forward Brier with the §15 statistical bar |
| **P4b** | Video earns its place | Controlled experiment — A: records-only; B: A + video features; C: video-only — on Brier & log loss over unseen future matches. B > A ⇒ CV joins prediction. **B ≤ A ⇒ video features remain for explanation/reporting only — the coaching product still ships; not a project-killer** |
| **(opt)** | External credibility | CoachAI Track 1 eleven-target benchmark (spec in hand; data chased separately) |

Rehearsal available *now*, no camera: ShuttleSet-derived tactical features vs a rating baseline on its 44 matches = P4b with perfect labels — the upper bound. If clean labels add no lift there, noisy CV features won't either; run it before buying a tripod. The correction loop is the method: plan 3–4 fine-tune iterations, not one.

## 24. Immediate next steps (in order)

1. **Ingestion spec:** versioned shot-label mapping (18+unknown → collapsed vocabulary, including the 過度/過渡 variant), symmetrization rule, future-column strip list, cross-dataset dedup pass, global timeline. (Rules 1, 2, 4 — the silent ruiners.)
2. **Evaluation harness** + coin-flip verification (Brier 0.2500) + log5 baseline number on club records — the first real number everything must beat.
3. **Records-only XGBoost (P4a)** on club data; benchmark family sanity-checked against the 14,722-match public baseline territory (76–80%).
4. **ShuttleSet rehearsal** of P4b (perfect-label upper bound).
5. **Camera purchase + P0** per the §5 spec; consent form + one-page policy drafted alongside.
6. TrackNetV2 dataset manual download; TrackNetV3 fine-tune prep.

---

*Everything in this document is falsifiable by the phase tests above; anything that fails them gets demoted, not defended. The system's promise is not that video predicts — it's that the claim will be tested honestly, and the product survives either answer.*
