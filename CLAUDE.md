The master plan is badminton-vision-system.md; the current increment plan is the approved
"Increment 1: The No-Camera Foundation". This file is the standing contract for
HOW code gets written here. When this file conflicts with convenience, this file wins.

## Non-negotiables (violating these corrupts experiments, not just style)

1. **The leak firewall is a module boundary.** Feature code consumes
   `HistoricalView` (strictly earlier matches) or `PreOutcomeView` (stripped
   target match) — never raw stroke frames. Never add a column to
   `PreOutcomeView`'s keep-list without updating the strip-table justification
   and its test in the same commit.
2. **Determinism is absolute.** All randomness flows from config seeds through
   `numpy.random.Generator`. NEVER use built-in `hash()` for anything
   persistent or assignment-bearing — it is salted per process. Use
   `hashlib.blake2b` (or sha256) for stable keyed hashing, e.g. side
   assignment: `int.from_bytes(blake2b(f"{seed}:{match_uid}".encode(), digest_size=8).digest(), "big")`.
   Re-running any experiment with the same config must be bit-identical.
3. **data/raw is read-only, forever.** Code never writes there. Anything in
   `data/derived` must be rebuildable from raw + curated + code; if it isn't,
   it's in the wrong directory.
4. **Every run writes a manifest** (git sha, config hash, label-map version,
   seed, timeline hash, package versions). No manifest → the run didn't happen.
5. **Fail loud, never patch silently.** Unexpected data (unmapped label, extra
   column, unparseable date) raises a typed exception from
   `badminton_vision.errors`; it is never coerced, defaulted, or skipped
   without an explicit, tested policy (the unknown shot policy is the model:
   documented, diverges from reference code on purpose, unit-tested).
6. **Additive schemas only.** `events_v1` fields are never removed or
   re-typed; breaking change ⇒ `events_v2` + new version string. Same for
   label YAMLs (`shot_labels_v2.yaml`, never edit v1 in place).

## Architecture & file organization

- **Dependency direction is one-way:**
  `schemas → ingest → records/ratings → features → eval → experiments → cli`.
  Lower layers never import higher ones. `ingest` never imports `eval`;
  `features` never imports `experiments`. If you need it the other way, the
  design is wrong — stop and note it in DECISIONS.md.
- **Thin CLI.** `cli.py` parses args and calls library functions; zero business
  logic, zero science in CLI modules. One `bviz` entrypoint, subcommands
  registered per domain (argparse from stdlib; do not add click/typer).
- **Pure cores, I/O at edges.** Feature computation, metrics, ratings, and
  symmetrization are pure functions of dataframes/values — no file reads, no
  globals, no clock access. Loaders and the harness own I/O.
- **One concept per module.** Target < ~300 lines per module; split when a
  module grows a second responsibility. No `utils.py` dumping ground — name
  the concept or it doesn't get a module.
- **No notebooks in `src/`.** Exploration goes in `notebooks/` (gitignored) or
  becomes a test/script. Nothing load-bearing lives in a notebook.
- **Curated inputs** (`player_aliases.csv`, `match_date_overrides.csv`) are
  git-tracked, header-documented, and every row carries a provenance note.

## Python style

- Python 3.12 only (`requires-python = "==3.12.*"`). Full type hints on every
  public function and all module-level names; mypy must pass.
- ruff is the formatter and linter (line length 100). No manual formatting
  opinions — if `ruff format` changes it, that's the style.
- Pydantic v2 for anything crossing a process/file boundary (event schemas,
  configs); frozen dataclass (or NamedTuple) for internal records like
  `MatchMeta`.
- `logging` in library code; `print` only in CLI output paths.
- Docstrings: one imperative summary line on every public function; add
  Args/Returns only when non-obvious. Comments explain why, never what.
- Constants live in `configs/*.yaml` or a module-level `Final` with a comment
  pointing at the config that should eventually own it — never magic numbers
  inline in logic.
- Custom exceptions in `errors.py`: `DataContractError`, `LabelMapError`,
  `LeakGuardError`, `ValidationGateError`. Catch narrowly; never bare `except`.
- Timezone-naive ISO dates everywhere (match granularity is a day); never mix
  `datetime` and `date` types in one column.

## Testing rules

- Markers: `@pytest.mark.unit` (default, no filesystem beyond `tmp_path`,
  < 1s each, run constantly) and `@pytest.mark.integration` (may read
  `data/raw`; the M-gate and clean-clone runs). `make test` runs unit;
  `make test-all` runs everything.
- Unit tests use tiny synthetic fixtures in `tests/fixtures/` — never real
  data. Integration tests assert the verified real counts (pin numbers to what
  the loader actually returned on inspection, with a comment if the paper's
  figure differs).
- Exactness where math guarantees it: the coin-flip Brier test asserts
  `== 0.25`, no tolerance. Everywhere else, explicit `pytest.approx` with a
  stated rationale.
- Every bug fix lands with a regression test in the same commit. Every
  "policy" (unknown labels, min-support NaN, strip list, antisymmetry,
  prefix-invariance) has a named test — if a policy has no test, it isn't a
  policy yet.
- Tests are seeded and order-independent; a test that flakes gets fixed or
  deleted the day it flakes.

## Git & process hygiene

- Small commits, one logical change each; message format
  `M<milestone>: <imperative summary>` (e.g. `M1: enforce preoutcome strip list`).
  Tag milestone completion `m0`…`m7` after its exit test passes.
- Never commit `data/raw`, `data/derived`, `db/*.sqlite`, run outputs, or
  `.venv`. If `git status` shows data, stop and fix `.gitignore` first.
- `DECISIONS.md` at repo root: a dated one-paragraph entry for every deviation
  from the increment plan or this file, and for every judgment call an agent
  makes that a future session must not silently reverse. Read it at session
  start along with this file.
- At the start of each session: `make check && make test` before writing code.
  Leave the tree green at session end; if you must stop mid-change, commit to
  a branch with a `WIP:` prefix.

## Mechanical enforcement (set up in M0, before any feature code)

`make check` = `ruff format --check` + ruff lint + mypy; `make test` = unit
pytest. Pre-commit runs the same. Add to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E","F","W","I","N","UP","B","SIM","RUF","PT","RET","ARG","ERA","T20"]
ignore = ["E501"]  # formatter owns line length

[tool.ruff.lint.per-file-ignores]
"**/cli.py" = ["T20"]  # print is the CLI's job; banned everywhere else

[tool.mypy]
python_version = "3.12"
strict = true
warn_unreachable = true
# pydantic plugin
plugins = ["pydantic.mypy"]

[tool.pytest.ini_options]
markers = [
  "unit: fast, no real data",
  "integration: reads data/raw",
]
addopts = "-m unit"
```

`.pre-commit-config.yaml`: ruff (format + lint), mypy, end-of-file-fixer,
check-added-large-files (blocks accidental data commits).

Dependencies note: `pyarrow` is REQUIRED (parquet writes in the harness).
`matplotlib` lives in the optional `viz` group and is needed for M6's
calibration plots — `uv sync --group viz` before running p4b.

## Anti-slop rules (every line must earn its existence)

Slop is volume exceeding information. Code is a liability; the default answer
to "should this exist?" is no.

- **No speculative abstraction.** No class where a function does; no base
  class or Protocol with one implementation; no factory/manager/handler
  unless two concrete variants exist today. Generality is added when the
  second use case arrives, not before.
- **No wrapper that only renames.** A function that calls one other function
  with the same arguments gets deleted.
- **Error handling is policy, not padding.** `try/except` only where a
  specific failure has a specific, tested policy. Never
  `except Exception: log and continue` — that is the fail-loud rule
  inverted. Never mask absence with defaults (`.get(k, 0)`) when absence is
  a bug; let it raise.
- **Validate at gates, trust inside.** Data validated at an ingestion
  boundary is not re-checked at every layer. No isinstance-cascades under
  full type hints.
- **Dead code dies immediately.** No `pass  # TODO` stubs, no commented-out
  code (ruff ERA enforces), no unused anything kept "just in case." Real
  future work goes in DECISIONS.md or the backlog, not as stub code.
- **Check before writing.** stdlib/pandas/numpy/sklearn first for any
  algorithm; the existing codebase first for any helper. Duplicate helpers
  are how logic silently forks.
- **Tests must be falsifiable.** A test that cannot fail — asserts a
  tautology, tests the mock, or snapshots current output without independent
  reasoning — is deleted. Mock only at true I/O boundaries; never mock to
  force green.
- **Diff discipline.** Touch only what the task requires. No drive-by
  renames or reformatting of untouched files. A session's diff must be
  reviewable in one sitting; if it balloons, stop and split.
- **No tone slop.** No emoji in code, CLI output, or docs. README states
  what exists and how to run it — no "comprehensive", "powerful",
  "seamless", no aspirational feature lists.
- **The deletion pass.** The final commit of every milestone is
  subtractive: for each piece, ask "which named test or plan requirement
  dies if I delete this?" No answer ⇒ delete.

## When unsure

Prefer the boring option; prefer failing loudly; prefer the master document's
principle over local cleverness; and when a real ambiguity remains, write the
question and your provisional choice into DECISIONS.md rather than deciding
silently.
