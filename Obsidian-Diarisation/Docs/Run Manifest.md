---
status: done
created: 2026-09-04
---

# Run Manifest

Part of the [[Evaluation Harness]]. Depends on: [[Harness Reporter]], [[Orchestrator]].

## What it does

Every harness run can now write a self-contained JSON manifest recording the exact
configuration that produced it — pipeline checkpoint, segmentation source, clustering
model, dataset split/condition, and DER settings — alongside the existing per-file CSV
and summary JSON from [[Harness Reporter]]. A separate aggregator scans a directory of
manifests and builds an on-demand comparison table across runs, so comparing results as
the segmentation source, clustering model, or pipeline steps are varied doesn't rely on
memory or hand-maintained notes.

This is deliberately flat-file based (JSON manifests + a generated CSV), not a database —
that trade-off can be revisited if run volume or query needs outgrow it.

## `harness/run_manifest.py`

`write_manifest(run_config: dict, summary: dict, runs_dir: Union[str, Path]) -> Path`

- `run_config` must contain: `pipeline_config_id`, `segmentation_source_id`,
  `clustering_model`, `extra_pipeline_steps` (a list, empty by default — a new pipeline
  stage is one more entry, not a schema change), `split`, `condition`, `der_collar`,
  `der_skip_overlap`. Missing any required field raises `RunManifestError` immediately;
  nothing is written.
- `summary` is the dict [[Harness Reporter]]'s `write_report()` already returns
  (der/overlap_der/jer/counting_mae/counting_exact_match_percent) — passed through as-is,
  never recomputed.
- `run_id` is a UTC timestamp plus an 8-hex-char sha256 hash of the full `run_config`
  (e.g. `20260907T153012Z-a1b2c3d4`) — meaningful at a glance, and stable per exact config.
- Writes `{runs_dir}/{run_id}.json` containing `{run_id, created_at, run_config, summary}`.
  Creates `runs_dir` if missing (same `mkdir(parents=True, exist_ok=True)` convention as
  the reporter).
- Never overwrites an existing manifest silently: a `run_id` collision (same config,
  same timestamp) raises `RunManifestError` rather than clobbering. This is deterministic
  collision detection, not a race-avoidance mechanism — it's the intended signal that an
  identical rerun happened, not a bug to route around.

## `harness/aggregate_runs.py`

`aggregate_runs(runs_dir: Union[str, Path], output_csv_path: Union[str, Path]) -> Path`

Scans every `*.json` manifest under `runs_dir`, flattens each into one row (`run_id`,
`created_at`, the `run_config` fields, the `summary` fields), and writes them as a CSV.
This table is generated on demand, not hand-maintained, and can be pointed at this vault's
`Docs/` folder as an output path when a human-readable snapshot is wanted. An empty
`runs_dir` produces a zero-byte CSV rather than an error.

## Orchestrator wiring

`run_harness()` (in `run_harness.py`) takes two new optional parameters:

- `clustering_model: Optional[str] = None`
- `runs_dir: Optional[Union[str, Path]] = None`

Both default to `None` so every pre-existing caller keeps working unchanged. When
`runs_dir` is `None`, no manifest is written at all — this is a small addition beyond the
ticket's original wording, needed to avoid a breaking signature change across existing
callers, rather than making both parameters required.

When `runs_dir` is provided, `run_harness()` assembles `run_config` itself from
`pipeline_config_id`, `segmentation_source.id`, the `clustering_model` param, a hardcoded
empty `extra_pipeline_steps: []`, and `config.{split,condition,der_collar,der_skip_overlap}`
— callers never assemble `run_config` by hand. `write_manifest()` is called after
`write_report()`, with the same "nothing partial on failure" property the reporter
already has: a mid-run exception (e.g. `OracleSegmentation`'s `NotImplementedError`)
leaves zero manifest files, same as it leaves zero report files.

`clustering_model` has no real source yet — nothing in `harness/` or the pipeline object
currently exposes a clustering identifier. Until real clustering-model selection exists,
the `__main__` CLI block in `run_harness.py` hardcodes a literal (`"pyannote-default"`)
for it, the same way it already hardcodes the pipeline checkpoint string. There is no
`--clustering-model` CLI flag yet; that belongs to whichever future ticket implements
real clustering-model selection. The CLI does have `--runs-dir` (default `"runs"`).

## Non-goals (still true)

- No new scoring math — consumes the reporter's summary output as-is.
- No implementation of clustering-model selection or the planned multispeaker-detection
  step — only the fields to record which ones were used.
- No SQLite/DB.
- No changes to `harness/reporter.py`'s existing output shape — the manifest is additive.

## Tests

`tests/test_run_manifest.py` and `tests/test_aggregate_runs.py` — all fabricated inputs,
no model/audio/dataset needed:

- Manifest round-trips a full `run_config` + `summary` exactly.
- Different `run_config`s produce different `run_id`s.
- An identical `run_config` at the same (frozen/mocked) timestamp raises
  `RunManifestError` on the second write and leaves the first manifest's contents
  unchanged.
- `runs_dir` is created if missing.
- Omitting a required `run_config` field (e.g. `clustering_model`,
  `segmentation_source_id`) raises before anything is written.
- The aggregator builds one row per manifest with correct config + metric values, and
  handles an empty `runs_dir` cleanly.

`tests/test_run_harness_integration.py` has two tests added
(`test_run_manifest_written_alongside_report`,
`test_run_manifest_not_written_on_mid_run_failure`) verifying the orchestrator wiring
end to end — correctly written and wired against the new `run_harness()` signature.

**Known gap:** these two integration tests could not be run to a pass/fail result in this
environment. The `full_pipeline` fixture shared across `tests/test_run_harness_integration.py`
(defined in `tests/conftest.py`) fails on this Windows machine with
`FileNotFoundError: Could not find file "trñ00"`. Root cause, confirmed during this
ticket: `tests/data/debug.train.lst` intentionally contains a non-ASCII filename
(`trñ00`, added upstream in commit `b41b176e`, "fix: fix support for non-ASCII
characters") to test non-ASCII path handling, and the matching `trñ00.wav` audio file is
correctly tracked in git and present on disk. The failure is `pyannote.database`'s file-list
reader decoding that UTF-8-encoded filename using Windows' default `cp1252` codepage
instead of UTF-8, producing a string that renders identically in a terminal but does not
match the actual file. This affects every test in that file, including five pre-existing
ones this ticket never touched — it is a pre-existing bug in `pyannote.database`'s
encoding handling on Windows, unrelated to and out of scope for this ticket. It should be
tracked as its own ticket if it needs fixing.

## Key files

- `harness/run_manifest.py` — `write_manifest`, `RunManifestError`
- `harness/aggregate_runs.py` — `aggregate_runs`
- `run_harness.py` — `run_harness()`'s new `clustering_model`/`runs_dir` params, `__main__`'s `--runs-dir` flag
- `tests/test_run_manifest.py`, `tests/test_aggregate_runs.py`
- `tests/test_run_harness_integration.py` — two new tests, blocked on the `full_pipeline` fixture encoding bug described above
