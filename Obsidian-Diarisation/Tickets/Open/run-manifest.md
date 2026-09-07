---
status: not-started
created: 2026-09-04
---

# Run Manifest

Part of the [[Evaluation Harness]]. Depends on: [[Harness Reporter]], [[Orchestrator]].

## Why

Right now a run only produces a per-file CSV and a corpus summary JSON (see [[Harness Reporter]]) — nothing records *how* that run was configured. As we start varying the segmentation source (standard vs. oracle), the clustering model, and adding new pipeline steps (e.g. a small model to flag multispeaker sections before clustering), we need each result tied to its exact config, or comparisons across runs become guesswork. This picks Option 4 from the reporting-organisation discussion: one self-contained JSON manifest per run, with an aggregator that builds a comparison table on demand — not a hand-maintained doc, not a DB (revisit SQLite only if run volume/query needs outgrow flat files).

## Scope

**1. `harness/run_manifest.py`** — a new component, called once by the [[Orchestrator]] right after `write_report()`.

- `write_manifest(run_config: dict, summary: dict, runs_dir: Union[str, Path]) -> Path`
  - `run_config` — everything needed to reproduce the run:
    - `pipeline_config_id` (already exists — the orchestrator's own param)
    - `segmentation_source_id` (`.id` off the `SegmentationSource` instance — this is where "standard" vs "oracle" is recorded; see [[Segmentation Source Interface]])
    - `clustering_model` — new field, a stable string identifying which clustering approach/model was used (does not exist as a concept anywhere in the harness yet — this ticket introduces it as an explicit, required manifest field rather than letting it hide inside `pipeline_config_id`)
    - `extra_pipeline_steps` — a list, empty by default, of stable string ids for any additional stage bolted onto the pipeline (e.g. the planned multispeaker-detection pre-clustering step). Deliberately a list, not fixed fields, so a new step is one more entry, not a schema change.
    - `split`, `condition`, `der_collar`, `der_skip_overlap` (off `HarnessConfig`)
  - `summary` — the dict `write_report()` already returns (der/overlap_der/jer/counting_mae/counting_exact_match_percent) — passed through as-is, not recomputed.
  - Generates a `run_id`: UTC timestamp + short hash of `run_config` (so re-running identical config twice doesn't silently collide, and the id is still meaningful at a glance — e.g. `20260904T153012Z-a1b2c3d4`).
  - Writes `{runs_dir}/{run_id}.json` containing `{run_id, created_at, run_config, summary}`. Creates `runs_dir` if missing (same `mkdir(parents=True, exist_ok=True)` convention as [[Harness Reporter]]).
  - Never overwrites an existing manifest file silently — collision (same config + same second) raises rather than clobbering.

**2. `harness/aggregate_runs.py`** — a small script/function that scans a `runs_dir` of manifest JSONs and builds a flat comparison table (one row per run, columns = the flattened `run_config` fields + summary metrics), written as CSV. This is the on-demand "give me a table" step — it is *generated*, not hand-maintained, and can be pointed at the vault (`Docs/`) as an output path when a human-readable snapshot is wanted.

**3. Orchestrator wiring** — `run_harness.py` gains a `run_config` (or equivalent explicit params) and `runs_dir` parameter, calls `write_manifest()` after `write_report()`, same "nothing partial on failure" property the reporter already has (a mid-run exception must leave zero manifest files, same as it leaves zero report files today).

## Non-goals

- No new scoring math — consumes [[Harness Reporter]]'s summary output as-is.
- No implementation of the multispeaker-detection step itself, or of real clustering-model selection logic — this ticket only adds the *fields to record* which ones were used. The actual feature work is separate, future tickets.
- No SQLite/DB — flat JSON files + a generated CSV table is the whole design for now.
- No changes to `harness/reporter.py`'s existing output shape — the manifest is additive, alongside the existing CSV/summary files, not a replacement.

## Tests first

Add `tests/test_run_manifest.py`:

1. `test_manifest_contains_full_run_config_and_summary` — write a manifest with a fabricated `run_config` (including `clustering_model` and a non-empty `extra_pipeline_steps` list) and summary dict, read it back, assert every field round-trips exactly.
2. `test_run_id_is_unique_per_config` — two different `run_config` dicts produce two different `run_id`s (and thus two files, no collision).
3. `test_identical_config_rerun_does_not_clobber` — calling `write_manifest` twice with the same `run_config` at effectively the same instant either produces two distinct files or raises clearly — assert it never silently overwrites the first manifest's contents.
4. `test_runs_dir_created_if_missing` — same `mkdir(parents=True, exist_ok=True)` behavior as the reporter.
5. `test_missing_required_run_config_field_raises` — omitting e.g. `clustering_model` or `segmentation_source_id` raises a clear error rather than writing a manifest with a silently missing field.

Add `tests/test_aggregate_runs.py`:

6. `test_aggregate_builds_one_row_per_manifest` — fabricate 2-3 manifest JSONs in a temp `runs_dir`, run the aggregator, assert the output table has one row per manifest with correct config + metric values.
7. `test_aggregate_handles_empty_runs_dir` — no manifests present, assert a clear result (empty table with headers, or a clear error) rather than a crash.

## Acceptance criteria

- All tests pass with fabricated inputs — no model, no audio, no real dataset needed (same bar as [[Harness Reporter]]'s test suite).
- A manifest is proven to hold the *actual* config a run used — not a hardcoded/default value — for at least `segmentation_source_id` and `clustering_model` (the two fields this project is about to start varying).
- Orchestrator integration: extend `tests/test_run_harness_integration.py` with one test confirming `run_harness()` now also produces a manifest file alongside the existing CSV/summary, and that a mid-run exception (e.g. `OracleSegmentation`'s `NotImplementedError`) leaves zero manifest files too.

## Implementation Notes

_(fill in as work progresses — decisions made, alternatives rejected, gotchas, key files touched)_
