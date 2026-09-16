---
status: in-progress
created: 2026-09-13
---

# T1: Harness reporting and manifest schema

Part of the oracle/ceiling analysis batch (see [[T2-post-clustering-refinement-hook]], [[T3-oracle-segmentation-provider]], [[T4-oracle-assignment-strategy]], [[T5-cross-condition-deltas]]). Largest ticket in the batch. Touches `scorer.py`, `reporter.py`, the summary writer, the run manifest, and `aggregate_runs.py` together.

## Goal

Make a run self-describing and make the ceiling metrics available.

## Scope

### Metrics
- Add missed detection, false alarm, and speaker confusion as separate columns. Global DER alone cannot distinguish "error moved upstream" from "assignment confusion", and that distinction is the finding.
- Rename `overlap_der` to `der_overlap_system`.
- Add `der_overlap_assigned`, scored over **T ∩ D** (ground-truth overlap regions intersected with regions the hypothesis emitted as multi-speaker). Derive the hypothesis overlap region from the hypothesis annotation, intersect with T and with the base UEM, following the existing `overlap_uem` pattern in `harness/scorer.py:24-56`.

### Region census
- Log duration of **T ∩ D**, **T \ D**, and **D \ T** per file and corpus-wide. Cheap, and it makes a surprising metric immediately diagnosable rather than a debugging session.

### Structured condition fields on the manifest
- `condition`: controlled vocabulary, one of `baseline`, `oracle_segmentation`, `oracle_assignment`, `nearest_centroid`
- `segmentation_source_id`: already exists in the cache key at `harness/runner.py:35-40`, needs surfacing
- `refinement_strategy`: one of `identity`, `oracle`, `nearest_centroid`
- `corpus`, `mic_condition`
- `git_commit`
- `counts_toward_results`: boolean, **defaults to false**

`--notes` stays as free text. The delta computation in T5 keys on `condition` only, never on notes.

## Acceptance criteria
- [x] All new metrics and fields appear in `per_file.csv`, `summary.json`, the run manifest, and `comparison.csv` (unit-verified; see Implementation Notes)
- [x] `FIELDNAMES` in `reporter.py:17` updated
- [ ] A re-run of the existing baseline produces DER unchanged at 17.05% -- **needs a real GPU/AMI run to verify, not yet executed**
- [ ] Region census on the baseline run is non-zero and the three regions (T∩D, T\D, D\T) are disjoint -- **needs a real GPU/AMI run to verify, not yet executed**

## Out of scope
Any new metric beyond those listed.

## Dependencies
Blocks [[T3-oracle-segmentation-provider]] and [[T2-post-clustering-refinement-hook]] (both feed the manifest fields this ticket defines). Can run in parallel with T2.

## Implementation Notes

(append here as work happens — decisions, rejected alternatives, gotchas, key files/functions touched)

- 2026-09-13: Wrote failing tests only (TDD red phase) across `tests/test_scorer.py`, `tests/test_reporter.py`, `tests/test_run_manifest.py`, `tests/test_aggregate_runs.py`. No source changes yet under `harness/`.
- Confirmed via `pyannote.metrics` source (`identification.py`, `matcher.py`, `base.py`) that `metric(reference, hypothesis, uem=uem, detailed=True)` returns both the ratio and the raw component durations (`"missed detection"`, `"false alarm"`, `"confusion"`, `"correct"`, `"total"`) in one call, so `score()` can get components without a second scoring pass or a second accumulator call — important since the existing tests assert each metric is called exactly once per file.
- Verified independently (Hungarian mapper re-optimizes the reference/hypothesis label mapping *within whatever region it's scored over*) that naive "swap two speaker labels" fixtures for `der_overlap_assigned` don't produce real confusion — the mapper just re-solves and calls it correct. The fixture that actually forces a nonzero, non-recoverable value needs a genuine detection shortfall (hypothesis emits fewer distinct speakers than the reference within T∩D), not a relabeling. Used a 3-speaker overlap where hypothesis only outputs 2 speakers in the intersection window.
- **Design decision (confirmed with the user before writing tests):** `der_overlap_assigned` is scored over a region (T∩D) distinct from both `der` (whole uem) and `der_overlap_system` (T), so it needs its own `DiarizationErrorRate` accumulator instance — otherwise `abs(metric)` would give the wrong corpus-level total for whichever metric's instance got reused. `score()` and `write_report()` both grow a 4th metric positional parameter (`der_overlap_assigned`), mirroring the existing `der`/`overlap_der`/`jer` pattern rather than refactoring to a dict/object param. This is a breaking signature change to both functions and to `run_harness.py`'s call sites (not yet updated — orchestrator wiring is part of the implementation pass, not this tests-only pass).
- Region census (`region_t_and_d`, `region_t_minus_d`, `region_d_minus_t`) has no corresponding `pyannote.metrics` accumulator (they're raw durations, not error rates) — tests assert the reporter sums these three fields itself across per-file rows for the corpus-level summary, rather than reading them from an `abs(metric)`-style object.
- `run_manifest.py`: existing `_run_config()` test fixture used `condition: "IHM"`, which predates T1's controlled vocabulary and isn't a valid value. Changed the fixture default to `condition: "baseline"` and moved `"IHM"` to the new `mic_condition` field — this touches the pre-existing manifest tests' fixture but not their assertions.
- `refinement_strategy` controlled vocabulary includes `"oracle"` even though T2 (this batch's other in-progress ticket) only implements `identity`/`nearest_centroid` — `oracle` is T4's strategy, but the manifest's vocabulary needs to accept it now so T4 doesn't have to touch `run_manifest.py` later.
- Confirmed `aggregate_runs.py` needs **no code change** for T1: `_flatten_manifest` already dynamically unions every `run_config`/`summary` key into CSV columns. Added a regression test (`test_aggregate_includes_t1_condition_and_metric_fields`) that passes green today, proving the plumbing already carries the new fields through rather than leaving this unverified.
- First run of the full T1 test set: 29 failed / 20 passed across `test_scorer.py`, `test_reporter.py`, `test_run_manifest.py`, `test_aggregate_runs.py` (the 20 passes are pre-existing behavior untouched by T1, plus the one new aggregate_runs regression test). All 29 failures are `TypeError` (signature mismatches in `score()`/`write_report()`/old `write_manifest()` not yet validating controlled vocab) or `KeyError` (new dict keys not yet produced) — confirms genuine red state, not test-authoring bugs.

- 2026-09-17: Implemented all three modules; full unit suite (`test_scorer.py`, `test_reporter.py`, `test_run_manifest.py`, `test_aggregate_runs.py`) is green: 49/49.
  - `harness/scorer.py`: `score()` now takes a 4th positional metric param `der_overlap_assigned`, matching the confirmed design. DER components come from a single `der(reference, hypothesis, uem=uem, detailed=True)` call (returns both the ratio, keyed by `pyannote.metrics.diarization.DER_NAME`, and raw component durations keyed by the `matcher.py` constants `"missed detection"`/`"false alarm"`/`"confusion"`) — this avoids a second scoring pass and keeps `der` called exactly once per file, preserving the existing "each metric called exactly once" contract. `der_overlap_system` (renamed from `overlap_der`) and `der_overlap_assigned` are still scored via plain `metric(..., uem=...)` calls, since only the component breakdown was needed for `der`. Region census durations use `Timeline.crop(mode="intersection")` / `Timeline.extrude()`.
  - `harness/reporter.py`: `write_report()` grows the matching 4th positional param. `FIELDNAMES` extended with the 7 new columns. Corpus summary reads `der`/`der_overlap_system`/`der_overlap_assigned`/`jer` via `abs(metric)` as before; region census fields have no `pyannote.metrics` accumulator (they're raw durations, not ratios), so the reporter sums them directly across `rows` for the corpus-level summary — the one place T1 deviates from "always read accumulator totals, never recompute from rows," and it's a deliberate, necessary exception (there's nothing to accumulate into).
  - `harness/run_manifest.py`: added `VALID_CONDITIONS` and `VALID_REFINEMENT_STRATEGIES` tuples and validation before the missing-fields check; `counts_toward_results` is merged into a fresh `run_config` dict (`{**run_config, "counts_toward_results": run_config.get(..., False)}`) before hashing/writing, so the run_id hash and the written manifest both reflect the defaulted value consistently. This meant one pre-existing test (`test_manifest_contains_full_run_config_and_summary`) needed updating — it asserted `written["run_config"] == run_config` (byte-identical), which is no longer true now that a field is always added; fixed to assert equality against `{**run_config, "counts_toward_results": False}` instead. This is a legitimate contract change caused by T1, not a bug in the old test.
  - `run_harness.py` (orchestrator): added a 4th `DiarizationErrorRate` instance for `der_overlap_assigned`, threaded through `score()`/`write_report()`. Wired `refinement_strategy` into the pipeline via T2's `harness.refinement.get_refinement_strategy()`, setting `pipeline.refinement` before constructing the `Runner` (confirmed with T2's implementer that `self.refinement` is a plain settable instance attribute on `SpeakerDiarization`, not init-only). Added `_current_git_commit()` (shells out to `git rev-parse --short HEAD`) for the new `git_commit` manifest field.
  - **Naming collision resolved (confirmed with user):** the existing `--condition` CLI flag means AMI mic condition (`IHM`/`SDM`), which predates T1 and collides with T1's new `condition` controlled vocabulary (`baseline`/`oracle_segmentation`/etc). Kept `--condition` meaning mic condition unchanged (feeds the new `mic_condition` manifest field) and added a new `--run-condition` flag (default `"baseline"`) for T1's `condition` field, to avoid breaking any existing script/doc that already calls `--condition IHM`. Also added `--refinement-strategy`, `--corpus`, and `--counts-toward-results` CLI flags.
  - Verified `tests/test_run_harness_integration.py`'s 9 errors (`FileNotFoundError` for a test audio fixture `trñ00.wav`) are pre-existing and unrelated to this work — reproduced identically via `git stash` against the pre-T1/T2 tree. Not fixed as part of this ticket; likely a checkout/encoding issue with a special-character filename on this machine.
  - **Not yet verified:** the two acceptance criteria requiring a real AMI/GPU run (exact 17.05% DER reproduction, non-zero disjoint region census on a real run) — these need `run_harness.py` executed against real data, which hasn't happened in this pass. Everything else is unit-test verified.
