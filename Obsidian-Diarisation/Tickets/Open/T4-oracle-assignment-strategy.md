---
status: in-progress
created: 2026-09-13
---

# T4: Oracle assignment strategy

Part of the oracle/ceiling analysis batch (see [[T1-harness-reporting-and-manifest-schema]], [[T2-post-clustering-refinement-hook]], [[T3-oracle-segmentation-provider]], [[T5-cross-condition-deltas]]). This is the decision-relevant ticket in the batch — baseline minus this result is the budget the intervention competes for.

## Goal

Measure the ceiling on post-clustering assignment, holding the real segmentation and real clusters fixed.

The strategy assigns each pair the cluster label that best matches ground truth. **It may only choose among clusters the pipeline actually produced.** It must not invent speakers the pipeline missed. That constraint is what makes the number an honest ceiling for an assignment intervention rather than a ceiling for the whole pipeline.

## Method
1. Compute an optimal mapping between hypothesis clusters and reference speakers, by total temporal overlap across the file
2. For each pair, determine its temporal support from the segmentation
3. Find the reference speaker with greatest overlap over that support
4. Assign the cluster that maps to that reference speaker
5. If the dominant reference speaker maps to no cluster, leave the pair unchanged

## Config

`oracle_scope`, with two values:
- `overlap_degraded` (**primary**): refine only pairs whose support is predominantly coincident with another speaker. Isolates the ceiling to the population the intervention targets.
- `all_pairs` (secondary): refine every pair. Gives the full post-clustering ceiling, useful as context.

Both are cheap once the strategy exists. Run both.

## Acceptance criteria
- [ ] Strategy implements the [[T2-post-clustering-refinement-hook]] interface
- [ ] On a fixture, `der_overlap_assigned` drops substantially against baseline
- [ ] Both `oracle_scope` values run and appear in `comparison.csv`
- [ ] The strategy never emits a cluster label absent from its input

## Dependencies
Blocked on [[T2-post-clustering-refinement-hook]]. Blocks [[T5-cross-condition-deltas]] (needs all conditions scored).

## Implementation Notes

(append here as work happens — decisions, rejected alternatives, gotchas, key files/functions touched)

- **Ticket has no "Tests first" section** (unlike some others in this batch). Proceeding by deriving tests from the Method/Config/Acceptance-criteria sections directly rather than stopping to ask, since this is a background/unattended run — documenting that choice here per the tdd-ticket skill's guidance to flag any deviation.
- **Signature problem discovered while reading T2's hook.** The fixed refinement interface is `strategy(embeddings, hard_clusters, soft_clusters, centroids, segmentations) -> hard_clusters` — there is no reference/ground-truth parameter, and no per-file hook at all: `run_harness.py` currently does `pipeline.refinement = get_refinement_strategy(name)` **once**, before the per-file loop. Oracle assignment fundamentally needs the reference `Annotation` for the file currently being scored, which isn't available at that point.
  - Rejected: smuggling reference into `segmentations` or a global/module-level variable — fragile, breaks the "operates on pairs and what clustering produced" contract T2 documents.
  - Chosen: `harness/refinement.py` exposes a **factory**, `make_oracle_strategy(reference, oracle_scope) -> Callable`, matching the normal 5-arg interface via closure. `run_harness.py`'s per-file loop, only when `refinement_strategy == "oracle"`, sets `pipeline.refinement = make_oracle_strategy(reference, oracle_scope)` immediately before `runner.run(file)` (identity/nearest_centroid runs are unaffected — `pipeline.refinement` is still set once, outside the loop, for those). This is a small, additive change to `run_harness.py`'s loop body, not a change to the T2 interface itself or to `speaker_diarization.py`.
- **Optimal hyp-cluster <-> reference-speaker mapping**: reused `pyannote.metrics.diarization.DiarizationErrorRate().optimal_mapping(reference, hypothesis, uem=None)` (returns `dict[hyp_label -> ref_label]`, computed by total temporal overlap via Hungarian matching) rather than hand-rolling an overlap matrix + assignment. Requires materializing `hard_clusters`+`segmentations` into a hypothesis `Annotation` first (helper `_reconstruct_hypothesis`), since that's what `optimal_mapping` takes.
- **Pair support**: helper `_pair_support(hard_clusters, segmentations)` builds a `Timeline` per `(chunk, local_speaker)` pair from that pair's active frames in `segmentations.data[chunk, :, local_speaker]`, mapped to real time via `segmentations.sliding_window[chunk]`.
- **`overlap_degraded` scope semantics** (not spelled out precisely in the ticket beyond "support is predominantly coincident with another speaker"): a pair is in-scope when, within its own temporal support, the reference annotation shows >=2 simultaneous speakers for more than half of that support's duration (i.e. the pair's audio is mostly overlapping speech in the ground truth). This directly matches "support predominantly coincident with another speaker" and is checkable purely from `reference.get_overlap()` cropped to the pair's support.
- **Never invents a missing cluster**: reverse mapping is built as `ref_speaker -> hyp_cluster` from `optimal_mapping`'s `hyp_cluster -> ref_speaker` dict. If the dominant reference speaker for a pair's support isn't a value in that dict (i.e. no hypothesis cluster maps to it), the pair is left unchanged (method step 5). Also: mapping only ever assigns labels that appeared in `hard_clusters` to begin with, since `optimal_mapping`'s hypothesis side is built only from clusters actually present.
- **`harness/refinement.py` (strategy) done and TDD'd**: `tests/test_oracle_refinement.py`, 9 tests, all real `pyannote.core` fixtures (no mocks) -- genuine red (`ImportError: cannot import name 'make_oracle_strategy'`) shown before implementing, then green after. Committed as `09b2d5e9`.
- **`run_harness.py`/manifest wiring is the remaining piece.** Plan: add an `oracle_scope: Optional[str] = None` param to `run_harness()` and a `--oracle-scope` CLI flag (choices `all_pairs`/`overlap_degraded`, default `all_pairs`); inside the per-file loop, when `refinement_strategy == "oracle"`, do `pipeline.refinement = make_oracle_strategy(reference, oracle_scope)` right before `runner.run(file)` for that file (identity/nearest_centroid keep the existing once-before-the-loop assignment, unaffected). Also add `oracle_scope` to the manifest's `run_config` dict (not to `REQUIRED_RUN_CONFIG_FIELDS` in `harness/run_manifest.py`, since it's meaningless/absent for non-oracle runs) so `aggregate_runs.py`'s generic flattening picks it up into `comparison.csv` automatically -- no changes needed there.
- **Gotcha found while writing the integration test**: `tests/test_run_harness_integration.py`'s own `full_pipeline` fixture is currently broken in this environment independent of T4 -- every one of its existing tests (not just new ones) errors with `FileNotFoundError: Could not find file "trñ00"` from `pyannote.database.file_finder`, before any of my code runs. Confirmed pre-existing/unrelated by running that file's tests unmodified. Replaced the planned fixture-based integration test with `tests/test_run_harness_oracle_wiring.py`, which mocks `AMIDatasetAdapter`/`Runner`/`make_oracle_strategy` instead -- proves run_harness.py really does call `make_oracle_strategy(reference, oracle_scope=...)` once per file with that file's own reference, and leaves identity/nearest_centroid's single up-front assignment untouched, without depending on the broken fixture.
- **`run_harness.py`/manifest wiring implemented and TDD'd**: added `oracle_scope: Optional[str] = None` param to `run_harness()`, an `--oracle-scope` CLI flag (choices `all_pairs`/`overlap_degraded`, default `all_pairs`), the per-file `pipeline.refinement = make_oracle_strategy(reference, oracle_scope=...)` branch (only when `refinement_strategy == "oracle"`), and `oracle_scope` added to the manifest's `run_config` dict (not to `REQUIRED_RUN_CONFIG_FIELDS`, since it's meaningless/None for non-oracle runs). `tests/test_run_harness_oracle_wiring.py` (2 tests) shown genuinely red first (`AttributeError: <module 'run_harness'> does not have the attribute 'make_oracle_strategy'`) then green. All 56 tests across `test_refinement.py`, `test_oracle_refinement.py`, `test_run_harness_oracle_wiring.py`, `test_run_harness_cli.py`, `test_run_manifest.py`, `test_aggregate_runs.py` pass with no regressions.
- **Note on concurrent T3 work**: while implementing this, `run_harness.py` was updated on disk mid-task by the concurrently-running T3 agent (added `--oracle-rttm`/`OracleSegmentation` wiring for oracle *segmentation*, a different ticket). Re-read the file before making further edits and applied T4's changes additively on top of T3's, rather than reverting/conflicting with them. Because we share one working directory, T3's own commit (`287ed25f`) ended up sweeping in my uncommitted `run_harness.py` oracle_scope edits alongside theirs (both agents' changes to the same file landed in one commit, authored under T3's message) -- the code content is correct and present either way, just noting the attribution mismatch here for the record rather than trying to rewrite shared history.
- **Real end-to-end verification run** (RTX 3060, real `pyannote/speaker-diarization-community-1` pipeline, `--device cuda`, tiny fixture AMI data at `tests/fixtures/ami/basic/IHM`, 2 files):
  - `--refinement-strategy oracle --oracle-scope all_pairs` (`--run-condition oracle_assignment`): ran end to end, manifest `runs/20260916T221535Z-59aa4f06.json`.
  - `--refinement-strategy oracle --oracle-scope overlap_degraded`: ran end to end, manifest `runs/20260916T221603Z-7e67cbbe.json`.
  - `--refinement-strategy identity` baseline comparator on the same fixture: manifest `runs/20260916T221648Z-9865c76b.json`.
  - Both `oracle_scope` values appear correctly in `runs/comparison.csv` with `refinement_strategy=oracle` and their respective `oracle_scope` values -- that acceptance criterion is met.
  - **Acceptance criterion "der_overlap_assigned drops substantially against baseline" is NOT demonstrated on this fixture, and can't be**: all three runs (oracle/all_pairs, oracle/overlap_degraded, identity baseline) report `der_overlap_assigned = 0.0` with `region_t_and_d = 0`. Checked why: the reference RTTM does contain real overlap (`ES2002a` has spk1/spk2 overlapping 3.0-5.0s, confirmed via `reference.get_overlap()`), but the real pipeline's hypothesis never predicted simultaneous local speakers anywhere in that window on this tiny 2-file fixture -- so `t ∩ d` (the region `der_overlap_assigned` is scored over) is empty for every refinement strategy alike, including baseline. This is a property of the fixture/pipeline-on-tiny-audio interaction, not a defect in the oracle strategy's logic (which is separately proven correct by the 9 unit tests in `tests/test_oracle_refinement.py` using hand-built overlap scenarios where reassignment demonstrably happens). Demonstrating this criterion for real needs a fixture (or the full AMI test split) where the pipeline actually detects overlapping speech somewhere -- flagging as open rather than claiming it's covered.
