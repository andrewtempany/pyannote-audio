# 04 — Segmentation source interface

Part of the [eval harness epic](TICKET-eval-harness.md). Depends on: [01](TICKET-01-segmentation-injection-seam.md).

## Scope

`harness/segmentation.py` — a small interface with two implementations:

- `BaselineSegmentation`: uses the pipeline's own segmentation. Populates nothing on the file dict; must not wrap or reimplement anything the pipeline already does.
- `OracleSegmentation`: stub only. Raises `NotImplementedError`. Docstring points at `pyannote.audio.pipelines.utils.oracle.oracle_segmentation(file, window, frames, num_speakers=None)` (confirmed to exist at that exact path/signature, returns a `(num_chunks, num_frames, num_speakers)` `SlidingWindowFeature`, already used for oracle clustering at `src/pyannote/audio/pipelines/clustering.py:712-715`) as the utility the real implementation will build on.

Each source needs exactly two things: an `id` string (feeds the runner's cache key, ticket 06) and a hook that may populate `file[CACHED_SEGMENTATION]` before the pipeline runs, using whichever mechanism ticket 01 established to make that key actually honored outside training.

## Non-goals

- Do not implement oracle-segmentation injection logic (future ticket).
- Do not have `BaselineSegmentation` invoke ticket 01's fix mechanism at all — baseline must be pipeline-default behavior, untouched.

## Tests first

Add `tests/test_segmentation.py`:

1. `test_baseline_id_is_stable_string` — `BaselineSegmentation().id` returns the same value across instantiations (it's going into a cache key — non-determinism here would silently break caching).
2. `test_baseline_populate_hook_is_noop` — call the populate hook on a file dict, assert the dict is unchanged (no `CACHED_SEGMENTATION` key added, nothing else mutated).
3. `test_baseline_end_to_end_runs_pipeline_normally` — run the real pipeline through `BaselineSegmentation`'s hook on one real file, assert the segmentation model was actually invoked (inverse of ticket 01's seam test — baseline must NOT trigger the injection path).
4. `test_oracle_raises_not_implemented` — instantiating or calling `OracleSegmentation`'s populate hook raises `NotImplementedError`.
5. `test_oracle_docstring_references_oracle_segmentation_util` — a simple string-contains check on the docstring/module for the `oracle_segmentation` import path, so the pointer doesn't silently rot if the utility moves.
6. `test_interface_contract` — both `BaselineSegmentation` and `OracleSegmentation` expose the same `id` property and populate-hook method signature (duck-type or ABC check), since the runner (ticket 06) must be able to swap them with no other code changes.

## Acceptance criteria

- All tests pass.
- Swapping `BaselineSegmentation` for `OracleSegmentation` in the runner requires no changes outside `segmentation.py` (verified structurally by test 6, and functionally once ticket 06's runner test exercises both).

## Implementation notes (resolved during build — 2026-09-04)

- `SegmentationSource` is an `abc.ABC` with an abstract `populate(self, pipeline, file)` method and a required `id` class attribute (`harness/segmentation.py`) — the interface contract is enforced by Python itself (subclassing requires implementing `populate`), not just checked by a duck-typing test.
- `populate()`'s signature is `(pipeline, file)`, not just `(file)` — even though the ticket's Scope text describes it as populating `file`, the mechanism from TICKET-01 (`pipeline.training = True`) needs access to the pipeline object itself, so both baseline and the future real oracle implementation share the same two-argument shape.
- `BaselineSegmentation.populate()` is confirmed (via `test_baseline_end_to_end_runs_pipeline_normally`, reusing TICKET-01's real offline pipeline fixture) to never touch `pipeline.training` — the segmentation model runs normally, unmodified.
- Reuses the `pipeline`/`one_file` fixtures from `tests/conftest.py` (moved there during this ticket from `test_segmentation_seam.py`) rather than rebuilding the offline-pipeline scaffolding again.
