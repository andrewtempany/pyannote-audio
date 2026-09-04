# 01 — Segmentation injection seam: make CACHED_SEGMENTATION actually work outside training

Part of the [eval harness epic](TICKET-eval-harness.md). Depends on: [00](TICKET-00-environment-setup.md).

## Problem

The epic's Component 4 assumed the injection seam for oracle segmentation was: `SpeakerDiarization.get_segmentations()` checks `file[self.CACHED_SEGMENTATION]` and skips the model if present. That check is real (`src/pyannote/audio/pipelines/speaker_diarization.py:305-329`, `CACHED_SEGMENTATION` is a `@property` returning `"training_cache/segmentation"`), but it is nested inside `if self.training:` (line 321). During normal inference (`pipeline.training` is falsy by default), the method unconditionally runs the segmentation model regardless of what's on `file` — the cached key is never even consulted. If this ticket is skipped, ticket 04's `OracleSegmentation` (and by extension the entire reason the interface exists) will not work later, and nobody will notice until that ticket is implemented.

## Scope

- Determine the minimal, safe way to make a pre-populated `file[CACHED_SEGMENTATION]` actually get used during a normal (non-training) `pipeline(file)` call. Candidate approaches to evaluate:
  - Temporarily setting `pipeline.training = True` around the call — check what else that flag gates elsewhere in `speaker_diarization.py` and in any base class, and confirm it has no other observable side effect on inference output.
  - Subclassing `SpeakerDiarization` and overriding `get_segmentations()` to drop the `self.training` gate.
  - Any other seam the pipeline exposes that isn't the `self.training`-gated one (re-check the surrounding code for alternatives before committing to a workaround).
- Document the decision and rationale directly in code (a short comment at the point of use is enough — this doesn't need a design doc).
- This ticket only needs to prove the seam works for an arbitrary pre-populated `SlidingWindowFeature`. It does not need to build the real oracle segmentation logic (that's a future ticket per the epic's non-goals) — a dummy/fake cached segmentation value is sufficient for the test.

## Non-goals

- Do not implement `oracle_segmentation` wiring (future ticket, per epic).
- Do not modify `models/` or `tasks/`.
- Do not change `CACHED_SEGMENTATION`'s value or any other pipeline default.

## Tests first

Add `tests/test_segmentation_seam.py`:

1. `test_reproduces_bug_cached_segmentation_ignored_outside_training` — build a real `SpeakerDiarization` pipeline instance (loaded pretrained, or the smallest fixture that instantiates the class), patch/spy `self._segmentation` (or whatever internal callable actually runs the model) with a call counter, set `file[pipeline.CACHED_SEGMENTATION]` to a dummy `SlidingWindowFeature`, call `get_segmentations(file)` with `pipeline.training` left at its default, and assert the spy **was** called (i.e., confirm the bug exists before fixing it — this test documents the failure mode and should be deleted or inverted once the fix lands, whichever keeps the suite meaningful).
2. `test_cached_segmentation_is_used_via_chosen_fix` — same setup, but apply whatever mechanism this ticket decides on (e.g., the `training=True` context, or the subclass override), and assert the spy was **not** called, and that the value returned by `get_segmentations()` is derived from the injected dummy value, not a freshly computed one.
3. `test_fix_has_no_side_effect_on_normal_output` — run the pipeline end-to-end on one real file with the fix's mechanism active but *without* pre-populating `CACHED_SEGMENTATION`, and diff the resulting `Annotation` against a normal run without the fix's mechanism active. They should be identical — this catches the case where flipping `training=True` silently changes something else (e.g., dropout, batching, augmentation).

## Acceptance criteria

- All three tests pass.
- The chosen mechanism is exposed as something ticket 04 can call cleanly (a documented function or context manager) — decide the shape here so ticket 04 doesn't have to re-derive it.
- A one- or two-line comment at the fix site explains why the workaround exists, since it will look surprising to a future reader who hasn't seen this investigation.

## Implementation notes (resolved during build — 2026-09-04)

- **No source patch to `speaker_diarization.py` was needed.** Confirmed via `pyannote/pipeline/pipeline.py:60-71` (the sister `pyannote-pipeline` package, not this fork) that `pipeline.training` is a plain bool attribute on the base `Pipeline` class — unrelated to `nn.Module.train()`/`.eval()` and their dropout/batchnorm effects, and it recursively propagates to sub-pipelines. The "fix" is a calling convention: temporarily set `pipeline.training = True` around the call that should honor `CACHED_SEGMENTATION`, then restore it. `get_embeddings()` already uses the identical `if self.training:` gate for its own caching, so this is an existing, intended seam — just undocumented for inference use.
- Tests live in `tests/test_segmentation_seam.py`. The fixtures that build a real, fully offline `SpeakerDiarization` pipeline (`protocol`, `trained_segmentation_model`, `trained_embedding_model`, `dummy_plda`, `pipeline`, `full_pipeline`, `one_file`) were later moved to `tests/conftest.py` since TICKET-04 and TICKET-06 need the identical fixture — don't recreate them locally in new test files, import via conftest.
- Building that offline pipeline required more scaffolding than expected: `PLDA` is always loaded in `SpeakerDiarization.__init__` regardless of clustering choice (built here from small, well-conditioned synthetic `.npz` matrices, not real weights), and `apply()` always extracts embeddings before clustering even under `OracleClustering` — so a real (tiny) embedding model is unavoidable for any full `apply()`-level test, not just the ones that look like they need one. The repo's own `SimpleEmbeddingModel` debug model doesn't accept the `weights=` mask kwarg that `get_embeddings()` always passes, so a local `_MaskAwareEmbeddingModel` subclass (in conftest.py) ignores it — fine for producing *a* deterministic embedding, not an accurate one, which is all these tests need.
- TICKET-04's `OracleSegmentation` stub already documents that the real implementation will need to invoke this `pipeline.training = True` mechanism itself when it's built — that ticket only reserves the interface shape, it doesn't wire the mechanism up yet.
