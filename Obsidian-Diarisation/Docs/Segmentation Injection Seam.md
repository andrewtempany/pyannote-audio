---
status: done
created: 2026-09-04
---

# Segmentation Injection Seam

> Part of the [[Evaluation Harness]]. This page documents a **finding about pre-existing `pyannote-audio` behavior** and the calling-convention decision this project made in response. **No source file inside `src/pyannote/audio` was modified.** The only new code is in this project's own `harness/segmentation.py` (see [[Segmentation Source Interface]]), which relies on the convention documented here.

## Why this exists

The harness needs to eventually swap the pipeline's own segmentation output for ground-truth ("oracle") segmentation, to measure how much diarisation error is attributable to the segmentation stage versus downstream stages. The design assumed `SpeakerDiarization` already had a usable injection seam for this: `get_segmentations()` checks `file[self.CACHED_SEGMENTATION]` and skips the segmentation model if that key is present.

That check is real — it lives in pre-existing pyannote-audio code at `src/pyannote/audio/pipelines/speaker_diarization.py:305-329` (`CACHED_SEGMENTATION` is a `@property` returning the string `"training_cache/segmentation"`) — but it is **nested inside `if self.training:`**. During a normal inference call (`pipeline.training` is falsy by default), `get_segmentations()` unconditionally runs the segmentation model regardless of what's already on `file` — `CACHED_SEGMENTATION` is never even consulted:

```python
# src/pyannote/audio/pipelines/speaker_diarization.py (pre-existing, unmodified)
if self.training:
    if self.CACHED_SEGMENTATION in file:
        segmentations = file[self.CACHED_SEGMENTATION]
    else:
        segmentations = self._segmentation(file, hook=hook)
        file[self.CACHED_SEGMENTATION] = segmentations
else:
    segmentations: SlidingWindowFeature = self._segmentation(file, hook=hook)
```

Left unaddressed, this would have made the entire reason [[Segmentation Source Interface]] exists silently not work — `OracleSegmentation` could populate `file[CACHED_SEGMENTATION]` all it wanted and the pipeline would ignore it outside training.

## The finding: this is a calling convention, not a bug to patch

Investigation (reading `pyannote-pipeline`'s base `Pipeline` class at `pyannote/pipeline/pipeline.py:60-71` — a separate sister package, not this fork) confirmed that `pipeline.training` is a **plain bool attribute** on the base `Pipeline` class. It is unrelated to `nn.Module.train()`/`.eval()` and their dropout/batch-norm effects, and it propagates recursively to sub-pipelines.

Conclusion: the fix is a **calling convention**, not a source patch. Temporarily setting `pipeline.training = True` around a call that should honor `CACHED_SEGMENTATION`, then restoring it afterward, makes the existing check fire — with no modification to `speaker_diarization.py` or any other pyannote-audio file. `get_embeddings()` already uses the identical `if self.training:` gate for its own caching, so this is an existing, intended seam in the pipeline — just previously undocumented for inference-time use.

**No source patch to `src/pyannote/audio/pipelines/speaker_diarization.py` was made or is needed.**

## Where this convention is actually used

[[Segmentation Source Interface]]'s `populate(pipeline, file)` hook is the place this convention gets invoked. `BaselineSegmentation.populate()` never touches `pipeline.training` (baseline must remain untouched, default pipeline behavior). The still-stubbed `OracleSegmentation.populate()` documents that its real implementation will need to set `pipeline.training = True` around setting `file[pipeline.CACHED_SEGMENTATION]` — that wiring is left for a future ticket; this seam only establishes and proves the mechanism works.

## Verification performed

Encoded in `tests/test_segmentation_seam.py` (all passing):

1. `test_reproduces_bug_cached_segmentation_ignored_outside_training` — confirms the described behavior: with `pipeline.training` at its default, a pre-populated `CACHED_SEGMENTATION` is ignored and the segmentation model runs anyway.
2. `test_cached_segmentation_is_used_via_chosen_fix` — with `pipeline.training = True` set around the call, confirms the segmentation model is *not* invoked and the pre-populated value is used instead.
3. `test_fix_has_no_side_effect_on_normal_output` — runs the pipeline end-to-end with and without the `training=True` mechanism active (but no pre-populated cache either way), and diffs the resulting `Annotation`s to confirm they're identical — ruling out the concern that flipping `training=True` silently changes dropout, batching, or augmentation behavior.

## Test scaffolding note (relevant to anyone extending these tests)

Building a real, fully offline `SpeakerDiarization` pipeline for these tests required more scaffolding than a simple mock:

- `PLDA` is always loaded in `SpeakerDiarization.__init__` regardless of clustering method, so the fixtures build one from small, well-conditioned synthetic `.npz` matrices (not real trained weights).
- `apply()` always extracts embeddings before clustering, even under `OracleClustering` — so a real (tiny) embedding model is unavoidable for any full `apply()`-level test.
- The repo's own `SimpleEmbeddingModel` debug model doesn't accept the `weights=` mask kwarg that `get_embeddings()` always passes; a local `_MaskAwareEmbeddingModel` subclass in `tests/conftest.py` ignores that kwarg — good enough for a deterministic (not accurate) embedding.

These fixtures (`protocol`, `trained_segmentation_model`, `trained_embedding_model`, `dummy_plda`, `pipeline`, `full_pipeline`, `one_file`) now live in `tests/conftest.py`, shared with [[Segmentation Source Interface]]'s and [[Orchestrator]]'s tests — do not recreate them locally in a new test file.
