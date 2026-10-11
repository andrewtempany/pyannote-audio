---
status: done
created: 2026-09-04
---

# Segmentation Source Interface

> Part of the [[Evaluation Harness]]. **`harness/segmentation.py` is code contributed by this project — not upstream `pyannote-audio`.** It directly depends on pre-existing pyannote-audio internals in two places, both cited below: the `CACHED_SEGMENTATION` seam on `SpeakerDiarization` (`src/pyannote/audio/pipelines/speaker_diarization.py`) and, for the still-stubbed oracle path, `pyannote.audio.pipelines.utils.oracle.oracle_segmentation` (`src/pyannote/audio/pipelines/utils/oracle.py`).

## Purpose

An injectable seam so the harness can later swap the pipeline's own neural segmentation output for ground-truth ("oracle") segmentation, without changing the [[Runner]] or the pipeline itself. This lets a future phase of the project measure how much diarisation error is fixable downstream of the segmentation stage.

## Where it fits

The [[Runner]] takes a `SegmentationSource` instance as a constructor argument and calls its `populate()` hook immediately before invoking the pipeline. Swapping `BaselineSegmentation` for `OracleSegmentation` requires no change outside this file — the [[Orchestrator]] simply passes a different instance in.

## Interface

```python
from harness.segmentation import BaselineSegmentation, OracleSegmentation, SegmentationSource
```

`SegmentationSource` is an `abc.ABC` (not just a documented duck-type contract — Python itself enforces it: subclassing requires implementing `populate`):

- `id: str` — a stable class attribute. Feeds the [[Runner]]'s cache key, so it must never vary across instantiations (non-determinism here would silently break caching).
- `populate(self, pipeline, file) -> None` — called before the pipeline is applied to `file`. May populate `file[pipeline.CACHED_SEGMENTATION]`; must not otherwise mutate `pipeline` or `file`.

Note the hook's signature is `(pipeline, file)`, not just `(file)` — even though earlier design notes described it as populating `file` alone, actually using `CACHED_SEGMENTATION` outside training requires the `pipeline.training = True` mechanism established in [[Segmentation Injection Seam]], which needs access to the pipeline object itself. Both implementations share this two-argument shape.

### `BaselineSegmentation`

```python
BaselineSegmentation().id  # "baseline"
```

Uses the pipeline's own segmentation — normal, unmodified behavior. `populate()` is a no-op: it doesn't touch `file`, and — confirmed by test — it **never touches `pipeline.training`**. The segmentation model runs exactly as it would with no harness involved at all. This must not wrap or reimplement anything the pipeline already does.

### `OracleSegmentation`

```python
OracleSegmentation().id  # "oracle"
OracleSegmentation().populate(pipeline, file)  # raises NotImplementedError
```

Still a stub. Its docstring points at the pre-existing utility the real implementation will build on:

> `pyannote.audio.pipelines.utils.oracle.oracle_segmentation(file, window, frames, num_speakers=None)` — confirmed to exist at that exact path/signature (`src/pyannote/audio/pipelines/utils/oracle.py`). It discretises a reference `Annotation` into the `(num_chunks, num_frames, num_speakers)` `SlidingWindowFeature` shape that `CACHED_SEGMENTATION` expects, and is already used for oracle *clustering* (pre-existing code, `src/pyannote/audio/pipelines/clustering.py`, around the `oracle_segmentation(file, window, frames=frames)` call) — so the shape contract is known-good.

Wiring this up — computing the value, setting `file[pipeline.CACHED_SEGMENTATION]`, and engaging the `pipeline.training = True` seam from [[Segmentation Injection Seam]] — is left to a future ticket. This stub only reserves the interface shape and fails loudly (`NotImplementedError`) rather than silently doing nothing or something wrong.

## How the runner consumes this

See [[Runner]]: the cache-existence check happens *before* `populate()` is ever called, so on a cache hit neither the segmentation source nor the pipeline is touched at all. This is also what makes `OracleSegmentation`'s `NotImplementedError` fail-fast rather than partial: it can only ever surface on a cache miss, before any pipeline call or cache write happens.

## Non-goals

- No oracle-segmentation injection logic is implemented yet (future ticket).
- `BaselineSegmentation` never invokes the [[Segmentation Injection Seam]] mechanism at all — baseline must stay pipeline-default behavior, untouched.
