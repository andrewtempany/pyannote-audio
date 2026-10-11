---
status: reference
scope: Pre-existing (upstream pyannote-audio)
source: src/pyannote/audio/pipelines/
---

# `pipelines/` — inference-time composition

> **Pre-existing (upstream pyannote-audio).** Nothing in this repo modifies these files. This note documents, in detail, the seam `harness/` plugs into — see **"Extension point used by harness/"** below. `harness/` itself is out of scope for this note (see the harness's own docs).

`pipelines/` is where trained `Model`s (see [[Models-Subpackage]]) get composed into end-to-end, user-facing pipelines (see [[Core-Pipeline]] for the base `Pipeline` class every one of these subclasses). ARCHITECTURE.md names the files at a glance; this note goes one level deeper into the shared utilities and the flagship `SpeakerDiarization` pipeline.

## Shared utilities (`pipelines/utils/`)

**`getter.py`** — the uniform loader pattern used in almost every pipeline `__init__`: accept an already-built instance, a string (local path or HF Hub id), or a dict of `from_pretrained`-style kwargs, and return a ready instance. `get_model` / `get_pipeline` / `get_plda` / `get_calibration` / `get_augmentation` / `get_devices`, with matching type aliases (`PipelineModel`, `PipelinePLDA`, `PipelineCalibration`, `PipelineAugmentation`) used in constructor signatures across the subpackage. `get_model` always calls `.eval()` on the result.

**`diarization.py`** — `set_num_speakers(num_speakers, min_speakers, max_speakers)` reconciles the three into a consistent triple (`num_speakers` forces `min == max`; raises if `min > max`). `SpeakerDiarizationMixin` bundles static helpers shared by any diarization-shaped pipeline:
- `optimal_mapping` — best bijective label mapping between a reference `Annotation` and a hypothesis, via `pyannote.metrics.diarization.DiarizationErrorRate.optimal_mapping`.
- `speaker_count` — frame-level instantaneous active-speaker count, via `Inference.trim` + `Inference.aggregate` (see [[Core-Inference]]).
- `to_annotation` — discrete `(frame, speaker)` binary array → continuous `Annotation`, via `utils.signal.Binarize`.
- `to_diarization` — discrete per-chunk segmentations + a precomputed instantaneous speaker `count` → one discrete per-speaker timeline, keeping only the `count[t]` most active speakers at each frame (`np.argsort(-activations)`).
- `classes()` — infinite `SPEAKER_00, SPEAKER_01, ...` generator, the default naming scheme when no reference is available to map against.

**`hook.py`** — the concrete callables passed as `hook=` to `pipeline(file, hook=...)` (see [[Core-Pipeline]]'s `setup_hook`): `ArtifactHook` (records each named step's artifact into `file[file_key][step_name]`), `ProgressHook` (Rich progress bar, one bar per step name), `TimingHook` (per-step wall-clock elapsed time into `file[file_key]`), and `Hooks` (compose several as one). Every concrete pipeline's `apply()` calls `hook(step_name, artifact, file=file, total=..., completed=...)` after each major stage — any custom hook (e.g. one an evaluation harness might add for instrumentation) needs to accept exactly that signature.

**`oracle.py`** — `oracle_segmentation(file, window, frames, num_speakers=None)`: builds a "ground-truth" segmentation `SlidingWindowFeature` by discretizing `file["annotation"]` (the reference) at the given chunk window and frame resolution, in the exact `(num_chunks, num_frames, num_speakers)` shape a real segmentation model's `Inference` output would have (see its docstring's `oracle = Model.from_pretrained("oracle")` sketch of the model it stands in for). If more speakers are requested than actually present, it pads with fake speaker labels; if fewer, it keeps the `num_speakers` most talkative ones. This function is reused twice: by `clustering.py`'s `OracleClustering` (below), and — per its own docstring and `harness/segmentation.py`'s `OracleSegmentation` stub — is the function a future oracle-segmentation-injection feature in `harness/` is meant to build on.

## `clustering.py`

`BaseClustering` (a `pyannote.pipeline.Pipeline` itself, so its own hyperparameters can be optimized) provides `set_num_clusters` (reconciles num/min/max clusters against how many embeddings are actually available) and `filter_embeddings` (drops NaN embeddings and speakers active less than `min_active_ratio` of a chunk before clustering). Concrete algorithms are exposed via the `Clustering` enum (e.g. the default `VBxClustering`, PLDA-scored via `utils/vbx.py`'s `cluster_vbx`, plus agglomerative-style and `OracleClustering` options).

**`OracleClustering`** is worth calling out specifically: instead of clustering embeddings, its `cluster()` calls `oracle_segmentation(file, window, frames=frames)` to get the true per-speaker activity, then uses `utils.permutation.permutate` to find the permutation that best aligns the pipeline's *own* segmentation output with the oracle one, deriving hard/soft cluster assignments (and, if real embeddings were supplied, centroids) from that alignment rather than from any actual distance-based clustering. This is the pre-existing "evaluate with ground-truth clustering, keep everything else real" mode; it's structurally the sibling of what an oracle *segmentation* source would do, just one pipeline stage later.

## `speaker_diarization.py` — the flagship `SpeakerDiarization` pipeline

Composes: a segmentation `Model` wrapped in an `Inference` (`self._segmentation`, `skip_aggregation=True` — segmentation output must stay per-chunk, un-aggregated, because clustering needs chunk-level speaker identities before they can be stitched together) → an embedding model + `PLDA` (unless `clustering="OracleClustering"`, though note the embedding model is still always constructed and always used at `apply()` time regardless of clustering choice — there's no clustering mode that skips extracting embeddings) → a `Clustering` algorithm instance.

**`DiarizeOutput`** (dataclass, the return type of `apply()` unless `legacy=True`):
- `speaker_diarization: Annotation` — may contain overlapping speech.
- `exclusive_speaker_diarization: Annotation` — the same timeline with overlap resolved down to at most one active speaker per frame, meant for feeding a downstream transcription step that can't handle overlap.
- `speaker_embeddings: np.ndarray | None` — one centroid per output speaker, row-order aligned to `speaker_diarization.labels()`.
- `.serialize()` — both annotations as lists of `{start, end, speaker}` dicts.
- `legacy=True` on the pipeline constructor makes `apply()` return the bare `speaker_diarization` `Annotation` instead of the full `DiarizeOutput`, for callers written against the pre-4.x API.

### `CACHED_SEGMENTATION` and the training-cache gate — **extension point used by harness/**

```python
@property
def CACHED_SEGMENTATION(self):
    return "training_cache/segmentation"

def get_segmentations(self, file, hook=None) -> SlidingWindowFeature:
    ...
    if self.training:
        if self.CACHED_SEGMENTATION in file:
            segmentations = file[self.CACHED_SEGMENTATION]
        else:
            segmentations = self._segmentation(file, hook=hook)
            file[self.CACHED_SEGMENTATION] = segmentations
    else:
        segmentations = self._segmentation(file, hook=hook)
    return segmentations
```

`get_segmentations()` only ever looks at `file[self.CACHED_SEGMENTATION]` when `self.training` is truthy — see [[Core-Pipeline]] for the full explanation of `pipeline.training` (it's a `pyannote.pipeline.Pipeline` base-class flag meant for hyperparameter-search speedups, unrelated to `nn.Module.train()`). Since `training` defaults to `False` and stays `False` outside of an actual `pyannote.pipeline.optimizer.Optimizer` run, **a pre-populated `file[CACHED_SEGMENTATION]` is silently ignored during ordinary `pipeline(file)` inference** — the segmentation model runs unconditionally.

`harness/segmentation.py`'s `SegmentationSource` interface (`BaselineSegmentation` = do nothing; `OracleSegmentation` = stub, will eventually call `oracle_segmentation()` above) is built around exploiting this seam: its `populate(pipeline, file)` hook is meant to set `file[pipeline.CACHED_SEGMENTATION]` *and* engage the `pipeline.training = True` context around the `pipeline(file)` call, so that a precomputed (e.g. oracle) segmentation is actually used instead of the real model's output. `get_embeddings()` in this same file has the identical `if self.training:` gate for its own `file["training_cache/embeddings"]` cache — anything that flips `pipeline.training = True` to reach the segmentation cache will, as a side effect, also make embedding extraction cache-aware across repeated calls on the same `file` object.

### `get_embeddings(file, binary_segmentations, exclude_overlap, hook)`

Extracts one embedding per `(chunk, speaker)` pair. When `exclude_overlap=True`, frames where more than one speaker is simultaneously active are zeroed out of the mask used to select audio for each speaker's embedding — but falls back to the full (non-exclusive) mask if too few clean frames remain to safely extract an embedding (`min_num_frames`, derived from the embedding model's `min_num_samples`). Has its own `if self.training:` cache (see above) keyed on `file["training_cache/embeddings"]`, valid across repeated calls as long as the segmentation model is powerset-based or the `segmentation.threshold` hyperparameter hasn't changed since the cached value was stored — this is the caching mechanism the upstream hyperparameter-search flow relies on for speed.

### `apply()` walkthrough

1. Validate/reconcile `num_speakers`/`min_speakers`/`max_speakers` (`set_num_speakers`). If the chosen clustering algorithm expects a fixed cluster count (e.g. KMeans-style) and none was given, infer it from `file["annotation"]` if present (i.e. during evaluation against a labeled protocol) or raise.
2. `get_segmentations(file, hook)` → per-chunk speaker-activity scores.
3. Binarize (skipped entirely for powerset models, whose output is already discrete) via `utils.signal.binarize` at the `segmentation.threshold` hyperparameter.
4. `speaker_count` → frame-level instantaneous active-speaker count. If it's zero everywhere, short-circuit and return an empty `DiarizeOutput` immediately (no speech detected).
5. `get_embeddings(...)`.
6. `self.clustering(embeddings=..., segmentations=..., num_clusters=num_speakers, min_clusters=..., max_clusters=..., file=file, frames=self._segmentation.model.receptive_field)` — note `file`/`frames` are threaded through purely so `OracleClustering` can call `oracle_segmentation` itself; other clustering algorithms ignore them.
7. Cap instantaneous speaker count at `max_speakers`, force-assign inactive speakers to a throw-away cluster (`-2`), then `reconstruct()` turns hard cluster assignments + raw segmentation back into one discrete per-speaker timeline (`to_diarization` under the hood).
8. `to_annotation()` twice: once for the full diarization, once after capping `count` at 1 active speaker per frame for the exclusive variant.
9. Relabel: if a reference `annotation` is present on `file`, map hypothesis labels onto reference speaker names via `optimal_mapping` (extra hypothesis speakers not in the reference keep their original label); otherwise rename to `SPEAKER_00, SPEAKER_01, ...` via `classes()`.
10. Reorder `centroids` to match the final label order before building the returned `DiarizeOutput`.

## Sibling pipelines (brief)

- **`voice_activity_detection.py`** — binary speech/non-speech pipeline built the same way (segmentation model → `Inference` → binarize/reconstruct), without the embedding/clustering stages.
- **`multilabel.py`** — generic multi-label segmentation pipeline (e.g. overlapped speech detection) reusing the same `Model` → `Inference` → binarize shape for an arbitrary fixed label set.
- **`speech_separation.py`** — wraps `ToTaToNet` (see [[Models-Subpackage]]) to produce separated per-speaker audio streams alongside diarization.
- **`speaker_verification.py`** — `PretrainedSpeakerEmbedding` wrapper (imported by `speaker_diarization.py` as `self._embedding`) around an embedding `Model`, exposing `.dimension`, `.metric`, `.min_num_samples`, and `__call__(waveform, masks=...)` for extracting one embedding per masked region.

## `pyannoteai/` — the local/hosted duality, concretely

ARCHITECTURE.md names the mechanism (`Pipeline.from_pretrained` routes to a local graph or a hosted API depending on the token); here's the concrete shape both wrappers share with `SpeakerDiarization`:

- **`local.py`**'s `Local` — wraps the official (closed-source) `pyannoteai.local.Pipeline` on-premise package. `apply(file, num_speakers=, min_speakers=, max_speakers=)` forwards to `self._pipeline.diarize(...)` (passing either the file's `"audio"` path or its in-memory `"waveform"`/`"sample_rate"`), then `_deserialize()` turns the returned `list[{start, end, speaker}]` dicts back into `pyannote.core.Annotation`s, returned as the same `DiarizeOutput` dataclass `SpeakerDiarization.apply()` uses (minus `speaker_embeddings`, which this wrapper doesn't populate).
- **`sdk.py`**'s `SDK` — same shape, but calls the hosted HTTP API via `pyannoteai.sdk.Client` instead of a local package; takes a `model` argument (default `"precision-2"`) to select which hosted model to hit.

Because both implement the exact same `apply()` signature and return type as the local `SpeakerDiarization` pipeline, `Pipeline.from_pretrained("pyannote/speaker-diarization-precision-2", token=pyannoteai_api_key)` is a drop-in replacement for `Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")` everywhere in user code — the branching on token type happens once, at `from_pretrained`, not at every call site.
