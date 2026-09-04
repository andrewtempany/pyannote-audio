---
status: done
created: 2026-09-04
---

# Runner

> Part of the [[Evaluation Harness]]. **`harness/runner.py` is code contributed by this project — not upstream `pyannote-audio`.** It directly wraps a pre-existing pyannote-audio pipeline call and consumes its pre-existing output type: `DiarizeOutput`, defined at `src/pyannote/audio/pipelines/speaker_diarization.py`. It also uses `pyannote.database.util.load_rttm` (a sister package, no in-repo path) for cache reads.

## Purpose

Given a loaded pipeline, a [[Segmentation Source Interface|segmentation source]], and one file, produce a hypothesis `Annotation` — invoking the (expensive) pipeline only when a cached result doesn't already exist on disk.

## Where it fits

Constructed and driven by the [[Orchestrator]] for every file yielded by the [[Dataset Adapter]]. Its output feeds directly into [[Scorer]].

## API

```python
from harness.runner import Runner

runner = Runner(pipeline, pipeline_config_id, segmentation_source, cache_dir)
hypothesis = runner.run({"uri": uri, "audio": audio_path})  # -> pyannote.core.Annotation
```

## `DiarizeOutput`: score the full annotation, not the exclusive one

Pre-existing pyannote-audio code (`src/pyannote/audio/pipelines/speaker_diarization.py`) defines `DiarizeOutput` with **three** fields, not two:

- `speaker_diarization` — full annotation (includes overlap).
- `exclusive_speaker_diarization` — overlap-removed, meant for downstream transcription.
- `speaker_embeddings` — optional, one embedding per speaker.

The runner scores `output.speaker_diarization`. Using `exclusive_speaker_diarization` would silently break overlap scoring, since overlap regions are removed from it entirely — and overlap is the whole point of this project's metrics.

## `pipeline_config_id` is caller-supplied, not derived

`Runner.__init__` takes `pipeline_config_id` as an explicit string parameter rather than trying to introspect the live `Pipeline` object. A loaded `SpeakerDiarization` pipeline holds non-serializable state (loaded models, a `PLDA` instance, etc.), so hashing it directly isn't practical. It's the caller's responsibility (the [[Orchestrator]]) to generate a stable id — e.g. an HF checkpoint id/revision, or a hash of instantiated hyperparameters — when it constructs the real pipeline.

## Caching

- **Cache key** = `sha256(f"{pipeline_config_id}|{segmentation_source.id}|{uri}")`.
- **Cache file path** = `{cache_dir}/{key}.rttm`.
- On a cache **hit**, the runner loads `pyannote.database.util.load_rttm(path)[uri]` and returns it — the pipeline is never invoked.
- On a cache **miss**, the runner calls `segmentation_source.populate(pipeline, file)`, then `pipeline(file)`, extracts `.speaker_diarization`, writes it via `hypothesis.write_rttm(f)`, and returns it.
- Writing uses the same `pyannote.database.util` RTTM utilities the [[Dataset Adapter]] uses for reading ground truth — kept consistent rather than inventing a second RTTM reader/writer path.

**The cache-existence check happens first, before `populate()` is even called.** This means a still-stubbed `OracleSegmentation`'s `NotImplementedError` (see [[Segmentation Source Interface]]) always surfaces cleanly on a cache miss — never partway through a run, never masked by a partial cache write.

## Testing note

Tests 1–6 in `tests/test_runner.py` run fully mocked (no network/model required) and cover: cache-key sensitivity to each of pipeline config / source id / uri; cache-miss behavior (pipeline invoked, RTTM written); cache-hit behavior (pipeline spy count zero, correct `Annotation` returned); full-vs-exclusive field selection; the populate-hook-called-before-pipeline contract; and oracle's fail-fast behavior. `test_runner_real_pipeline_one_file` is the one integration test that exercises a real, credentialed `community-1` run — it's marked `pytest.mark.integration` (registered in the project's root `pyproject.toml`) and is skipped, not silently omitted, when no HF token is available; `pytest -m "not integration"` is the CI-safe filter.

## Non-goals

- No scoring logic — that's [[Scorer]], called by the [[Orchestrator]].
- No dataset parsing.
- Does not implement `OracleSegmentation`'s real logic; it must work correctly with either source via the interface, including surfacing the stub's failure loudly and early.
