---
status: reference
scope: Pre-existing (upstream pyannote-audio)
source: src/pyannote/audio/core/pipeline.py
---

# `core.Pipeline`

> **Pre-existing (upstream pyannote-audio).** Nothing in this repo modifies this file. This note contains the `pipeline.training` extension point that `harness/` relies on — see the dedicated section below, marked **extension point used by harness/**.

`Pipeline` (`src/pyannote/audio/core/pipeline.py`) subclasses `pyannote.pipeline.Pipeline` (`_Pipeline`, from the separate `pyannote-pipeline` PyPI package — a sister library, not part of this repo, that provides generic hyperparameter-optimizable "pipeline" scaffolding: `Uniform`/`Categorical`/`ParamDict` parameters, `.instantiate()`, `.freeze()`, and the `Optimizer`). `core.Pipeline` adds everything pyannote-audio-specific on top: HF Hub loading, `Model`/`Inference` attribute tracking, file preparation, and the `__call__` entrypoint.

## `from_pretrained`: config resolution

`Pipeline.from_pretrained(checkpoint, ...)` accepts a dict (already-parsed config), a local `config.yaml` path or containing directory, or a HuggingFace Hub id, and always ends up with a `config` dict shaped like:

```yaml
pipeline:
  name: SpeakerDiarization        # resolved via get_class_by_name, default module pyannote.pipeline.blocks
  params: {...}                   # constructor kwargs
dependencies: {pyannote.audio: ">=4.0"}   # version-checked via check_dependencies
freeze: {...}                     # pipeline.freeze(...)
params: {...}                     # pipeline.instantiate(...) — hyperparameter values
preprocessors: {...}
device: cpu
```

Notable mechanics:

- **`expand_subfolders`**: recursively walks the config and replaces any string value of the form `"$model/{subfolder}"` with `{"checkpoint": model_id, "subfolder": ..., "revision": ..., "token": ..., "cache_dir": ...}` — this is how a single HF repo (e.g. `pyannote/speaker-diarization-community-1`) can host one `config.yaml` whose `segmentation`/`embedding`/`plda` parameters each point at a *subfolder* of the same repo, resolved lazily by `get_model`/`get_plda` (see [[Pipelines-Subpackage]]) rather than needing separate repos.
- Legacy configs with a top-level `version:` key are rewritten into `dependencies: {pyannote.audio: version}` for backward compatibility.
- `preprocessors:` entries can be a `{name, params}` dict (dynamically imported class, default module `pyannote.audio`), a path to a `database.yml` (wrapped in `pyannote.database.FileFinder`), or a path template — assigned to `pipeline.preprocessors`, consumed by `prepare_one` below.

## `_models` / `_inferences`: attribute interception

`Pipeline.__init__` creates two `OrderedDict`s, `_models` and `_inferences`, alongside `pyannote.pipeline.Pipeline`'s own `_parameters`/`_pipelines`/`_instantiated` bookkeeping. `__setattr__` is overridden so that assigning an `nn.Module` (a `Model`) or a `BaseInference` (an `Inference`) instance to any attribute name transparently routes it into the matching dict instead of `self.__dict__` — and removes any prior registration of that name from the *other* dicts, so an attribute can't accidentally be both a hyperparameter and a model. `__getattr__`/`__delattr__` mirror this. Practically: a concrete pipeline's `__init__` just writes `self.segmentation_model = get_model(...)` or `self._segmentation = Inference(model, ...)`, and never has to manually register it anywhere else — `Pipeline.to(device)` (below) walks all three dicts (`_pipelines`, `_models`, `_inferences`) to move everything to a device uniformly.

## File preparation — `prepare_one`

Before `apply()` ever sees a file, `prepare_one(file, preload=False)`:

1. `Audio.validate_file(file)` — canonicalizes it (see [[Core-Audio-IO]]).
2. Wraps it in a `pyannote.database.ProtocolFile` with `lazy=self.preprocessors` if the pipeline has any configured (e.g. converting a `uri` into an actual file path via `FileFinder`).
3. If `preload=True`, eagerly loads the waveform into `file["waveform"]`/`file["sample_rate"]` via `Audio()(file)` and removes any `channel` key (channel selection already happened).

## `__call__`: the pipeline entrypoint

`pipeline(file, preload=False, **kwargs)` is what users actually call. It:

- Fixes RNG reproducibility (`fix_reproducibility`) for the pipeline's device.
- Auto-instantiates with `self.default_parameters()` if the pipeline hasn't been `.instantiate()`d yet (warns when doing so; raises `RuntimeError` if `default_parameters()` isn't implemented or isn't accepted).
- Accepts either a single `AudioFile` or a `list[AudioFile]`. For a list: prepares every file, rejects duplicate `uri`s, and dispatches to `_apply_batch` — which uses the pipeline's own `apply_batch()` if it defines one (native batch support), otherwise calls `apply()` per file in sequence, yielding `(file, prediction)` pairs either way.
- For a single file: `prepare_one` then `self.apply(file, **file.get("pipeline_kwargs", {}), **kwargs)` — **`apply()` is the method every concrete pipeline subclass must implement**; it is the pipeline-level equivalent of `Model.forward`.
- Reports `track_pipeline_apply` telemetry (file duration, `num_speakers` if diarization) after every apply.

## `to()` / `cuda()`

Moves every nested sub-pipeline (`_pipelines`), model (`_models`), and inference (`_inferences`) to a `torch.device`, and records `self.device`.

## `setup_hook` and the hook contract

`Pipeline.setup_hook(file, hook=None)` returns `partial(hook or noop, file=file)` — every concrete `apply()` calls this once at the top and then invokes the resulting callable as `hook(step_name, step_artifact, total=..., completed=...)` after each major internal step. See `pipelines/utils/hook.py`'s `ArtifactHook` / `ProgressHook` / `TimingHook` / `Hooks` for the concrete hook implementations users pass in (documented in [[Pipelines-Subpackage]]).

## `pipeline.training` — **extension point used by harness/**

`training` is **not defined in this file** — it's a plain `bool` property inherited from the external `pyannote.pipeline.Pipeline` base class (`pyannote/pipeline/pipeline.py` in the sister `pyannote-pipeline` package, not this repo): defaults to `False`, and setting it recursively propagates to any nested `_pipelines` (e.g. a clustering sub-pipeline). It is **unrelated to `nn.Module.train()`/`.eval()`** — it does not touch dropout, batchnorm, or any other inference-mode behavior.

Its documented upstream purpose is hyperparameter-search speedup: `pyannote.pipeline.optimizer.Optimizer` sets `pipeline.training = True` for the duration of an optimization trial and back to `False` afterward. `SpeakerDiarization.get_segmentations()` and `SpeakerDiarization.get_embeddings()` (`pipelines/speaker_diarization.py`, see [[Pipelines-Subpackage]]) both gate a caching behavior behind `if self.training:` — while `True`, they check `file[self.CACHED_SEGMENTATION]` / `file["training_cache/embeddings"]` before recomputing, so repeated trials against the same file during a hyperparameter search don't re-run the (expensive) neural network every time a downstream hyperparameter changes.

**Outside of an actual `Optimizer` run, `pipeline.training` is always `False` by default**, so `get_segmentations()` unconditionally runs the segmentation model regardless of what's on `file` — the cache key is never even consulted during normal inference. `harness/` (see `harness/segmentation.py` and `TICKET-01-segmentation-injection-seam.md` in the repo root) discovered this and repurposes the flag outside of any real `Optimizer`: it deliberately sets `pipeline.training = True` around a single `pipeline(file)` call to force `get_segmentations()` to honor a pre-populated `file[pipeline.CACHED_SEGMENTATION]` (an oracle or otherwise precomputed segmentation), then restores it. This is documented, pre-existing upstream behavior being used for a purpose (test-time segmentation injection) the seam happens to also support — not a patch to `speaker_diarization.py` itself. See [[Pipelines-Subpackage]]'s `SpeakerDiarization` section for the exact gated methods.
