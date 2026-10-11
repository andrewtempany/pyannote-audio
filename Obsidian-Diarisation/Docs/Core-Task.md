---
status: reference
scope: Pre-existing (upstream pyannote-audio)
source: src/pyannote/audio/core/task.py
---

# `core.Task`

> **Pre-existing (upstream pyannote-audio).** Nothing in this repo modifies this file.

`Task` (`src/pyannote/audio/core/task.py`) is a `lightning.LightningDataModule` subclass and is the other half of the `Model`/`Task` pairing described in [[Core-Model]]: a `Task` is "a problem plus a dataset" (e.g. "voice activity detection on AMI"), responsible for turning a `pyannote.database.Protocol` into training/validation minibatches, and for supplying the loss/metric contract a paired `Model` trains against. This file also defines `Problem`, `Resolution`, `Specifications`, and `UnknownSpecificationsError` — all imported into `core/model.py` and used throughout the codebase.

## `Specifications`

A dataclass carrying everything a `Model` needs to know about what it's predicting, independent of architecture:

- `problem: Problem` — `BINARY_CLASSIFICATION`, `MONO_LABEL_CLASSIFICATION`, `MULTI_LABEL_CLASSIFICATION`, `REPRESENTATION`, `REGRESSION`.
- `resolution: Resolution` — `FRAME` (sequence of per-frame predictions, e.g. segmentation) or `CHUNK` (one vector per chunk, e.g. embedding).
- `duration` / `min_duration` — (maximum/minimum) training chunk duration in seconds.
- `warm_up: (float, float)` — seconds on the left/right of each chunk that the model processes but that are excluded from loss computation and from inference aggregation (see [[Core-Inference]]).
- `classes`, `powerset_max_classes`, `permutation_invariant` — classification-specific; `powerset` (a `cached_property`) is `True` when `powerset_max_classes` is set, and `num_powerset_classes` computes how many "at most k simultaneous classes" subsets that implies (used by multi-speaker frame-level tasks like diarization, where a single frame's label is one of the possible *subsets* of active speakers rather than one speaker).
- Also iterable/sized as a 1-tuple of itself, so code that handles multi-task models (`tuple[Specifications]`) can treat single-task models uniformly — see `utils/multi_task.map_with_specifications`, used pervasively in `Inference` and `Model`.

A `Model` can have either one `Specifications` (single task) or a tuple of them (multi-task, e.g. joint separation+diarization in `ToTaToNet`/`PixIT`).

## `prepare_data()`: protocol → cached numpy arrays

This is the heaviest method in the file and the main ETL step between a `pyannote.database.Protocol` and trainable minibatches. Called once, only on the main process (`global_rank 0`), it iterates the protocol's `train()` (and `development()`, if the protocol has validation) files and builds a compact set of numpy arrays cached to disk via `np.savez_compressed`:

- `audio-path`, `audio-metadata` (per-file database/subset/scope + any other str/int metadata key the protocol provides, each interned into small integer codes), `audio-annotated` (total annotated duration per file), `annotations-regions`/`audio-regions-ids` (annotated time spans, with the start/end index into the flat array recorded per file), `annotations-segments`/`audio-segments-ids` (the actual labeled segments, similarly indexed), plus `metadata-values` and per-scope label vocabularies (`metadata-{database}-labels`, `metadata-labels`).
- Segments shorter than the task's `duration` are dropped from `annotated-regions` (can't fit a full training chunk).
- Speaker labels are tracked at up to three scopes — `file`, `database`, `global` — mirroring `pyannote.database`'s `scope` protocol field; this is what lets the same task work with datasets where speaker identity is only meaningful within one file vs. globally consistent across a corpus.
- Integer columns use the smallest safe numpy dtype (`get_dtype` picks `i1`/`i2`/`i4`/`i8` based on the max value actually present) to keep the cache small for large datasets.
- `post_prepare_data(prepared_data)` is a subclass hook for task-specific additions (e.g. discovering `max_speakers_per_chunk` from the data) called just before saving.
- If `cache` is not provided, a temp file is used instead — meaning by default every fresh `Task()` instance re-derives this from scratch; passing an explicit `cache` path makes repeated experiments skip re-scanning the protocol.

`setup(stage)` is the counterpart run on *every* process/device: it re-loads the cached `.npz` (path broadcast across ranks during `"fit"` via `self.trainer.strategy.broadcast`) into `self.prepared_data`, and asserts the protocol name matches what's cached (protects against accidentally reusing another protocol's cache).

## Minibatch generation contract

`Task` does not implement dataset iteration itself — it defines the *interface* that concrete tasks (`tasks/segmentation/*.py`, `tasks/embedding/arcface.py`, `tasks/separation/PixIT.py`) must fill in:

- `train__iter__()` / `train__len__()` back a `TrainDataset` (an `IterableDataset` thin wrapper).
- `val__getitem__(idx)` / `val__len__()` back a `ValDataset` (a regular indexable `Dataset`), only used when `self.has_validation`.
- `collate_fn(batch, stage=...)` batches individual samples.
- `train_dataloader()`/`val_dataloader()` wrap those in a `torch.utils.data.DataLoader` with the task's `batch_size`/`num_workers`/`pin_memory`. `num_workers` is forced to 0 on macOS with Python ≥ 3.8 (a known multiprocessing incompatibility) if the caller didn't explicitly set it.

## Default training/validation step

`common_step(batch, batch_idx, stage)` provides a default `training_step`/`validation_step` for single-task setups (raises `NotImplementedError` for multi-task — those must override it):

1. Forward pass: `y_pred = self.model(batch["X"])`.
2. Optional per-frame `weight` (looked up via `batch[self.weight]` if the task defines a `weight` attribute — training only), defaulting to all-ones.
3. **Warm-up masking**: zeroes the loss weight for the leftmost/rightmost `warm_up` seconds of each chunk (converted to a frame count via `warm_up / duration * num_frames`) — this is the training-side counterpart of the warm-up trimming `Inference` does at inference time (see [[Core-Inference]]).
4. `default_loss()` picks binary cross-entropy (binary/multi-label) or NLL (mono-label) automatically from `specifications.problem`.
5. Returns `None` (skips the batch) if the loss is `NaN`; otherwise logs `loss/{stage}` and returns `{"loss": loss}`.

## Validation metric

`default_metric()` is abstract (subclasses must implement it); `Task.metric` (`cached_property`) wraps whatever it returns in a `torchmetrics.MetricCollection`, and `setup_validation_metric()` (called from `Model.setup()`) attaches it to the model as `model.validation_metric`. `val_monitor` exposes `(metric_name, "max"|"min")` for wiring up Lightning's `ModelCheckpoint`/`EarlyStopping` callbacks.

## Concrete tasks

`tasks/segmentation/speaker_diarization.py` (`SpeakerDiarization`, via the `SegmentationTask` mixin in `tasks/segmentation/mixins.py`), `voice_activity_detection.py`, `multilabel.py`; `tasks/embedding/arcface.py` (ArcFace/metric-learning loss for `XVector`-style models); `tasks/separation/PixIT.py` (joint separation+diarization, pairs with `ToTaToNet`) — each fills in the `train__iter__`/`collate_fn`/`default_metric` contract above for its specific problem. See [[Models-Subpackage]] for which `Model` architecture each is meant to pair with.
