---
status: reference
scope: Pre-existing (upstream pyannote-audio)
source: src/pyannote/audio/core/model.py
---

# `core.Model`

> **Pre-existing (upstream pyannote-audio).** Nothing in this repo modifies this file. See [[Welcome]] and the repo-root `ARCHITECTURE.md` for the wider map; this note goes deeper on the piece ARCHITECTURE.md only sketches in its `core/` table.

`Model` (`src/pyannote/audio/core/model.py`) is the base class every neural network architecture in `models/` subclasses (`PyanNet`, `SSeRiouSS`, `XVector`, `ToTaToNet`, ...). It is a thin `lightning.LightningModule`, so it gets checkpointing, device management, and the Lightning training loop for free — this class layers pyannote-specific concerns on top: task binding, HF Hub load/save, receptive-field bookkeeping, and layer freeze/unfreeze helpers for fine-tuning.

## Task binding and `Specifications`

A `Model` is constructed with `sample_rate`, `num_channels` (mono only — asserted), and an optional `task: Task`. Setting `model.task = some_task` invalidates any cached `specifications`. `specifications` (a `Specifications` or `tuple[Specifications]`, dataclass defined in `core/task.py` — see [[Core-Task]]) describes what the model is being asked to predict: problem type (binary / mono-label / multi-label classification, representation, regression), output resolution (frame vs chunk), chunk duration, warm-up, and — for classification — the class list and optional `powerset` encoding.

- If `model.task` is set, `model.specifications` simply proxies `task.specifications`.
- If not (e.g. a model loaded standalone for inference, with no `Task` attached), it falls back to a `_specifications` value set directly (`model.specifications = ...`) and raises `UnknownSpecificationsError` if neither is available.

## `build()` and `setup()`: adding task-dependent layers

Concrete architectures do **not** add their final classifier/activation layers in `__init__`. Instead they override `build()`, called from `Model.setup(stage)`:

1. `task.trainer` is set (`stage == "fit"` only), then `task.setup(stage)` runs (loads cached prepared data — see [[Core-Task]]).
2. The model's current `state_dict()` is snapshotted, `build()` is called (adds e.g. `self.classifier = nn.Linear(...)`, `self.activation = self.default_activation()`), then the pre-`build()` state dict is reloaded with `strict=False` — this is what lets a pretrained checkpoint's backbone weights survive being re-specialized for a new task/label set. A `RuntimeError` mentioning `"size mismatch"` is caught and turned into a warning suggesting a two-stage fine-tune instead of a hard failure.
3. Newly added modules (the diff between the module set before/after `build()`) are moved to the model's device and recorded as `self.task_dependent` (a list of names) — useful for `freeze_up_to`/`unfreeze_up_to`.
4. If a `task` is attached, `task.model = self`, then `task.setup_loss_func()` and `task.setup_validation_metric()` run.

`default_activation()` picks sigmoid (binary/multi-label) or log-softmax (mono-label) automatically from `specifications.problem`; architectures call it from their own `build()`.

## Receptive field

`Model.receptive_field` (a `cached_property`) turns three subclass-provided hooks — `num_frames(num_samples)`, `receptive_field_size(num_frames)`, `receptive_field_center(frame)` — into a `pyannote.core.SlidingWindow` in seconds. This is how `Inference` (see [[Core-Inference]]) knows how to align frame-level model output back onto the original waveform's timeline. Every segmentation/embedding architecture must implement these three hooks; they usually delegate to a shared front-end block (SincNet, or the generic `conv1d_*` helpers in `utils/receptive_field.py` for wav2vec/Conv1D-based models — see [[Models-Subpackage]]).

## Checkpointing and `from_pretrained`

- `on_save_checkpoint` stashes pyannote-specific metadata under a `"pyannote.audio"` key in the Lightning checkpoint dict: library version, the model's own `(module, class)` for later dynamic re-import, and its `specifications`.
- `on_load_checkpoint` restores `specifications` from that metadata, then calls `self.setup()` so task-dependent layers (classifier, activation) exist *before* Lightning restores the state dict onto them.
- `Model.from_pretrained(checkpoint, ...)` (classmethod) resolves `checkpoint` — a local file/dir/`BytesIO`, or a HuggingFace Hub id — into a checkpoint blob (via `download_from_hf_hub` for the Hub case), version-checks it (`check_dependencies`), dynamically imports the concrete class named in the checkpoint's own metadata (`import_module` + `getattr`, so `Model.from_pretrained` never needs an explicit registry of architectures), and calls `Klass.load_from_checkpoint(...)`. If loading `strict=True` fails specifically because of a `loss_func` key (some tasks, e.g. ArcFace, have their own trainable loss weights not needed at inference time), it retries `strict=False` with a warning rather than failing outright. Records `_otel_origin` (local/HF org/generic HF) for telemetry.

## Delegation to `Task`

`train_dataloader`, `training_step`, `val_dataloader`, `validation_step` on `Model` are one-line delegations to the identically-named `Task` methods — by design the model has no idea how it is being trained; see [[Core-Task]].

## Freezing helpers

Two independent freezing APIs, both returning the list of module names they touched and raising `ValueError` on an unknown name:

- `freeze_up_to(module_name)` / `unfreeze_up_to(module_name)` — walk modules in `ModelSummary(self, max_depth=-1)` order (i.e., the order `model.summary("full")` would print) and freeze/unfreeze everything up to and including the named module. Intended for architectures that are roughly sequential (front-end → recurrent → classifier).
- `freeze_by_name(modules)` / `unfreeze_by_name(modules)` — freeze/unfreeze specific named submodules directly (accepts a single name or a list), independent of ordering.

## Where this fits at runtime

A `Model` is rarely called directly outside of training — inference code wraps it in an `Inference` (sliding-window application + aggregation, see [[Core-Inference]]), and a `Pipeline` (see [[Core-Pipeline]]) owns one or more such `Inference` instances plus any other post-processing (clustering, etc.). No extension point relevant to `harness/` lives in this file — the harness's segmentation-injection seam is at the `Pipeline`/`SpeakerDiarization` layer, not here.
