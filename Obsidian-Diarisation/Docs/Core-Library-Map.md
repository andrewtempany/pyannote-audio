---
status: reference
scope: Pre-existing (upstream pyannote-audio)
---

# Core library map

> **Pre-existing (upstream pyannote-audio).** Index note only — every page linked below documents code under `src/pyannote/audio/` that this repo has not modified (verified via `git diff main -- src/`). Start with the repo-root `ARCHITECTURE.md` for the one-page overview; these notes go deeper on the pieces it only sketches.

## `core/` — the four foundational abstractions

- [[Core-Model]] — `Model`, the `nn.Module`/`LightningModule` base every architecture subclasses: task binding, `build()`/`setup()`, checkpointing, `from_pretrained`, freeze/unfreeze.
- [[Core-Task]] — `Task`, the `LightningDataModule` base pairing a "problem" with a `pyannote.database.Protocol`: `Specifications`, `prepare_data()`, the minibatch-generation contract, default training/validation steps.
- [[Core-Pipeline]] — `Pipeline`, wrapping the external `pyannote.pipeline.Pipeline`: `from_pretrained` config resolution, model/inference attribute tracking, `__call__`, and the `pipeline.training` flag **harness/ plugs into**.
- [[Core-Audio-IO]] — `Audio`: decoding (torchcodec), resampling, cropping, the `torchcodec`-unavailable fallback.
- [[Core-Inference]] — `Inference`: sliding-window application of a `Model` to long audio, overlap-add aggregation, warm-up handling.

## Subpackages

- [[Models-Subpackage]] — `models/`: `PyanNet`, `SSeRiouSS` (segmentation), `XVector`/wespeaker (embedding), `ToTaToNet` (joint separation+diarization), and how each pairs with a `tasks/` counterpart.
- [[Pipelines-Subpackage]] — `pipelines/`: shared utilities (`getter.py`, `hook.py`, `oracle.py`, `clustering.py`'s `SpeakerDiarizationMixin`), the flagship `SpeakerDiarization` pipeline in full, sibling pipelines, and the local/hosted (`pyannoteai/`) duality.

## Extension point used by `harness/`

`harness/` (a separate, new evaluation package in this repo — not documented here, see its own docs) hooks into upstream code at exactly one seam: `SpeakerDiarization.get_segmentations()`'s `if self.training:` gate on `file[pipeline.CACHED_SEGMENTATION]`, driven by the `pipeline.training` flag inherited from the external `pyannote.pipeline.Pipeline` base class. Full detail in [[Core-Pipeline]] (the flag itself) and [[Pipelines-Subpackage]] (the gated methods, `CACHED_SEGMENTATION`, and `oracle_segmentation`).
