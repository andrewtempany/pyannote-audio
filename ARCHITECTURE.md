# pyannote-audio — Codebase Overview

*Generated 2026-07-21 for onboarding / handoff purposes. Reflects the state of the `main` branch at commit `b749285c`.*

## 1. Purpose

`pyannote-audio` is an open-source Python toolkit for **speaker diarization** (answering "who spoke when" in an audio recording), built on PyTorch. It provides three things:

1. **Reusable building blocks** (`Model`, `Task`, `Pipeline`, `Audio`, `Inference`) for training and running audio ML models — segmentation, speaker embedding, and speech separation.
2. **Pretrained, fine-tunable models and pipelines** distributed via the HuggingFace Hub (`hf.co/pyannote`), including the open-source `speaker-diarization-community-1` pipeline.
3. **A thin client for pyannoteAI's premium hosted API** (`speaker-diarization-precision-2`), so the same `Pipeline.from_pretrained(...)` call can transparently run locally or against a paid cloud service depending on the token supplied.

The project is maintained by **pyannoteAI** (commercial company) as the open-source front door to a freemium product: local/open weights are "good", the hosted API is "better", and the codebase is engineered so switching between the two is a one-line change (see `README.md`'s `community-1` vs `precision-2` examples).

## 2. Requirements

- **Python** ≥ 3.10 (CI matrix tests 3.10–3.12; `.python-version` pins 3.10 for local dev).
- **PyTorch** ≥ 2.8, **torchaudio** ≥ 2.8, **torchcodec** ≥ 0.7 (audio decoding; requires system **ffmpeg** to be installed — this is a common source of setup failures, see `fix: handle torchcodec unavailability` in recent commits).
- **pytorch-lightning** (`lightning`) for training loops, multi-GPU support.
- Sister libraries from the same ecosystem: `pyannote-core`, `pyannote-database`, `pyannote-metrics`, `pyannote-pipeline` — these are separate PyPI packages under the same `pyannote` namespace and define core data structures (`Annotation`, `Segment`, `Timeline`) and the pipeline/hyperparameter-optimization framework this repo builds on.
- `huggingface-hub` for downloading pretrained weights; most pretrained models/pipelines gate access behind accepting user conditions on the HF model card, so a HF access token is required for anything beyond fully local/offline use.
- `pyannoteai-sdk` — official client for the hosted premium API, optional at the point of use (only needed if a pyannoteAI API key is passed instead of an HF token).
- Optional extras (`pyproject.toml`): `test` (pytest, papermill for notebook tests), `doc` (Sphinx), `cli` (typer — see below), `separation` (asteroid, transformers), `dev` (ipython).
- Packaging: `hatchling` + `hatch-vcs` (version derived from git tags), package lives under `src/pyannote/audio` (implicit namespace package `pyannote`).

## 3. High-level architecture

```
                      ┌─────────────────────────┐
                      │  pyannote.audio.Pipeline │   ← user-facing entry point
                      │   .from_pretrained(id)   │
                      └────────────┬─────────────┘
                                   │ branches on the kind of token passed
              ┌────────────────────┼────────────────────────┐
              ▼                                              ▼
   local pipeline graph                          pyannoteai.{local,sdk}.*
   (pipelines/speaker_diarization.py,             thin wrappers that call the
    voice_activity_detection.py, ...)             hosted pyannoteAI API instead
              │                                    of running anything locally
              ▼
   composed of pyannote.pipeline.Pipeline
   "blocks", each wrapping one or more
   pyannote.audio.core.Model instances
              │
              ▼
   ┌───────────────────────────────┐
   │        core.Model             │  nn.Module + task/spec metadata,
   │  (PyanNet, SSeRiouSS, ...)    │  HF Hub load/save, ONNX-ish "specs"
   └───────────────┬───────────────┘
                    │ trained via
                    ▼
   ┌───────────────────────────────┐
   │        core.Task              │  pytorch-lightning LightningDataModule
   │ (SpeakerDiarization, VAD, ...) │  wrapping a pyannote.database Protocol
   └───────────────────────────────┘
                    │ reads audio via
                    ▼
   ┌───────────────────────────────┐
   │        core.io.Audio          │  torchcodec-backed audio loading,
   └───────────────────────────────┘  resampling, cropping
```

### `core/` — the four foundational abstractions (`src/pyannote/audio/core/`)

| File | Role |
|---|---|
| `model.py` | `Model` base class (`nn.Module` subclass). Adds HF Hub `from_pretrained`/push, checkpoint (de)serialization, task-specification metadata (`Specifications`), layer freeze/unfreeze helpers used for fine-tuning. |
| `task.py` | `Task` base class (`pl.LightningDataModule`). Defines the train/val data pipeline against a `pyannote.database` `Protocol`, chunking/batching strategy, and the loss/metric contract a `Model` must implement for that task. |
| `pipeline.py` | `Pipeline` base class wrapping `pyannote.pipeline.Pipeline`. Handles `from_pretrained` (resolving HF Hub ids, YAML pipeline configs, or local pyannoteAI dispatch), instantiating sub-models, hyperparameter defaults. |
| `io.py` | `Audio` class — decoding (via `torchcodec`), resampling, mono-downmixing, cropping by `Segment`. |
| `inference.py` | `Inference`/`BaseInference` — slides a `Model` over long audio with configurable window/step, stitches window-level outputs back into full-file predictions. |
| `calibration.py`, `plda.py`, `callback.py` | Score calibration, PLDA scoring for speaker verification, and Lightning training callbacks, respectively. |

### `models/` — neural network architectures (`src/pyannote/audio/models/`)

- `segmentation/` — `PyanNet` (SincNet + LSTM) and `SSeRiouSS` (wav2vec2/self-supervised-frontend based) — frame-level speaker activity models used for diarization/VAD/OSD.
- `embedding/` — `XVector` and a `wespeaker/` port — fixed-dimension speaker embedding models used for clustering speakers.
- `separation/` — `ToTaToNet` — speech source separation model (isolating overlapping speakers into separate audio streams), used by the `speech_separation` pipeline and the `PixIT` joint task.
- `blocks/` — shared reusable layers (`sincnet.py` filterbank front-end, `pooling.py` statistics pooling).

### `tasks/` — training-time definitions (`src/pyannote/audio/tasks/`)

Mirrors `models/`: `segmentation/` (`SpeakerDiarization`, `VoiceActivityDetection`, `OverlappedSpeechDetection` via `multilabel.py`), `embedding/` (`arcface.py` — ArcFace/metric-learning loss for embeddings), `separation/` (`PixIT.py` — joint separation+diarization training). Each `Task` subclass pairs with the `Model` it trains and knows how to build minibatches from a `pyannote.database.Protocol`.

### `pipelines/` — inference-time composition (`src/pyannote/audio/pipelines/`)

- `speaker_diarization.py` — the flagship pipeline: segmentation model → local speaker embeddings → clustering → (optional) exclusive-speaker post-processing. Returns a `DiarizeOutput` dataclass (annotation + exclusive annotation + optional speaker embeddings).
- `voice_activity_detection.py`, `multilabel.py`, `speech_separation.py`, `speaker_verification.py`, `clustering.py` — supporting/alternate pipelines and the clustering algorithms (agglomerative, etc.) diarization depends on.
- `pyannoteai/local.py` and `pyannoteai/sdk.py` — wrappers implementing the *same* `Pipeline` interface but delegating to pyannoteAI's on-prem package or hosted HTTP API respectively. This is the mechanism that makes `precision-2` a drop-in replacement for `community-1` in user code.
- `utils/` — shared helpers: HF hub model/pipeline resolution (`getter.py`), progress reporting (`hook.py`), oracle (ground-truth) pipelines for evaluation (`oracle.py`), diarization-output shaping (`diarization.py`).

### `telemetry/` — usage metrics (`src/pyannote/audio/telemetry/`)

OpenTelemetry-based, opt-out (`metrics_enabled: true` in `config.yaml`) tracking of model/pipeline init and pipeline-apply events, exported via OTLP to `otel.pyannote.ai`. Worth flagging explicitly to anyone evaluating this codebase for privacy-sensitive deployments — it phones home by default.

### `utils/` — cross-cutting helpers

Loss functions, permutation-invariant training utilities (`permutation.py`, `powerset.py`), reproducibility/seeding, receptive-field computation, dependency checking (`dependencies.py` — likely home of the recent `torchcodec` availability guard), HF Hub download helpers.

### CLI (`__main__.py`, entry point `pyannote-audio`)

A `typer` app (extra: `cli`) exposing:
- `download` — fetch a pretrained pipeline/model
- `apply` — run a pipeline over audio file(s)
- `benchmark` — evaluate a pipeline against a `pyannote.database` protocol (DER/JER metrics)
- `optimize` — hyperparameter search via `pyannote.pipeline.Optimizer`
- `strip` — presumably strip a checkpoint down for distribution

### Supporting content

- `tutorials/` — the primary user-facing documentation, as runnable notebooks (training, applying models/pipelines, adding a custom model/task, speaker verification, etc.). `tests/test_run_notebooks.py` actually executes these via `papermill` as part of the test suite, so they double as integration tests.
- `notebook/` — smaller example notebooks (augmentation, inference, sharing to HF Hub, freezing layers).
- `doc/` — Sphinx scaffold (`doc/source/`), thin — most real documentation lives in the README, tutorials, and `FAQ.md`/`.faq/`.
- `tests/` — pytest suite (`tasks/`, `utils/` subpackages, plus top-level files for inference, IO, clustering, metrics, training, and notebook execution). Also `test_import_lib.py`, likely guarding against optional-dependency import regressions (relevant given `separation` is an optional extra using `asteroid`/`transformers`).
- `.faq/`, `faq.yml`, `FAQ.md` — structured FAQ content, possibly auto-published somewhere.

## 4. Notable design decisions / conventions

- **HF Hub as the distribution mechanism** for both weights and pipeline configs (YAML files referencing model IDs) — `from_pretrained` is the universal loading API across `Model` and `Pipeline`.
- **Local/hosted duality baked into `Pipeline.from_pretrained`**: the same call signature routes to a local `nn.Module` graph or to `pyannoteai/{local,sdk}.py` HTTP wrappers depending on what kind of `token` is passed. This is the core of the commercial strategy and the main architectural seam to understand before modifying pipeline loading code.
- **Lightning-based training** (`Task` = `LightningDataModule`) keeps training loop concerns (multi-GPU, checkpointing) out of this repo's own code and delegates to `pytorch-lightning`.
- **`pyannote.database.Protocol`** is the abstraction for "a dataset with train/dev/test splits and known annotations" — this repo has no bundled datasets; it consumes protocols defined elsewhere (typically in per-dataset sibling repos, per the `pyannote-database` ecosystem convention).
- **Telemetry is on by default.**

## 5. Where to start reading, by goal

| Goal | Start here |
|---|---|
| Understand end-user inference API | `README.md`, `src/pyannote/audio/core/pipeline.py`, `src/pyannote/audio/pipelines/speaker_diarization.py` |
| Understand model architectures | `src/pyannote/audio/models/segmentation/PyanNet.py`, `models/embedding/xvector.py` |
| Understand training | `src/pyannote/audio/core/task.py`, `tasks/segmentation/speaker_diarization.py`, `tutorials/training_a_model.ipynb` |
| Understand local-vs-hosted split | `src/pyannote/audio/core/pipeline.py` (`from_pretrained`), `src/pyannote/audio/pipelines/pyannoteai/{local,sdk}.py` |
| CLI usage | `src/pyannote/audio/__main__.py` |

## 6. Starting points for improvement review

These are observations from a structural read, not a full audit — worth treating as hypotheses to verify, not findings:

1. **`core/` is doing a lot** — `task.py` (32KB) and `pipeline.py`/`model.py`/`inference.py` (each 20–25KB) are large, dense modules mixing several responsibilities (data loading, batching, loss wiring, HF I/O, hyperparameter defaults). Worth checking whether any of these have grown organically past the point of being easily testable in isolation.
2. **Optional-dependency handling is a recurring pain point** — the recent commit history (`a4280d71 fix: fix #2032`, `42931805 fix: handle torchcodec unavailability with clear RuntimeError messages`) suggests `torchcodec`/ffmpeg availability has caused user-facing breakage more than once. `utils/dependencies.py` is the current mitigation; worth checking whether other optional extras (`separation`'s `asteroid`/`transformers`) have equally clear failure modes.
3. **Telemetry defaults and disclosure** — telemetry is opt-out rather than opt-in and isn't mentioned in the README's quickstart. Worth confirming this is adequately disclosed for the intended audience (research/production users, some in regulated or offline environments).
4. **Local vs. hosted code path divergence** — `pyannoteai/local.py` and `pyannoteai/sdk.py` both wrap `Pipeline` and return the same `DiarizeOutput`, but live fairly separately from the "real" local pipeline in `speaker_diarization.py`. Worth checking how much behavior (error handling, output shaping, hook/progress support) is duplicated vs. shared, since drift between the three would be a confusing user-facing inconsistency.
5. **Test coverage shape** — the test suite mixes real unit tests with notebook execution via `papermill` as an integration-test substitute. That's a reasonable pattern but means notebook edits can silently become load-bearing test fixtures; worth checking whether that's documented for contributors.
6. **`setup.py` alongside `pyproject.toml`** — the repo has both a `pyproject.toml` (hatchling/hatch-vcs, the apparent source of truth) and a `setup.py`. Worth checking whether `setup.py` is legacy/vestigial and safe to remove, or still serves a purpose.
