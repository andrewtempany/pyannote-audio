---
status: reference
scope: Pre-existing (upstream pyannote-audio)
source: src/pyannote/audio/models/
---

# `models/` — neural network architectures

> **Pre-existing (upstream pyannote-audio).** Nothing in this repo modifies these files.

Every file under `src/pyannote/audio/models/` defines a concrete `nn.Module` subclassing `core.Model` (see [[Core-Model]]). The contract every architecture fulfils: implement `forward(waveforms) -> scores`, implement `build()` to add the final classifier/activation once `task`/`specifications` are known, and implement the three receptive-field hooks (`num_frames`, `receptive_field_size`, `receptive_field_center`) so `Model.receptive_field` and `Inference` can map between input samples and output frames.

## `models/segmentation/`

**`PyanNet.py`** — SincNet → LSTM → stack of `Linear`+LeakyReLU → classifier + activation. Frame-resolution output, `(batch, frame, classes)`. The SincNet front-end (`models/blocks/sincnet.py`) does the raw-waveform-to-features work and owns all the receptive-field math — `PyanNet`'s own `num_frames`/`receptive_field_size`/`receptive_field_center` are one-line delegations to `self.sincnet`'s equivalents. The LSTM can be "monolithic" (one multi-layer `nn.LSTM`) or split into a `ModuleList` of single-layer LSTMs with inter-layer dropout (`monolithic=False` in the `lstm` config) — the latter exists specifically to make it possible to probe/hook individual LSTM layers. `dimension` (the classifier's output width) is either `len(specifications.classes)` or, when the task uses powerset encoding, `specifications.num_powerset_classes`.

**`SSeRiouSS.py`** — "Self-Supervised Representation for Speaker Segmentation": same LSTM → linear → classifier tail as `PyanNet`, but the front-end is a pretrained self-supervised speech model from `torchaudio` (default `"WAVLM_BASE"`) instead of SincNet. `wav2vec_layer=-1` (default) averages all of the pretrained model's transformer layers with learnable per-layer weights; a specific layer index can be selected instead. `wav2vec_frozen` optionally freezes the pretrained front-end's weights during training. Because the front-end isn't SincNet, receptive-field math instead goes through the generic `conv1d_num_frames`/`conv1d_receptive_field_size`/`conv1d_receptive_field_center` helpers in `utils/receptive_field.py`.

**`debug.py`** — a minimal, fast segmentation architecture used only by the test suite so tests can exercise a real, trainable/loadable `Model` without downloading pretrained weights.

## `models/embedding/`

**`xvector.py`** (`XVector`, `XVectorMFCC`) — MFCC (or SincNet) front-end → TDNN stack → `StatsPool` (mean+std pooling over time, `models/blocks/pooling.py`) → a fixed-`dimension` embedding vector. Chunk-resolution output (one vector per chunk, not per frame) — used for speaker embedding/verification, and trained via the ArcFace metric-learning loss in `tasks/embedding/arcface.py`.

**`wespeaker/`** — a ported ResNet-based embedding architecture (`resnet.py`), with `convert.py` a one-off script for converting original WeSpeaker project checkpoints into this repo's format.

**`debug.py`** — same role as the segmentation debug model, for embedding-dependent tests.

## `models/separation/`

**`ToTaToNet.py`** — joint speech-separation + speaker-diarization model:

```
                   /--------------\
Conv1D Encoder --------+--- DPRNN --X------- Conv1D Decoder
WavLM -- upsampling --/                 \--- Avg pool -- Linear -- Classifier
```

A Conv1D encoder/decoder pair (from `asteroid_filterbanks`) does the actual audio separation via a DPRNN (from `asteroid.masknn`); a WavLM branch, upsampled to the encoder's frame resolution, is fused in and pooled/classified to also produce per-source speaker-activity predictions. This is the model behind `pipelines/speech_separation.py` and is trained by the joint `tasks/separation/PixIT.py` task. Both `asteroid` and `transformers` are optional dependencies here, guarded by module-level `try/except ImportError` flags (`ASTEROID_IS_AVAILABLE`, `TRANSFORMERS_IS_AVAILABLE`) — this is one of the "optional-dependency handling" spots the repo-root `ARCHITECTURE.md` flags as worth double-checking for a clear failure mode if the extra isn't installed (unlike `torchcodec`, which raises a clear `RuntimeError` at point of use — see [[Core-Audio-IO]] — it's worth verifying `ToTaToNet` fails equally clearly rather than with a raw `NameError`/`ImportError` deep in `forward()`).

## `models/blocks/`

Shared layers reused across the above: `sincnet.py` (the SincNet learnable-filterbank front-end used by `PyanNet` and optionally `XVector`), `pooling.py` (`StatsPool`, used by `XVector`).

## Pairing with `tasks/`

There is no enforced type linking a specific `Model` architecture to a specific `Task` — `Model.task` accepts any `Task` and vice versa (see [[Core-Model]] / [[Core-Task]]) — but by convention/usage:

| Model | Trained by |
|---|---|
| `PyanNet`, `SSeRiouSS` | `tasks/segmentation/speaker_diarization.py`, `voice_activity_detection.py`, `multilabel.py` |
| `XVector`, `XVectorMFCC`, wespeaker `ResNet` | `tasks/embedding/arcface.py` |
| `ToTaToNet` | `tasks/separation/PixIT.py` |

## Where this fits at runtime

At inference time, models are never instantiated directly by pipeline code — `pipelines/utils/getter.py`'s `get_model()` loads them (by instance, path, HF id, or `from_pretrained` kwargs dict) and pipelines wrap the result in an `Inference` (see [[Core-Inference]]) before use. See [[Pipelines-Subpackage]] for how `SpeakerDiarization` composes a segmentation model, an embedding model, and clustering into the full diarization pipeline.
