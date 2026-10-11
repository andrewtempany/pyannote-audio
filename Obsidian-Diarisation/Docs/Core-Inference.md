---
status: reference
scope: Pre-existing (upstream pyannote-audio)
source: src/pyannote/audio/core/inference.py
---

# `core.Inference`

> **Pre-existing (upstream pyannote-audio).** Nothing in this repo modifies this file.

`Inference` (`src/pyannote/audio/core/inference.py`) is the layer between a trained `Model` (see [[Core-Model]]) and a full-length audio file: it slides the model over long audio in overlapping chunks, batches the forward passes, and stitches the per-chunk outputs back into one continuous prediction. `BaseInference` is just an empty marker superclass — `Pipeline.__setattr__` (see [[Core-Pipeline]]) special-cases any `BaseInference` instance into `self._inferences` so `Pipeline.to(device)` can move it along with everything else.

## Construction

`Inference(model, window="sliding"|"whole", duration=None, step=None, pre_aggregation_hook=None, skip_aggregation=False, skip_conversion=False, device=None, batch_size=32)`:

- Puts `model` in `eval()` mode and moves it to `device` (defaults to `model.device`).
- `window="sliding"` (default) processes the file in overlapping chunks and aggregates; `window="whole"` runs one forward pass over the entire file/chunk (warns if used with a frame-resolution model — likely bad results and high memory use for long audio).
- `duration` defaults to the model's training chunk duration (warns if overridden, since inference at a different chunk size than training "might lead to suboptimal results").
- `step` defaults to the model's left warm-up duration if non-zero, else 10% of `duration`; raises if `step > duration` (would leave gaps between chunks).
- **Powerset → multilabel conversion**: if the model's `specifications.powerset` is set and `skip_conversion` is not, output is automatically converted from the powerset encoding to plain multi-label via `utils.powerset.Powerset` — most pipeline code downstream never has to know powerset encoding exists.
- `pre_aggregation_hook`: an optional callable applied to raw per-chunk output just before overlap-add aggregation (e.g. to reshape/post-process before averaging).

## `infer(chunks)` — the raw forward pass

Sends a batch of chunks through the model, converts any CUDA out-of-memory `RuntimeError` (detected via `lightning.pytorch.utilities.memory.is_oom_error`) into a `MemoryError` suggesting a smaller `batch_size`, and applies the powerset→multilabel `conversion` before returning numpy arrays.

## `slide(waveform, sample_rate, hook)` — the windowing algorithm

The core of sliding-window inference:

1. Computes `window_size`/`step_size` in samples, unfolds the waveform into overlapping chunks via `torch.Tensor.unfold`.
2. Handles a possible last, incomplete chunk separately (zero-padded to `window_size`).
3. Batches chunks through `infer()` in groups of `batch_size`, calling `hook(completed=, total=)` after each batch — this is what powers `ProgressHook` (see [[Pipelines-Subpackage]]).
4. Aggregates all per-chunk outputs via `Inference.aggregate` (below) unless `skip_aggregation`, the model outputs one vector per chunk (`Resolution.CHUNK`), or the task is `permutation_invariant` with no `pre_aggregation_hook` (raw per-chunk speaker-labeled output can't be naively averaged across chunks without first resolving the permutation — that's what pipelines like `SpeakerDiarization` do explicitly via clustering, rather than at the `Inference` layer).
5. Trims the padding added to the last chunk back off before returning.

Every step above is wrapped in `utils.multi_task.map_with_specifications`, the recurring pattern (also used in `Model`) for applying the same code uniformly whether the model has one `Specifications` or a tuple of them (multi-task models).

## `Inference.aggregate` (static) — overlap-add

Turns `(num_chunks, num_frames_per_chunk, num_classes)` raw scores plus a target frame resolution into one continuous `(num_frames, num_classes)` `SlidingWindowFeature`:

- Optional Hamming windowing (`hamming=True`) down-weights each chunk's edges before summation, so overlapping chunks blend smoothly rather than having hard chunk-boundary discontinuities.
- A `warm_up` mask (near-zero, not exactly zero — `epsilon`) further suppresses the outer `warm_up` seconds of each chunk from contributing to the aggregate, mirroring the same warm-up region `Task.common_step` excludes from the training loss (see [[Core-Task]]).
- Frames with contributions from zero chunks are set to `missing` (default `NaN`) rather than `0`, so downstream code can distinguish "genuinely zero score" from "never covered by any chunk".
- `skip_average=True` returns the raw weighted sum instead of the normalized average — used where the caller wants to combine with additional weighting itself (e.g. `pipelines/utils/diarization.py`'s `to_diarization`, see [[Pipelines-Subpackage]]).

## `Inference.trim` (static)

Chops a fixed *ratio* (`warm_up`, default 10% each side) of frames off both ends of every chunk in a raw `(num_chunks, num_frames, num_classes)` score array — a simpler sibling to the aggregation-time warm-up masking, used e.g. by `SpeakerDiarizationMixin.speaker_count` before counting instantaneous active speakers, so warm-up-region noise doesn't skew the count.

## `__call__` / `crop`

`inference(file, hook=None)` runs `slide` (or a single whole-file forward pass) over an entire file. `inference.crop(file, chunk, hook=None)` restricts inference to one `Segment` (or the bounding box of several) — used wherever only a small region's scores are needed rather than the whole file. Its docstring calls out an important idiom: if the model needs warm-up, the caller must *extend* the requested chunk by `warm_up` on each side and then re-`.crop()` the result down to the region of actual interest, or the model never gets warmed up before the region that matters.
