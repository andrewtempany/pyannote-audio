---
status: reference
scope: Pre-existing (upstream pyannote-audio)
source: src/pyannote/audio/core/io.py
---

# `core.io.Audio`

> **Pre-existing (upstream pyannote-audio).** Nothing in this repo modifies this file.

`Audio` (`src/pyannote/audio/core/io.py`) is the single entry point for turning an "audio file" — in whatever form it's provided — into a `(channel, time)` `torch.Tensor` waveform plus sample rate. It's used directly by `Model.audio` (constructed as `Audio(sample_rate=..., mono="downmix")` in `Model.__init__`), by `Inference` and `SpeakerDiarization.get_embeddings` for cropping chunks, and anywhere else in the codebase that needs to read audio.

## Decoding backend and the `torchcodec` guard

Decoding uses `torchcodec` (`AudioDecoder`); resampling uses `torchaudio.functional.resample`. The `torchcodec` import is wrapped in `try/except` at module load time: if it fails, `TORCHCODEC_AVAILABLE` is set `False` and a `UserWarning` is emitted immediately at import time (not lazily) explaining the two ways forward — fix the `torchcodec`/ffmpeg install, or only ever pass audio pre-loaded in memory as `{"waveform": tensor, "sample_rate": int}`. Every code path that actually needs to decode from a path (`Audio.__call__`, `Audio.crop`, `get_audio_metadata`) re-checks `TORCHCODEC_AVAILABLE` and raises a clear `RuntimeError` with the same "provide waveform dict instead" guidance if it's missing — this is the concrete mitigation behind ARCHITECTURE.md's note about `torchcodec`/ffmpeg being a recurring source of user-facing breakage.

## `AudioFile` — the accepted input shapes

`AudioFile = str | Path | IOBase | Mapping`. `Audio.validate_file(file)` canonicalizes any of these into a `Mapping` with a `"uri"` key and either an `"audio"` key (path-like or open stream) or both `"waveform"` (a `(channel, time)` tensor) and `"sample_rate"`. An optional `"channel"` key (zero-indexed int) selects a single channel from multi-channel audio. This is the shape every other function in the file — and every `Pipeline.prepare_one` call (see [[Core-Pipeline]]) — assumes after validation.

## Core operations

- `Audio(sample_rate=None, mono=None)` — `sample_rate=None` keeps native rate; `mono` is `"random"` (pick one channel) or `"downmix"` (average all channels) for multi-channel input.
- `downmix_and_resample(waveform, sample_rate, channel=None)` — channel selection, then downmix, then resample, in that order.
- `get_duration(file)` — from the loaded waveform's sample count if already in memory, otherwise from `torchcodec`'s header metadata (`AudioStreamMetadata.duration_seconds_from_header`) without decoding the full file.
- `get_num_samples(duration, sample_rate=None)` — deterministic `round(duration * sample_rate)`, used everywhere chunk boundaries need to be converted between seconds and samples.
- `__call__(file)` — loads and returns the *whole* file's waveform.
- `crop(file, segment, mode="raise"|"pad")` — the performance-critical path: for in-memory waveforms, slices directly; for on-disk files, uses `torchcodec`'s `get_samples_played_in_range(start, end)` to read only the requested span rather than decoding the whole file. Out-of-bounds requests either raise (`mode="raise"`, default) or are clamped and zero-padded (`mode="pad"`). After decoding, it reconciles the actual number of samples returned against the "expected" count computed from `segment.duration` and tolerates a ±1-sample rounding discrepancy (trimming or padding one sample) before raising on anything larger — a defensive check against off-by-one issues at chunk boundaries.

## `get_audio_metadata`

A "protocol preprocessor" — meant to be registered under a pipeline's `preprocessors:` config key (see [[Core-Pipeline]]) or a `pyannote.database` protocol so that `AudioStreamMetadata` (duration, sample rate, etc.) is computed once and cached on the file dict, speeding up later repeated random-access `Audio.crop` calls (e.g. inside a training dataloader that crops many chunks per file).
