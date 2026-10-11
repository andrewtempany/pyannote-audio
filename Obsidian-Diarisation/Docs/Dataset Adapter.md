---
status: done
created: 2026-09-04
---

# Dataset Adapter

> Part of the [[Evaluation Harness]]. **`harness/datasets.py` is code contributed by this project — not upstream `pyannote-audio`.** It builds on two pre-existing *sister* packages that are dependencies of this repo but live outside it (not under `src/pyannote/audio`, so there is no in-repo file path to cite): `pyannote.core.Annotation`/`Timeline` (the data types) and `pyannote.database.util.load_rttm`/`load_uem` (the RTTM/UEM parsers this adapter delegates to rather than hand-rolling its own).

## Purpose

`AMIDatasetAdapter` is the only harness component that knows about file paths, RTTM parsing, and UEM parsing. It turns an AMI split (train/dev/test) and condition (IHM/SDM) into an iterable of `(uri, reference, uem)` tuples — the ground truth the rest of the harness scores against.

## Where it fits

Constructed from a [[Harness Config]] instance and iterated by the [[Orchestrator]], which pairs each `(uri, reference, uem)` with a hypothesis produced by the [[Runner]] and hands both to the [[Scorer]].

## API

```python
from harness.datasets import AMIDatasetAdapter, DatasetError

adapter = AMIDatasetAdapter(config)  # config: harness.config.HarnessConfig
for uri, reference, uem in adapter:
    ...  # reference: pyannote.core.Annotation, uem: pyannote.core.Timeline
```

`DatasetError` (a `RuntimeError` subclass) is raised, naming the exact file(s) involved, when expected data is missing.

## Directory convention

A harness-specific convention, **not a real-AMI given** — chosen to mirror this repo's own existing test-fixture convention (`tests/data/debug.{split}.rttm`):

```
{data_root}/{condition}/{split}.rttm
{data_root}/{condition}/{split}.uem
```

One combined, multi-uri RTTM/UEM file per (condition, split). If a real AMI layout on disk differs from this, the directory-resolution logic in `AMIDatasetAdapter.__iter__` is the only place that needs to change — RTTM/UEM parsing itself is delegated entirely to `pyannote.database.util.load_rttm`/`load_uem`.

The adapter iterates the **UEM's** uris as the authoritative "files to score" list. A uri present in the UEM with no matching RTTM entry raises `DatasetError` naming both file paths and the missing uri.

## Multi-channel IHM design decision

AMI's IHM condition has multiple raw per-participant headset-channel audio files per meeting, but the ground-truth RTTM/UEM are inherently **meeting-level** — one `Annotation` covering every speaker in the meeting, independent of which physical mic channel later gets decoded. This adapter's uri space is therefore exactly the set of uris in the (condition, split)'s RTTM/UEM: **one `(uri, reference, uem)` per meeting, never per channel.**

The adapter never touches audio at all — its output tuple has no audio path in it. Resolving a uri to a concrete audio file (e.g. a pre-mixed "Mix-Headset" `.wav`) is deferred to the [[Runner]]/[[Orchestrator]] layer; see [[Orchestrator]]'s `_audio_path()` helper for where that decision actually landed.

## Gotcha for anyone comparing `Annotation`s in tests

`pyannote.core.Annotation.__eq__` compares internal track identifiers, not just segment/label content. `load_rttm` assigns track ids from a pandas row index (`0`, `1`, `2`, ...), so a hand-built comparison `Annotation` using default/auto track naming will compare **unequal** even when segments and labels match exactly. `tests/test_datasets.py` works around this by comparing `itertracks(yield_label=True)` content instead of full `Annotation` equality — do the same in any test that compares annotations built via different code paths. (See also [[Scorer]]'s related gotcha about constructing genuinely overlapping tracks.)

## Non-goals

- No model loading, no pipeline invocation.
- Read-only — no RTTM/UEM writing.
- Does not attempt to download AMI; a data root missing the expected files fails with a clear, specific error, not a generic Python traceback.
