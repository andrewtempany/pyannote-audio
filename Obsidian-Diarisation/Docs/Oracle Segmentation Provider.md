---
status: done
created: 2026-09-13
---

# Oracle Segmentation Provider

> Part of the oracle/ceiling analysis batch (see [[Segmentation Injection Seam]], [[Segmentation Source Interface]], [[Oracle Assignment Strategy]]). Implements the real `OracleSegmentation` behind the interface [[Segmentation Source Interface]] reserved as a stub.

## Purpose

Supplies ground-truth speech/overlap segmentation through the harness's injectable segmentation seam, so a run can be scored with the pipeline's own neural segmentation stage replaced by ground truth — measuring how much diarization error is attributable to the segmentation stage versus everything downstream of it.

## Why `oracle_segmentation()` didn't need a source patch

`pyannote.audio.pipelines.utils.oracle.oracle_segmentation(file, window, frames, num_speakers=None)` (`src/pyannote/audio/pipelines/utils/oracle.py`) already does exactly what's needed: given `file["annotation"]` (a reference `Annotation`), a `window` (chunk duration/step), and `frames` (output resolution), it discretises the reference into the `(num_chunks, num_frames, num_speakers)` `SlidingWindowFeature` shape that `CACHED_SEGMENTATION` expects. It's already used this way for oracle *clustering* (`src/pyannote/audio/pipelines/clustering.py:712`), so the shape contract was known-good before this ticket touched anything. No modification to any file under `src/pyannote/audio` was needed or made.

The `window`/`frames` convention matches what `SpeakerDiarization.apply()` itself uses when calling `oracle_segmentation` for oracle clustering (`speaker_diarization.py:663`, `clustering.py:710-712`):

```python
window = SlidingWindow(step=pipeline._segmentation.step, duration=pipeline._segmentation.duration)
frames = pipeline._segmentation.model.receptive_field
```

Both are available off any `SpeakerDiarization` pipeline instance.

## Interface

```python
from harness.segmentation import OracleSegmentation
from pyannote.database.util import load_rttm

reference_lookup = load_rttm("path/to/reference.rttm")  # uri -> Annotation
source = OracleSegmentation(reference_lookup=reference_lookup)
source.id  # "oracle" -- feeds the runner's cache key (harness/runner.py)
```

`OracleSegmentation` conforms to the same `SegmentationSource` interface as `BaselineSegmentation` (see [[Segmentation Source Interface]]): a stable `id` class-level-shaped attribute and a `populate(pipeline, file) -> None` hook called by [[Runner]] before the pipeline is applied. Because `id` differs from `BaselineSegmentation`'s (`"oracle"` vs `"baseline"`), the runner's cache key automatically distinguishes oracle runs from baseline runs with no other wiring needed.

### `__init__(reference_lookup=None)`

Takes a `Mapping[str, Annotation]` — uri to reference Annotation — exactly the shape `pyannote.database.util.load_rttm()` returns. This is deliberately decoupled from any specific RTTM's provenance: the caller (e.g. `run_harness.py`'s CLI, via `--oracle-rttm`) is responsible for parsing whichever reference RTTM it wants used (an `only_words` RTTM, a standard collapsed RTTM, or any other reference annotation source), mirroring how `pipeline_config_id` ownership is left to the caller in `harness/runner.py`. Defaults to an empty dict if omitted, so `populate()` still fails fast (see below) rather than silently doing nothing.

### `populate(pipeline, file)`

1. Looks up `file["uri"]` in the `reference_lookup` (raises `KeyError` if absent — fail-fast, matching the seam doc's requirement that a bad/missing reference never silently no-ops).
2. Refuses the run if `pipeline._expects_num_speakers` is True — see [[#The `_expects_num_speakers` guard]].
3. Builds `window`/`frames` from the pipeline's own segmentation `Inference` object, then computes `oracle_segmentation(scaffold, window, frames)` on a **shallow copy** of `file` carrying `annotation`, and injects the result into `file[pipeline.CACHED_SEGMENTATION]`.
4. Wraps the call in the `pipeline.training = True` seam established in [[Segmentation Injection Seam]] (required for `CACHED_SEGMENTATION` to actually be consulted at inference time), restoring the pipeline's original `training` value in a `try/finally` — so a failure partway through never leaves the pipeline stuck in training mode.

### Why the reference goes on a shallow copy

The runner's `file` dict doesn't otherwise carry a reference — it's threaded separately into `score()` — so `populate()` must supply one for `oracle_segmentation()` to read. It does that on a copy:

```python
scaffold = dict(file)
scaffold["annotation"] = reference
file[pipeline.CACHED_SEGMENTATION] = oracle_segmentation(scaffold, window, frames)
```

`file` itself therefore **never carries an `annotation` key at any point**. This replaced an earlier set-then-restore-in-`finally` approach, which worked correctly but depended on the `finally` running and on nothing observing `file` in between. A copy removes both dependencies; there is no window in which the key exists.

Verified safe before changing: `oracle_segmentation()` ([oracle.py:64-68](../../src/pyannote/audio/pipelines/utils/oracle.py)) reads only `"duration"` (falling back to `Audio.get_duration`) and `"annotation"`, and never writes to the mapping it is given — so a shallow copy carries everything it needs.

Equivalence was confirmed against the previous implementation on four real AMI meetings from the reference the recorded oracle runs used, at production geometry (593 frames): byte-identical arrays in every case. That check must be run against a **cold cache** — `Runner.run()` returns the cached RTTM *before* `populate()` is reached, so a warm-cache comparison reads back the very file it is comparing against and passes unconditionally.

### The `_expects_num_speakers` guard

`populate()` raises `RuntimeError` when `pipeline._expects_num_speakers` is True (`KMeansClustering` and `OracleClustering` both set `expects_num_clusters = True`).

**This guard is a refusal of an unsupported combination, not leak prevention.** Speaker-count-driven clustering needs a speaker count and the harness supplies none: `run_harness.py` builds `file` as `{"uri", "audio"}` and `Runner.run()` calls the pipeline with no `num_speakers`. Without the guard, `SpeakerDiarization.apply()` falls through to its own `ValueError: num_speakers must be provided ...` from inside the library (`speaker_diarization.py:600-607`). Failing at the harness level instead names the actual problem.

Where `k` should come from — oracle count, a fixed value, or an estimate — is a design decision with a different experiment behind each option, and is a **prerequisite for the clustering sweep** rather than something this class should decide implicitly. Note that using the oracle count makes the run an oracle-segmentation-*plus*-oracle-count condition (Confound A in [[Oracle Segmentation Seam]]) and must be recorded as such.

## Using it in a real run

`run_harness.py`'s CLI selects `OracleSegmentation` when `--run-condition oracle_segmentation` is passed, building its `reference_lookup` from `--oracle-rttm <path>`:

```bash
python run_harness.py --data-root <data-root> --condition IHM \
    --run-condition oracle_segmentation --oracle-rttm <path/to/only_words.rttm>
```

The manifest's `condition` field (not `segmentation_source_id`, which stays `"oracle"` for cache-key purposes) is what `runs/comparison.csv` shows as `oracle_segmentation`, via the existing `--run-condition`/manifest wiring already present in `run_harness.py` before this ticket.

No `only_words` RTTM has been produced yet for the real converted AMI data directory (`~/Code/AMI-diarization-setup/harness-data/IHM/`, see [[AMI Corpus Setup]]) — only the standard collapsed RTTM/UEM pair `AMIDatasetAdapter` already reads. Producing one is a follow-on: either a `convert_for_harness.sh`-style conversion step targeting AMI's word-level annotations, or a small loader alongside `AMIDatasetAdapter`. Until that lands, `--oracle-rttm` must point at a manually prepared or fixture-scale RTTM.

## Tests

- `tests/test_segmentation.py`:
  - `test_oracle_populate_sets_cached_segmentation_matching_fixture_rttm` — builds `OracleSegmentation` from a small synthetic RTTM fixture (`tests/data/oracle_only_words.rttm`) and checks the resulting `CACHED_SEGMENTATION` array matches `oracle_segmentation()` called directly on the same reference, frame-for-frame.
  - `test_oracle_populate_restores_training_flag_after_success` / `_after_failure` — confirm `pipeline.training` is always restored, success or failure.
  - `test_interface_contract` — `OracleSegmentation`/`BaselineSegmentation` share the same `populate` signature and are both `SegmentationSource` subclasses with distinct, stable `id`s.
  - These use a lightweight fake pipeline (`_FakePipelineForOracle`) rather than the shared `tests/conftest.py` `pipeline`/`full_pipeline` fixtures, which are currently broken in this Windows environment for an unrelated reason (a `FileNotFoundError` on a non-ASCII test fixture filename — tracked separately, not a segmentation bug).
- `tests/test_segmentation_non_mutating.py`:
  - `test_populate_never_assigns_annotation_onto_the_caller_file` — uses a `_RecordingDict` that logs every `__setitem__`, because a plain dict cannot distinguish "never set" from "set then restored" once the `finally` has run. Asserting the post-state of `file` would pass on either implementation.
  - `test_populate_preserves_a_pre_existing_annotation_key` — a caller-supplied `annotation` is left untouched rather than overwritten and restored.
  - `test_populate_result_is_unchanged_by_the_shallow_copy` — output equivalence, restated beside the mutation test.
  - `test_guard_is_retained_and_docstring_describes_refusal_not_leak_prevention` — pins the `RuntimeError` and the corrected docstring framing.
- `tests/test_run_harness_oracle_segmentation.py::test_oracle_segmentation_run_appears_in_comparison_csv` — unit-level (mocks `Runner`/`AMIDatasetAdapter` only) proof that a run using `OracleSegmentation` completes, scores, writes a manifest with `condition: oracle_segmentation` and `segmentation_source_id: oracle`, and appears correctly in `comparison.csv` after `aggregate_runs()`.
- `tests/test_run_harness_integration.py::test_oracle_segmentation_run_completes_scores_and_tagged_in_manifest` — the real end-to-end counterpart (drives the actual `full_pipeline` fixture and a new `tests/fixtures/ami/basic/IHM/test.only_words.rttm` fixture); currently blocked by the same pre-existing environment issue as every other `full_pipeline`-dependent test in that file, not by anything in this feature.
- `tests/test_runner.py` and `tests/test_run_harness_integration.py` had three pre-existing assertions (`pytest.raises(NotImplementedError)`, from when `OracleSegmentation` was a stub) updated to `pytest.raises(KeyError)`, since a no-arg `OracleSegmentation()` now fails fast via an empty `reference_lookup` instead of an unconditional `NotImplementedError`.

## Corrected claim: the guard does not prevent a leak

Recorded because this doc and the class docstring both once asserted a mechanism that turned out to be wrong, and the wrong version cost a round of misdirected planning.

**The claim, as it stood:** the guard exists because `file["annotation"]` is briefly present on the real `file` dict; `SpeakerDiarization.apply()` would read that key for the true speaker count *before it was restored*, silently contaminating the condition with oracle speaker count. A shallow copy would therefore remove the leak, make the guard unnecessary, and unblock KMeans.

**Why it was wrong:** there is no race. `populate()` runs to completion — `finally` included — before `Runner.run()` calls `self._pipeline(file)` on the next line. By the time `apply()` reads the key at `speaker_diarization.py:601`, it is already gone. The guard was defensive redundancy over a restore that already worked, not protection against a live leak.

**What the shallow copy actually changed:** nothing about any leak. It is a hygiene improvement (see above), and it is worth having on its own terms. But removing the guard alongside it would have converted a clear harness-level `RuntimeError` into a `ValueError` thrown from inside the library. **KMeans stays blocked either way** — the real blocker is structural and described under the guard section above.

**How the error propagated:** the class docstring stated the sequencing incorrectly, and a later reader took that description at face value rather than checking it against `runner.py`. The docstring now carries a short retraction at the point of the claim, so the same text cannot mislead again. Same failure mode as the mechanism errors recorded in §9 of [[Oracle 2x2 Combined Cells]]: a stated mechanism accepted without verifying the call sequence.

**Note on coverage:** the guard had no test when this was found. `tests/test_segmentation_non_mutating.py::test_guard_is_retained_and_docstring_describes_refusal_not_leak_prevention` now pins both the `RuntimeError` and the docstring framing.

## Non-goals / follow-ons

- No `only_words` RTTM conversion pipeline for the real AMI corpus yet (see above).
- No source patch to `src/pyannote/audio` was made or needed — the entire feature lives in `harness/segmentation.py` and `run_harness.py`.
- **Wiring `num_speakers` through the harness is not done**, and is a prerequisite for any clustering sweep involving KMeans. See the guard section above for why it is a design decision rather than a mechanical change.
