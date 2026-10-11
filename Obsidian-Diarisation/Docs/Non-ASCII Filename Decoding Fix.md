---
status: done
created: 2026-09-17
---

# Non-ASCII filename decoding fix (Windows test setup)

> Two related fixes, both confined to `tests/conftest.py`. Neither touches
> `diarisation-env/`, `src/pyannote/audio/`, `harness/`, or `tests/data/debug.train.lst`.
> Together they're what makes `tests/test_segmentation_seam.py` and
> `tests/test_run_harness_integration.py` runnable to a genuine pass/fail result on
> Windows at all — before this, both files failed at fixture setup, before any test
> code ran.

## What this fixes

Two independent, pre-existing problems, both specific to running this repo's test
suite on Windows, both surfaced only once you try to actually run
`tests/test_segmentation_seam.py` / `tests/test_run_harness_integration.py` here
(CI only runs on `ubuntu-latest`, which hits neither):

1. **Non-ASCII filenames in `.lst` files decode wrong.** `pyannote.database`'s
   file-list reader opens `.lst` files without an explicit encoding, so Python falls
   back to the platform's default codepage — `cp1252` on Windows, not UTF-8.
2. **PyTorch DataLoader workers can't start.** `SpeakerDiarizationTask` and
   `SupervisedRepresentationLearningWithArcFace` default to `num_workers > 0` on this
   machine, and Windows' `spawn`-based multiprocessing can't pickle the dynamically
   created protocol class these fixtures use.

## Problem 1: UTF-8 decoding forced at file-list load time

### Root cause

`tests/data/debug.train.lst` intentionally contains a non-ASCII filename (`trñ00`,
added upstream in commit `b41b176e`, "fix: fix support for non-ASCII characters") to
exercise non-ASCII path handling; the matching `trñ00.wav` is tracked in git and
present on disk.

The actual decode happens in the installed `pyannote.database` package, at
`pyannote/database/loader.py`'s `load_lst()`:

```python
def load_lst(file_lst):
    with open(file_lst, mode="r") as fp:   # no encoding= -> platform default
        lines = fp.readlines()
    return [line.strip() for line in lines]
```

No `encoding=` argument means Python falls back to
`locale.getpreferredencoding(False)`, which is `cp1252` on Windows. The raw bytes on
disk for the filename are `b"tr\xc3\xb100"` (UTF-8 for `trñ00`). Decoded as cp1252,
that becomes `"trÃ±00"` — a string that can render similarly in some terminals but
does not match the actual `trñ00.wav` file. `pyannote.database.FileFinder` then raises
`FileNotFoundError` while resolving it (that's where the error surfaces —
`file_finder.py:126` — but the bad decode happens earlier, at list-load time).

There's a second, unused copy of `load_lst` in `pyannote/database/util.py` with an
identical body; it's dead code, never imported by `custom.py`, not part of the call
path. The real call path is `pyannote/database/custom.py`'s `subset_entries()`
generator (used while building the `Debug` protocol during `registry.load_database()`),
which calls `load_lst(...)`. `custom.py` imports that name via `from .loader import
load_lst` — a direct binding to the function object, made at `custom.py` import time.

### How the fix works

`tests/conftest.py` defines a drop-in replacement, `_force_utf8_load_lst()`, identical
to the original except for `encoding="utf-8"`, and monkeypatches it onto **both**
module references — `pyannote.database.loader.load_lst` and
`pyannote.database.custom.load_lst` — inside `pytest_sessionstart`, before
`registry.load_database(...)` runs. Patching only `loader.load_lst` would not be
enough: `custom.py` already holds its own reference to the original function object
from its `from .loader import load_lst` import, so that reference has to be
reassigned too.

```python
def _force_utf8_load_lst(file_lst):
    with open(file_lst, mode="r", encoding="utf-8") as fp:
        lines = fp.readlines()
    return [line.strip() for line in lines]


def pytest_sessionstart(session):
    import pyannote.database.custom as _database_custom
    import pyannote.database.loader as _database_loader

    _database_loader.load_lst = _force_utf8_load_lst
    _database_custom.load_lst = _force_utf8_load_lst

    from pyannote.database import registry
    registry.load_database("tests/data/database.yml")
```

## Problem 2: DataLoader workers and Windows `spawn`

### Root cause

Once problem 1 stopped blocking fixture setup, `tests/test_segmentation_seam.py` and
`tests/test_run_harness_integration.py` reach real model training
(`trainer.fit(model)`, inside `conftest.py`'s `_fit()` helper) and failed there with:

```
_pickle.PicklingError: Can't pickle <class 'pyannote.database.registry.Debug'>:
attribute lookup Debug on pyannote.database.registry failed
```

`SpeakerDiarizationTask.__init__` (`src/pyannote/audio/core/task.py`) defaults
`num_workers` to `multiprocessing.cpu_count() // 2`, which is `> 0` on typical
machines. PyTorch's `DataLoader` then spawns worker *processes*. Windows has no
`fork()`, so `multiprocessing` uses `spawn`, which requires everything passed to a
worker process to be picklable by dotted module path. `registry.load_database()`
creates the `Debug` protocol class dynamically at runtime and attaches it to the
`pyannote.database.registry` module after the fact, so the pickler can't resolve it by
attribute lookup in a fresh child process.

This is not new or specific to this repo's own code — `task.py` already has an
identical carve-out for macOS, which has the same class of `spawn`-by-default
multiprocessing limitation:

```python
# src/pyannote/audio/core/task.py (pre-existing, unmodified)
if (
    num_workers > 0
    and sys.platform == "darwin"
    and sys.version_info[0] >= 3
    and sys.version_info[1] >= 8
):
    warnings.warn(
        "num_workers > 0 is not supported with macOS and Python 3.8+: "
        "setting num_workers = 0."
    )
    num_workers = 0
```

Windows just isn't covered by that existing platform check. CI never encounters this
because `.github/workflows/test.yml` only runs on `ubuntu-latest`, which uses `fork`.

### How the fix works

Rather than extend the `src/pyannote/audio/core/task.py` platform check (out of scope
for test-setup-only changes), `tests/conftest.py` passes `num_workers=0` explicitly at
the two places it constructs tasks, with a comment pointing at the macOS precedent
above:

```python
@pytest.fixture(scope="session")
def trained_segmentation_model(protocol):
    # num_workers=0: ... same class of limitation as the existing macOS carve-out
    # in task.py:291-301 ...
    task = SpeakerDiarizationTask(protocol, num_workers=0)
    return _fit(SimpleSegmentationModel(task=task), task)


@pytest.fixture(scope="session")
def trained_embedding_model(protocol):
    task = SupervisedRepresentationLearningWithArcFace(protocol, num_workers=0)
    return _fit(_MaskAwareEmbeddingModel(task=task), task)
```

These are tiny, single-step, in-process debug models — worker parallelism buys nothing
here regardless of platform.

### Scope of effect

This fix only reaches tests that go through `conftest.py`'s
`trained_segmentation_model` / `trained_embedding_model` / `pipeline` / `full_pipeline`
fixtures. Some other pre-existing test files build their own tasks locally with
`num_workers` hardcoded in the test file itself (e.g. `tests/inference_test.py` and
`tests/tasks/test_reproducibility.py` both use `num_workers=4` directly) — those are
untouched by this fix and still hit the same `PicklingError` on Windows. That's a
separate, still-open gap in those files, not addressed here.

## Verification

`tests/test_segmentation_seam.py` — all 3 pass, including
`test_reproduces_bug_cached_segmentation_ignored_outside_training`, which asserts a
pre-populated `CACHED_SEGMENTATION` is ignored while `pipeline.training` is `False` —
confirming this fix didn't change, and doesn't mask, the seam behaviour documented in
[[Segmentation Injection Seam]].

`tests/test_run_harness_integration.py` — 8 of 10 pass, including the two tests this
was originally blocking end-to-end verification for:
`test_run_manifest_written_alongside_report` and
`test_run_manifest_not_written_on_mid_run_failure`. The remaining 2 failures
(`test_per_file_csv_has_one_row_per_file_with_all_four_metrics`,
`test_corpus_summary_has_all_required_fields`) are unrelated to either fix here — both
fail on a field-naming mismatch between the test's expected key (`overlap_der`) and
what the harness summary dict actually produces (`der_overlap_system` /
`der_overlap_assigned`). That's a separate, real gap, left for its own ticket.

Full suite, same scope before and after (excluding `tests/test_run_notebooks.py`,
which fails to collect on this machine for an unrelated reason — `papermill` isn't
installed in this venv, a pre-existing gap in `diarisation-env`): passing count went
from 199 to 203 on identical scope, with the FAILED count unchanged and ERROR count
dropping from 10 to 6. The 4 tests that moved from ERROR to PASSED are exactly the
ones that route through the fixtures this fix touches:
`test_segmentation.py::test_baseline_end_to_end_runs_pipeline_normally` and all 3
`test_segmentation_seam.py` tests. Remaining failures/errors elsewhere in the suite
(`test_train.py`, `inference_test.py`, `tasks/test_reproducibility.py`) are pre-existing,
confirmed via `git log` to predate this branch, and build tasks outside these fixtures
— unaffected by either fix, and out of scope here.

## Related

- [[Segmentation Injection Seam]] — the seam behaviour that
  `test_segmentation_seam.py` verifies; this fix is what makes that verification
  actually run on Windows.
- [[oracle-segmentation-seam-no-op]] — was blocked on
  `tests/test_segmentation_seam.py` running to a genuine pass; unblocked by this fix.
