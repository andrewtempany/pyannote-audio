---
status: not-started
created: 2026-09-17
---

# Non-ASCII filename decoding bug (Windows)

## Goal

Fix `pyannote.database`'s file-list reader so it decodes non-ASCII filenames correctly on
Windows, unblocking the harness integration tests that currently can't run to a pass/fail
result on this machine.

## Symptom

`tests/conftest.py`'s `full_pipeline` fixture (shared by every test in
`tests/test_run_harness_integration.py`) fails with:

```
FileNotFoundError: Could not find file "trñ00"
```

## Root cause (confirmed during [[Oracle Ceiling Metrics]])

`tests/data/debug.train.lst` intentionally contains a non-ASCII filename (`trñ00`, added
upstream in commit `b41b176e`, "fix: fix support for non-ASCII characters") to test
non-ASCII path handling, and the matching `trñ00.wav` is correctly tracked in git and
present on disk. `pyannote.database`'s file-list reader decodes that UTF-8-encoded
filename using Windows' default `cp1252` codepage instead of UTF-8, producing a string
that renders identically in a terminal but does not match the actual file on disk.

## Impact

Affects every test in `tests/test_run_harness_integration.py`, including tests that
predate T1/T2's changes. Specifically blocked
`test_run_manifest_written_alongside_report` and
`test_run_manifest_not_written_on_mid_run_failure` from being run to a pass/fail result
during T1 — they were written and are believed correct, but not independently verified
end-to-end on this machine.

## Scope (when picked up)

- Locate the file-list reading code in `pyannote.database` responsible for decoding
  filenames and force UTF-8 decoding regardless of platform default codepage.
- Since this lives in the `pyannote.database` dependency (not this repo's own code),
  determine whether the fix belongs upstream, as a pinned patch, or as a
  workaround/monkeypatch local to this repo's test setup.
- Once fixed, run the previously-blocked integration tests
  (`test_run_manifest_written_alongside_report`,
  `test_run_manifest_not_written_on_mid_run_failure`) plus the full
  `test_run_harness_integration.py` suite to confirm all pass.

## Non-goals

- Not blocking any currently-open ticket (T3/T4/T5 don't depend on these specific
  integration tests) — this is a correctness/coverage gap, not a hard blocker.
