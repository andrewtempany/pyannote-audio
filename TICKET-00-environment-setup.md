# 00 — Environment setup: fork installed and importable

Part of the [eval harness epic](TICKET-eval-harness.md). Blocking prerequisite for every other sub-ticket.

## Problem

The dev venv (`diarisation-env`) currently has vanilla `pyannote.audio==3.4.0` installed from PyPI, not this fork's `src/pyannote/audio`. Anything that does `import pyannote.audio` today silently runs upstream code, not this repo's code. The venv is also Python 3.9, while `pyproject.toml` requires `>=3.10`. `pyproject.toml` additionally pins floors (`pyannote-core>=6.0.1`, `pyannote-metrics>=4.0.0`) that are newer than what's actually installed (`pyannote.core==5.0.0`, `pyannote.metrics==3.2.1`). None of the later tickets are trustworthy until this is fixed, since every claim about this fork's API was verified by reading source, not by importing it.

## Scope

- Recreate (or repair) the dev environment on Python >=3.10.
- Install this fork editable (`pip install -e .` or equivalent via `uv`), so `import pyannote.audio` resolves into `src/pyannote/audio`.
- Resolve the dependency floor mismatch: either upgrade `pyannote.core`/`pyannote.metrics` to satisfy `pyproject.toml`, or if the floors are wrong, correct them — check which is true rather than assuming.
- `PYANNOTE_SKIP_DEPENDENCY_CHECK=1` set for local dev (per the epic's Environment section) and confirmed to have no other effect.

## Non-goals

- Do not touch `models/` or `tasks/`.
- Do not upgrade unrelated dependencies beyond what's needed to satisfy `pyproject.toml`'s own floors.

## Tests first (write these before touching the environment)

Add `tests/test_environment.py`:

1. `test_pyannote_audio_resolves_to_this_fork` — import `pyannote.audio`, resolve `pyannote.audio.__file__` with `os.path.realpath`, assert it sits under the repo's `src/` directory (not `site-packages`).
2. `test_python_version_meets_pyproject_floor` — parse the `requires-python` constraint from `pyproject.toml` and assert `sys.version_info` satisfies it (don't hardcode "3.10" — read it from the file, so this test stays correct if the floor changes).
3. `test_installed_dependency_versions_satisfy_pyproject_floors` — for `pyannote.core`, `pyannote.metrics` (and any other pinned sister packages), parse the floor from `pyproject.toml` and assert the installed version (via `importlib.metadata.version`) satisfies it.

These tests should **fail** on the current environment — run them first to confirm they reproduce the problem, then fix the environment until they pass.

## Acceptance criteria

- `pytest tests/test_environment.py` passes.
- `python -c "import pyannote.audio; print(pyannote.audio.__file__)"` prints a path under this repo's `src/`.
- `python smoke.py` (with a valid token and clip) runs without an import or version-check error.

## Implementation notes (resolved during build — 2026-09-04)

- Rebuilt `diarisation-env` on Python 3.10.19 via `pyenv install 3.10.19` (matches `.python-version`, which was set but not actually installed). The old venv was 3.9.6.
- Used `uv venv diarisation-env --python 3.10.19` + `uv pip install --python diarisation-env/bin/python -e ".[test,cli,dev]" python-dotenv` rather than plain `pip`, since `uv.lock` already exists for this project.
- **No `pyproject.toml` dependency-floor edits were needed.** The editable install resolved `pyannote-core==6.0.1` and `pyannote-metrics==4.1` on its own, both already satisfying the `>=6.0.1`/`>=4.0.0` floors — the earlier version mismatch (installed versions below the pinned floors) was purely a symptom of the broken venv, not a wrong pin.
- `PYANNOTE_SKIP_DEPENDENCY_CHECK` was confirmed (via `src/pyannote/audio/utils/dependencies.py`) to have exactly one effect: gating `check_dependencies()`'s raise-vs-warn behavior. No other code path reads it.
