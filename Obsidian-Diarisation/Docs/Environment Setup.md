---
status: done
created: 2026-09-04
---

# Environment Setup (harness dev environment)

> Part of the [[Evaluation Harness]]. This page documents the *development environment* the harness runs in — it is process/tooling documentation, not a `harness/*.py` module. Nothing here is upstream `pyannote-audio` code; it's the setup this project's contributed harness code depends on to actually exercise this fork's `src/pyannote/audio` rather than a PyPI copy.

## Purpose

Before any harness component can be trusted, the dev environment has to actually be running *this fork's* `src/pyannote/audio`, on a Python version that meets `pyproject.toml`'s floor, with dependency versions that satisfy the same file's pins. Getting this wrong doesn't fail loudly — `import pyannote.audio` silently resolves to whatever's on `sys.path`, which can be a normal PyPI install that behaves subtly differently from this repo's source.

## What was wrong, and what fixed it

The dev venv (`diarisation-env`) was originally:
- Running vanilla `pyannote.audio==3.4.0` from PyPI, not this fork's `src/pyannote/audio`.
- Python 3.9, while `pyproject.toml` requires `>=3.10`.
- Carrying `pyannote.core==5.0.0` / `pyannote.metrics==3.2.1`, both below `pyproject.toml`'s pinned floors (`pyannote-core>=6.0.1`, `pyannote-metrics>=4.0.0`).

Resolution:
- Rebuilt `diarisation-env` on Python **3.10.19** via `pyenv install 3.10.19` (this matched the repo's `.python-version`, which was set but had never actually been installed locally — the old venv was 3.9.6).
- Used `uv venv diarisation-env --python 3.10.19` followed by `uv pip install --python diarisation-env/bin/python -e ".[test,cli,dev]" python-dotenv`, rather than plain `pip`, since a `uv.lock` already existed for this project.
- **No edits to `pyproject.toml`'s dependency floors were needed.** The editable install resolved `pyannote-core==6.0.1` and `pyannote-metrics==4.1` on its own — both already satisfy the pins. The earlier version mismatch was a symptom of the broken venv, not a wrong pin in `pyproject.toml`.

After this, `import pyannote.audio; pyannote.audio.__file__` resolves under this repo's `src/`, not `site-packages`.

## `PYANNOTE_SKIP_DEPENDENCY_CHECK`

This fork carries a dev version string that fails `pyannote.audio`'s own dependency/version check on import. Setting `PYANNOTE_SKIP_DEPENDENCY_CHECK=1` in the environment avoids that failure.

Its effect was confirmed by reading `src/pyannote/audio/utils/dependencies.py` (pre-existing pyannote-audio code): it gates exactly one thing — whether `check_dependencies()` raises or warns. No other code path reads this variable. [[Harness Config]] (`harness/config.py`) sets this via `os.environ.setdefault(...)` at config-load time, so callers using the harness don't need to remember to set it manually — see that doc for why this is one of the few places the harness deliberately does mutate global process environment.

## Verifying the environment

Three checks, encoded as `tests/test_environment.py` (kept as regression tests, not one-off scripts):

1. `test_pyannote_audio_resolves_to_this_fork` — imports `pyannote.audio`, resolves `__file__` with `os.path.realpath`, asserts it's under the repo's `src/` directory.
2. `test_python_version_meets_pyproject_floor` — parses `requires-python` out of `pyproject.toml` itself (not hardcoded) and checks `sys.version_info` against it.
3. `test_installed_dependency_versions_satisfy_pyproject_floors` — for `pyannote.core`, `pyannote.metrics`, and other pinned sister packages, parses the floor from `pyproject.toml` and checks the installed version via `importlib.metadata.version`.

```bash
pytest tests/test_environment.py
python -c "import pyannote.audio; print(pyannote.audio.__file__)"   # should print a path under this repo's src/
python smoke.py   # with a valid HF token and clip — should run without an import/version-check error
```

## Scope notes

- `models/` and `tasks/` were not touched by this environment work.
- No dependencies were upgraded beyond what `pyproject.toml`'s own floors required.
