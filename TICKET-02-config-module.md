# 02 — Config module

Part of the [eval harness epic](TICKET-eval-harness.md). Depends on: [00](TICKET-00-environment-setup.md).

## Scope

`harness/config.py` — a single place holding: AMI data root path, split (train/dev/test), AMI condition (default `IHM`, overridable to `SDM`), metric settings (collar, skip_overlap — even though the epic pins these, keep them as named config values rather than magic numbers scattered across `scorer.py`), cache directory, and HF token loading via `python-dotenv` from a gitignored `.env`.

## Non-goals

- No dataset parsing (that's ticket 03).
- No model loading.
- Do not name the module `dotenv.py` (shadows the real package — the epic already calls this out).

## Tests first

Add `tests/test_config.py`:

1. `test_missing_data_root_raises_clear_error` — construct config with a nonexistent path, assert it raises with a message that names the missing path (not a generic `FileNotFoundError` with no context).
2. `test_default_condition_is_ihm` — construct config without specifying condition, assert it resolves to `IHM`.
3. `test_condition_overridable_to_sdm` — construct config with `condition="SDM"`, assert it's honored.
4. `test_invalid_condition_rejected` — construct config with an unsupported condition string, assert a clear error (don't silently accept typos).
5. `test_hf_token_loaded_from_dotenv` — write a temp `.env` file with a fake token, point config's loader at it, assert the token is read into config (not into `os.environ` globally, unless that's the deliberate design — decide and test that behavior explicitly).
6. `test_cache_dir_created_if_missing` — point config at a cache dir that doesn't exist yet, assert it's created (or assert config raises clearly if the design is "must pre-exist" — pick one and test it, don't leave it implicit).
7. `test_pyannote_skip_dependency_check_env_var_documented` — not a runtime test necessarily, but confirm somewhere (env or config) that `PYANNOTE_SKIP_DEPENDENCY_CHECK` is set/checked as the epic requires; a simple assertion that the harness sets it if absent, or a comment/README note if it's left to the user — decide and encode the decision in a test.

## Acceptance criteria

- All tests pass.
- No I/O beyond reading `.env` and checking/creating the cache dir — no model loading, no network calls.

## Implementation notes (resolved during build — 2026-09-04)

- `HarnessConfig` is a frozen dataclass with a `.load(...)` classmethod factory (`harness/config.py`), not a bare constructor — validation (missing data root, invalid condition) happens in `load()` before the object is built.
- **HF token loading deliberately does not touch `os.environ`.** Uses `dotenv_values(path)` (returns a dict) rather than `load_dotenv(path)` (which would mutate the process environment). Decision: loading harness config should never leak a secret into global process state as a side effect — callers that need the token read it off `config.hf_token` explicitly.
- **`PYANNOTE_SKIP_DEPENDENCY_CHECK` is the opposite case and *must* mutate `os.environ`** (via `os.environ.setdefault(...)`), since pyannote's own `check_dependencies()` reads it via `os.getenv` internally (confirmed in TICKET-00) — there's no config-object plumbing that could substitute for a real env var here.
- **Gotcha caught during testing:** the `cache_dir` parameter's default value resolves relative to the process's current working directory. Tests that didn't pass an explicit `cache_dir` left a stray `.harness_cache/` directory in the repo root as a side effect of just running the test suite. Fixed in the tests (every test now passes an explicit `tmp_path`-based `cache_dir`), not in `harness/config.py` — a caller-facing tool defaulting to a cwd-relative cache dir is reasonable behavior; it's test isolation that needed fixing.
- Errors raise `ConfigError` (a `ValueError` subclass), always with the offending path or value included in the message text, per the ticket's "not a bare `FileNotFoundError`" requirement.
