---
status: done
created: 2026-09-04
---

# Harness Config

> Part of the [[Evaluation Harness]]. **`harness/config.py` is code contributed by this project — not upstream `pyannote-audio`.** Its one dependency on pre-existing pyannote code is indirect: it sets `PYANNOTE_SKIP_DEPENDENCY_CHECK`, an environment variable read internally by `check_dependencies()` in pre-existing code at `src/pyannote/audio/utils/dependencies.py` (see [[Environment Setup]]).

## Purpose

A single place holding every value the rest of the harness needs to run: where the AMI data lives, which split/condition to score, metric settings, where to cache hypotheses, and the HuggingFace token needed to load `community-1`. Every other harness component takes a `HarnessConfig` instance rather than reading paths or env vars itself.

## Where it fits

`HarnessConfig` is constructed once at the top of a run (by the [[Orchestrator]], or directly by a caller) and passed into [[Dataset Adapter]] and read by [[Orchestrator]] when constructing the metric accumulators and `Runner`.

## API

```python
from harness.config import HarnessConfig

config = HarnessConfig.load(
    data_root="/path/to/ami",
    split="test",           # default "test"
    condition="IHM",        # default "IHM"; "SDM" also supported
    cache_dir=".harness_cache",
    dotenv_path=".env",     # optional; None means no HF token loaded
    der_collar=0.0,
    der_skip_overlap=False,
)
```

`HarnessConfig` is a **frozen dataclass** with a `.load(...)` classmethod factory — not a bare constructor. All validation (missing data root, invalid condition) happens inside `load()`, before the object is built, so an invalid `HarnessConfig` can never exist.

Fields: `data_root`, `split`, `condition`, `cache_dir`, `der_collar`, `der_skip_overlap`, `hf_token`.

## Behavior worth knowing

- **Missing data root** raises `ConfigError` (a `ValueError` subclass) naming the offending path — never a bare `FileNotFoundError` with no context. The harness never attempts to download AMI itself.
- **Condition** defaults to `IHM` (clean headset audio, used to validate the harness end to end) and can be switched to `SDM` (far-field). Anything else raises `ConfigError` — typos are rejected, not silently accepted.
- **Cache dir** is created (`mkdir(parents=True, exist_ok=True)`) if it doesn't already exist — the design decision here is "create it," not "must pre-exist."
  - Gotcha: if `cache_dir` isn't passed explicitly, it defaults to `.harness_cache` relative to the process's **current working directory**. This is reasonable for a CLI tool but means tests must always pass an explicit (`tmp_path`-based) `cache_dir`, or they'll leave a stray directory in the repo root as a side effect of just running the suite.
- **HF token loading deliberately does not touch `os.environ`.** It uses `dotenv_values(path)` (returns a plain dict) rather than `load_dotenv(path)` (which mutates the process environment). This is a deliberate design choice: loading harness config should never leak a secret into global process state as a side effect. Callers that need the token read it explicitly off `config.hf_token`.
- **`PYANNOTE_SKIP_DEPENDENCY_CHECK` is the opposite case, and *does* mutate `os.environ`** — via `os.environ.setdefault("PYANNOTE_SKIP_DEPENDENCY_CHECK", "1")` — because pyannote-audio's own `check_dependencies()` reads it via `os.getenv` internally; there's no config-object plumbing that could substitute for a real process env var here. See [[Environment Setup]] and [[Segmentation Injection Seam]]'s neighboring context.
- Errors always raise `ConfigError`, always with the offending path or value included in the message text.

## Non-goals

- No dataset/RTTM/UEM parsing — that's [[Dataset Adapter]].
- No model loading.
- The module is deliberately not named `dotenv.py` (would shadow the real `python-dotenv` package it imports from).
- No I/O beyond reading `.env` and checking/creating the cache dir.
