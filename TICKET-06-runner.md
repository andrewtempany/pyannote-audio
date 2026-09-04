# 06 — Runner

Part of the [eval harness epic](TICKET-eval-harness.md). Depends on: [00](TICKET-00-environment-setup.md), [01](TICKET-01-segmentation-injection-seam.md), [04](TICKET-04-segmentation-source-interface.md).

## Scope

`harness/runner.py` — given a loaded pipeline, a segmentation source (ticket 04), and one file's audio, produce a hypothesis `Annotation`, with disk-cached RTTM keyed by a hash of (pipeline config + segmentation-source `id` + file uri).

Confirmed field names on `DiarizeOutput` (`src/pyannote/audio/pipelines/speaker_diarization.py:63-75`): it has **three** fields, not two — `speaker_diarization` (full), `exclusive_speaker_diarization` (overlap-removed), `speaker_embeddings` (optional). Score `output.speaker_diarization`. Using `exclusive_speaker_diarization` silently breaks overlap scoring, per the epic.

## Non-goals

- No scoring logic here (that's ticket 05, called by the orchestrator, not the runner).
- No dataset parsing.
- Do not implement `OracleSegmentation`'s real logic — the runner just needs to work correctly when handed either source via ticket 04's interface, including the still-stubbed oracle (which should fail loudly and early, not partway through a run).

## Tests first

Add `tests/test_runner.py`:

1. `test_cache_key_includes_pipeline_config_source_id_and_uri` — construct two runners differing only in segmentation-source `id`, same file, assert they produce different cache keys/paths. Repeat varying only pipeline config, then only uri.
2. `test_cache_miss_invokes_pipeline_and_writes_rttm` — empty cache dir, run once, assert the pipeline was invoked (spy/counter) and a cache RTTM file now exists at the expected path.
3. `test_cache_hit_skips_pipeline_and_loads_rttm` — pre-populate the cache (from test 2's output or a fixture RTTM), run again with the same key, assert the pipeline spy's call count is **zero** and the returned `Annotation` matches the cached RTTM's content.
4. `test_returns_full_annotation_not_exclusive` — mock the pipeline call to return a `DiarizeOutput` with deliberately distinguishable `speaker_diarization` vs `exclusive_speaker_diarization` values, assert the runner's returned hypothesis matches `speaker_diarization`.
5. `test_runner_uses_segmentation_source_populate_hook` — pass a fake segmentation source whose populate hook is a spy, assert it's called with the file dict before the pipeline runs.
6. `test_oracle_source_fails_fast_not_silently` — pass `OracleSegmentation` (still `NotImplementedError` per ticket 04), assert the runner surfaces that error clearly rather than caching a bad/partial result or masking it.
7. `test_runner_real_pipeline_one_file` (marked slow/integration, requires a valid HF token and real audio fixture) — run the actual loaded `community-1` pipeline through `BaselineSegmentation` on one short real clip, confirm a valid non-empty `Annotation` is produced and cached, and that a second call is fast (cache hit) and pipeline-invocation-free — this is the one test that exercises tickets 00, 01, and 04 together for real.

## Acceptance criteria

- Tests 1-6 pass fully mocked, no network/model required — fast, run on every commit.
- Test 7 passes when run manually/in CI with credentials available; documented as such (e.g., a pytest marker) rather than silently skipped without explanation.

## Implementation notes (resolved during build — 2026-09-04)

- **`pipeline_config_id` is a caller-supplied stable string, not derived by introspecting the live `Pipeline` object.** A loaded `SpeakerDiarization` pipeline holds non-serializable state (loaded models, a `PLDA` instance, etc.), so hashing it directly isn't practical. `Runner.__init__` takes `pipeline_config_id` as an explicit parameter; TICKET-08's orchestrator is responsible for generating one (e.g. an HF checkpoint id/revision, or a hash of instantiated hyperparameters) when it constructs the real pipeline. This wasn't spelled out in the original ticket text and is worth knowing before wiring up the orchestrator.
- Cache key = `sha256(f"{pipeline_config_id}|{segmentation_source.id}|{uri}")`; cache file path = `{cache_dir}/{key}.rttm`.
- **The cache-hit path never touches the segmentation source or the pipeline at all** — the cache-existence check happens first, before `segmentation_source.populate()` is even called. This is also what makes the oracle-stub fail-fast test correct: `populate()` (and its `NotImplementedError`) is only ever reached on a cache miss, so a `NotImplementedError` always surfaces before any pipeline call or cache write — never partway through, never masked.
- Cache format: hypothesis `Annotation.write_rttm()` on write, `pyannote.database.util.load_rttm(path)[uri]` on read — same utility TICKET-03 uses for the dataset adapter, kept consistent rather than inventing a second RTTM reader.
- Added a `[tool.pytest.ini_options] markers` entry in the **project's** `pyproject.toml` (not scoped to `harness/`) registering `integration`, so `@pytest.mark.integration` doesn't warn and `pytest -m "not integration"` becomes a valid CI filter. Small and additive, but worth knowing it touched a shared project file.
