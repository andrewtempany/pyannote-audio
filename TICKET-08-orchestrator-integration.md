# 08 — Orchestrator (integration)

Part of the [eval harness epic](TICKET-eval-harness.md). Depends on: [03](TICKET-03-dataset-adapter.md), [04](TICKET-04-segmentation-source-interface.md), [05](TICKET-05-scorer.md), [06](TICKET-06-runner.md), [07](TICKET-07-reporter.md) — all prior tickets complete.

## Scope

`run_harness.py` — wire adapter (03) → runner (06) → scorer (05) → reporter (07) for a given split and config (02). One command produces both output files. This ticket owns constructing the three metric accumulator instances (`der`, `overlap_der`, `jer`) exactly once at the start of the run and threading them through every `score()` call, per ticket 05's resolved contract.

This is the one ticket whose acceptance test is a genuine end-to-end run — everything below it should already be unit-tested in isolation, so this test is about wiring, not about re-verifying component-level correctness.

## Non-goals

- No new scoring/parsing/caching logic — pure composition of the already-built components.
- No AMI download.

## Tests first

Add `tests/test_run_harness_integration.py` (marked slow/integration; requires a small real or realistic AMI dev subset — 2-3 files is enough — and, since it exercises the runner, a valid HF token):

1. `test_end_to_end_produces_csv_and_summary_files` — run the orchestrator against the small subset, assert both output files exist and are non-empty.
2. `test_per_file_csv_has_one_row_per_file_with_all_four_metrics` — assert row count matches file count, and every row has der/overlap_der/jer/count fields populated (not NaN/missing).
3. `test_corpus_summary_has_all_required_fields` — assert accumulated DER, overlap-DER, JER, counting MAE, and exact-match % are all present.
4. `test_metrics_configured_correctly_end_to_end` — cross-check at least one file's per-file DER against an independently hand-computed value on that fixture, confirming collar=0/skip_overlap=False actually took effect through the full wiring (not just in ticket 05's isolated tests).
5. `test_rerun_is_cache_accelerated` — run twice, assert the second run's wall-clock time is substantially lower (or, more robustly, assert via a pipeline-invocation spy/counter that the second run made zero pipeline calls) — confirms ticket 06's caching survives being driven through the full orchestrator, not just in isolation.
6. `test_segmentation_source_is_swappable_via_config` — run once with `BaselineSegmentation` selected, assert it completes; run once with `OracleSegmentation` selected, assert the orchestrator surfaces the `NotImplementedError` clearly (not a stack trace three layers deep) rather than partially writing output files.

## Acceptance criteria

All six original epic acceptance criteria, now checked against the real wired system:

1. Running the orchestrator on an AMI split + valid data path produces a per-file CSV and a corpus summary with all four metrics. *(tests 1-3)*
2. DER and JER configured with `collar=0`/`skip_overlap=False`; corpus DER/overlap-DER/JER come from three separate single-accumulated metric objects. *(test 4, plus ticket 05's unit tests)*
3. Overlap-region DER scored only over reference overlap regions intersected with the UEM. *(covered by ticket 05, re-confirmed here via test 4)*
4. Runner scores the full annotation and caches hypotheses so a second scoring run doesn't invoke the pipeline. *(test 5)*
5. Segmentation source is injectable: baseline works, oracle is a stubbed `NotImplementedError`. *(test 6)*
6. Counting reported per file and summarised as MAE and exact-match %. *(tests 2-3)*

## Implementation notes (resolved during build — 2026-09-04)

- **Deviation from the ticket's literal test description, made deliberately:** the ticket frames `tests/test_run_harness_integration.py` as requiring a valid HF token (i.e. the real `community-1` pipeline). This environment has none. Instead, all six tests drive the orchestrator with the same real, fully offline `SpeakerDiarization` pipeline from `tests/conftest.py` (`full_pipeline`, shared with TICKET-01/04/06) against real (short, reused) audio fixtures added at `tests/fixtures/ami/basic/IHM/audio/{ES2002a,ES2002b}.wav` (copies of the repo's existing `tests/data/dev00.wav`/`dev01.wav`). Every real component (`HarnessConfig`, `AMIDatasetAdapter`, `Runner`, `score()`, `write_report()`) and a real `Pipeline` object are genuinely exercised end to end — only the model weights are tiny/untrained rather than `community-1`'s. TICKET-06's own `test_runner_real_pipeline_one_file` remains the one place an actual credentialed run gets exercised (and it's still just a `pytest.skip`, not run, in this environment).
- **`uri -> audio path` resolution, deferred by TICKET-03, is decided here:** `{data_root}/{condition}/audio/{uri}.wav` — a small orchestrator-owned helper (`_audio_path()` in `run_harness.py`), not part of the dataset adapter's contract.
- `run_harness(config, pipeline, pipeline_config_id, segmentation_source, per_file_csv_path, summary_path) -> dict` in `run_harness.py` (repo root, not under `harness/`, per the epic's suggested layout). Constructs `der`/`overlap_der`/`jer` once, loops the adapter, calls `runner.run()` then `score()` per file, and calls `write_report()` exactly once at the end — so a mid-loop exception (e.g. `OracleSegmentation`'s stub) naturally leaves zero output files written, with no special-case error handling needed for that guarantee.
- A minimal `argparse`-based CLI lives behind `if __name__ == "__main__":`, loading a real `community-1` pipeline via `Pipeline.from_pretrained` — this satisfies the epic's "one command produces both output files" goal, but is untested (no credentials available); the tested surface is the `run_harness()` function itself, which takes any pre-built pipeline object.
- **Real finding, not a wiring bug:** the first attempt at `test_metrics_configured_correctly_end_to_end` compared the orchestrator's reported per-file DER against a DER recomputed from the hypothesis *reloaded from the RTTM cache* — and failed by ~0.00005 (`0.45568` vs `0.45573`). Cause: `Annotation.write_rttm()` formats timestamps at `%.3f` (millisecond) precision, so a round-tripped hypothesis differs by sub-millisecond amounts from the in-memory object `Runner.run()` actually scores (it scores the hypothesis *before* writing it to cache). Fixed the test — not `run_harness.py`/`runner.py`, which were both correct — to re-run the same deterministic (eval-mode) pipeline call directly instead of reloading from the cache file. Worth knowing for anyone later writing a test that compares a scored value against a cache-reloaded `Annotation`.
