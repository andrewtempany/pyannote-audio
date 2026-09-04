---
status: done
created: 2026-09-04
---

# Orchestrator

> Part of the [[Evaluation Harness]]. **`run_harness.py` is code contributed by this project — not upstream `pyannote-audio`.** It wires together this project's own harness components and loads a pre-existing pyannote-audio pipeline via `pyannote.audio.Pipeline.from_pretrained` (`src/pyannote/audio/core/pipeline.py`), plus `pyannote.metrics.diarization.DiarizationErrorRate`/`JaccardErrorRate` (sister package, no in-repo path).

## Purpose

Pure composition: wire [[Dataset Adapter]] → [[Runner]] → [[Scorer]] → [[Harness Reporter]] for a given split and config, so one call (or one CLI invocation) produces both output files. This is the one place that constructs the three corpus-level metric accumulators (`der`, `overlap_der`, `jer`) exactly once per run and threads them through every [[Scorer]] call, per that module's resolved accumulator contract.

## Where it fits

Sits above every other harness component — it is the only piece that touches all of them. See [[Evaluation Harness]] for the full data-flow diagram.

## API

```python
from run_harness import run_harness

summary = run_harness(
    config,                 # harness.config.HarnessConfig
    pipeline,                # a loaded pyannote.audio SpeakerDiarization pipeline
    pipeline_config_id,      # stable string, e.g. an HF checkpoint id/revision
    segmentation_source,     # harness.segmentation.SegmentationSource instance
    per_file_csv_path,
    summary_path,
) -> dict
```

Internally:

```python
der = DiarizationErrorRate(collar=config.der_collar, skip_overlap=config.der_skip_overlap)
overlap_der = DiarizationErrorRate(collar=config.der_collar, skip_overlap=config.der_skip_overlap)
jer = JaccardErrorRate(collar=config.der_collar, skip_overlap=config.der_skip_overlap)

runner = Runner(pipeline, pipeline_config_id, segmentation_source, config.cache_dir)
adapter = AMIDatasetAdapter(config)

rows = []
for uri, reference, uem in adapter:
    file = {"uri": uri, "audio": str(_audio_path(config, uri))}
    hypothesis = runner.run(file)
    row = score(reference, hypothesis, uem, der, overlap_der, jer)
    row["uri"] = uri
    rows.append(row)

return write_report(rows, der, overlap_der, jer, per_file_csv_path, summary_path)
```

`write_report()` (see [[Harness Reporter]]) is called exactly once, at the very end, after every file has been scored. This means a mid-loop exception — e.g. [[Segmentation Source Interface|`OracleSegmentation`]]'s still-stubbed `NotImplementedError` — naturally leaves **zero** output files written, with no special-case error handling needed to guarantee that.

## `uri` → audio path resolution

[[Dataset Adapter]] deliberately never resolves a uri to an audio file (it only knows RTTM/UEM). That decision was made here instead, via a small orchestrator-owned helper:

```python
def _audio_path(config: HarnessConfig, uri: str) -> Path:
    return config.data_root / config.condition / "audio" / f"{uri}.wav"
```

i.e. `{data_root}/{condition}/audio/{uri}.wav`. If the real AMI layout differs, this is the place to change it.

## CLI entry point

A minimal `argparse` CLI lives behind `if __name__ == "__main__":` in `run_harness.py`. It loads a real `community-1` pipeline via `pyannote.audio.Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=config.hf_token)` and runs it through `BaselineSegmentation`:

```bash
python run_harness.py \
  --data-root /path/to/ami --split test --condition IHM \
  --cache-dir .harness_cache --dotenv .env \
  --per-file-csv per_file.csv --summary summary.json
```

This CLI path is **untested** — no credentialed environment was available at build time. The tested surface is the `run_harness()` function itself, which accepts any pre-built pipeline object, so it works equally well against a real `community-1` pipeline or a small offline test pipeline.

## How this was tested without real AMI data or an HF token

`tests/test_run_harness_integration.py` drives `run_harness()` with the same real, fully offline `SpeakerDiarization` pipeline used by [[Segmentation Injection Seam]] and [[Segmentation Source Interface]]'s tests (`full_pipeline` in `tests/conftest.py`), against real (short, reused) audio fixtures at `tests/fixtures/ami/basic/IHM/audio/`. Every real harness component (`HarnessConfig`, `AMIDatasetAdapter`, `Runner`, `score()`, `write_report()`) and a real `Pipeline` object are genuinely exercised end to end — only the model weights are tiny/untrained rather than `community-1`'s actual weights. [[Runner]]'s own `test_runner_real_pipeline_one_file` remains the one place an actual credentialed run gets exercised, and it's skipped (not run) in this environment.

Six integration tests cover: both output files get produced and are non-empty; per-file row count matches file count with all metrics populated; the corpus summary has all required fields; at least one file's DER is cross-checked against an independently hand-computed value (confirming `collar=0`/`skip_overlap=False` actually took effect through the full wiring); a second run makes zero pipeline calls (cache-accelerated); and swapping to `OracleSegmentation` surfaces its `NotImplementedError` cleanly rather than partially writing output files.

### A real finding surfaced while testing (not a wiring bug)

An early version of the DER cross-check test compared the orchestrator's reported per-file DER against a DER recomputed from the hypothesis *reloaded from the RTTM cache* — and it failed by about `0.00005` (`0.45568` vs `0.45573`). Cause: `Annotation.write_rttm()` formats timestamps at millisecond (`%.3f`) precision, so a round-tripped hypothesis differs by sub-millisecond amounts from the in-memory object [[Runner]] actually scores (scoring happens *before* the RTTM write). The test was fixed — not `run_harness.py` or `runner.py`, which were both already correct — to re-run the same deterministic pipeline call directly instead of reloading from the cache file. Worth remembering for any future test that compares a scored value against a cache-reloaded `Annotation`.

## Non-goals

- No new scoring, parsing, or caching logic — pure composition of the already-built components.
- No AMI download.
