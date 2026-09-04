---
status: done
created: 2026-09-04
---

# Evaluation Harness

> **Everything under `harness/` and `run_harness.py` is code contributed by this project.** None of it is upstream `pyannote-audio`. It is a measurement tool built *on top of* the pre-existing library to score the `pyannote.audio` `speaker-diarization-community-1` pipeline against ground truth. Where a harness component depends on a pre-existing pyannote class or interface, that dependency is named explicitly (with file path) in the relevant subsection and in each component's own doc page.

This is the index/overview doc for the harness. Each component also has its own doc page under `Docs/` with full detail; this page covers what the harness is for, how the pieces fit together, and how to run it.

Originally built from a set of tickets (`TICKET-eval-harness.md` and `TICKET-00` through `TICKET-08`, still at the repo root) predating this vault's ticket workflow. All nine tickets were verified complete before conversion — see "Verification" below.

## Why this exists

We run `pyannote.audio`'s `speaker-diarization-community-1` pipeline on meeting audio (AMI corpus) and need to measure how well it does. Diarisation answers "who spoke when." The pipeline itself is neural (segmentation → embeddings → clustering) and is **not** touched by this harness — the harness only runs the pipeline over labelled audio and scores its output.

One design constraint shaped the architecture: a later phase of the project will swap the pipeline's segmentation output for ground-truth ("oracle") segmentation, to measure how much error is fixable downstream of the segmentation stage. So the segmentation source is an injectable input to the harness, not baked into the run — see [[Segmentation Source Interface]].

## Pipeline / data flow

```
AMI RTTM/UEM on disk
        │
        ▼
harness/datasets.py  (AMIDatasetAdapter)  ──yields──▶  (uri, reference, uem)
        │
        ▼
run_harness.py orchestrator ──constructs file dict {"uri", "audio"}──▶
        │
        ▼
harness/runner.py (Runner.run) ──calls──▶ segmentation_source.populate(pipeline, file)
        │                                          │
        │                                 (harness/segmentation.py:
        │                                  BaselineSegmentation = no-op,
        │                                  OracleSegmentation = stub)
        ▼
   pipeline(file)  [pre-existing pyannote.audio SpeakerDiarization pipeline]
        │
        ▼
   DiarizeOutput.speaker_diarization  ──cached as RTTM──▶  hypothesis Annotation
        │
        ▼
harness/scorer.py (score()) ──uses (reference, hypothesis, uem) + shared──▶
        │                     DER / overlap-DER / JER accumulators
        ▼
   per-file metrics dict
        │
        ▼
harness/reporter.py (write_report()) ──▶ per-file CSV + corpus summary JSON
```

The orchestrator (`run_harness.py`, see [[Orchestrator]]) owns wiring all of this together and constructing the three corpus-level metric accumulators exactly once per run.

## Components (doc pages)

| Component | File | Doc |
|---|---|---|
| Environment setup | dev venv / editable install | [[Environment Setup]] |
| Segmentation injection seam | `src/pyannote/audio/pipelines/speaker_diarization.py` (pre-existing, calling-convention finding) | [[Segmentation Injection Seam]] |
| Config | `harness/config.py` | [[Harness Config]] |
| Dataset adapter | `harness/datasets.py` | [[Dataset Adapter]] |
| Segmentation source interface | `harness/segmentation.py` | [[Segmentation Source Interface]] |
| Scorer | `harness/scorer.py` | [[Scorer]] |
| Runner | `harness/runner.py` | [[Runner]] |
| Reporter | `harness/reporter.py` | [[Harness Reporter]] |
| Orchestrator | `run_harness.py` | [[Orchestrator]] |

## Metrics produced

For each file, and accumulated corpus-wide:

- **DER (overall)** — `pyannote.metrics.diarization.DiarizationErrorRate(collar=0, skip_overlap=False)`.
- **Overlap-region DER** (the project's primary metric) — the same DER computation restricted to reference overlap regions intersected with the file's UEM.
- **JER** — `pyannote.metrics.diarization.JaccardErrorRate(collar=0, skip_overlap=False)`.
- **Speaker counting** — per file: reference speaker count, hypothesis speaker count, absolute difference; corpus-wide: MAE and exact-match %.

DER/overlap-DER/JER are corpus-level ratios, not averages of per-file ratios — see [[Scorer]] for why that matters and how it's enforced.

## How to run it

```bash
python run_harness.py \
  --data-root /path/to/ami \
  --split test \
  --condition IHM \
  --cache-dir .harness_cache \
  --dotenv .env \
  --per-file-csv per_file.csv \
  --summary summary.json
```

This loads `community-1` via `pyannote.audio.Pipeline.from_pretrained` (needs an accepted HF token, see [[Harness Config]]), runs `BaselineSegmentation`, and writes both output files. Re-running with the same config and cache dir is fast — hypotheses are cached to disk by content hash (see [[Runner]]).

To call the harness programmatically instead of via the CLI, use `run_harness.run_harness(config, pipeline, pipeline_config_id, segmentation_source, per_file_csv_path, summary_path)` directly — this is the tested surface; the `argparse` CLI wrapper is a thin, untested convenience over it (no credentialed test environment was available at build time).

## What the harness deliberately does not do

- No training, fine-tuning, or edits under `models/`/`tasks/`.
- No DER/JER math of its own — everything metric-related delegates to `pyannote.metrics`.
- Does not use the `pyannote-audio benchmark` CLI (it can't produce overlap-region DER or counting error, and has no oracle seam).
- Does not implement oracle-segmentation injection logic yet — `OracleSegmentation` is an interface + documented stub (`NotImplementedError`). See [[Segmentation Source Interface]].
- Does not download AMI. The data root is supplied by the caller; a missing/empty root fails with a clear error rather than attempting a fetch.

## Verification (how "done" was confirmed for this doc conversion)

Before converting any ticket to documentation, each was checked against the actual repo state rather than trusting its claimed status:

- Every `harness/*.py` file described by a ticket exists and its contents were read and cross-checked against that ticket's Scope and Implementation Notes.
- `run_harness.py` (the orchestrator, TICKET-08) exists at the repo root as described.
- The full test suite for the harness (`tests/test_environment.py`, `test_segmentation_seam.py`, `test_config.py`, `test_datasets.py`, `test_segmentation.py`, `test_scorer.py`, `test_runner.py`, `test_reporter.py`, `test_run_harness_integration.py`) was run: **55 passed, 1 skipped** (the one skip is `test_runner_real_pipeline_one_file`, deliberately gated on real HF credentials per TICKET-06 — documented as a `pytest.mark.integration`-style manual/CI-only test, not a silent skip).
- The TICKET-01 finding (that `SpeakerDiarization.get_segmentations()`'s `CACHED_SEGMENTATION` check is nested inside `if self.training:`) was independently re-confirmed by reading `src/pyannote/audio/pipelines/speaker_diarization.py` directly — the code matches the ticket's description exactly, and no source patch exists in that file (the fix is a calling-convention decision documented in `harness/segmentation.py`, not a change to pyannote-audio itself).
- The `oracle_segmentation` utility citation (TICKET-04) was independently confirmed at `src/pyannote/audio/pipelines/utils/oracle.py` and its use for oracle clustering at `src/pyannote/audio/pipelines/clustering.py` (around the `oracle_segmentation(file, window, frames=frames)` call).

All nine tickets (`TICKET-00` through `TICKET-08`) were confirmed complete and converted. The original `TICKET-*.md` files remain in place at the repo root, unmodified, per this vault's workflow (they predate it and only stay put — the vault holds the documentation).
