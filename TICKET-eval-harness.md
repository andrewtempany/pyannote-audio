# Build a speaker-diarisation evaluation harness around pyannote.audio

> **Status: EPIC.** This document is the design record and rationale for the harness. It is no longer worked directly — implementation happens in the sub-tickets below, each scoped to be independently testable (TDD: tests written before implementation) and independently verifiable. Read this doc first for context, then work the sub-tickets in order; later ones depend on earlier ones.
>
> | # | Ticket | Depends on |
> |---|---|---|
> | 00 | [Environment setup](TICKET-00-environment-setup.md) | — |
> | 01 | [Segmentation injection seam](TICKET-01-segmentation-injection-seam.md) | 00 |
> | 02 | [Config module](TICKET-02-config-module.md) | 00 |
> | 03 | [Dataset adapter](TICKET-03-dataset-adapter.md) | 02 |
> | 04 | [Segmentation source interface](TICKET-04-segmentation-source-interface.md) | 01 |
> | 05 | [Scorer](TICKET-05-scorer.md) | 00 |
> | 06 | [Runner](TICKET-06-runner.md) | 00, 01, 04 |
> | 07 | [Reporter](TICKET-07-reporter.md) | 05 |
> | 08 | [Orchestrator (integration)](TICKET-08-orchestrator-integration.md) | 03, 04, 05, 06, 07 |
>
> Two corrections found during scoping review, both folded into the sub-tickets below: (1) the codebase currently has this fork's `src/pyannote/audio` *not* installed into the dev venv (`diarisation-env` has vanilla PyPI `pyannote.audio==3.4.0` instead) and the venv's Python 3.9 doesn't meet `pyproject.toml`'s `>=3.10` floor — see ticket 00. (2) `SpeakerDiarization.get_segmentations()`'s `CACHED_SEGMENTATION` check (the seam this epic assumed oracle injection would use) is nested inside `if self.training:` — it does **not** fire during normal inference, contrary to this doc's Component 4 description below. See ticket 01, which must resolve this before the segmentation interface (ticket 04) or runner (ticket 06) are implemented.

## Context (read first, no prior knowledge assumed)

We run the **pyannote.audio `speaker-diarization-community-1`** pipeline on meeting audio and need a **scoring harness** to measure how well it does. "Diarisation" answers *who spoke when*. The pipeline runs in stages: a neural segmentation model, then speaker embeddings, then clustering. We are **not** touching the neural code. This ticket is purely about building the code that runs the pipeline over labelled audio and scores the output against ground truth.

The harness is the reusable measurement tool for the whole project, so correctness and clean structure matter more than speed.

One design constraint drives the architecture: later we will swap the pipeline's **segmentation output** for ground-truth ("oracle") segmentation to measure how much error is fixable downstream. So the segmentation source must be an **injectable input**, not baked into the run. Build the interface now; a full oracle implementation is a later ticket.

## Goal

A harness that, given an AMI split and a data path, runs `community-1` over each file and produces per-file and corpus-level scores as CSV/JSON.

## Scope for this ticket

- Full **baseline** run mode (unmodified `community-1`).
- All four metrics below, correctly configured.
- A **segmentation-source interface** so oracle segmentation can be dropped in later. Implement the baseline source; leave oracle as a documented stub that raises `NotImplementedError`.
- Hypothesis caching so re-scoring does not re-run the expensive pipeline.

## Non-goals (do not do these)

- No training, fine-tuning, or edits under `models/` or `tasks/`.
- **Do not reimplement DER/JER maths.** Use `pyannote.metrics`.
- **Do not use the `pyannote-audio benchmark` CLI.** It cannot produce overlap-region DER or counting error and has no oracle seam.
- Do not implement oracle-segmentation injection logic yet. Interface + stub only.

## Components

Build four small, single-responsibility pieces plus a thin orchestrator.

**1. Dataset adapter** (`datasets.py`)
- Input: an AMI split (train/dev/test) and a root data path (supplied by the user via config, see Environment).
- Output: an iterable of `(uri, reference, uem)` where:
  - `uri` is a file id / audio path,
  - `reference` is a `pyannote.core.Annotation` built from the ground-truth RTTM,
  - `uem` is a `pyannote.core.Timeline` from the official UEM (the "score this region" map).
- This is the only component that knows about file paths, RTTM parsing, and UEM parsing.
- Start with **one AMI condition**. Default to **IHM** (clean headset audio) to validate the harness end to end, and make the condition a config value so **SDM** (far-field) can be selected later.

**2. Runner** (`runner.py`)
- Input: a loaded pipeline, a segmentation source, and one file's audio.
- Output: a hypothesis `Annotation`.
- **Cache hypotheses to disk** as RTTM, keyed by a hash of (pipeline config + segmentation-source id + file uri). On a cache hit, skip the pipeline and load the RTTM. This makes re-scoring instant.
- `community-1` returns a `DiarizeOutput` object that contains **two** annotations: a full one and an **exclusive** (overlap-removed) one. **Score the full annotation.** Using the exclusive one silently breaks overlap scoring. Verify the exact field name against the codebase (see Verification).

**3. Scorer** (`scorer.py`)
- A pure function: `(reference, hypothesis, uem) -> dict` of metric values. No I/O, no model loading.
- Metrics and exact settings:

  - **DER (overall)** via `pyannote.metrics.diarization.DiarizationErrorRate`. Configure `collar=0` and `skip_overlap=False`. A nonzero collar or skipping overlap inflates the score and breaks comparability with published pyannote numbers, and overlap is the whole point of the project so it is never skipped.
  - **Overlap-region DER (primary metric)**: the same DER computation but with the evaluation region **restricted to reference overlap**. Derive the overlap timeline from the reference (the ≥2-speaker regions), intersect it with the file's UEM, and pass that as the `uem` argument to the metric.
  - **JER** via `pyannote.metrics.diarization.JaccardErrorRate`, with the same `collar=0` and `skip_overlap=False`. `JaccardErrorRate` subclasses `DiarizationErrorRate` and already defaults to both, so passing them is belt-and-braces — but pass them explicitly anyway so nobody "aligns" JER to a different collar later.
  - **Speaker counting**: per file, compare the number of distinct speaker labels in reference vs hypothesis. Report the two counts and their absolute difference.

- **Accumulation rule (important):** DER, overlap-DER, and JER are corpus-level ratios. Feed every file into **one** metric object each and read the total at the end. Do **not** average per-file DERs. Counting error is different: it is per-file, and the corpus summary is **MAE** and **exact-match %** across files.

- **Three metric objects, not two (easy bug):** overall DER and overlap-DER must be **separate `DiarizationErrorRate` instances**. A `pyannote.metrics` object accumulates its components internally on every call, so calling one instance twice per file — once with the full UEM, once with the overlap-only UEM — silently sums both regions into one accumulator and produces a meaningless corpus total. Construct `der`, `overlap_der`, and `jer` once at the start of the run and call each exactly once per file.

**4. Reporter** (`reporter.py`)
- Collect per-file rows into a table and write:
  - a per-file CSV: `uri, der, overlap_der, jer, count_ref, count_hyp, count_error`,
  - a corpus summary (CSV or JSON): accumulated DER, accumulated overlap-DER, accumulated JER, counting MAE, counting exact-match %.

**Orchestrator** (`run_harness.py`)
- Wire adapter → runner → scorer → reporter for a given split and config. One command produces both output files.

**Segmentation source interface** (`segmentation.py`)
- A small interface with two implementations planned:
  - `BaselineSegmentation`: uses the pipeline's own segmentation (normal behaviour). Implement this.
  - `OracleSegmentation`: will build segmentation from the reference. Stub it, raise `NotImplementedError`, and leave a docstring describing intent.
- The runner takes the source as a parameter so the oracle drop-in later is a one-line swap.
- **Do not over-engineer the baseline.** The real injection seam already exists in the pipeline: `SpeakerDiarization.get_segmentations()` checks `file[self.CACHED_SEGMENTATION]` and only runs the segmentation model if that key is absent. So the interface needs just two things per source: an **`id`** string (feeds the cache key) and a hook that may **populate that key on the file dict** before the pipeline is applied. `BaselineSegmentation` populates nothing and simply lets the pipeline run — it must not wrap or reimplement anything. `OracleSegmentation` will later fill the key instead.
- The oracle stub's docstring should point at the existing utility it will build on: `pyannote.audio.pipelines.utils.oracle.oracle_segmentation(file, window, frames, num_speakers=None)`, which discretises a reference `Annotation` into the `(num_chunks, num_frames, num_speakers)` `SlidingWindowFeature` that `CACHED_SEGMENTATION` expects. It is already used for oracle *clustering*, so the shape contract is known-good. Do not implement the wiring in this ticket.

## Suggested layout

```
harness/
  __init__.py
  config.py         # data root, split, AMI condition, metric settings, cache dir
  datasets.py       # AMI RTTM/UEM -> (uri, reference, uem)
  segmentation.py   # segmentation-source interface + baseline impl + oracle stub
  runner.py         # pipeline + source + audio -> hypothesis, with caching
  scorer.py         # (ref, hyp, uem) -> metrics dict
  reporter.py       # rows -> per-file CSV + corpus summary
run_harness.py      # entry point
```

## Environment (assume mostly configured, but confirm)

- A HuggingFace token is required to load `community-1`, with the model conditions accepted on its HF page. Load it from a gitignored `.env` via `python-dotenv`. Do not name any file `dotenv.py`.
- Set `PYANNOTE_SKIP_DEPENDENCY_CHECK=1` (the repo is a fork with a dev version string that fails pyannote's version check).
- Audio decoding needs system **ffmpeg shared libraries** (not the static build) for torchcodec.
- The **AMI data path is supplied by the user** via `config.py`. Do not attempt to download AMI. If the path is missing, fail with a clear message.

## Verification against the actual codebase (do not trust these from memory)

Before relying on them, confirm in the installed pyannote source:
- The exact `DiarizeOutput` field for the full vs exclusive annotation.
- The `Annotation` method that returns overlap (≥2-speaker) regions, and whether it returns a `Timeline`.
- The `Timeline` API for intersecting the overlap timeline with the UEM, and the exact mode argument it needs. Confirm the result is a `Timeline` suitable to pass straight to a metric's `uem=` argument.
- Whether any oracle utility already exists under the pipelines package that the oracle stub should later build on.

If any differs from this ticket, follow the codebase and note the discrepancy in a comment.

## Acceptance criteria

1. Running the orchestrator on an AMI split + valid data path produces a per-file CSV and a corpus summary with all four metrics.
2. DER and JER are configured with `collar=0` and `skip_overlap=False`, and corpus DER/overlap-DER/JER come from three separate single-accumulated metric objects, not per-file averages and not one object called twice.
3. Overlap-region DER is scored only over reference overlap regions intersected with the UEM.
4. The runner scores the **full** annotation from `DiarizeOutput`, and caches hypotheses so a second scoring run does not invoke the pipeline.
5. The segmentation source is injectable: baseline works, oracle is a stubbed `NotImplementedError` with a docstring.
6. Counting is reported per file and summarised as MAE and exact-match %.
