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
| Run manifest / cross-run comparison | `harness/run_manifest.py`, `harness/aggregate_runs.py` | [[Run Manifest]] |
| Oracle/ceiling-analysis metrics & manifest fields | `harness/scorer.py`, `harness/reporter.py`, `harness/run_manifest.py` | [[Oracle Ceiling Metrics]] |
| Post-clustering refinement hook | `harness/refinement.py`, `src/pyannote/audio/pipelines/speaker_diarization.py` | [[Post-Clustering Refinement Hook]] |
| Oracle segmentation provider | `harness/segmentation.py` (`OracleSegmentation`) | [[Oracle Segmentation Provider]] |
| Oracle assignment strategy | `harness/refinement.py` (`make_oracle_strategy`) | [[Oracle Assignment Strategy]] |
| Cross-condition deltas table | `harness/aggregate_runs.py` (`build_cross_condition_table`) | [[Cross-Condition Deltas]] |

## Metrics produced

For each file, and accumulated corpus-wide:

- **DER (overall)** — `pyannote.metrics.diarization.DiarizationErrorRate(collar=0, skip_overlap=False)`, plus its component breakdown (missed detection, false alarm, confusion) as of [[Oracle Ceiling Metrics]].
- **`der_overlap_system`** (renamed from `overlap_der`) — the same DER computation restricted to reference overlap regions (T) intersected with the file's UEM.
- **`der_overlap_assigned`** — added by [[Oracle Ceiling Metrics]]: DER restricted to reference overlap **intersected with hypothesis-detected overlap** (T ∩ D), isolating post-clustering assignment error from detection error.
- **Region census** — added by [[Oracle Ceiling Metrics]]: duration of T∩D, T\D, D\T, per file and corpus-wide.
- **JER** — `pyannote.metrics.diarization.JaccardErrorRate(collar=0, skip_overlap=False)`.
- **Speaker counting** — per file: reference speaker count, hypothesis speaker count, absolute difference; corpus-wide: MAE and exact-match %.

DER/der_overlap_system/der_overlap_assigned/JER are corpus-level ratios, not averages of per-file ratios — see [[Scorer]] and [[Oracle Ceiling Metrics]] for why that matters and how it's enforced. Region census durations are the one exception (summed across files, since there's no ratio to preserve) — see [[Oracle Ceiling Metrics]].

## How to run it

```bash
python run_harness.py \
  --data-root /path/to/ami \
  --split test \
  --condition IHM \
  --cache-dir .harness_cache \
  --dotenv .env \
  --per-file-csv per_file.csv \
  --summary summary.json \
  --run-condition baseline \
  --refinement-strategy identity \
  --runs-dir runs \
  --notes "baseline community-1, no modifications"
```

This loads `community-1` via `pyannote.audio.Pipeline.from_pretrained` (needs an accepted HF token, see [[Harness Config]]), runs `BaselineSegmentation`, applies the post-clustering refinement strategy selected by `--refinement-strategy` (see [[Post-Clustering Refinement Hook]]; default `identity`, a no-op), and writes both output files plus a run manifest. Re-running with the same config and cache dir is fast — hypotheses are cached to disk by content hash (see [[Runner]]).

Note `--condition` (AMI mic condition, e.g. `IHM`/`SDM`) and `--run-condition` (oracle/ceiling-analysis condition, e.g. `baseline`/`oracle_segmentation`) are two different flags — see [[Oracle Ceiling Metrics]] for why.

To call the harness programmatically instead of via the CLI, use `run_harness.run_harness(config, pipeline, pipeline_config_id, segmentation_source, per_file_csv_path, summary_path)` directly — this is the tested surface; the `argparse` CLI wrapper is a thin, untested convenience over it (no credentialed test environment was available at build time).

## Clustering model selection

The pipeline's clustering stage is selectable per run, via two flags that reach the
pipeline through **two different seams**. Understanding why they differ is the whole
of this section.

```bash
python run_harness.py --data-root /path/to/ami \
  --clustering-model vbx \
  --clustering-param threshold=0.7 --clustering-param Fa=0.1
```

### Vocabulary

`--clustering-model` takes a closed vocabulary, defined as `VALID_CLUSTERING_MODELS`
in `harness/run_manifest.py` alongside the other controlled vocabularies:

| Value | Class | Notes |
|---|---|---|
| `pyannote-default` | `VBxClustering` | The default. Whatever the checkpoint ships. |
| `vbx` | `VBxClustering` | The same class, chosen explicitly. |
| `agglomerative` | `AgglomerativeClustering` | |
| `kmeans` | `KMeansClustering` | Selectable but **not runnable** - see below. |

**`pyannote-default` is `VBxClustering`, not `AgglomerativeClustering`.** Several
project notes claimed otherwise; that claim is false. community-1 ships
`clustering: VBxClustering` with `threshold: 0.6, Fa: 0.07, Fb: 0.8`, and those are the
values behind the recorded baseline DER `0.17048543579940637`. It follows that
`pyannote-default` and `vbx` are **one condition, not two**: same class, same
hyperparameters, same cache key. They differ only in whether the choice is stated
explicitly, which matters for reading a sweep's manifests, not for what runs.

`OracleClustering` is a real `Clustering` enum member but is deliberately **absent**
from the vocabulary. Its absence is written down in `run_manifest.py` precisely
because adding it would be a one-line change: an oracle clustering available as an
ordinary sweep point would produce impossibly good numbers under a manifest
indistinguishable from any other run.

### The two seams

From `src/pyannote/audio/core/pipeline.py:275-294`, the class and its hyperparameters
are fixed at different moments:

- **The class, at construction.** `from_pretrained` reads
  `config["pipeline"]["params"]` and calls `Klass(**params)`, so the clustering class
  is decided before the object exists. `SpeakerDiarization.__init__`
  (`speaker_diarization.py:283-295`) then does three things with the name: the
  `Clustering` enum lookup, the `VBxClustering`-only special case that supplies
  `self._plda` (VBx's `__init__` takes a positional `plda` with no default, so it
  cannot be built from a bare name), and setting `_expects_num_speakers` from
  `clustering.expects_num_clusters`.

  Selection therefore happens at construction, **not** by assigning
  `pipeline.clustering` on a loaded pipeline. That assignment sets one of the three
  and leaves `klustering` and `_expects_num_speakers` describing the previous class.
  Nothing crashes; the pipeline simply disagrees with itself, and
  `_expects_num_speakers` governs whether speaker counts are plumbed through.

- **The hyperparameters, via `pipeline.instantiate()`.** This is the sanctioned seam
  and already how `0.6/0.07/0.8` reach the default VBx instance, so a sweep point does
  not need the pipeline rebuilt. It also fails loudly for free: base
  `Pipeline.instantiate` raises `ValueError: parameter '<name>' does not exist`, so a
  typo cannot be silently ignored. `VBxClustering` overrides `__call__` rather than
  `cluster` (`clustering.py:572`), but that is irrelevant to this route -
  `instantiate` acts on the base class's descriptor machinery, not on the clustering
  entry point. Verified rather than assumed.

### How the class override is applied

`Pipeline.from_pretrained` has a **fixed signature** (`checkpoint, revision,
hparams_file, subfolder, token, cache_dir`) with no `**kwargs`, so there is no seam
for overriding a construction parameter - passing `clustering=` raises `TypeError`.
Since editing `src/pyannote/` is out of scope,
`run_harness._from_pretrained_with_clustering` reproduces `from_pretrained`'s own
sequence over the shipped config: read `config.yaml`, swap
`pipeline.params.clustering`, `expand_subfolders` against the real checkpoint id (so
`$model/segmentation`, `$model/embedding` and `$model/plda` still resolve - handing
`from_pretrained` a config *dict* instead would set `model_id = Path.cwd()` and break
that), construct `SpeakerDiarization(**params)`, then apply the shipped `params`.

**The default condition never goes through that helper.** `build_pipeline` branches:
when the requested class is the shipped one, it makes the untouched pre-ticket
`Pipeline.from_pretrained(checkpoint, token=...)` call. That branch is what guarantees
the baseline condition and its cache key cannot have moved.

One deliberate deviation from the library: the shipped config instantiates VBx's
`threshold`/`Fa`/`Fb`, and `instantiate` raises on a parameter the target class does
not declare - so handing agglomerative VBx's `Fa` would make every non-default
selection crash on load. Shipped defaults are therefore filtered to the names the
target class actually declares, and the drop is **printed**, not silent:

```
note: AgglomerativeClustering does not declare Fa, Fb -- shipped value(s) not applied
```

That matters because `threshold` exists on both VBx and agglomerative with *different*
meanings and ranges (`Uniform(0.5, 0.8)` vs `Uniform(0.0, 2.0)`), so a run that
inherits it is not running that class's own default. An explicit `--clustering-param`
is applied afterwards and always wins.

### `kmeans` is selectable but not runnable

`KMeansClustering.expects_num_clusters` is `True`, and `SpeakerDiarization.apply()`
requires `num_speakers` in that case (`speaker_diarization.py:600-607`). The harness
builds `file` as `{"uri", "audio"}` and passes no count, so `build_pipeline` **refuses
it at selection time** with a `ValueError` naming the class and the missing count.
Deciding where *k* comes from - oracle count, fixed, or estimated - is a separate
ticket.

Refused at selection rather than left to the library for two reasons. The library
would raise part-way through the corpus, after minutes of GPU work. And more
seriously: `Runner.run()` sets `pipeline.training = True`, and the library's guard
falls back to `len(file["annotation"].labels())` when an annotation is present. Today
the harness keeps `annotation` off the file dict so that fallback cannot fire - but if
it ever did, a kmeans run would silently take its speaker count from the ground truth
and report an oracle-count result under an ordinary baseline manifest.
`harness/segmentation.py` already guards the oracle-segmentation path this way; this
covers the baseline path too.

### Cache keys

Clustering configuration is part of the **final-hypothesis** cache key and deliberately
absent from the **intermediate** key - see [[Pre-Clustering Cache]], which owns that
key and its reasoning. The practical consequence for a sweep: every point gets its own
RTTM, while all points share one cached copy of segmentation and embeddings. Verified
on the real pipeline: `threshold` 0.6 -> 0.7 moves the final key
(`190b62c3...` -> `d12f731e...`) while the intermediate key stays `c636fc56...`.

## What the harness deliberately does not do

- No training, fine-tuning, or edits under `models/`/`tasks/`.
- No DER/JER math of its own — everything metric-related delegates to `pyannote.metrics`.
- Does not use the `pyannote-audio benchmark` CLI (it can't produce overlap-region DER or counting error, and has no oracle seam).
- Does not implement oracle-segmentation injection logic yet — `OracleSegmentation` is an interface + documented stub (`NotImplementedError`). See [[Segmentation Source Interface]].
- Does not download AMI. The data root is supplied by the caller; a missing/empty root fails with a clear error rather than attempting a fetch.
- Does not supply `num_speakers`, so clustering methods requiring a speaker count (`KMeansClustering`, `OracleClustering`) cannot be run - selection refuses them rather than failing mid-corpus. See "Clustering model selection" above.
- Does not run clustering sweeps. `--clustering-model`/`--clustering-param` make them possible; choosing and running the points is separate work.

## Verification (how "done" was confirmed for this doc conversion)

Before converting any ticket to documentation, each was checked against the actual repo state rather than trusting its claimed status:

- Every `harness/*.py` file described by a ticket exists and its contents were read and cross-checked against that ticket's Scope and Implementation Notes.
- `run_harness.py` (the orchestrator, TICKET-08) exists at the repo root as described.
- The full test suite for the harness (`tests/test_environment.py`, `test_segmentation_seam.py`, `test_config.py`, `test_datasets.py`, `test_segmentation.py`, `test_scorer.py`, `test_runner.py`, `test_reporter.py`, `test_run_harness_integration.py`) was run: **55 passed, 1 skipped** (the one skip is `test_runner_real_pipeline_one_file`, deliberately gated on real HF credentials per TICKET-06 — documented as a `pytest.mark.integration`-style manual/CI-only test, not a silent skip).
- The TICKET-01 finding (that `SpeakerDiarization.get_segmentations()`'s `CACHED_SEGMENTATION` check is nested inside `if self.training:`) was independently re-confirmed by reading `src/pyannote/audio/pipelines/speaker_diarization.py` directly — the code matches the ticket's description exactly, and no source patch exists in that file (the fix is a calling-convention decision documented in `harness/segmentation.py`, not a change to pyannote-audio itself).
- The `oracle_segmentation` utility citation (TICKET-04) was independently confirmed at `src/pyannote/audio/pipelines/utils/oracle.py` and its use for oracle clustering at `src/pyannote/audio/pipelines/clustering.py` (around the `oracle_segmentation(file, window, frames=frames)` call).

All nine tickets (`TICKET-00` through `TICKET-08`) were confirmed complete and converted. The original `TICKET-*.md` files remain in place at the repo root, unmodified, per this vault's workflow (they predate it and only stay put — the vault holds the documentation).
