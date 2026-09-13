---
status: done
created: 2026-09-13
---

# Run notes in the comparison table

Related: [[Run Manifest]], [[AMI Corpus Setup]].

## What this covers

Every harness run can carry a free-text description of what it actually was — e.g. "oracle
segmentation experiment," "overlap detection done with a random forest model," "DBSCAN used as
the clustering step instead of the pipeline default." This closes a real gap: `clustering_model`
and `segmentation_source_id` in the run manifest are still mostly hardcoded placeholder strings
in this codebase (real segmentation/clustering-model selection isn't wired up yet), so without a
notes field, rows in `runs/comparison.csv` were only distinguishable by DER number and a
timestamp — nothing said *what changed* between two runs.

## How to use it

Pass `--notes` on every `run_harness.py` invocation:

```
python run_harness.py --data-root <path> --condition IHM --notes "DBSCAN used as the clustering step instead of the pipeline default"
```

Even an unmodified baseline run should say so explicitly (e.g. `--notes "baseline community-1,
no modifications"`) rather than leaving it blank — a blank notes field should never be read as
"nothing unusual happened," only as "nobody recorded what happened." The
[[track-harness-run]] skill enforces asking for this on every run rather than guessing.

The value flows straight into the run manifest's `run_config.notes` and appears as a `notes`
column in `runs/comparison.csv` (produced by `harness/aggregate_runs.py`), positioned among the
other config columns, before the metric columns (der, jer, etc.).

## How it fits together

- `run_harness.py`'s `main()` exposes `--notes` (default `""`), forwarded through
  `run_harness()`'s `notes` keyword parameter into the `run_config` dict passed to
  `write_manifest`.
- `harness/run_manifest.py` needed no changes — `run_config` was never schema-constrained beyond
  `REQUIRED_RUN_CONFIG_FIELDS`, so an optional `notes` key just gets stored and round-tripped like
  any other field.
- `harness/aggregate_runs.py`'s `_flatten_manifest` calls `row.setdefault("notes", "")` between
  spreading `run_config` and `summary`, so manifests written before this feature existed (with no
  `notes` key at all) still produce a `notes` column with an empty value instead of a missing
  column or a `KeyError`.

## Gotchas

- `run_config` has no fixed schema beyond the required fields — it's easy to assume adding a new
  optional field requires touching `run_manifest.py`'s validation, but it doesn't. Only
  `aggregate_runs.py`'s flattening step needed to change, because CSV output needs every row to
  have every column even when a source manifest doesn't have the key.
- The `notes` field is genuinely load-bearing right now because `clustering_model` and
  `segmentation_source_id` don't yet reflect real model swaps — until those are wired up to
  actual selection logic, `notes` is the only place an experiment's real identity is recorded at
  all.

## Follow-ons

- Wire up real clustering-model/segmentation-model selection (DBSCAN, random forest overlap
  detector, etc.) so `clustering_model`/`segmentation_source_id` reflect the actual run instead of
  a hardcoded placeholder — `notes` remains useful afterward as free-text context, but shouldn't
  need to carry the entire burden of describing the run once structured fields exist.
- The run manifest still doesn't capture the resolved HF model revision/commit hash or library
  versions (see the earlier gap flagged when [[track-harness-run]] was created) — a manifest with
  identical `pipeline_config_id` across two runs can't currently prove identical model weights
  were used.
