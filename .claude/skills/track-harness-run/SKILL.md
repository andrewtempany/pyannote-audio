---
name: track-harness-run
description: Use whenever the user wants to run the diarization model/harness on a dataset (e.g. AMI) and have the result tracked for comparison later — "run the model on X", "score community-1 against Y", "run the harness", "run a test on the corpus". Ensures the run goes through run_harness.py (not an ad hoc script), a manifest with a plain-language description of the experiment (--notes: oracle segmentation, DBSCAN clustering, random forest overlap detection, etc.) is written, and runs/comparison.csv is refreshed afterward so results stay comparable across runs.
---

# Tracking a harness run

This skill exists because it's easy to run the model in a way that produces a DER number but no
record of how it was produced. `harness/run_manifest.py` and `harness/aggregate_runs.py` already
solve that — this skill just makes sure they're actually used every time, instead of being
bypassed by a shortcut script or a bare pipeline call.

## The failure mode this prevents

Test runs against AMI were done before without confirming a `runs/*.json` manifest came out the
other end. Once that happens, the run's DER number floats free of the config that produced it —
no record of which checkpoint, split, condition, or DER settings were used — and it can't be
placed in `runs/comparison.csv` next to other runs. A number with no manifest is not a trackable
result, no matter how correct it looks.

## Workflow

### 1. Confirm the invocation goes through `run_harness.py`

Never call the pipeline directly, and never write a new one-off script that duplicates what
`run_harness.py` already does. The only supported entry point is:

```
python run_harness.py --data-root <path-to-harness-data> --condition IHM [--split test] [--device cuda] --notes "<what this run actually is>"
```

- `--data-root` must point at the *converted* harness-ready layout (see
  `Obsidian-Diarisation/Docs/AMI Corpus Setup.md`) — `{data_root}/{condition}/{split}.rttm`,
  `.uem`, and `audio/{uri}.wav` — not the raw AMI-diarization-setup checkout.
- Do not pass `--runs-dir` unless the user explicitly wants manifests written somewhere other
  than the default `runs/` directory. Leaving it out is what makes the manifest happen — it is
  not optional plumbing.
- Windows/Git Bash note: use `$HOME` (bash) or `"$HOME\..."`/full paths, not `~`, if the shell
  doesn't expand it — a literal `~/...` passed to `--data-root` fails with `ConfigError` before
  the model even loads.

### 1a. Always fill in `--notes` with what this run actually is

`clustering_model` and `segmentation_source_id` in the manifest are still mostly hardcoded
placeholders in this codebase (see `run_harness.py`'s `main()`) — they do not yet reliably
describe what a given experiment used. `--notes` is the one field that captures that in plain
language, and it's the difference between a row in `runs/comparison.csv` being interpretable
months later versus just being a number next to a timestamp. Always ask the user (or state
plainly, if it's obvious from context) what the run actually was, and pass it verbatim, e.g.:

- `--notes "oracle segmentation experiment"`
- `--notes "overlap detection done with a random forest model"`
- `--notes "DBSCAN used as the clustering step instead of the pipeline default"`
- `--notes "baseline community-1, no modifications"` — even an unmodified baseline run should
  say so explicitly, rather than leaving notes blank and forcing a reader to infer "nothing
  unusual" from an absence.

Do not invent or guess a notes string if you don't actually know what was run differently —
ask the user rather than leaving it blank or fabricating a plausible-sounding description.

### 2. Run it and watch for the manifest, not just the DER number

The command prints a `summary` dict at the end (der, overlap_der, jer, ...) — that's necessary
but not sufficient. After it finishes, check that a new file landed in `runs/`:

```
ls runs/
```

If `runs/` doesn't contain a new `<timestamp>-<hash>.json`, the run did not get tracked — stop and
find out why (wrong working directory, `--runs-dir` overridden, or a duplicate `run_id` collision
raising `RunManifestError`) before telling the user the run is done.

### 3. Refresh the comparison table

Every tracked run should be reflected in one place the user can open and compare. After confirming
the manifest exists, regenerate `runs/comparison.csv`:

```
python -c "from harness.aggregate_runs import aggregate_runs; print(aggregate_runs(runs_dir='runs', output_csv_path='runs/comparison.csv'))"
```

This is cheap and idempotent — always safe to rerun, even if nothing changed. Do this after every
tracked run, not just the first one.

### 4. Report the result against prior runs, not in isolation

When telling the user the outcome, don't just report the new DER — open `runs/comparison.csv` (or
read the new row) and say how it compares to the most recent prior run with the same `condition`/
`split`, if one exists. Quote the `notes` values of the runs being compared, not just their DER
numbers — "17.05% vs the baseline run's 17.0%" is a much weaker comparison than "17.05% (DBSCAN
clustering) vs the baseline run's 17.0% (unmodified community-1 defaults)." A single row's number
is much less useful than a number tied to what produced it.

### 5. Flag manifest gaps, don't silently work around them

The manifest currently records `pipeline_config_id`, `segmentation_source_id`, `clustering_model`,
`split`, `condition`, and the DER settings — but not the resolved model revision/commit hash
(`Pipeline.from_pretrained` tracks the latest version of the HF repo at run time) or library
versions. If the user is relying on the manifest to distinguish runs precisely and this gap
matters for what they're doing, say so explicitly rather than implying the manifest is a complete
fingerprint of the run.

## Anti-patterns to catch yourself doing

- Running the pipeline through a scratch script "just to check a number" and reporting that
  number as if it were a tracked result.
- Treating the printed `summary` dict as proof the run was tracked — it prints regardless of
  whether `runs_dir` is set or whether the manifest write actually succeeded.
- Skipping the `runs/comparison.csv` refresh because "it's just one more run" — the table is only
  useful if it's kept current every time, not reconstructed later from memory.
- Reporting a DER number without checking it against the last comparable run.
- Leaving `--notes` blank because the run "seemed obvious" — six months from now nothing about a
  row in `runs/comparison.csv` is obvious except what's written in its `notes` column.
- Guessing or fabricating a notes description instead of asking the user what the run actually
  changed.
