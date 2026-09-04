---
status: done
created: 2026-09-04
---

# Reporter

> Part of the [[Evaluation Harness]]. **`harness/reporter.py` is code contributed by this project — not upstream `pyannote-audio`.** It has no direct pyannote-audio dependency itself: it only consumes the shape of dicts/objects produced by [[Scorer]] (a project-contributed module) and standard-library `csv`/`json`.

## Purpose

Turn a run's per-file scoring rows, plus the run's three corpus-level metric accumulator instances, into two output artifacts: a per-file CSV and a corpus summary JSON.

## Where it fits

Called exactly once, at the end of a run, by the [[Orchestrator]] — after every file has been scored via [[Scorer]].

## API

```python
from harness.reporter import write_report

summary = write_report(rows, der, overlap_der, jer, per_file_csv_path, summary_path)
```

- `rows` — a list of dicts matching [[Scorer]]'s `score()` output shape, each with `uri` added by the caller.
- `der`, `overlap_der`, `jer` — the same accumulator instances [[Scorer]] was called with throughout the run; this component reads their final totals (`abs(metric)`), it does **not** recompute them.
- Returns the summary dict as well as writing it — a convenience beyond the original ask, since the caller (the [[Orchestrator]]) needs the corpus numbers for its own end-of-run reporting anyway, and re-parsing the JSON it just wrote would be redundant.

## Output files

- **Per-file CSV** — via `csv.DictWriter`, fixed column order: `uri, der, overlap_der, jer, count_ref, count_hyp, count_error` (the module constant `FIELDNAMES`).
- **Corpus summary** — JSON (the design allowed CSV or JSON; JSON was chosen because it round-trips exact float values and typed fields without the CSV string-parsing ambiguity the per-file CSV already has to live with). Contains `der`, `overlap_der`, `jer` (all `abs(metric)` on the caller-supplied accumulators), `counting_mae`, and `counting_exact_match_percent`.

## Why corpus totals come from the accumulators, not from averaging rows

Corpus DER is a ratio over the whole corpus (total errors / total reference duration), not a mean of per-file ratios — averaging per-file DER values would be a different (wrong) number. This is enforced by construction: `der`/`overlap_der`/`jer` in the summary come from `abs(metric)` on the caller-owned accumulator objects that were shared across every [[Scorer]] call in the run, never recomputed from `rows`. The test suite deliberately uses accumulator totals that *disagree* with what averaging the rows would produce, so a future edit that accidentally started recomputing from `rows` would fail loudly rather than silently drifting.

## Counting summary

- `counting_mae` — mean of each row's `count_error`.
- `counting_exact_match_percent` — percentage of rows with `count_error == 0`.

## Error handling

Calling `write_report()` with **zero rows** raises `ReporterError` (a `ValueError` subclass) *before* either output file is created or written — confirmed neither `per_file_csv_path` nor `summary_path` exists after a `ReporterError`. A failed report never leaves a partial or misleading file behind.

## Non-goals

- No scoring math — consumes [[Scorer]]'s output shape as-is.
- No pipeline/model/dataset involvement — inputs are rows and metric-total objects only.
