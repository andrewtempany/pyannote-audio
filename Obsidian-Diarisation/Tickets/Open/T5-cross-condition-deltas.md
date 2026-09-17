---
status: in-progress
created: 2026-09-13
---

# T5: Cross-condition deltas in aggregate_runs.py

Part of the oracle/ceiling analysis batch (see [[T1-harness-reporting-and-manifest-schema]], [[T2-post-clustering-refinement-hook]], [[T3-oracle-segmentation-provider]], [[T4-oracle-assignment-strategy]]). Final ticket in the batch — needs all four conditions scored (baseline, oracle segmentation, oracle assignment, nearest centroid).

## Goal

One table, computed rather than transcribed.

## Scope

**Step 0 (new, do first): add corpus-wide DER component breakdown to the summary.**
`comparison.csv` does not currently carry corpus-wide missed detection / false alarm /
confusion -- this must land before the rest of T5 is possible, not be computed ad hoc
inside T5 itself.

- **Correction from an earlier pass of this ticket:** `harness/scorer.py` already computes
  `missed_detection` / `false_alarm` / `confusion` and writes them to the *per-file* CSV
  (`FIELDNAMES` in `harness/reporter.py` already lists all three) -- no new field needs
  adding there. The actual gap is narrower: `write_report()`'s corpus-level `summary` dict
  (`harness/reporter.py`, around the `summary = {...}` block) never surfaces a corpus-wide
  total for these three.
- **Do not sum the per-file CSV rows to get the corpus total.** Confirmed by reading
  `harness/scorer.py`'s `score()`: each file's `der(reference, hypothesis, uem=uem,
  detailed=True)` call both accumulates into the shared `der` object's running total *and*
  returns that single call's own component breakdown as the per-file row. This is the same
  per-file-ratio-vs-accumulator-total distinction the reporter's docstring already warns
  about for `der`/`overlap_der`/`jer` (see [[Scorer]], [[Harness Reporter]]) -- durations
  are additive so summing rows would likely give the same number as the accumulator here,
  but doing it via a fresh sum breaks the "always read from the accumulator, never
  recompute" invariant the rest of the reporter is built around, and there's no need to
  duplicate that work. Get the corpus totals from the same shared `der` accumulator object
  already passed into `write_report()`, via `der.accumulated_["missed detection"]`,
  `der.accumulated_["false alarm"]`, `der.accumulated_["confusion"]` (confirmed available:
  `DiarizationErrorRate.metric_components()` -> `['total', 'correct', 'false alarm',
  'missed detection', 'confusion']`).
- `harness/run_manifest.py` / `harness/aggregate_runs.py` need no schema change --
  `summary` is passed through as-is and `_flatten_manifest` already unions whatever keys
  `summary` contains into the CSV row. Confirm this by running an update and checking the
  three new columns appear in `comparison.csv` automatically.
- Note: existing runs already in `comparison.csv` were recorded before this field existed
  -- they will show blank cells for these three columns, same as `duration_seconds` does
  for pre-fix runs. Re-running those four tracked conditions is optional, not required,
  to unblock the rest of T5 -- but the cross-condition table can't show component
  breakdown for a condition until it's been (re)run after this step lands.

**Then, the original scope:**
- Filter `comparison.csv` rows on `counts_toward_results`
- Group by `condition` (and `oracle_scope`, where present -- see resolved question below)
- Emit a table of DER, `der_overlap_system`, `der_overlap_assigned`, and the component breakdown (missed detection, false alarm, speaker confusion) per condition
- Compute the two key deltas:
  - baseline minus oracle segmentation (downstream budget)
  - baseline minus oracle assignment (assignment budget)

## Resolved decisions

- **Entry point and output format.** A new function `build_cross_condition_table(runs_dir,
  ...) -> str` in `harness/aggregate_runs.py`, alongside the existing `aggregate_runs()` --
  it reads the already-refreshed `comparison.csv` (or takes `runs_dir` and calls
  `aggregate_runs()` itself first), and returns a Markdown table as a string. Add a small
  CLI entry point (e.g. `python -m harness.aggregate_runs --cross-condition-table`, or a
  new tiny script alongside `run_experiment.sh`) that calls it and prints the result to
  stdout -- easy to paste directly into a doc, PR description, or ticket without
  reformatting. No new file is written by default; if a saved copy is wanted later, that's
  a follow-on, not part of this ticket.

- **Which `oracle_assignment` row feeds the "assignment budget" delta: report both.**
  Two scopes are tracked with different DER: `all_pairs` (der=0.1608, T4's fuller
  secondary/context ceiling) and `overlap_degraded` (der=0.1658, T4's primary/targeted
  scope) -- see `runs/comparison.csv`. Rather than picking one, T5 computes and displays
  the assignment-budget delta for both scopes side by side, clearly labeled by
  `oracle_scope`. Confirmed on real data (both deltas positive, meaning oracle helped in
  both cases): `baseline - oracle_assignment(all_pairs)` = 0.1705 - 0.1608 = +0.00969;
  `baseline - oracle_assignment(overlap_degraded)` = 0.1705 - 0.1658 = +0.00470.
- **Sign convention: `baseline - oracle`, positive means oracle improved (lowered) DER.**
  This reads as "how much DER is recoverable" in plain English with no double-negative.
  Confirmed on real data: `baseline - oracle_segmentation` = 0.17049287 - 0.17048544 =
  +0.0000074 (near-zero, correctly reads as "oracle segmentation barely helped, most
  error isn't in the segmentation stage"); both oracle_assignment deltas are also
  positive (values above) -- both ceilings correctly show oracle helping, never
  reporting a spurious negative "oracle made it worse" for these runs.

## Acceptance criteria
- [x] `write_report()`'s summary includes corpus-wide missed detection / false alarm / confusion, sourced from the `der` accumulator's `accumulated_` totals, never recomputed by summing per-file rows
- [x] The three new columns appear in `comparison.csv` automatically after a run, with no `aggregate_runs.py` changes needed -- verified end to end in a temp dir
- [x] The table regenerates from `comparison.csv` with no manual entry
- [x] Both `oracle_assignment` scopes (`all_pairs` and `overlap_degraded`) appear as separate, clearly labeled rows/deltas -- neither is silently dropped or averaged together
- [x] Deltas are computed as `baseline - oracle`, documented inline (docstring/comment) so a positive value unambiguously means oracle improved DER
- [x] `build_cross_condition_table()` exists in `harness/aggregate_runs.py`, is callable from a CLI entry point, and prints a Markdown table to stdout

## Out of scope
Commentary or conclusions. This ticket ends at the table; interpretation and the go/no-go decision are written by Andrew, not the agent.

## Dependencies
Needs [[T1-harness-reporting-and-manifest-schema]] (fields to group/filter on), [[T3-oracle-segmentation-provider]], and [[T4-oracle-assignment-strategy]] all scored.

## Implementation Notes

- 2026-09-17: **Step 0 landed.** `harness/reporter.py` now adds corpus-wide
  `missed_detection` / `false_alarm` / `confusion` to `write_report()`'s summary, read from
  `der.accumulated_` via a new `_DER_COMPONENT_KEYS` mapping (summary snake_case name ->
  pyannote's space-separated key, e.g. `missed_detection` -> `"missed detection"`).
  Verified against a real `DiarizationErrorRate`: after scoring, `accumulated_` holds
  `{'false alarm', 'correct', 'missed detection', 'confusion', 'total'}` and the component
  totals reconcile with `abs(der)` (miss 2.0 + confusion 4.0 over total 10.0 = 0.6 DER).
- **Gotcha: use `.get(key, 0.0)`, not `[key]`.** A direct lookup broke 9 pre-existing
  reporter tests whose `_FakeMetric` has an empty `accumulated_`. Rather than rewrite those
  tests, the reporter tolerates a metric object with no components yet -- which is also the
  honest behaviour for an accumulator that has scored zero files.
- Tests deliberately set accumulator component totals that **disagree** with what summing
  the per-file rows would give (11.5/22.5/33.5 vs rows summing to 3.0 each), so a future
  edit that started summing rows fails loudly. A second test gives each of the four metric
  objects distinct component totals, catching a wrong-accumulator wiring bug.
- **Step 1 landed.** `build_cross_condition_table(comparison_csv_path) -> str` in
  `harness/aggregate_runs.py`, plus a `main()` CLI: `python -m harness.aggregate_runs
  --cross-condition-table` refreshes the CSV then prints the Markdown table.
- **Gotcha found only by running it on real data: `oracle_scope` leaks a meaningless
  default.** `run_harness.py`'s `--oracle-scope` defaults to `all_pairs` and is written to
  *every* manifest, so the table initially rendered `oracle_segmentation | all_pairs`,
  implying a scope choice nobody made. Fixed with `_effective_scope()`, which blanks the
  scope for any condition other than `oracle_assignment`. Regression test:
  `test_oracle_scope_ignored_for_conditions_that_do_not_use_it`.
- Verified end-to-end in a temp dir that a summary carrying the three new component fields
  flows through `write_manifest` -> `aggregate_runs` -> `build_cross_condition_table` with
  **no schema change to `run_manifest.py` or `aggregate_runs.py`**, confirming the ticket's
  prediction that `_flatten_manifest` unions whatever `summary` contains.
- **Known limitation:** all five runs currently in `runs/comparison.csv` were scored before
  Step 0, so their component columns render as `--`. The numbers are not missing or broken;
  those conditions simply need re-running to populate them. Deltas and all other columns
  are unaffected.
- Key files: `harness/reporter.py` (`_DER_COMPONENT_KEYS`, summary block),
  `harness/aggregate_runs.py` (`build_cross_condition_table`, `_effective_scope`,
  `_build_delta_lines`, `_render_markdown_table`, `main`),
  `tests/test_reporter.py` (2 new tests, `_FakeMetric` grew `accumulated_`),
  `tests/test_cross_condition_table.py` (new, 11 tests).
- Test results: 122 passed, 1 skipped (the pre-existing HF-token-gated integration test)
  across the harness suite -- no regressions.
