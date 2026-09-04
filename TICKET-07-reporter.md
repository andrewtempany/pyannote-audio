# 07 — Reporter

Part of the [eval harness epic](TICKET-eval-harness.md). Depends on: [05](TICKET-05-scorer.md).

## Scope

`harness/reporter.py` — collect per-file scorer output rows into a table and write:
- a per-file CSV: `uri, der, overlap_der, jer, count_ref, count_hyp, count_error`,
- a corpus summary (CSV or JSON): accumulated DER, accumulated overlap-DER, accumulated JER (read from the same `der`/`overlap_der`/`jer` accumulator instances ticket 05 requires the caller to own — the reporter reads their final totals, e.g. via `abs(metric)`, it does not recompute them), counting MAE, counting exact-match %.

## Non-goals

- No scoring math (consumes ticket 05's output shape as-is).
- No pipeline/model/dataset involvement — this component takes rows and metric-total inputs only.

## Tests first

Add `tests/test_reporter.py`:

1. `test_per_file_csv_has_expected_columns` — fabricate a few rows (dicts matching ticket 05's `score()` output shape), write, read the CSV back, assert exact column order/names: `uri, der, overlap_der, jer, count_ref, count_hyp, count_error`.
2. `test_per_file_csv_values_match_input_rows` — assert values round-trip correctly (including float formatting doesn't lose meaningful precision).
3. `test_corpus_summary_includes_accumulated_der_overlap_der_jer` — pass in fake metric objects whose `abs(...)`/total-reading method returns known fixed values, assert the summary file contains exactly those values (not recomputed from the per-file rows — this is the direct test that the reporter respects the "accumulate via one instance" contract rather than re-deriving corpus DER by averaging rows, which the epic explicitly forbids).
4. `test_counting_mae_computed_correctly` — rows with known `count_error` values, assert MAE matches the hand-computed mean.
5. `test_counting_exact_match_percentage_computed_correctly` — rows with a known number of `count_error == 0` cases, assert the percentage matches.
6. `test_empty_rows_raises_clear_error` — call the reporter with zero rows, assert a clear error rather than a divide-by-zero traceback or a silently empty/garbage summary.
7. `test_output_files_written_to_configured_paths` — assert the CSV and summary land at the paths the reporter was given, not hardcoded elsewhere.

## Acceptance criteria

- All tests pass with fabricated inputs — no model, no audio, no real dataset needed.
- Corpus summary values in the tests are shown to come from the accumulator totals, not from averaging per-file values (test 3 is the load-bearing one here — the epic calls this exact bug out as "easy" to introduce).

## Implementation notes (resolved during build — 2026-09-04)

- `write_report(rows, der, overlap_der, jer, per_file_csv_path, summary_path) -> dict` in `harness/reporter.py`. Per-file CSV via `csv.DictWriter` with a fixed `FIELDNAMES` order; corpus summary as JSON (the ticket allowed CSV or JSON — JSON chosen since it round-trips exact float values and typed fields without the CSV-string-parsing ambiguity the per-file CSV already has to deal with).
- Corpus `der`/`overlap_der`/`jer` values come from `abs(metric)` on the caller-supplied accumulator objects — never recomputed from `rows`. `test_corpus_summary_includes_accumulated_der_overlap_der_jer` deliberately uses accumulator totals that disagree with what averaging the rows would produce, so the test would fail loudly if a future edit accidentally started recomputing from rows.
- Empty `rows` raises `ReporterError` (a `ValueError` subclass) **before** either output file is created or written — confirmed neither `per_file_csv_path` nor `summary_path` exists after a `ReporterError`, so a failed report never leaves a partial/misleading file behind.
- `write_report()` also returns the summary dict (a small convenience beyond the ticket's literal ask — the caller, TICKET-08's orchestrator, needs the corpus numbers for its own end-of-run reporting anyway, and reading them back by re-parsing the JSON it just wrote would be redundant).
