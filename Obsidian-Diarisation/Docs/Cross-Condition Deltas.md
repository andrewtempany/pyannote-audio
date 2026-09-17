---
status: done
created: 2026-09-13
---

# Cross-Condition Deltas

> Part of the [[Evaluation Harness]]. Extends [[Run Manifest]] and [[Oracle Ceiling Metrics]]. **All code here is project-contributed** (`harness/aggregate_runs.py`, `harness/reporter.py`) — no changes to upstream `pyannote-audio`.

## Why this exists

The oracle/ceiling-analysis batch produces five separate tracked runs — baseline, nearest-centroid, both oracle-assignment scopes, and oracle-segmentation — each landing as its own row in `runs/comparison.csv`. The batch's actual finding (how much DER is recoverable downstream of the network, and where) is a set of *deltas between rows*, not any single row. Before this, that comparison had to be read off the raw CSV by eye and computed by hand — this page documents the function that computes it instead.

## `build_cross_condition_table()`

```python
from harness.aggregate_runs import build_cross_condition_table

table = build_cross_condition_table(comparison_csv_path)
```

Reads an already-aggregated `comparison.csv` (see [[Run Manifest]]'s `aggregate_runs()`), filters to rows where `counts_toward_results` is `true` (exploratory/debugging runs are excluded by design — the table always reflects the deliberately-tracked batch, never everything ever run), and renders a Markdown table plus the budget deltas below it. Returns a plain string, ready to paste into a doc, PR description, or ticket without reformatting.

CLI: `python -m harness.aggregate_runs --cross-condition-table` refreshes `comparison.csv` and prints the table to stdout in one step.

### Table columns

`Condition`, `Oracle scope`, `DER`, `DER overlap (system)`, `DER overlap (assigned)`, `Missed detection`, `False alarm`, `Confusion` — one row per tracked run, values formatted to 4 decimal places (or `--` for a blank cell).

### Grouping: `oracle_scope` is shown only where it means something

`oracle_assignment` runs come in two scopes — `all_pairs` and `overlap_degraded` — that measure genuinely different ceilings (see [[Oracle Assignment Strategy]]) and are always reported as separate rows, never averaged together.

`run_harness.py`'s `--oracle-scope` flag defaults to `all_pairs` and gets written into *every* manifest regardless of condition, which meant the table initially rendered `oracle_segmentation | all_pairs` — implying a scope choice nobody actually made for a condition where scope is meaningless. Fixed with `_effective_scope()`, which blanks the scope column for any condition other than `oracle_assignment`.

### Budget deltas

```python
def _build_delta_lines(rows): ...
```

Computes, for each budget-relevant condition present in the filtered rows:

- **Downstream budget** = `baseline_der - oracle_segmentation_der` — how much DER is fixable by better segmentation.
- **Assignment budget** = `baseline_der - oracle_assignment_der`, reported **separately for each `oracle_scope`** present.

**Sign convention: `baseline - oracle`, so a positive value means the oracle condition lowered DER** — reads as "how much DER is recoverable" in plain English, with no double-negative to trip over. A negative value honestly reports the condition made things worse, rather than being clipped to zero or hidden.

If no baseline row is present in the filtered set, the table says so explicitly (`_No baseline run recorded -- budget deltas cannot be computed._`) rather than silently omitting the deltas section.

## Prerequisite: corpus-wide DER component totals in the summary

Before this table could show a meaningful component breakdown per condition, `write_report()`'s corpus-level `summary` dict (in [[Harness Reporter]] / [[Oracle Ceiling Metrics]]) needed `missed_detection`/`false_alarm`/`confusion` totals — it previously only wrote these per-file, not corpus-wide.

**These are read from the `der` accumulator's `accumulated_` dict** (`der.accumulated_.get("missed detection", 0.0)` etc, via a `_DER_COMPONENT_KEYS` name-mapping table), never by summing the per-file CSV rows — the same accumulator-not-average rule [[Scorer]] and [[Harness Reporter]] already establish for `der`/`der_overlap_system`/`jer`. Durations are additive, so summing rows would likely produce the same number here, but computing it via a fresh sum would violate the "always read from the accumulator" invariant the rest of the reporter is built around for no benefit. `.get(key, 0.0)` (not `[key]`) tolerates an accumulator that has scored zero files, rather than raising.

No schema change was needed in `harness/run_manifest.py` or `harness/aggregate_runs.py` for this — `summary` is passed through as-is, and `_flatten_manifest` already unions whatever keys `summary` contains into `comparison.csv`'s columns. Verified end-to-end in a temp dir: a summary carrying the three new fields flows through `write_manifest` → `aggregate_runs` → `build_cross_condition_table` with zero code changes to either of those two files.

**Known limitation:** the five runs that were already tracked before this landed were scored before the component totals existed in the summary, so their `Missed detection`/`False alarm`/`Confusion` cells render as `--` in the table — not missing or broken, just recorded before the field existed. Re-running a condition populates it; nothing else in the table (DER, the deltas) is affected.

## Results on real data (AMI IHM test split, 16 meetings)

```
| Condition            | Oracle scope     | DER    | DER overlap (system) | DER overlap (assigned) |
| baseline             | --               | 0.1705 | 0.3339                | 0.2195                 |
| nearest_centroid     | --               | 0.1705 | 0.3339                | 0.2195                 |
| oracle_assignment    | all_pairs        | 0.1608 | 0.3137                | 0.1920                 |
| oracle_assignment    | overlap_degraded | 0.1658 | 0.3189                | 0.1993                 |
| oracle_segmentation  | --               | 0.1705 | 0.3339                | 0.2195                 |
```

**Budget deltas** (`baseline - oracle`; positive = oracle lowered DER):

- Downstream budget (baseline − oracle segmentation): **+0.0000** — essentially zero. Almost none of the error is fixable by better segmentation; whatever headroom exists is downstream of it.
- Assignment budget (baseline − oracle assignment) `[overlap_degraded]`: **+0.0047** (17.05% → 16.58%) — the primary, honest ceiling for a realistic post-clustering fix: this scope restricts the oracle to exactly the population (overlap-degraded pairs) a real classifier could target.
- Assignment budget (baseline − oracle assignment) `[all_pairs]`: **+0.0097** (17.05% → 16.08%) — a looser secondary ceiling that also lets the oracle correct pairs that were never overlap-degraded to begin with; useful as context, not the number to hold a real intervention to.
- `nearest_centroid` matching baseline exactly (not just approximately) is expected, not a bug: clustering already assigns each pair to its own nearest centroid by construction, so a pure nearest-centroid pass on top of already-converged clustering is close to a no-op. See [[Post-Clustering Refinement Hook]].

**These numbers are data, not a decision.** Section 8 of the original oracle/ceiling-analysis spec — the pre-registered go/no-go threshold and the decision record measuring these deltas against it — is explicitly out of scope for this table and for any agent: that call belongs to the project owner, made transparently, not derived here.

## Non-goals

- No commentary or conclusions about what the deltas mean for the go/no-go decision — the table ends at the numbers.
- No new metrics beyond corpus-wide component totals — those already existed per-file; this only surfaced their corpus-level accumulator totals.
- No persisted/saved table file by default — `build_cross_condition_table()` returns a string; saving it anywhere is left to the caller.

## Key files

- `harness/aggregate_runs.py` — `build_cross_condition_table`, `_effective_scope`, `_build_delta_lines`, `_render_markdown_table`, `main` (`--cross-condition-table` CLI flag)
- `harness/reporter.py` — `_DER_COMPONENT_KEYS`, corpus-summary block
- `tests/test_reporter.py` — 2 new tests (corpus-wide component totals, wrong-accumulator-wiring guard)
- `tests/test_cross_condition_table.py` — 11 tests covering grouping, scope-blanking, delta signs, missing-baseline handling, and Markdown rendering
