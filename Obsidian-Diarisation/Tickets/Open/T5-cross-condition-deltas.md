---
status: not-started
created: 2026-09-13
---

# T5: Cross-condition deltas in aggregate_runs.py

Part of the oracle/ceiling analysis batch (see [[T1-harness-reporting-and-manifest-schema]], [[T2-post-clustering-refinement-hook]], [[T3-oracle-segmentation-provider]], [[T4-oracle-assignment-strategy]]). Final ticket in the batch — needs all four conditions scored (baseline, oracle segmentation, oracle assignment, nearest centroid).

## Goal

One table, computed rather than transcribed.

## Scope
- Filter `comparison.csv` rows on `counts_toward_results`
- Group by `condition`
- Emit a table of DER, `der_overlap_system`, `der_overlap_assigned`, and the component breakdown (missed detection, false alarm, speaker confusion) per condition
- Compute the two key deltas:
  - baseline minus oracle segmentation (downstream budget)
  - baseline minus oracle assignment (assignment budget)

## Acceptance criteria
- [ ] The table regenerates from `comparison.csv` with no manual entry

## Out of scope
Commentary or conclusions. This ticket ends at the table; interpretation and the go/no-go decision are written by Andrew, not the agent.

## Dependencies
Needs [[T1-harness-reporting-and-manifest-schema]] (fields to group/filter on), [[T3-oracle-segmentation-provider]], and [[T4-oracle-assignment-strategy]] all scored.

## Implementation Notes

(append here as work happens — decisions, rejected alternatives, gotchas, key files/functions touched)
