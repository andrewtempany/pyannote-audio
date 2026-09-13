---
status: not-started
created: 2026-09-13
---

# T4: Oracle assignment strategy

Part of the oracle/ceiling analysis batch (see [[T1-harness-reporting-and-manifest-schema]], [[T2-post-clustering-refinement-hook]], [[T3-oracle-segmentation-provider]], [[T5-cross-condition-deltas]]). This is the decision-relevant ticket in the batch — baseline minus this result is the budget the intervention competes for.

## Goal

Measure the ceiling on post-clustering assignment, holding the real segmentation and real clusters fixed.

The strategy assigns each pair the cluster label that best matches ground truth. **It may only choose among clusters the pipeline actually produced.** It must not invent speakers the pipeline missed. That constraint is what makes the number an honest ceiling for an assignment intervention rather than a ceiling for the whole pipeline.

## Method
1. Compute an optimal mapping between hypothesis clusters and reference speakers, by total temporal overlap across the file
2. For each pair, determine its temporal support from the segmentation
3. Find the reference speaker with greatest overlap over that support
4. Assign the cluster that maps to that reference speaker
5. If the dominant reference speaker maps to no cluster, leave the pair unchanged

## Config

`oracle_scope`, with two values:
- `overlap_degraded` (**primary**): refine only pairs whose support is predominantly coincident with another speaker. Isolates the ceiling to the population the intervention targets.
- `all_pairs` (secondary): refine every pair. Gives the full post-clustering ceiling, useful as context.

Both are cheap once the strategy exists. Run both.

## Acceptance criteria
- [ ] Strategy implements the [[T2-post-clustering-refinement-hook]] interface
- [ ] On a fixture, `der_overlap_assigned` drops substantially against baseline
- [ ] Both `oracle_scope` values run and appear in `comparison.csv`
- [ ] The strategy never emits a cluster label absent from its input

## Dependencies
Blocked on [[T2-post-clustering-refinement-hook]]. Blocks [[T5-cross-condition-deltas]] (needs all conditions scored).

## Implementation Notes

(append here as work happens — decisions, rejected alternatives, gotchas, key files/functions touched)
