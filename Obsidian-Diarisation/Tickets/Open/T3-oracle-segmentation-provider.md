---
status: not-started
created: 2026-09-13
---

# T3: Oracle segmentation provider

Part of the oracle/ceiling analysis batch (see [[T1-harness-reporting-and-manifest-schema]], [[T2-post-clustering-refinement-hook]], [[T4-oracle-assignment-strategy]], [[T5-cross-condition-deltas]]).

## Goal

Supply ground-truth speech and overlap regions through the harness's existing injectable segmentation interface (see [[Segmentation Injection Seam]] and [[Segmentation Source Interface]]), so a run can be scored with the network's segmentation stage replaced by ground truth.

## Scope
- Build segmentation from the `only_words` RTTMs in the shape the pipeline expects
- Conform to the same interface contract as the model-based source, so the cache key picks up a distinct `segmentation_source.id` automatically (see `harness/runner.py:35-40`)
- Timeboxed first step: assess whether `pipelines/utils/oracle.py` can be used directly before writing anything custom

## Acceptance criteria
- [ ] Provider passes the same interface contract as the model source
- [ ] A fixture file produces segmentation matching its RTTM
- [ ] Run completes, scores, and appears in `comparison.csv` as `oracle_segmentation`

## Dependencies
Needs the manifest fields from [[T1-harness-reporting-and-manifest-schema]] (`condition`, `segmentation_source_id`). Can start once T1 lands.

## Implementation Notes

(append here as work happens — decisions, rejected alternatives, gotchas, key files/functions touched)
