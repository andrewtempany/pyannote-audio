---
status: in-progress
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

- **Ticket has no "Tests first" section** (unlike some others in this batch). Proceeding by deriving tests from the Method/Config/Acceptance-criteria sections directly rather than stopping to ask, since this is a background/unattended run — documenting that choice here per the tdd-ticket skill's guidance to flag any deviation.
- **Signature problem discovered while reading T2's hook.** The fixed refinement interface is `strategy(embeddings, hard_clusters, soft_clusters, centroids, segmentations) -> hard_clusters` — there is no reference/ground-truth parameter, and no per-file hook at all: `run_harness.py` currently does `pipeline.refinement = get_refinement_strategy(name)` **once**, before the per-file loop. Oracle assignment fundamentally needs the reference `Annotation` for the file currently being scored, which isn't available at that point.
  - Rejected: smuggling reference into `segmentations` or a global/module-level variable — fragile, breaks the "operates on pairs and what clustering produced" contract T2 documents.
  - Chosen: `harness/refinement.py` exposes a **factory**, `make_oracle_strategy(reference, oracle_scope) -> Callable`, matching the normal 5-arg interface via closure. `run_harness.py`'s per-file loop, only when `refinement_strategy == "oracle"`, sets `pipeline.refinement = make_oracle_strategy(reference, oracle_scope)` immediately before `runner.run(file)` (identity/nearest_centroid runs are unaffected — `pipeline.refinement` is still set once, outside the loop, for those). This is a small, additive change to `run_harness.py`'s loop body, not a change to the T2 interface itself or to `speaker_diarization.py`.
- **Optimal hyp-cluster <-> reference-speaker mapping**: reused `pyannote.metrics.diarization.DiarizationErrorRate().optimal_mapping(reference, hypothesis, uem=None)` (returns `dict[hyp_label -> ref_label]`, computed by total temporal overlap via Hungarian matching) rather than hand-rolling an overlap matrix + assignment. Requires materializing `hard_clusters`+`segmentations` into a hypothesis `Annotation` first (helper `_reconstruct_hypothesis`), since that's what `optimal_mapping` takes.
- **Pair support**: helper `_pair_support(hard_clusters, segmentations)` builds a `Timeline` per `(chunk, local_speaker)` pair from that pair's active frames in `segmentations.data[chunk, :, local_speaker]`, mapped to real time via `segmentations.sliding_window[chunk]`.
- **`overlap_degraded` scope semantics** (not spelled out precisely in the ticket beyond "support is predominantly coincident with another speaker"): a pair is in-scope when, within its own temporal support, the reference annotation shows >=2 simultaneous speakers for more than half of that support's duration (i.e. the pair's audio is mostly overlapping speech in the ground truth). This directly matches "support predominantly coincident with another speaker" and is checkable purely from `reference.get_overlap()` cropped to the pair's support.
- **Never invents a missing cluster**: reverse mapping is built as `ref_speaker -> hyp_cluster` from `optimal_mapping`'s `hyp_cluster -> ref_speaker` dict. If the dominant reference speaker for a pair's support isn't a value in that dict (i.e. no hypothesis cluster maps to it), the pair is left unchanged (method step 5). Also: mapping only ever assigns labels that appeared in `hard_clusters` to begin with, since `optimal_mapping`'s hypothesis side is built only from clusters actually present.
