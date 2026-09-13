---
status: in-progress
created: 2026-09-13
---

# T2: Post-clustering refinement hook, plus nearest-centroid validation

Part of the oracle/ceiling analysis batch (see [[T1-harness-reporting-and-manifest-schema]], [[T3-oracle-segmentation-provider]], [[T4-oracle-assignment-strategy]], [[T5-cross-condition-deltas]]).

## Context

The unit of assignment is the `(chunk, local_speaker)` pair, not the frame. `get_embeddings` (`pipelines/speaker_diarization.py:332-478`) pools within each chunk and local speaker before returning; `embeddings` and `hard_clusters` both have shape `(num_chunks, local_num_speakers, dimension)` / `(num_chunks, local_num_speakers)`. Frame-level identity is produced downstream by `reconstruct()` (lines 480-528) from the pair labels. Every refinement strategy in this batch operates on pairs.

## Goal

Create the single extension point every strategy in this batch plugs into, and prove it works.

The seam sits between the clustering call and the inactive-speaker masking in `SpeakerDiarization.apply`:

```python
hard_clusters, soft_clusters, centroids = self.clustering(...)   # line 640
hard_clusters = self.refinement(                                  # new
    embeddings, hard_clusters, soft_clusters, centroids, segmentations
)
hard_clusters[inactive_speakers] = -2                             # line 680
```

Everything downstream, including `reconstruct()`, the relabeling at lines 715-740, and the centroid reordering at lines 763-773, is untouched.

## Scope
- **Recover `soft_clusters`.** Line 640 currently discards it into `_`. It is the per-cluster posterior for every pair, already computed, and the most useful feature a later classifier could have. One character of code.
- Define the refinement strategy interface, taking the five arguments above (`embeddings`, `hard_clusters`, `soft_clusters`, `centroids`, `segmentations`) and returning a `hard_clusters` array of identical shape.
- Default strategy is `identity`, returning its input unchanged.
- Implement `nearest_centroid`: assign each pair to the centroid nearest its embedding by cosine similarity.
- Strategy selectable by harness config, flowing into the `refinement_strategy` manifest field from [[T1-harness-reporting-and-manifest-schema]].

## Acceptance criteria
- [ ] With `identity`, the baseline run reproduces 17.05% **exactly**, not approximately
- [ ] With `nearest_centroid`, the run completes and DER lands within roughly a point of baseline (a large swing means the plumbing is wrong)
- [ ] Both runs land in `comparison.csv` with correct `condition` values

## Implementer notes
- Check whether the harness invokes the `legacy=True` path. If it does, `DiarizeOutput` never leaves `apply` and centroids are unavailable downstream. Verify before assuming.
- `OracleClustering` returns `centroids=None` when no embeddings are supplied. The interface needs to handle a null-centroid case explicitly rather than crashing.

## Out of scope
The oracle strategy — that is [[T4-oracle-assignment-strategy]].

## Dependencies
Can run in parallel with [[T1-harness-reporting-and-manifest-schema]]. Blocks [[T4-oracle-assignment-strategy]].

## Implementation Notes

(append here as work happens — decisions, rejected alternatives, gotchas, key files/functions touched)

- 2026-09-13: Wrote failing unit tests only (TDD red phase) in `tests/test_refinement.py`. No source changes yet — `src/pyannote/audio/pipelines/speaker_diarization.py` and `harness/` are untouched.
- Confirmed real shapes before writing fixtures: `embeddings` (num_chunks, local_num_speakers, dimension), `hard_clusters` (num_chunks, local_num_speakers), `soft_clusters` (num_chunks, local_num_speakers, num_clusters) per `clustering.py`, `centroids` (num_speakers, dimension) or `None` for `OracleClustering`.
- Proposed interface: `harness/refinement.py` with plain functions `identity(embeddings, hard_clusters, soft_clusters, centroids, segmentations)` and `nearest_centroid(...)` (same signature), a `REFINEMENT_STRATEGIES` dict registry, and `get_refinement_strategy(name) -> Callable` that raises `KeyError` on an unknown name. Matches the plain-function house style of `harness/scorer.py` / `harness/reporter.py` — no classes.
- soft_clusters recovery (`hard_clusters, _, centroids = self.clustering(...)` → capture instead of discard) is a one-line change at the `apply()` call site, not something a unit test on `harness/refinement.py` can directly exercise without a real pipeline run. Wrote a unit-level test instead asserting the strategy interface has a genuine (non-vestigial) `soft_clusters` parameter that accepts the real posterior shape — documented this tradeoff inline in the test. True end-to-end confirmation that line 640 actually captures it is deferred to the implementation step / an integration test, out of scope for this pass.
- `nearest_centroid(centroids=None, ...)` open decision (flagged in ticket's implementer notes): chose no-op fallback to identity rather than raising, so a config-selected `nearest_centroid` run doesn't hard-crash when it happens to hit an `OracleClustering` condition. Documented as a choice in the test, not inferred — a raising `ValueError` would be equally defensible and could be swapped later without ticket scope creep.
- Config selection tested narrowly against `get_refinement_strategy`/`REFINEMENT_STRATEGIES` directly rather than wiring into `harness/config.py`'s `HarnessConfig` dataclass — wiring the config field itself is implementation, not this pass's scope.
- First run of `pytest tests/test_refinement.py -v` fails at collection with `ModuleNotFoundError: No module named 'harness.refinement'`, as expected — confirms the red state is real (module doesn't exist yet), not a scaffolding bug.
