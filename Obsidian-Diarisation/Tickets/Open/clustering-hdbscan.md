---
status: not-started
created: 2026-09-25
---

# Clustering method: HDBSCAN (exploratory)

## Goal

Add HDBSCAN as a `BaseClustering` subclass and evaluate it, including what its noise-point
behaviour does to diarisation.

Sub-ticket of [[clustering-method-comparison]]. **P3, exploratory. Expected to underperform.**

## Why include a method expected to lose

Two reasons, and the second is the real one.

**It is the only density-based method worth trying.** Plain DBSCAN is not: its `eps` parameter
is exactly as sensitive as the agglomerative threshold, so it trades one hyperparameter for
another with no structural gain. HDBSCAN removes the global `eps` by building a hierarchy over
varying density, which is the only version of the idea that adds anything.

**Noise points are the interesting part.** Density methods label low-density points as **noise,
cluster −1**, rather than forcing every point into a cluster. Every other method in this epic
assigns every embedding somewhere, whether or not the embedding is any good.

That connects directly to this project's original framing. Overlap-degraded embeddings are
blended voices that sit between centroids — low density, by construction. A method that can
decline to assign them is doing something none of the others can.

And the harness already handles it: `hard_clusters < 0` pairs are skipped in the refinement
loop ([refinement.py:110-111, 221-222](../../../harness/refinement.py)) and masked downstream.
Noise points flow through with no special handling needed.

So the question is concrete: **do HDBSCAN's noise points coincide with the overlap-degraded
pairs, and does declining to assign them help or hurt DER?** Either answer is a paragraph.

## Implementation

Subclass `BaseClustering`, implement `cluster(embeddings, min_clusters, max_clusters,
num_clusters)`. `hdbscan` is a separate package on older setups; `sklearn.cluster.HDBSCAN`
exists in recent scikit-learn. **Check which is available before adding a dependency** — the
container's network is allowlisted and an unavailable package is a hard stop, not a detour.

Three interface problems to solve explicitly, not paper over:

- **`num_clusters` cannot be honoured.** HDBSCAN determines its own cluster count and has no
  mechanism to hit a target. Raise a clear error if `num_clusters` is supplied rather than
  silently ignoring it.
- **`min_clusters` / `max_clusters` cannot be honoured either.** Same reasoning. Document the
  behaviour rather than approximating it.
- **No centroids, no soft assignments.** `nearest_centroid` already no-ops when `centroids` is
  None ([refinement.py:48-49](../../../harness/refinement.py)), so returning None is safe. Per-
  cluster mean embeddings could be computed as a convenience, but noise points must be excluded
  from any such mean. State which you did.

## Hyperparameters

| Parameter | Suggested range | Role |
| --- | --- | --- |
| `min_cluster_size` | 5 to 50 | Smallest group counted as a speaker |
| `min_samples` | 1 to 20 | Conservativeness; higher means more noise |
| `cluster_selection_method` | `eom`, `leaf` | Flat clusters from the hierarchy |
| `cluster_selection_epsilon` | 0.0 to 0.5 | Merges clusters below this distance |

`min_samples` is the noise lever and therefore the axis this ticket exists to explore.

## The metric this ticket needs that the others do not

**Noise rate.** Report, at every swept point:

- fraction of (chunk, local_speaker) pairs labelled −1
- total duration covered by noise-labelled pairs
- what fraction of those pairs are **overlap-degraded** by
  `_is_overlap_degraded` ([refinement.py:137](../../../harness/refinement.py))

That third number is the whole point. If noise points are disproportionately overlap-degraded,
HDBSCAN is identifying the hard cases for free, and that is a finding regardless of its DER.

**Note the DER interaction:** noise pairs become unassigned, which converts potential confusion
into missed detection. DER weights those equally, so a method that declines hard cases can look
neutral in DER while behaving very differently. **Report the three components separately** —
aggregate DER will hide this.

## Sweep design

1. **Defaults.** One run. Establishes the baseline and the default noise rate.
2. **`min_samples`**, 6 points, 1 to 20. The noise-rate sweep, with the overlap-degraded
   fraction at each point.
3. **`min_cluster_size`**, 6 points, at the best `min_samples`.
4. **`cluster_selection_method`**, two cells, at the best of the above.

## Prediction

Recorded before running. Do not tune toward it.

- **HDBSCAN underperforms agglomerative by 3 to 8 DER points.** Speaker embeddings on a
  hypersphere form roughly isotropic blobs of similar density — not the varying-density,
  arbitrary-shape structure density methods are built for.
- **Noise points are disproportionately overlap-degraded**, by a factor of at least 2 relative
  to their share of all pairs. This is the prediction worth testing; if it fails, the connection
  to the project's framing does not hold and the method is simply a poor fit.
- **DER's missed-detection component rises with `min_samples`** while confusion falls, roughly
  trading off, so aggregate DER moves less than either component.
- **Speaker count is over-estimated**, because HDBSCAN fragments clusters it is unsure about.

## Acceptance criteria

1. A routing test asserts the pipeline holds the HDBSCAN class when selected.
2. Supplying `num_clusters` raises a clear error rather than being ignored.
3. Noise rate, noise duration, and the overlap-degraded fraction of noise points are reported
   at every swept point.
4. DER's three components are reported separately at every point, not just aggregate DER.
5. DER-versus-`min_samples` curve over at least 6 points.
6. Every run writes a manifest recording `clustering_model` and the full hyperparameter set.
7. **A negative result is an acceptable outcome and must be reported as one.** Do not tune
   toward parity with agglomerative.

## Out of scope

- **Plain DBSCAN.** Assessed and rejected above; if HDBSCAN is unavailable, report that and
  stop rather than substituting DBSCAN.
- No oracle conditions.
- No changes to how the harness treats `cluster < 0`. The existing masking is the behaviour
  under test.

## Deliverable

The DER numbers against agglomerative's 17.05%, the noise-rate curves, the overlap-degraded
fraction of noise points, and one paragraph on whether declining to assign low-density
embeddings helps, hurts, or is neutral.

The noise-point analysis is the more valuable half even if DER is poor. Lead with it.

The interpretation is the deliverable. Do not append recommended next steps.

## Implementation Notes

_Append while the work happens. Record which HDBSCAN implementation was used and whether it
needed installing._

## On completion

Fold into [[clustering-method-comparison]]'s results table and write
`Docs/Clustering Method - HDBSCAN.md`, leading with the noise-point behaviour rather than the
DER ranking.

## See also

- [[clustering-method-comparison]] — parent epic.
- [[Oracle Assignment Strategy]] — `_is_overlap_degraded`, reused here to classify noise points.
- [[Post-Clustering Refinement Hook]] — how `cluster < 0` pairs are already handled.
- [[clustering-model-selection]] — selection and override plumbing.
