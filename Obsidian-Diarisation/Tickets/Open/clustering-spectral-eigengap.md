---
status: not-started
created: 2026-09-25
---

# Clustering method: spectral with eigengap count estimation

## Goal

Add spectral clustering as a `BaseClustering` subclass, with an eigengap-based speaker-count
estimator, and evaluate both.

Sub-ticket of [[clustering-method-comparison]]. **P1.** Also unblocks
[[clustering-kmeans-estimated-k]], which needs the count estimator this ticket builds.

## Why

Spectral clustering is the other standard approach in the diarisation literature alongside
agglomerative, so its absence is a gap in the comparison rather than an exotic addition.

Mechanically it differs where it matters: it builds an affinity graph over all embeddings and
partitions it globally, where agglomerative makes irreversible local merge decisions. A wrong
early merge in AHC cannot be undone; spectral has no equivalent failure mode.

The eigengap heuristic also **revives the speaker-counting lever from the original project
proposal**, which quietly decayed into a reported metric. Counting has been tracked all project
(12/16 exact, MAE 0.25 at baseline) without anything being done about it. This is the first
ticket that acts on it.

## Two deliverables, evaluated separately

This is the part that makes the ticket worth more than one DER number.

### A. The count estimator, evaluated on its own

The eigengap estimator predicts a speaker count per file from the affinity matrix spectrum.
**That prediction can be scored against the reference count directly, without running
clustering at all.**

Evaluate it as its own result: MAE and exact-match across the 16 meetings, against the
pipeline's current 0.25 / 12-of-16. This is cheap, needs no DER run, and produces a standalone
finding about whether the pipeline's counting can be improved.

Do this **first**. If the estimator counts worse than the current 0.25 MAE, that is worth
knowing before it is wired into two clustering methods.

### B. Spectral clustering DER

The usual sweep, using the estimator from A for k.

## Implementation

Subclass `BaseClustering`, implement
`cluster(embeddings, min_clusters, max_clusters, num_clusters)`
([clustering.py:44, 330](../../../src/pyannote/audio/pipelines/clustering.py) for the contract).
Respect `num_clusters` when supplied and fall back to the eigengap estimate otherwise, so the
same class serves both a self-counting and a supplied-k mode.

`sklearn.cluster.SpectralClustering` handles the partitioning. The eigengap routine is small:
build the affinity matrix, take its Laplacian eigenvalues, find the largest gap in the sorted
spectrum below `max_clusters`.

**Return `centroids`** where possible — compute per-cluster mean embeddings — so the refinement
hook and `nearest_centroid` remain usable. Spectral has no natural centroid, but the mean of
each cluster's embeddings is a reasonable one and keeps the interface complete.

## Hyperparameters

Design these as explicit parameters so they can be swept:

| Parameter | Suggested range | Role |
| --- | --- | --- |
| `affinity` | cosine, RBF | How the similarity matrix is built |
| `gamma` | log-spaced | RBF width, if RBF is used |
| `max_clusters` | 2 to 12 | Ceiling on the eigengap search |
| `refinement` | on / off | See below |

**Affinity matrix refinement.** The standard spectral diarisation recipe applies a sequence of
operations to the affinity matrix before eigendecomposition: row-wise thresholding,
symmetrisation, diffusion, row-max normalisation. This materially changes results in the
published literature.

Implement it as a **single on/off flag**, not as four independently swept steps. The
combinatorics are not affordable and the individual steps are not separately interesting here.
Report refined against unrefined as a two-point comparison.

## Sweep design

1. **Count estimator alone** — MAE and exact-match against reference counts, for each affinity
   type and with refinement on and off. No DER runs. Cheapest and most informative step.
2. **Spectral at defaults**, k from the best estimator configuration from step 1.
3. **`max_clusters`** sensitivity, 2 to 12.
4. **Affinity and refinement**, four cells (cosine/RBF × refined/unrefined), each at its best
   `max_clusters`.

## Prediction

Recorded before running. Do not tune toward it.

- **The eigengap estimator is worse than the pipeline's current counting.** Baseline is MAE 0.25
  and 12/16 exact, which is already good; eigengap is a coarse heuristic and AMI meetings have
  3 to 5 speakers, where the spectrum is short and gaps are noisy. Expect MAE in the 0.5 to 1.5
  range.
- **Affinity refinement helps more than the affinity type.** The refinement sequence exists in
  the literature precisely because raw cosine affinity is noisy.
- **Spectral DER lands within ±2 points of 17.05%**, dominated by counting error rather than by
  assignment quality.
- **If the estimator's MAE comes in below 0.25, treat it as a suspected artifact** and check it
  is not reading the reference. That would beat a tuned production pipeline with a heuristic,
  which is possible but should be doubted first.

## Acceptance criteria

1. The count estimator is evaluated **standalone** against reference counts, before any DER run.
2. A routing test asserts the pipeline holds the spectral class when selected.
3. `cluster()` respects `num_clusters` when supplied and uses the eigengap estimate otherwise.
   Both paths are tested.
4. What the class returns for `centroids` and `soft_clusters` is documented.
5. Counting results reported as MAE and exact-match, comparable to the baseline's 0.25 / 12-of-16.
6. DER-versus-`max_clusters` curve, plus the four-cell affinity/refinement table.
7. Every run writes a manifest recording the clustering class and full hyperparameter set.
8. Any configuration recommended over the default is reported on the **held-out** half of the
   split from [[agglomerative-hyperparameter-review]].

## Out of scope

- No oracle conditions. In particular the estimator is never compared against, or seeded with,
  the reference count inside a scored run — only scored against it afterwards as a metric.
- No per-step ablation of the affinity refinement sequence. One flag.
- No changes to `BaseClustering` or to existing clustering classes.

## Deliverable

The standalone counting result, the DER curves, the affinity/refinement table, and one
paragraph on whether spectral clustering and eigengap counting are competitive with the
shipped agglomerative default.

The interpretation is the deliverable. Do not append recommended next steps.

## Implementation Notes

_Append while the work happens._

## On completion

Fold into [[clustering-method-comparison]]'s results table and write
`Docs/Clustering Method - Spectral.md` covering the class, the eigengap estimator, and its
standalone counting performance. The estimator is reusable and needs documenting on its own
terms, not only as part of spectral clustering.

## See also

- [[clustering-method-comparison]] — parent epic.
- [[clustering-kmeans-estimated-k]] — depends on the estimator this ticket builds.
- [[clustering-model-selection]] — selection and override plumbing.
- [[pre-clustering-embedding-cache]] — without it each sweep point is a full GPU run.
