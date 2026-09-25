---
status: not-started
created: 2026-09-25
---

# Clustering method: KMeans with estimated k

## Goal

Run the already-implemented `KMeansClustering` using an estimated speaker count, and measure
how sensitive DER is to counting error.

Sub-ticket of [[clustering-method-comparison]]. **P2. Blocked by
[[clustering-spectral-eigengap]]**, which builds the count estimator.

## Why it is blocked, and why that is the right blocker

`KMeansClustering` is implemented but has `expects_num_clusters = True`
([clustering.py:496](../../../src/pyannote/audio/pipelines/clustering.py)), and the harness
supplies no speaker count: [run_harness.py:122](../../../run_harness.py) builds
`file = {"uri": ..., "audio": ...}` and `runner.run(file)` calls the pipeline with no
`num_speakers`.

There are three possible sources for k. Only one is in scope:

| Source | Status |
| --- | --- |
| Oracle count from the reference | **Excluded.** Injecting ground truth into the pipeline is an oracle condition, and those are closed. |
| Fixed k per run | Not a method, just a probe. Useful for the sensitivity sweep below, not as a reported system. |
| **Estimated k** | **In scope.** From the eigengap estimator in [[clustering-spectral-eigengap]]. |

So KMeans cannot be evaluated as a system until the estimator exists. That is a genuine
dependency, not sequencing preference.

## The experiment worth running

KMeans on its own is not expected to be competitive, and that is fine — see the prediction. Its
value is as an instrument.

**Measure DER as a function of counting error.** Run KMeans with k set to the estimate, and then
at the estimate ±1 and ±2, and plot DER against the offset. That quantifies something the
project has tracked all the way through without ever measuring: **how much a speaker-count
error actually costs in DER.**

That number matters beyond KMeans. It tells you what improving the pipeline's counting would be
worth, which is the question behind the counting regression recorded in
[[Oracle Segmentation Findings Report]] §4 and behind the eigengap work in the spectral ticket.
No other method in the epic isolates it, because every other method couples counting and
assignment together.

**Note carefully:** the offsets are relative to the **estimated** count, not the reference
count. Sweeping around the true count would be an oracle condition. Sweeping around the
estimate is a sensitivity analysis on a quantity the system produces for itself.

## Hyperparameters

`KMeansClustering` is thin. The axes are:

| Parameter | Range | Role |
| --- | --- | --- |
| k offset | −2 to +2 from the estimate | The sensitivity sweep above |
| `n_init` | 1, 10, 50 | Restarts; guards against poor initialisation |
| `metric` | cosine (inherited from `BaseClustering`) | Leave fixed |

Check what the class actually exposes before designing the sweep —
[clustering.py:483-548](../../../src/pyannote/audio/pipelines/clustering.py) — and note that
`cluster()` handles the cosine case specially at line 540.

## Sweep design

1. **KMeans at the estimated k**, `n_init` at default. One run. The headline number.
2. **k offset sweep**, five points, −2 to +2. The sensitivity result.
3. **`n_init`**, three points, at offset 0. Cheap; confirms results are not initialisation noise.

Record DER and its three components, plus the estimated and effective k per file, at every
point.

## Prediction

Recorded before running. Do not tune toward it.

- **KMeans underperforms agglomerative, by 2 to 5 DER points.** It forces spherical,
  roughly equal-sized clusters, and AMI speaker turn distributions are strongly unbalanced —
  a chair or presenter often holds a large majority of speech. That mismatch is structural, not
  a tuning problem.
- **DER degrades asymmetrically around the estimate.** Over-estimating k should cost less than
  under-estimating: splitting one speaker into two produces confusion on part of their speech,
  while merging two speakers produces confusion on all of the smaller one's. Expect the +
  direction to be the shallower side.
- **The per-unit cost of a counting error is 1 to 3 DER points.**
- **`n_init` matters little** beyond the default, on embeddings this well-separated.

A result where KMeans *beats* agglomerative would be a surprise worth checking the cache key
over before believing.

## Acceptance criteria

1. A routing test asserts the pipeline holds a `KMeansClustering` instance when selected.
2. k is sourced from the estimator, never from the reference. A test asserts the reference is
   not consulted on the k path.
3. The estimated k and the effective k are both recorded per file in the manifest or the
   per-file CSV. A run whose k source is unrecorded is uninterpretable later.
4. DER-versus-k-offset curve over five points, with offset 0 marked.
5. The per-unit DER cost of a counting error is stated as a number with its direction
   asymmetry, not just plotted.
6. Every run writes a manifest recording `clustering_model`, the k source, and the full
   hyperparameter set.

## Out of scope

- **No oracle k.** Not as a condition, not as a ceiling, not as a diagnostic.
- No modification to `KMeansClustering`.
- No alternative count estimators. This ticket consumes the one from
  [[clustering-spectral-eigengap]]; comparing estimators is separate work.

## Deliverable

The headline DER against agglomerative's 17.05%, the k-offset sensitivity curve, and one
paragraph stating the DER cost per unit of counting error and whether it is symmetric.

The sensitivity result is the more valuable half. Lead with it.

The interpretation is the deliverable. Do not append recommended next steps.

## Implementation Notes

_Append while the work happens._

## On completion

Fold into [[clustering-method-comparison]]'s results table. The k-sensitivity result deserves
its own short doc — `Docs/Speaker Count Sensitivity.md` — because it is reusable evidence about
the value of counting accuracy, independent of KMeans.

## See also

- [[clustering-method-comparison]] — parent epic.
- [[clustering-spectral-eigengap]] — builds the count estimator this ticket depends on.
- [[Oracle Segmentation Findings Report]] — §4, the counting regression that motivates the
  sensitivity sweep.
- [[clustering-model-selection]] — selection and override plumbing.
