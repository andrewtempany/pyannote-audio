---
status: not-started
created: 2026-09-25
---

# Hyperparameter review of the shipped default (VBx)

## Goal

Measure how sensitive DER and speaker counting are to the **shipped default clustering
configuration's** three hyperparameters — `Fa`, `Fb` and `threshold` — on AMI IHM test, and
establish whether community-1's defaults can be improved for this corpus without retraining
anything.

Sub-ticket of [[clustering-method-comparison]]. **P0 — highest value per hour in the epic.**

## PREMISE CORRECTION, 2026-09-25 — VBx is the baseline, not an alternative

This ticket was originally written as "evaluate VBx, a strong alternative that is already
implemented". **That premise was false**, and the ticket has been rewritten around the correct
one.

`community-1` clusters with `VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6)`. Verified from the
checkpoint's shipped `config.yaml` and independently from the live instantiated pipeline; both
recorded in [[Pre-Clustering Cache]] and now in every run manifest as `clustering_config`. The
vault previously asserted `AgglomerativeClustering` was the default. It is not, and
agglomerative has never been run in this project.

**What changes as a result:**

- There is **no "does VBx beat the default"** question. VBx at defaults *is* the default, and
  its result is already recorded: **DER 17.05**, count MAE 0.25, 12/16 exact. The original
  sweep step 1 ("one run at the shipped configuration ... the most important result in the
  ticket") is therefore **already done** — it is the baseline, and re-running it only
  reproduces 0.17048543579940637.
- The valuable job is the one the vault wrongly assigned to
  [[agglomerative-hyperparameter-review]]: **reviewing the shipped default's own settings.**
  That is what this ticket now is.
- The **"zero implementation cost" claim still holds**, and more strongly than before: the
  class is not merely implemented, it is already instantiated and running on every baseline
  run. Only `--clustering-param` overrides are needed.

## Why this one first

It reviews the configuration that produces every number this project has reported. Nothing
about those defaults has been questioned so far, and they were tuned end-to-end by
community-1's authors jointly with their own segmentation model on their own tuning data —
nothing guarantees they are optimal for AMI IHM specifically.

There is a concrete pointer that they may be mis-set. Under oracle segmentation, speaker
counting *regressed*: exact matches fell 12/16 to 7/16 and MAE rose 0.25 to 0.69, with
**over-counting in all 9 mismatching meetings and never under-counting**
([[Oracle Segmentation Findings Report]] §4). One reading is that the default sits where it
does partly to compensate for its own segmenter's missed speech. **That hypothesis is not
testable here** — confirming it would need a sweep under oracle segmentation, and no new
oracle runs are being done. It stands as recorded motivation only.

Mechanically, VBx models speaker turns as a **sequence with transition probabilities**. That
matters for this project: [[Oracle Ceiling Metrics]] established that temporal context from the
pipeline's existing Bayesian HMM does not extend to overlap assignment — but that sequence
modelling is already inside the clustering stage of the baseline, rather than being something
this epic would add.

## Hyperparameters

[clustering.py:568-570](../../../src/pyannote/audio/pipelines/clustering.py):

| Parameter | Range | Role |
| --- | --- | --- |
| `threshold` | `Uniform(0.5, 0.8)` | Threshold for the AHC initialisation VBx refines from |
| `Fa` | `Uniform(0.01, 0.5)` | Acoustic scaling — how strongly frame evidence drives assignment |
| `Fb` | `Uniform(0.01, 15.0)` | Speaker regularisation — resistance to introducing new speakers |

**`Fb` is the one to watch.** It controls how readily the model posits an additional speaker,
which makes it the direct lever on the speaker-counting behaviour this project has been
tracking. Its range spans three orders of magnitude, so sweep it **log-spaced**, not linearly.

> **Direction VERIFIED from the update equations, 2026-09-25 — higher `Fb` means FEWER
> speakers.** Checked because the counting hypothesis leans on it and the reading had come from
> the parameter's one-line description ("Speaker regularization coefficient Fb controls the
> final number of speakers", `vbx.py:51`), which does not state a direction.
>
> `Fb` appears in exactly two places in the VB loop, always as the ratio `Fa/Fb`
> ([vbx.py:109-112](../../../src/pyannote/audio/utils/vbx.py)):
>
> ```python
> invL  = 1.0 / (1 + Fa / Fb * gamma.sum(axis=0, keepdims=True).T * Phi)  # (17)
> alpha = Fa / Fb * invL * gamma.T.dot(rho)                               # (16)
> ```
>
> `alpha` is the speaker mean and `invL` its posterior variance. Raising `Fb` shrinks `Fa/Fb`,
> which drives `invL` toward 1 (maximum uncertainty) and scales `alpha` toward 0 (means collapse
> to the prior). Evaluated at Fa=0.07 with the other terms held fixed, alpha falls monotonically
> 0.199 -> 0.038 and invL rises 0.003 -> 0.811 as Fb goes 0.01 -> 15.0.
>
> Collapsed means make the per-speaker likelihoods at `log_p_` (:113-115, eq 23) converge, so
> responsibilities `gamma` (:125) flatten and `pi = sum(gamma)` (:126-128) spreads instead of
> concentrating. Speakers stop being separable and are absorbed. The surviving count is read as
> `sp > 1e-7` at [clustering.py:621](../../../src/pyannote/audio/pipelines/clustering.py).
>
> **So the hypothesis stands as written and needs no rewrite.** Shipped `Fb = 0.8` sits near the
> bottom of `[0.01, 15.0]`, i.e. WEAK regularisation / low reluctance to add a speaker, which is
> consistent with the over-counting seen under oracle segmentation
> ([[Oracle Segmentation Findings Report]] §4). The sweep should therefore expect speaker count
> to fall monotonically as `Fb` rises.
>
> Caveat worth recording: a synthetic end-to-end check of `cluster_vbx` did NOT reproduce the
> merge behaviour, because the fixture never converged to distinct speakers (effective count
> stayed at the initialisation of 10 regardless of `Fb`). The direction above is established
> from the equations and from the monotonic behaviour of (16)/(17), NOT from a working
> end-to-end simulation. A real sweep is what will confirm it on this corpus.

**The shipped defaults are already known** (this is the default method, so they were recorded
when the cache-key work read them off the live pipeline):

| Parameter | Shipped value |
| --- | --- |
| `threshold` | 0.6 |
| `Fa` | 0.07 |
| `Fb` | 0.8 |

Note `Fb = 0.8` sits near the **bottom** of its `[0.01, 15.0]` range, which is another reason to
sweep it log-spaced: a linear sweep would put almost every point far above the default and
barely probe the region around it.

## Implementation caution

`VBxClustering` overrides `__call__` ([clustering.py:572](../../../src/pyannote/audio/pipelines/clustering.py))
rather than implementing `cluster()` like `AgglomerativeClustering` and `KMeansClustering` do.
Since VBx is the default, this is the path every baseline run has always taken — but the
override still has consequences for *overrides*, which no baseline run exercises.

Two consequences to verify rather than assume:

- Hyperparameter overrides from [[clustering-model-selection]] must actually reach it. Assert a
  set value is readable off the instance and that it changes output.
- It may not produce `centroids` or `soft_clusters` in the same form the refinement hook
  expects. `nearest_centroid` already no-ops when `centroids` is None
  ([refinement.py:48-49](../../../harness/refinement.py)), so the interface tolerates absence,
  but **check what VBx actually returns** before combining it with any refinement strategy.
  Run with `--refinement-strategy identity` first.

## Sweep design

Run in this order. Each stage informs the next.

1. **Defaults — ALREADY DONE, do not re-run.** The shipped configuration is the baseline:
   **DER 17.05**, MD 2925.05, FA 1099.07, Conf 1212.15, count MAE 0.25, 12/16 exact
   (`runs/20260925T083743Z-a1f4391c.json`). Mark it on every curve as the reference point. A
   re-run only reproduces 0.17048543579940637 at the cost of a GPU run.
2. **`Fb`**, log-spaced, 10 points across `[0.01, 15.0]`, other parameters at default. Report
   DER **and** speaker counting at every point.
3. **`Fa`**, 8 points across `[0.01, 0.5]`, at default `Fb` and at the best `Fb` from step 2.
4. **`threshold`**, 8 points across `[0.5, 0.8]`, at the best `Fa`/`Fb`.

Record DER and its three components, count MAE and exact-match at every point.

## Prediction

Recorded before running. Do not tune toward it.

- **The shipped defaults are near-optimal for AMI IHM, within 0.5 DER points of the best point
  found.** The authors tuned on meeting-domain data and AMI is the canonical meeting corpus, so
  a large gap would be surprising. (Note this replaces a prediction comparing VBx to
  agglomerative's "17.05" — 17.05 *is* VBx, so that comparison was vacuous.)
- **`Fb` moves speaker count monotonically** — higher regularisation means fewer speakers
  posited — and moves DER less, because count errors partly trade missed detection against
  confusion.
- **Some `Fb` counts better than the default's 12/16 exact / MAE 0.25.** `Fb` is an explicit
  prior over speaker introduction, and the default was not tuned for AMI specifically.
- **A gain larger than 1 DER point is a suspected artifact.** Check the cache key and the
  routing before believing it — in particular confirm the final-hypothesis key changed between
  sweep points (see [[Pre-Clustering Cache]]). This project has twice produced a plausible number that measured
  something other than what it claimed.

## Acceptance criteria

1. Shipped default hyperparameter values confirmed off the live pipeline and recorded in
   Implementation Notes before any sweep (expected: `threshold=0.6, Fa=0.07, Fb=0.8`).
2. A routing test asserts the pipeline holds a `VBxClustering` instance when selected.
3. A test asserts a hyperparameter override reaches the instance and changes the output.
4. What VBx returns for `centroids` and `soft_clusters` is documented, and any refinement
   interaction is stated rather than assumed.
5. DER-versus-`Fb` curve over at least 10 log-spaced points, with the default marked.
6. Speaker count MAE and exact-match reported at every swept point.
7. Every run writes a manifest recording `clustering_model` and the full hyperparameter set.
8. Any configuration recommended over the default is reported on the **held-out** half of the
   test split. This ticket runs first, so it **establishes** that split: write it into the
   Implementation Notes before the first sweep and do not revisit it. [[clustering-method-comparison]]
   and [[agglomerative-hyperparameter-review]] reuse it.

## Out of scope

- No oracle conditions.
- No modification to `VBxClustering` itself. Configuration only. If it appears broken, report
  that rather than patching it.
- No refinement strategies beyond `identity` until item 4 is settled.

## Deliverable

The three sweep curves with the shipped default marked, the counting numbers, and one
paragraph stating whether the shipped defaults can be improved on held-out data and by how
much. If they cannot, **that is the finding** — it says community-1's defaults generalise to
AMI, which bounds what hyperparameter work can contribute.

The interpretation is the deliverable. Do not append recommended next steps.

## Implementation Notes

_Append while the work happens. Defaults first._

## On completion

Fold into [[clustering-method-comparison]]'s results table and write a short
`Docs/Clustering Method - VBx.md` covering what VBx does, what its hyperparameters control, and
how it behaved on AMI IHM.

## See also

- [[clustering-method-comparison]] — parent epic.
- [[agglomerative-hyperparameter-review]] — an untested ALTERNATIVE method, not the default;
  reuses the tune/report split this ticket establishes.
- [[clustering-model-selection]] — selection and override plumbing.
- [[pre-clustering-embedding-cache]] — without it each sweep point is a full GPU run.
