---
status: not-started
created: 2026-09-25
---

# Agglomerative clustering: evaluation of an alternative method

## Goal

Evaluate `AgglomerativeClustering` — an **alternative** clustering method that this project has
never run — on AMI IHM test, and measure how its DER and speaker counting respond to its three
hyperparameters.

## PREMISE CORRECTION, 2026-09-25 — agglomerative is NOT the shipped default

This ticket was written as "review the shipped default's settings". **That premise was false.**

`community-1` clusters with `VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6)`, verified from the
checkpoint's shipped `config.yaml` and from the live instantiated pipeline. `AgglomerativeClustering`
has **never been run in this project** and contributes to none of its recorded numbers.

**What changes as a result:**

- This is no longer "the cheapest test of the core research question" or a P0 review of the
  baseline. That job is a review of VBx's own hyperparameters, and it now lives in
  [[clustering-vbx]].
- This ticket becomes a **method comparison entry** under [[clustering-method-comparison]]:
  one more alternative, evaluated against the VBx baseline of DER 17.05.
- **There is no "default configuration" here to improve on.** The Coordinator finding below is
  now the central fact rather than a footnote: `--clustering-model agglomerative` yields
  `threshold=0.6` inherited from VBx's config, which is *not* agglomerative's own default and is
  not a meaningful baseline. Every sweep point must set `threshold` explicitly.
- **Recommended re-prioritisation: P0 → P2**, and a rename to `clustering-agglomerative` for
  consistency with the epic's other method tickets. See the report accompanying this change.

## Why run it at all

Agglomerative clustering is the other standard approach in the diarisation literature alongside
VBx, and it is **already implemented** with `expects_num_clusters = False`
([clustering.py:310](../../../src/pyannote/audio/pipelines/clustering.py)), so it needs no
speaker-count plumbing. Mechanically it is genuinely distinct from the baseline: it treats
embeddings as an unordered set and makes irreversible greedy merges, where VBx models speaker
turns as a sequence with transition probabilities. That makes it a fair test of whether the
baseline's sequence modelling is actually earning its place.

A note on motivation that **no longer applies to this ticket**: the oracle-segmentation counting
regression ([[Oracle Segmentation Findings Report]] §4) was cited here as evidence "the default
threshold may be mis-set". Since the default is VBx, that motivation has moved to
[[clustering-vbx]]. It says nothing about agglomerative.

## The hyperparameters

[clustering.py:322-328](../../../src/pyannote/audio/pipelines/clustering.py):

| Parameter | Range | Notes |
| --- | --- | --- |
| `threshold` | `Uniform(0.0, 2.0)` | Distance cutoff on the dendrogram. Primary axis. |
| `min_cluster_size` | `Integer(1, 20)` | Small clusters are absorbed. Secondary axis. |
| `method` | `Categorical(["average", "centroid", "complete", "median", "single", "ward", "weighted"])` | Linkage. Tertiary. |

Two behaviours worth knowing before designing the sweep:

- **`metric` and `method` interact.** [clustering.py:371-375](../../../src/pyannote/audio/pipelines/clustering.py): when `metric == "cosine"` and `method` is one of `centroid`, `median` or `ward`, embeddings are converted and the linkage is computed with `metric="euclidean"` instead. So those three linkages are not directly comparable to the others on the same distance scale, and a threshold that is sensible for one family may be meaningless for the other. **Sweep threshold separately per linkage family.**
- **`min_cluster_size` is adjusted at runtime.** [clustering.py:359-362](../../../src/pyannote/audio/pipelines/clustering.py) reduces it to `max(1, round(0.1 * num_embeddings))` when there are few embeddings. On short meetings the configured value may not be the value used. Record the effective value, not just the requested one.

**There are no "shipped defaults" for this class.** community-1 ships a VBx configuration, so
agglomerative has no blessed setting here. Record instead (a) the class's own declared defaults
from `clustering.py:322-328`, and (b) what `--clustering-model agglomerative` actually
instantiates — currently `threshold=0.6` **inherited from VBx's config**, see the Coordinator
finding below. Put both in the Implementation Notes, and be explicit in every curve about which
is which.

## Prerequisites

Both from [[clustering-method-comparison]]:

- `pre-clustering-embedding-cache` — without it each sweep point costs 47 to 71 minutes and a
  ten-point sweep is an overnight job. With it, minutes.
- `clustering-model-selection` — the path for hyperparameter overrides to reach the pipeline.

If the cache seam slips, this ticket can still run at reduced resolution (5 threshold points
instead of 15) overnight. Say so rather than waiting.

## Method

**Use the tune/report split established by [[clustering-vbx]]**, which runs first. Do not
invent a second split: the epic's cells must be comparable. There is no dev audio on disk, so
the 16 test meetings are the only data available, and selecting and reporting on the same 16 is
fitting. If [[clustering-vbx]] has not yet fixed the split when this ticket starts, fix it there
rather than here.

**Sweep in this order, one axis at a time:**

1. **`threshold`**, holding `method` and `min_cluster_size` at the class's declared defaults.
   12 to 15 points across the declared `Uniform(0.0, 2.0)` range. **Set it explicitly at every
   point**, including the first — the inherited 0.6 is not a baseline. This is the main result.
2. **`min_cluster_size`**, at the best threshold from step 1. The full 1 to 20 range is cheap.
3. **`method`**, across all seven linkages, each at its own best threshold from a short
   per-linkage sweep. Respect the cosine/euclidean split noted above.

**Record for every point:** DER and its three components, speaker count MAE, exact-match count,
and the effective `min_cluster_size`. The per-file CSVs already carry what is needed.

## Prediction

Recorded before running. Do not tune toward it.

- **Tuned agglomerative lands within ±1.5 DER points of the VBx baseline's 17.05.** Both are
  mature methods on a corpus in their intended domain, so a large gap either way would be
  surprising. (This replaces a prediction about "the default threshold" being near-optimal —
  there is no agglomerative default here to be optimal.)
- **The inherited `threshold=0.6` is NOT near the agglomerative optimum.** It was tuned for a
  different algorithm on a different distance scale, so a sweep should move away from it. If
  0.6 turns out to be optimal for agglomerative too, treat that as a coincidence to check, not
  a confirmation.
- **The DER curve is flatter than the counting curve.** DER trades missed detection against
  confusion as the threshold moves, which partly self-cancels; speaker count does not
  self-cancel and should move monotonically — a higher threshold merges more, so fewer
  speakers.
- **`min_cluster_size` matters less than `threshold`** on 30-to-50 minute meetings, because the
  runtime adjustment at line 359 only binds on short files.

**A large improvement would be a surprise and should be treated as one.** If a sweep point beats
the VBx baseline by more than 1 DER point, check for a scoring or wiring artifact before believing
it — the same standard applied to the oracle result. This project has twice produced a
plausible number that measured the wrong thing.

## Acceptance criteria

1. Both the class's declared defaults AND what `--clustering-model agglomerative` actually
   instantiates are recorded in the Implementation Notes before any sweep runs, with the
   inherited-from-VBx `threshold=0.6` flagged as such.
2. The tune/report split from [[clustering-vbx]] is reused unchanged, and named in this ticket.
3. A DER-versus-threshold curve over at least 12 points, with the **VBx baseline 17.05** marked
   as the reference line and the inherited 0.6 marked as a non-baseline.
4. Speaker count MAE and exact-match reported at every swept point, not just DER.
5. Any recommended configuration is reported on the **held-out** half, with the selection half
   named. A configuration selected and reported on the same 16 meetings is not a result and
   must not be presented as one.
6. Every run writes a manifest recording the hyperparameter values used.
7. If no configuration beats the **VBx baseline** on held-out data, **that is the finding**.
   Report it plainly. A negative result is directly meaningful: it says the baseline's sequence
   modelling earns its place against a strong set-based alternative.

## Deliverable

The threshold curve, the counting curve, a short table of the best configurations per axis, and
one paragraph stating whether tuned agglomerative beats, matches or loses to the VBx baseline on
held-out data.

The interpretation is the deliverable. Do not append recommended next steps.

## Out of scope

- No sweeps under oracle segmentation. The co-adaptation hypothesis stays untested and is cited
  as motivation only.
- No other clustering methods. Those are [[clustering-method-comparison]].
- **No review of the shipped default's hyperparameters.** That is [[clustering-vbx]].
- No changes to `AgglomerativeClustering` itself. This is configuration, not modification.

## Coordinator finding, 2026-09-25 — `agglomerative` starts at a threshold borrowed from VBx

Recorded here because this ticket is the one that would be misled by it. Verified live on the
real pipeline after [[clustering-model-selection]] landed.

`--clustering-model agglomerative` produces `AgglomerativeClustering(threshold=0.6)` — **not
that class's own default.** The 0.6 comes from the shipped community-1 config, whose `params.clustering`
block was written for **VBxClustering** (the real default; see the premise correction in
[[clustering-model-selection]]). Selection applies the shipped values filtered to the names the
target class declares, so `threshold` carries over while `Fa`/`Fb` are dropped, with a printed
note:

```
note: AgglomerativeClustering does not declare Fa, Fb -- shipped value(s) not applied
agglomerative threshold: 0.6
desc: AgglomerativeClustering(threshold=0.6)
```

**Why this is a trap rather than a detail.** `threshold` exists on both classes with *different
meanings and different ranges*: `Uniform(0.5, 0.8)` on VBx versus `Uniform(0.0, 2.0)` on
agglomerative. 0.6 sits mid-range for VBx and low for agglomerative, so an unswept
`agglomerative` run is not "agglomerative at its default" — it is agglomerative at a value
inherited from a different algorithm.

**What this ticket must do about it:** treat 0.6 as an arbitrary starting point, not a baseline
worth reporting as "the default". State the shipped-vs-declared distinction explicitly when
recording the sensitivity profile, and set `threshold` explicitly via `--clustering-param` for
every sweep point rather than relying on the inherited value for the first one.

## Implementation Notes

_Append while the work happens, not at the end. Record the shipped defaults and the tune/report
split here first._

## On completion

Rewrite as reference documentation describing how agglomerative clustering behaves on AMI IHM
relative to the VBx baseline: which parameters matter, over what range, and how DER and counting
respond. Set `status: done` and move to
`Obsidian-Diarisation/Docs/Clustering Method - Agglomerative.md`.

## See also

- [[clustering-method-comparison]] — the epic this feeds.
- [[clustering-vbx]] — the SHIPPED DEFAULT's hyperparameter review; establishes the tune/report
  split this ticket reuses.
- [[Oracle Segmentation Findings Report]] — §4, the counting regression that motivates this.
- [[Oracle Ceiling Metrics]] — what post-clustering intervention can add, for context on scale.
- [[Run Manifest]] — where hyperparameter values need recording.
