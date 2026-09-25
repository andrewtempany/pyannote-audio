---
status: not-started
created: 2026-09-25
---

# EPIC: Clustering method comparison

## Goal

Replace the pipeline's clustering stage with alternative methods and measure what each one
does to DER and to speaker counting, on AMI IHM test under **baseline segmentation only**.

This is the project's primary intervention arm. Everything to date has been ablation —
measuring what is *possible*. This measures what a *method* achieves, which is what the
research question actually asks: how far can diarisation be improved by intervening at the
clustering stage without retraining any neural component.

## Explicitly out of scope

- **No new oracle runs of any kind.** Not oracle segmentation, not oracle assignment, not
  `OracleClustering`. The oracle experiment is closed. Existing oracle results may be *cited*
  as motivation or as a ceiling, but nothing new is run.
- **No retraining.** Every method operates on the frozen pipeline's embeddings.
- **No new corpora.** AMI IHM test, 16 meetings. SDM is separate work.

## Prerequisite: pre-clustering cache

**This epic is not feasible without it.** A scored run currently costs 47 to 71 minutes because
segmentation and embedding extraction are redone every time. A ten-point hyperparameter sweep
is therefore 8 to 12 hours, and this epic involves several such sweeps. Without caching the
compute alone exceeds the time remaining.

Segmentation and embeddings do not depend on clustering. Cache them per file, keyed on
`(pipeline_config_id, segmentation_source.id, uri)`, and every clustering variation becomes
CPU-seconds.

**The seam is *before* clustering, not the refinement hook.** Refinement runs after clustering
and is too late. The injection point sits between embedding extraction and the clustering
call in `SpeakerDiarization.apply()`.

**Treat it with the same caution as the segmentation seam.** That seam failed silently and cost
a full invalid experiment. This ticket must carry a stated prediction (a cached run and a cold
uncached run produce byte-identical hypotheses) and a test that fails before the work lands.
See [[oracle-segmentation-seam-no-op]] for what happens when it does not.

Spawn as: `pre-clustering-embedding-cache`.

## Prerequisite: clustering selection in the harness

`run_harness.py:257-260` hardcodes `clustering_model = "pyannote-default"` with a comment
noting that no real selection exists yet. The manifest field is already in the schema, so the
provenance slot is waiting.

Needs a `--clustering-model` argument, a controlled vocabulary in `run_manifest.py`, and
wiring to instantiate the chosen clustering into the pipeline. Hyperparameter overrides need a
path through too, since every sweep depends on it.

**Watch for the same routing trap** that [[Oracle 2x2 Combined Cells]] found in item 2: an
exact-equality test that falls through to a default branch produces a run which completes,
writes a valid manifest claiming one method, and silently scores another. Add a regression
test per method asserting the pipeline actually holds the class the manifest names.

Spawn as: `clustering-model-selection`.

## What is already in the codebase

`Clustering` enum, [clustering.py:759-763](../../../src/pyannote/audio/pipelines/clustering.py):

| Class | `expects_num_clusters` | Status |
| --- | --- | --- |
| `VBxClustering` | False | **The shipped default.** `Fa=0.07, Fb=0.8, threshold=0.6` from community-1's own `config.yaml`; this is what produced the 17.05 baseline. Covered by [[clustering-vbx]]. |
| `AgglomerativeClustering` | False | Implemented, k-free, **untested here**. An alternative, not the default. Zero implementation cost. Covered by [[agglomerative-hyperparameter-review]]. |
| `KMeansClustering` | True | Implemented but needs a speaker count the harness does not supply. |
| `OracleClustering` | True | Out of scope. |

## Method list

Ordered by value per hour. Each becomes its own sub-ticket against the selection interface.

---

### P0 — `VBxClustering` (the shipped default — hyperparameter review, not a new method)

**Corrected 2026-09-25.** This entry previously described VBx as an untested alternative. It is
not: `VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6)` is what community-1 actually ships and runs,
verified from the shipped `config.yaml` and from the live instantiated pipeline. Every baseline
number in this vault, including DER 17.05, was produced by it.

So there is no "does VBx beat the default" question to answer — VBx *is* the default. The P0 job
is instead a **hyperparameter review of the shipped configuration**: how sensitive DER and speaker
counting are to `Fa`, `Fb` and `threshold` around the values that produced 17.05. That is the
cheapest test of this project's core question, and it is now owned by [[clustering-vbx]].

Variational Bayes HMM over embeddings, the standard strong baseline in DIHARD and VoxSRC
evaluations, so the baseline itself already anchors against published practice. It models speaker
turns as a sequence with transition probabilities.

`expects_num_clusters = False` ([clustering.py:552](../../../src/pyannote/audio/pipelines/clustering.py)),
so it runs with no counting work.

Hyperparameters ([clustering.py:568-570](../../../src/pyannote/audio/pipelines/clustering.py)):

- `threshold` — `Uniform(0.5, 0.8)`, the AHC initialisation threshold
- `Fa` — `Uniform(0.01, 0.5)`, acoustic scaling
- `Fb` — `Uniform(0.01, 15.0)`, speaker regularisation

Note `__call__` is overridden rather than `cluster` (line 572), so it does not follow the same
path as the other methods. Check how hyperparameter overrides reach it.

**This is the single highest-value item in the epic.** No implementation, a strong published
lineage, and a mechanism genuinely distinct from the default.

---

### P1 — Spectral clustering with eigengap count estimation

The other standard approach in the diarisation literature alongside agglomerative. Builds an
affinity matrix from embedding similarity and partitions the resulting graph, which is a
different failure mode from greedy merging: it considers global structure rather than making
irreversible local merge decisions.

The eigengap heuristic estimates the number of speakers from the spectrum of the affinity
matrix. That matters twice over: it makes spectral clustering k-free, **and it supplies the
count estimator that unblocks KMeans**. Implementing this once serves two methods.

Subclass `BaseClustering`, implement `cluster(embeddings, min_clusters, max_clusters,
num_clusters)`. `sklearn.cluster.SpectralClustering` plus a small eigengap routine.

This also revives the speaker-counting lever from the original project proposal, which had
decayed into a reported metric.

---

### P2 — `AgglomerativeClustering` (untested alternative)

Added to this list 2026-09-25. It was previously absent because the vault wrongly recorded it
as the shipped default, so it looked like the baseline rather than a candidate. It is not: it
has never been run in this project.

Greedy agglomerative merging over embeddings, treating them as an unordered set. That makes it
the natural contrast to the VBx baseline, which models turns as a sequence — so this is a fair
test of whether the baseline's sequence modelling earns its place.

`expects_num_clusters = False` ([clustering.py:310](../../../src/pyannote/audio/pipelines/clustering.py)),
so it runs with no counting work, and it is already implemented. Zero implementation cost.

Hyperparameters ([clustering.py:322-328](../../../src/pyannote/audio/pipelines/clustering.py)):
`threshold` `Uniform(0.0, 2.0)`, `min_cluster_size` `Integer(1, 20)`, `method` (seven linkages).

**Trap:** `--clustering-model agglomerative` yields `threshold=0.6` inherited from the shipped
VBx config, which is *not* agglomerative's own default and sits low on its range. Set
`threshold` explicitly at every sweep point. Owned by [[clustering-agglomerative]].

---

### P3 — `KMeansClustering` with estimated k

Already implemented but blocked: `expects_num_clusters = True`, and the harness never supplies
`num_speakers`. `run_harness.py:122` builds `file = {"uri": ..., "audio": ...}` and
`runner.run(file)` calls the pipeline with no count.

With oracle excluded, the only honest source is an estimator, so **this depends on P1 landing
first**. Record the k source in the manifest, not just the method name, or the run is
uninterpretable later.

Worth running mainly as a contrast: KMeans forces spherical, equally-sized clusters, which is
a poor fit for unbalanced speaker turn distributions. Expect it to underperform. A method that
fails for an understood reason is worth a paragraph.

---

### P4 — HDBSCAN (exploratory)

Density-based, k-free, no global `eps` to tune.

**Expect it to underperform, and include it anyway.** Speaker embeddings on a hypersphere form
roughly isotropic blobs, which is not the shape density methods are built for. Plain DBSCAN is
worse still: its `eps` is exactly as sensitive as the agglomerative threshold, so it trades one
hyperparameter for another with no gain. If either goes in, it is HDBSCAN.

The interesting property is that density methods label low-density points as **noise, cluster
-1**. The harness already skips `cluster < 0` pairs in the refinement loop and masks them
downstream, so unassigned pairs flow through without special handling. Those noise points are
plausibly the overlap-degraded embeddings sitting between centroids, which connects directly
to the project's original framing.

Whether that helps or hurts is a real question and the answer is informative either way.

---

### P5 — Affinity propagation, Bayesian GMM

Both k-free. Include only if P0 to P4 land with time to spare.

Bayesian GMM with a Dirichlet process prior was a stronger candidate before `VBxClustering`
turned up in the codebase; VBx now occupies that niche with a better pedigree. Demoted
accordingly.

## Methodology

**Report curves, not tuned points.** For each method, sweep its main hyperparameter and plot
DER against it with the shipped default marked. A point comparison between a tuned method and
an untuned one is meaningless, and a sensitivity profile is more informative anyway: for a
retraining-free intervention, fragility matters as much as peak performance.

**Split the test set for any tuning claim.** There is no dev audio on disk, so tuning on all 16
meetings and reporting on all 16 is fitting and reporting on the same data. Split the meetings
into two halves, select on one, report on the other, and fix the split before looking at any
result. Same runs, different aggregation — the per-file CSVs already support it. See
[[clustering-vbx]], which runs first and therefore establishes the split this epic reuses.

**Record speaker counting alongside DER for every cell.** The harness already computes count
MAE and exact-match. Methods differ more in how many speakers they find than in how well they
assign them, and counting is where the original proposal's second lever lives.

**Sixteen meetings is small.** Differences below roughly 0.3 DER points will not be resolvable.
Say so rather than ranking methods separated by less than that.

## Results table this epic produces

| Method | k source | DER | MD | FA | Conf | Count MAE | Exact | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **VBx (shipped default)** | self, threshold | 17.05 | 2925.05 | 1099.07 | 1212.15 | 0.25 | 12/16 | **the baseline.** `Fa=0.07, Fb=0.8, threshold=0.6` |
| VBx (tuned) | self, threshold | | | | | | | from [[clustering-vbx]] |
| Agglomerative | self, threshold | | | | | | | untested alternative; from [[agglomerative-hyperparameter-review]] |
| Spectral | eigengap | | | | | | | |
| KMeans | eigengap | | | | | | | |
| HDBSCAN | self | | | | | | | |

Plus one sensitivity curve per method.

## Sub-tickets

Spawn in this order. Each is small against the interface once the prerequisites land.

1. `pre-clustering-embedding-cache` — prerequisite, blocks everything
2. `clustering-model-selection` — prerequisite, blocks everything
3. `clustering-vbx` — P0, no implementation (the shipped default's own review)
4. `clustering-spectral-eigengap` — P1, also unblocks P3
5. `clustering-agglomerative` — P2, no implementation
6. `clustering-kmeans-estimated-k` — P3
7. `clustering-hdbscan` — P4
8. P5 only if time allows

## Acceptance criteria for the epic

1. Every run records `clustering_model` and, where applicable, the k source in its manifest.
   No run may claim a method it did not use — verified by a per-method routing test.
2. The comparison table above is populated for at least P0 and P1, with counting metrics.
3. At least one sensitivity curve per tested method.
4. Any claim that a non-default configuration improves on the default is reported on the
   held-out half of the split, with the selection half named.
5. Differences below 0.3 DER points are described as unresolved rather than ranked.

## See also

- [[clustering-vbx]] — the SHIPPED DEFAULT's own hyperparameter sweep; runs first and
  establishes the tune/report split this epic reuses.
- [[agglomerative-hyperparameter-review]] — an untested alternative method, not the default.
- [[Oracle 2x2 Combined Cells]] — the routing trap to avoid, and the assignment ceilings that
  bound what post-clustering work can add.
- [[Oracle Segmentation Findings Report]] — the 13.09 pt segmentation budget that bounds all of
  this, and the counting regression that motivates the counting metrics.
- [[Run Manifest]] — the `clustering_model` field, currently hardcoded.
- [[Evaluation Harness]] — where clustering selection has to hook in.
