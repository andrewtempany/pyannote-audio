---
status: done
created: 2026-09-25
aliases: [clustering-agglomerative, agglomerative-hyperparameter-review]
---

# Clustering Method - Agglomerative

## What it is

`AgglomerativeClustering` (`src/pyannote/audio/pipelines/clustering.py`, pre-existing pyannote code, unchanged by this project) is an **alternative** clustering method. It is **not** the shipped default. community-1 ships `VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6)`, and that is what produced the baseline DER 17.05.

It works by greedy hierarchical merging over the embeddings, treated as an unordered set:
1. Every embedding starts as its own cluster.
2. The closest pair is merged until the dendrogram distance exceeds `threshold`.
3. Clusters smaller than `min_cluster_size` are absorbed into the nearest large one.

It is k-free (`expects_num_clusters = False`), so the harness can run it with no speaker count.

Select it with `--clustering-model agglomerative`. **Always set `threshold` explicitly.** Without it, selection inherits `threshold=0.6` from VBx's shipped config, and that value means something different on VBx's scale.

## Hyperparameters

| Parameter | Declared range | Effect on AMI IHM |
|---|---|---|
| `threshold` | `Uniform(0.0, 2.0)` | The main lever: U-shaped DER curve, sweet spot about 0.65–0.75 for centroid/average |
| `min_cluster_size` | `Integer(1, 20)` | Matters a lot: 1–6 over-counts by 9–44 speakers on average; 12–20 best |
| `method` | average, centroid, complete, median, single, ward, weighted | Only centroid and average reach VBx's level |

**Two behaviours to know:**
- **Centroid, median and ward use a different distance.** On cosine they switch to euclidean distance on unit-normalised embeddings, so their thresholds aren't on the same scale as the others'.
- **Ward's scale is far larger still.** Its last merges on a meeting sit at about 6–40, against about 1.1 for centroid. A threshold grid built for the other linkages cuts ward into fragments.
- **The runtime shrink of `min_cluster_size` never fires here.** It caps the setting at `max(1, round(0.1 * num_embeddings))`, but the smallest AMI test meeting has 850 training embeddings, so the configured value is always the value used.

## How it behaved

Measured by [[Clustering Benchmark Results]] (79-point sweep, tune/report split):

- **Best (selected on the tune half):** centroid, min_cluster_size 12, threshold 0.75.
  - DER 16.87 overall (tune 16.10, report 17.54), count MAE 0.188, 13/16 exact.
  - Report-half Δ vs VBx: −0.20 [−0.65, +0.31]. **No clear difference** from VBx.
- **The same threshold with min_cluster_size 20** ties exactly on the tune half and reports 17.19 with perfect report-half counting. It wasn't selected, because choosing on report data isn't allowed.
- **Speaker counting is more balanced than VBx** (signed bias +0.06 vs +0.25).
- **It's fragile.** Threshold ±0.05 costs up to about 1 point, and ±0.15 costs 3–4. At ≥ 1.10 every meeting collapses to one cluster: 2 output speakers (via the pipeline's overlap padding), DER 59.90%.
- **Other linkages are well behind:**
  - complete, best 19.99 at 1.0;
  - weighted and median, best about 21.7 at 0.6;
  - single, best 28.46 at 0.2, collapsing to one cluster from 0.6 up.
- **Ward was not evaluated on its own scale.** Doing so needs thresholds of about 4–20, outside the declared range.
- **Like every clustering method in this pipeline, it changes only confusion and the speaker count.** Missed speech, false alarm and overlap detection are fixed by segmentation.

## See also

- [[Clustering Benchmark Results]]: full numbers, charts, error profiles.
- [[clustering-method-comparison]]: the epic.
- [[clustering-vbx]]: the shipped default's own review.
- [[Evaluation Harness]]: selection and `--clustering-param`.
