---
status: done
created: 2026-09-25
aliases: [clustering-hdbscan]
---

# Clustering Method - HDBSCAN

## What it is

`HDBSCANClustering` (`src/pyannote/audio/pipelines/clustering.py`) is a density-based, k-free clustering method added by this project. It builds a hierarchy over varying density and reads flat clusters off it. Points that don't fit any dense clump are labelled **noise (outliers)**. Plain DBSCAN was rejected: its global `eps` is as sensitive as the agglomerative threshold.

Select it with `--clustering-model hdbscan`. It uses `sklearn.cluster.HDBSCAN` (scikit-learn 1.9, already installed); no new dependency.

## How `cluster()` works

1. Unit-normalise the embeddings and run HDBSCAN with `metric="euclidean"`, which on the unit sphere is monotonic in cosine distance.
2. **Fold outliers in.** Every embedding labelled noise (any negative label) is reassigned to the nearest non-noise cluster centroid by cosine. So `cluster()` never returns a negative label, and the pipeline downstream sees an ordinary clustering.
3. **If every point is noise, everything goes to cluster 0.**
4. Renumber labels 0..K-1. As with every `BaseClustering` subclass, `assign_embeddings` then assigns every embedding, including the ones the filter held back, to its nearest centroid. The class returns those centroids.

`num_clusters`, `min_clusters` and `max_clusters` are **not honoured**: HDBSCAN chooses its own count. The pipeline only ever passes the trivial bounds 1 and N here.

## Hyperparameters

| Parameter | Declared range | Meaning |
|---|---|---|
| `min_cluster_size` | `Integer(2, 200)` | Smallest group counted as a speaker |
| `min_samples` | `Integer(0, 200)` | Neighbourhood size for density; **0 means "same as min_cluster_size"** (sklearn's None) |
| `cluster_selection_method` | `eom`, `leaf` | How flat clusters are read off the hierarchy |

## Diagnostics

After each call, the instance holds `last_diagnostics`:
- per-file counts: training embeddings, outliers, clusters found, embeddings held back by the pipeline's filter, and mean overlap fraction for outliers, inliers and held-back embeddings;
- one row per training embedding (`chunk_idx, speaker_idx, is_outlier, assigned_cluster, overlap_fraction`).

`run_harness.py --clustering-diagnostics-dir DIR` writes these to `diagnostics.json` and `diagnostics_rows.csv`; the benchmark driver renames the latter `outliers.csv`.

**Overlap fraction** is the share of that local speaker's active frames, in that chunk, where 2 or more speakers are active, taken from the binarised segmentation. The subclass collects this by overriding `filter_embeddings`: it calls the base version unchanged and records indices on the side. No other class is affected.

## How it behaved

Measured by [[Clustering Benchmark Results]]:

- **Best (selected on the tune half):** `leaf`, min_cluster_size 40, min_samples default.
  - DER 16.86 overall (tune 16.48, report 17.18).
  - 14/16 meetings counted exactly, 8/8 on the report half.
  - Report-half Δ vs VBx: **−0.55 [−1.08, −0.14]**, the only method with an interval excluding zero.
  - **The caveat:** it was 0.22 *worse* than VBx on the tune half, and lost 4 points on IS1009a. Treat it as promising, not as an established improvement.
- **Main weakness: under-counting.** It merges a speaker on 2 meetings, and JER is 0.2432 against VBx's 0.2219. The speaker-balanced score pays for a merged quiet speaker even when DER doesn't.
- **Very sensitive to settings:**
  - min_cluster_size below about 20 fragments speakers (DER 33–98%);
  - `eom` plateaus around 24–25%;
  - min_samples 1 or 5 makes about half of all embeddings outliers (DER 65–69%).

**Outliers at the best point:**
- **Count:** 18.3% of training embeddings, ranging 3.1% (ES2004c) to 36.8% (TS3003d) by meeting.
- **Overlap:** outliers carry about twice the overlap of non-outliers (mean overlap fraction 0.206 vs 0.107). So HDBSCAN's "can't place this" signal does pick out overlap-touched embeddings. That is the hook for the planned outlier-classifier ticket.
- **The filter's blind spot:** the most heavily overlapped embeddings never reach HDBSCAN at all. The pipeline's `filter_embeddings` holds back any embedding with less than 20% solo speech in its chunk, which is 38.1% of active embeddings on this corpus. Those are assigned afterwards by nearest centroid, for every method.

## See also

- [[Clustering Benchmark Results]]: full numbers, charts, outlier table per meeting.
- [[clustering-method-comparison]]: the epic.
- [[Oracle Assignment Strategy]]: `_is_overlap_degraded`, an alternative overlap classification not used here.
