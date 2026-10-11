---
status: done
created: 2026-09-25
aliases: [clustering-spectral-eigengap]
---

# Clustering Method - Spectral

## What it is

`SpectralEigengapClustering` (`src/pyannote/audio/pipelines/clustering.py`) is spectral clustering with an **eigengap** estimate of the number of speakers. It's added by this project and named to avoid clashing with sklearn's `SpectralClustering`.

It builds an affinity graph over all embeddings and partitions it globally, so unlike agglomerative it makes no irreversible early merges. The eigengap makes it k-free (`expects_num_clusters = False`).

Select it with `--clustering-model spectral`.

## How `cluster()` works

1. **Affinity:** unit-normalise and take the cosine similarity matrix. Zero negative similarities, keep the strongest `ceil(pruning × N)` entries per row (self-similarity included, so at least 1), and zero the rest. Symmetrise with `(A + A.T) / 2`. This is the `affinity()` method.
2. **Eigengap count:** take the smallest `max_speakers + 1` eigenvalues of the normalised Laplacian `I − D^-½ A D^-½` (`scipy.linalg.eigh` with `subset_by_index`). Then k = 1 + argmax of the consecutive gaps, so 1 ≤ k ≤ `max_speakers`. This is the `estimate_num_clusters()` method.
3. **k resolution:** a supplied `num_clusters` overrides the estimate, and k is then clamped to [`min_clusters`, `max_clusters`]. The pipeline only passes 1 and N, so the clamp never binds here.
4. **Partition:** sklearn `SpectralClustering(n_clusters=k, affinity="precomputed", random_state=42)` on the pruned affinity. If k = 1, everything goes to cluster 0.
5. **Labels and centroids:** labels are renumbered 0..K-1. There are no outliers. Centroids are the per-cluster means computed by `BaseClustering.assign_embeddings`.

`last_diagnostics` holds the estimated k, the eigenvalues and the wall time. The benchmark logs the k range per point.

**Speed:** about 2.5 s per AMI meeting (up to about 3,500 training embeddings). Dense `eigh` is fine and no sparse solver is needed. Synthetic timing: N = 6000 takes 12 s.

## Hyperparameters

| Parameter | Declared range | Meaning |
|---|---|---|
| `pruning` | `Uniform(0.01, 1.0)` | Fraction of strongest similarities kept per row; 1.0 = no pruning |
| `max_speakers` | `Integer(2, 20)` | Upper limit for the eigengap search |

## How it behaved

Measured by [[Clustering Benchmark Results]]:

- **Best (selected on the tune half):** pruning 0.20, max_speakers 10.
  - DER 17.16 overall (tune 16.35, report 17.86), count MAE 0.25, 13/16 exact.
  - Report-half Δ vs VBx: +0.13 [−0.87, +1.45]. **No clear difference.**
- **Stable plateau:** pruning 0.10–0.30 gives DER 17.16–17.24 with identical counting.
- **Too much pruning (0.05):** the estimate drifts up (k up to 6).
- **Too little:** the graph is too connected and the eigengap under-counts. At pruning 1.00 k reaches 1 on one meeting, giving DER 26.3, and 32.4 on the report half.
- **`max_speakers` didn't matter** (6, 10, 15 identical): the estimate stayed between 2 and 4 on every meeting.
- **As a standalone speaker counter** at pruning 0.20: MAE 0.25 and 13/16 exact, against the shipped pipeline's 0.25 and 12/16. It errs by **under-counting** (3 meetings), where VBx over-counts.
  - This is the estimator that could supply k to `KMeansClustering` ([[clustering-kmeans-estimated-k]]): call `estimate_num_clusters(affinity(embeddings))`.
- **Weaknesses:** the highest JER of the methods tested (0.2552 vs VBx 0.2219), and large losses on IS1009d (+6.0) and IS1009a (+3.9), both under-counted.

## See also

- [[Clustering Benchmark Results]]: full numbers and charts.
- [[clustering-method-comparison]]: the epic.
- [[clustering-kmeans-estimated-k]]: the consumer of the estimator.
