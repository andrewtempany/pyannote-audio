---
status: in-progress
created: 2026-10-08
type: results
---

# Closeout Results

Single results document for the pre-writing closeout ([[closeout-0-master]]). Sections below are owned by their tickets: this first pass (closeout 1) fills the cross-condition table (IHM rows), the classifier transcription, the bootstrap intervals and the convergent-failure analysis. Closeouts 2, 3 and 4 append under their own headings at the bottom.

Conventions used throughout:
- All runs are the 16 AMI IHM **test** meetings. DER is in %.
- **DER aggregation** is total error divided by total reference speech across the 16 meetings: `sum(MD+FA+Conf) / sum(MD+FA+Conf / per-file DER)`. It is **not** the mean of per-file DERs. This is `harness.assignment_runs._der` (`harness/assignment_runs.py:161`), applied to `error` and `total` columns built in `per_file()` (`:143-147`).
- **Intervals** are 90% percentile bootstrap over meetings (resample 16 of 16 with replacement), 2000 draws, seed 0, 5th and 95th percentiles. "Paired" means: in every draw the same resampled meetings are used for the method and for the VBx baseline, and the quantity is DER(method) minus DER(baseline) on those meetings (Δ vs the VBx baseline's own per-meeting scores).
- "Nearest centroid" means `nc_target` (`703683ce`, 16.89) everywhere. The September run `32241285` (16.79) is a footnoted variant only.

---

## Cross-condition table

Settings per method are the ones selected on IHM (clustering benchmark tune half, and k=25 chosen on dev pair accuracy), carried across unchanged. Measured cells: DER [90% CI of the absolute DER over the 16 meetings]. Each Δ vs VBx [paired 90% CI] sits in the table under it.

| Condition | VBx (default) | Agglomerative | HDBSCAN | Spectral | Nearest centroid | kNN | Ward (thr 8)* |
|---|---|---|---|---|---|---|---|
| AMI IHM (published) | 17.0 | — | — | — | — | — | — |
| AMI IHM (measured) | 17.05 [15.31, 18.74] | 16.87 [14.99, 18.68] | 16.86 [15.01, 18.68] | 17.16 [15.27, 19.05] | 16.89 [15.09, 18.64] | 16.85 [15.02, 18.63] | 17.47 [15.33, 19.66] |
| Δ vs VBx, paired | — | −0.18 [−0.43, +0.08] | −0.19 [−0.55, +0.13] | +0.11 [−0.45, +0.87] | −0.16 [−0.26, −0.06] | −0.20 [−0.35, −0.06] | +0.42 [−0.40, +1.36] |
| run id | `0fa3b41d` | `da03276c` | `b15caf86` | `fda86985` | `703683ce` | `e2ef8a62` | `ef90e3f5` |
| AMI SDM (published, not run by us; see caveat below) | 19.9 | — | — | — | — | — | — |
| AMI SDM (measured) | 19.91 [18.07, 21.74] | 20.33 [17.99, 22.48] | 19.55 [17.51, 21.49] | 20.60 [18.02, 23.31] | 19.72 [17.84, 21.59] | 19.66 [17.76, 21.58] | 21.88 [18.92, 24.66] |
| Δ vs VBx, paired (SDM) | — | +0.42 [−0.37, +1.18] | −0.36 [−0.75, −0.00] | +0.69 [−0.37, +1.94] | −0.19 [−0.32, −0.06] | −0.25 [−0.40, −0.11] | +1.97 [+0.23, +3.75] |
| SDM − IHM, paired per meeting | +2.86 [+2.39, +3.40] | +3.46 [+2.50, +4.37] | +2.69 [+2.24, +3.19] | +3.44 [+2.44, +4.57] | +2.83 [+2.36, +3.39] | +2.81 [+2.34, +3.39] | +4.41 [+2.60, +6.43] |
| run id (SDM) | `70c72cb4` | `94c45184` | `ffde933d` | `993c6760` | `3a04d861` | `00b82017` | `32122096` |

\* Ward (threshold 8, min_cluster_size 12) is the tune-selected setting from closeout 3 section 3 (IHM run `ef90e3f5`). It was added after closeout 2's ticket was written, so it is not in the ticket's list of methods. The 19.9 is from the repo README benchmark table and its scoring settings are not stated there; details and the other SDM caveats are under "Closeout 2 — SDM" below.

Notes on the table:
- Settings: agglomerative `AGG-1a-07` (centroid, min_cluster_size=12, threshold=0.75); HDBSCAN `HDB-2b-04` (leaf, min_cluster_size=40, min_samples=0); spectral `SPC-3a-03` (pruning=0.20, max_speakers=10); nearest centroid and kNN (k=25) are refinement strategies applied to the VBx clustering. Source: `runs/sweeps/clustering-benchmark/SWEEP_INDEX.md`, [[Assignment Classifier Results]].
- Published figures (17.0 IHM, 19.9 SDM) are quoted from the repo `README.md:84-101` benchmark table ("Benchmark (last updated in 2025-09)", community-1 column), **not** from the Hugging Face model card, which was not fetched. That table does not state collar or overlap handling; "collar 0, overlap scored" is an inference from IHM 17.0 matching our 17.05, not a quote.
- Per-cell sources, CSV paths and the commands that produced the intervals are in "Bootstrap intervals" below. Every point estimate here was recomputed from the per-file CSV and matches the run manifest's `summary.der` to 4 decimals (see that section).
- The interval on a single absolute DER (about ±1.7 points) is wide because the 16 meetings differ a lot in difficulty (10.05 to 23.17 for VBx). The paired Δ intervals are narrower for methods that differ from VBx only on a few voices (nearest centroid, kNN), and wide for methods that re-cluster (spectral).
- **Footnote — September all-voices variant (16.79).** Run `20260917T054753Z-32241285`, DER 16.79 [14.96, 18.53]; Δ vs VBx −0.26 [−0.39, −0.14]. It is not a column. Its scope: `harness/refinement.py:33-66` (the issue text cites 33-69; the function itself ends at line 66, 67-69 are a blank line and the next constant):
  ```python
  def nearest_centroid(embeddings, hard_clusters, soft_clusters, centroids, segmentations):
      ...
      similarity = embeddings_norm @ centroids_norm.T
      nearest = np.argmax(similarity, axis=-1)
      return nearest.reshape(num_chunks, local_num_speakers)
  ```
  It returns a label for **every** `(chunk, local_speaker)` pair by plain argmax over centroids, including the clean voices that the clustering had already placed (and inactive slots). `nc_target` (`harness/assignment_strategies.py:100-101` with `_apply_on_targets` at `:94-97`) relabels only the held-back target pairs: `out = m.hard_clusters.copy(); out[m.target] = decided[m.target]`. The paired difference between the two variants is −0.10 [−0.16, −0.05] (16.79 minus 16.89). Closeout 3 §5 reran it on the current cache (run `20261008T044106Z-868efc2d`, 16/16 intermediate hits): `summary.der` 0.16792363760650816, identical to September, and identical per meeting (max abs difference 0.0 over 16).

---

## Classifier results

Transcribed from [[Assignment Classifier Results]]; full table in `runs/sweeps/assignment-classifier/REPORT.md`, ledger `RUN_INDEX.md`. Test, 16 meetings; DER in %, Conf in s. Pair accuracy is correct / labelled target pairs; "correct" is the oracle's dominant-reference-speaker label (see the source doc for why it is not DER-aligned on overlapped voices).

### Pair accuracy by overlap bin, DER and ΔDER

| Clustering | Strategy | Pair acc [CI] | Acc by overlap bin (0 / ≤.25 / ≤.5 / >.5) | DER | Conf | `der_overlap_assigned` | ΔDER [CI] |
|---|---|---|---|---|---|---|---|
| VBx | as-is (identity) | 44.2 [40.8, 47.3] | 63.8 / 72.1 / 70.1 / 31.2 | 17.05 | 1212 | 0.2195 | – |
| VBx | nearest centroid (`nc_target`) | 67.2 [65.5, 69.1] | 78.2 / 82.1 / 79.3 / 60.3 | 16.89 | 1164 | 0.2172 | −0.16 [−0.26, −0.06] |
| VBx | one-per-chunk NC (`constrained_target`) | 44.2 | = as-is | 17.05 | 1212 | 0.2195 | 0 (identical by construction) |
| VBx | nearest neighbours k=25 | 68.6 [66.9, 70.4] | 80.2 / 83.3 / 80.8 / 61.5 | 16.85 | 1151 | 0.2163 | −0.20 [−0.35, −0.06] |
| VBx | trees, argmax | 72.1 [70.6, 73.9] | 79.3 / 84.3 / 81.2 / 67.2 | 17.16 | 1247 | 0.2307 | +0.11 [−0.08, +0.33] |
| VBx | MLP, argmax | 72.7 [71.1, 74.6] | 80.0 / 83.7 / 80.8 / 68.0 | 17.22 | 1265 | 0.2342 | +0.17 [−0.01, +0.36] |
| VBx | MLP + one-per-chunk | 45.6 [42.7, 48.5] | 65.7 / 73.1 / 71.9 / 32.4 | 16.91 | 1169 | 0.2125 | −0.14 [−0.25, −0.03] |
| VBx | pooled trees + one-per-chunk | 45.6 | 65.4 / 73.0 / 71.6 / 32.5 | 16.95 | 1182 | 0.2148 | −0.10 [−0.20, 0.00] |
| HDBSCAN | as-is (= nearest centroid) | 72.7 [70.8, 74.7] | 84.2 / 86.7 / 82.4 / 61.9 | 16.86 | 1153 | 0.2132 | – |
| HDBSCAN | one-per-chunk NC | 53.5 [49.9, 56.8] | 72.2 / 79.6 / 74.4 / 33.8 | 16.94 | 1179 | 0.2125 | +0.08 [−0.01, +0.19] |
| HDBSCAN | nearest neighbours k=25 | 72.9 [71.0, 75.0] | 84.8 / 86.9 / 82.4 / 62.1 | 16.93 | 1175 | 0.2162 | +0.07 [0.00, +0.16] |
| HDBSCAN | trees, argmax | 75.5 [73.9, 77.4] | 84.1 / 86.9 / 82.6 / 67.4 | 17.11 | 1231 | 0.2247 | +0.25 [+0.15, +0.37] |
| HDBSCAN | MLP, argmax | 75.9 [74.3, 77.7] | 85.0 / 86.8 / 82.5 / 67.7 | 17.14 | 1239 | 0.2271 | +0.28 [+0.15, +0.42] |
| HDBSCAN | trees + one-per-chunk | 53.9 [50.4, 57.2] | 72.8 / 80.9 / 75.7 / 33.8 | 16.92 | 1172 | 0.2125 | +0.06 [+0.01, +0.13] |

Invariant on every run: MD 2925.0 s, FA 1099.1 s and the region census equal the clustering's identity run, so every DER change is a confusion change. Speaker count is unchanged (MAE/bias 0.250/+0.250 for VBx, 0.188/−0.188 for HDBSCAN).

Prediction recorded in advance and carried across as is: nearest-centroid pair accuracy is lower in high-overlap bins than in the `0` bin. **Held**: VBx 78 → 60 (78.2 → 60.3), HDBSCAN 84 → 62 (84.2 → 61.9).

### Share of the oracle ceiling

VBx only; **no oracle run exists for HDBSCAN**, so no HDBSCAN share is reported. Ceilings: all-pairs oracle 0.97 DER points, overlap-degraded oracle 0.47.

| Strategy (VBx) | Share of 0.97 | Share of 0.47 |
|---|---|---|
| nearest neighbours k=25 | 20% | 42% |
| nearest centroid | 16% | 33% |
| MLP + one-per-chunk | 14% | 30% |
| argmax classifiers | negative | negative |

(Share = −ΔDER / ceiling. Oracle ceilings reproduced with intervals in "Bootstrap intervals".)

### Permutation importance by feature group (dev, trees model)

Drop in dev pair accuracy (points) when the group is permuted jointly, 5 repeats. Source `runs/sweeps/assignment-classifier/feature_importance.csv`.

| Group | Features | VBx | HDBSCAN |
|---|---|---|---|
| Similarity | cosine to centroid, rank, margin, soft score, kNN similarity (k=10) | 31.9 | 42.7 |
| Temporal | label agreement of matched neighbours ±1 / ±5 chunks, candidate activity ±30 s | 2.7 | 1.4 |
| Constraint / size | candidate already taken in chunk, candidate's share of the meeting | 0.8 | 0.0 |
| Quality | overlap fraction, clean ratio, active ratio, embedding norm | 0.5 | 0.2 |

Retraining without a group (dev): without temporal 71.9 / 76.2 (−0.2 / −0.1 points); without similarity 57.8 / 50.1.

### Spot-checks against `REPORT.md`

Three cells were compared with `runs/sweeps/assignment-classifier/REPORT.md` (or its source CSV where REPORT.md does not carry the quantity):

| # | Cell in the Docs page | Docs value | Source value | Result |
|---|---|---|---|---|
| 1 | VBx `nc_target`, acc by overlap bin (REPORT.md line 9) | 78.2 / 82.1 / 79.3 / 60.3 | 78.2 / 82.1 / 79.3 / 60.3 | match |
| 2 | HDBSCAN trees (`classifier:hdbscan-hgb`) ΔDER [CI] (REPORT.md line 23) | +0.25 [+0.15, +0.37] | +0.25 [+0.15, +0.37] | match |
| 3 | VBx kNN k=25 oracle-ceiling share (REPORT.md line 13) | 20% / 42% | 20% / 42% | match |

Additionally, the importance table is **not** in REPORT.md (no importance text there). It was checked against `feature_importance.csv`: `similarity` VBx `acc_drop_mean` 0.3194 (doc 31.9), HDBSCAN 0.4273 (doc 42.7): match.

A cell-by-cell comparison of all 14 rows of the results table (accuracy, bins, DER, Conf, `der_overlap_assigned`, ΔDER) against REPORT.md found no value mismatch. The only differences are formatting: the doc writes `[0.00, +0.16]` where REPORT.md writes `[+0.00, +0.16]` (HDBSCAN kNN), and the `constrained_target` VBx row is abbreviated in the doc ("= as-is") and has no separate REPORT.md row at that spelling.

---

## Bootstrap intervals

Every interval below was computed from the named per-file CSV, 16 meetings, 2000 draws, seed 0, by `scratchpad/closeout1_bootstrap.py` (path under Implementation Notes in [[closeout-1-analysis]]). 

**Function reused:** `harness/assignment_runs.py:150` `bootstrap(values_a, values_b, stat, draws=BOOT_DRAWS, seed=BOOT_SEED)`, with `BOOT_DRAWS, BOOT_SEED = 2000, 0` at `:42`, `stat = _der` (`:161`) and per-meeting columns from `per_file()` (`:143`). It draws `rng.integers(0, n, n)` per draw from `np.random.default_rng(0)`, so every call resamples the same meeting indices (over `uri` sorted alphabetically) and any two intervals are paired. Absolute intervals use the repo's own trick for the level interval (`point_stats`, `:169`, zeros at `:181`): the baseline arm is all zeros / ones, so `stat(b) − stat(a)` is `DER` itself.

**Which function the original clustering benchmark used, and why it was not reused:** `harness/clustering_benchmark_report.py:57` `paired_bootstrap_delta` bootstraps over the **8-meeting report half** (`half="report"`, with `assert len(uris) == 8`), not over 16 meetings, and uses `rng.choice(uris)`. Its DER aggregation (`_der`, `:52`) is the same total-error / total-duration quantity. The 16-meeting convention with `rng.integers` is the one in the classifier ledger, so that function was reused. Consequence: the Δ intervals for agglomerative, HDBSCAN and spectral below are **new 16-meeting intervals**, not the report-half intervals quoted in the clustering benchmark.

**Sanity check that the reused function is the ledger's function:** recomputing Δ vs VBx for `nc_target` and kNN k=25 gives [−0.26, −0.06] and [−0.35, −0.06], identical to `RUN_INDEX.md`.

**Aggregation and reproduction of the headline:** the point estimate in every row is recomputed from the CSV as total error / total reference speech; it equals the run manifest's `summary.der` to 4 decimals in every row (column "manifest"). Nothing failed to reproduce.

| Result | Run id | Per-file CSV (exact path) | DER (CSV) | DER (manifest `summary.der`) | Absolute 90% CI | Δ vs VBx baseline (paired) |
|---|---|---|---|---|---|---|
| VBx baseline | `20261003T044356Z-0fa3b41d` | `runs/sweeps/assignment-classifier/test_runs/vbx-identity/per_file.csv` | 17.0485 | 17.0485 | [15.31, 18.74] | – |
| Agglomerative `AGG-1a-07` | `20261003T003611Z-da03276c` | `runs/sweeps/clustering-benchmark/all_points_per_file.csv`, rows `point_id=AGG-1a-07` (16 rows) | 16.8678 | 16.8678 | [14.99, 18.68] | −0.181 [−0.43, +0.08] |
| HDBSCAN `HDB-2b-04` | `20261003T012018Z-b15caf86` | `runs/sweeps/clustering-benchmark/all_points_per_file.csv`, rows `point_id=HDB-2b-04` | 16.8551 | 16.8551 | [15.01, 18.68] | −0.193 [−0.55, +0.13] |
| Spectral `SPC-3a-03` | `20261003T012326Z-fda86985` | `runs/sweeps/clustering-benchmark/all_points_per_file.csv`, rows `point_id=SPC-3a-03` | 17.1600 | 17.1600 | [15.27, 19.05] | +0.111 [−0.45, +0.87] |
| Nearest centroid `nc_target` | `20261003T050432Z-703683ce` | `runs/sweeps/assignment-classifier/test_runs/vbx-nc_target/per_file.csv` | 16.8927 | 16.8927 | [15.09, 18.64] | −0.156 [−0.26, −0.06] |
| kNN k=25 (VBx) | `20261003T055818Z-e2ef8a62` | `runs/sweeps/assignment-classifier/test_runs/vbx-knn_target_k25/per_file.csv` | 16.8499 | 16.8499 | [15.02, 18.63] | −0.199 [−0.35, −0.06] |
| **NC all-voices, September (variant)** | `20260917T054753Z-32241285` | `per_file_nearest_centroid_t5.csv` (repo root; see note) | 16.7924 | 16.7924 | [14.96, 18.53] | −0.256 [−0.39, −0.14] |
| Oracle all_pairs (ceiling 0.97) | `20260916T233858Z-1db2ba34` | `per_file_oracle_all_pairs.csv` (repo root; see note) | 16.0807 | 16.0807 | [14.37, 17.76] | −0.968 [−1.26, −0.72] |
| Oracle overlap_degraded (ceiling 0.47) | `20260917T005714Z-c93fb9f9` | `per_file_oracle_overlap_degraded.csv` (repo root; see note) | 16.5795 | 16.5795 | [14.94, 18.17] | −0.469 [−0.64, −0.30] |
| HDBSCAN identity (classifier ledger) | `20261003T044452Z-b96eddd3` | `runs/sweeps/assignment-classifier/test_runs/hdbscan-identity/per_file.csv` | 16.8551 | 16.8551 | [15.01, 18.68] | −0.193 [−0.55, +0.13] (same numbers as `b15caf86`, as expected: same clustering) |

Notes:
- **Recorded headline (the 17.05 / 16.87 / 16.86 / 17.16 / 16.89 / 16.85 / 16.79 values) reproduces from the CSVs at the decimals shown.** VBx baseline 17.0485 also matches `baseline_reference.json` `metrics.all.der` 0.17048543579940634, and the VBx baseline rows in `all_points_per_file.csv` (`point_id=VBX-baseline`) have max per-file DER difference 0.0 from the classifier work's `vbx-identity` CSV. The Δ for kNN reads −0.199 here and −0.20 elsewhere (rounding).
- **Where the September run's per-file CSV came from.** The manifest `runs/20260917T054753Z-32241285.json` holds only summary totals, not per-meeting rows, and the CSV is not labelled with a run id. The repo-root file `per_file_nearest_centroid_t5.csv` (16 rows) was identified as that run's output because its sums equal the manifest exactly: MD 2925.0495, FA 1099.0744, Conf 1133.4700 (manifest `summary.missed_detection`, `false_alarm`, `confusion`: difference 0.0), and its DER recomputes to 0.16792363760650816, the manifest's `summary.der` to every digit. Identification is by value match, not by a recorded link.
- **Oracle CSVs** (repo root) are matched to the manifests by DER only (the oracle manifests do not carry MD/FA/Conf totals): 16.0807 and 16.5795 match `summary.der` of `1db2ba34` and `c93fb9f9` to 4 decimals. Same caveat: matched by value.
- **Scope difference for `32241285` vs `nc_target`:** see the footnote under the cross-condition table (`refinement.py:33-66` relabels every pair; `assignment_strategies.py:94-101` only target pairs). Paired difference between the two variants: −0.100 [−0.163, −0.046].
- **Reading of the intervals.** Only nearest centroid, kNN, the September variant and the oracles have a paired interval excluding 0 against VBx; agglomerative, HDBSCAN and spectral cross 0. All gains for the non-oracle methods are below the project's 0.3 resolution; the largest non-oracle point estimate, the September variant, is −0.26. Note the 16-meeting HDBSCAN interval [−0.55, +0.13] is wider than any single-meeting-driven story would suggest: see the IS1009b analysis below. The oracle ceilings are the only results that clear 0.3 clearly, and their lower bound on the overlap-degraded ceiling (−0.30) touches it.
- Missing CSVs: none. All ten rows have a 16-row per-file CSV.

---

## Convergent failure analysis

Question: do the clustering methods fail on the same meetings? Data: `runs/sweeps/clustering-benchmark/all_points_per_file.csv` (rows for `VBX-baseline`, `AGG-1a-07`, `HDB-2b-04`, `SPC-3a-03`; 16 meetings each); script `scratchpad/closeout1_convergent.py` and `closeout1_confusion_only.py`; no audio, no GPU. Meeting properties from the reference `~/Code/AMI-diarization-setup/harness-data/IHM/test.rttm` and `test.uem` (these are descriptive properties of the test set; nothing was selected or tuned using them).

### Per-meeting DER (%)

Per-file DER from the CSV `der` column ×100 (error / that meeting's reference speech).

| Meeting | VBx | Agglomerative (`da03276c`) | HDBSCAN (`b15caf86`) | Spectral (`fda86985`) |
|---|---|---|---|---|
| EN2002a | 21.90 | 21.84 | 22.07 | 21.82 |
| EN2002b | 20.74 | 22.52 | 20.79 | 20.54 |
| EN2002c | 17.54 | 17.44 | 17.55 | 17.40 |
| EN2002d | 23.17 | 23.42 | 23.45 | 23.27 |
| ES2004a | 18.76 | 18.11 | 18.14 | 18.01 |
| ES2004b | 12.61 | 11.79 | 11.79 | 11.96 |
| ES2004c | 12.35 | 12.10 | 12.11 | 11.91 |
| ES2004d | 18.67 | 18.59 | 18.59 | 18.91 |
| IS1009a | 19.65 | 19.61 | 23.70 | 23.57 |
| IS1009b | 15.19 | 13.85 | 12.71 | 12.85 |
| IS1009c | 10.05 | 9.84 | 9.85 | 9.91 |
| IS1009d | 16.45 | 16.03 | 16.12 | 22.49 |
| TS3003a | 17.14 | 17.50 | 18.95 | 18.95 |
| TS3003b | 11.97 | 11.46 | 11.38 | 11.38 |
| TS3003c | 13.34 | 12.99 | 13.06 | 12.98 |
| TS3003d | 21.07 | 20.56 | 20.39 | 20.29 |
| unweighted mean | 16.91 | 16.73 | 16.92 | 17.27 |

(The unweighted mean differs from the pooled DER in the table above, as it must.)

### Rank correlation between methods

Spearman rank correlation of the 16 per-meeting DERs:

| | VBx | Agglo | HDBSCAN | Spectral |
|---|---|---|---|---|
| VBx | 1.00 | 0.98 | 0.94 | 0.85 |
| Agglo | 0.98 | 1.00 | 0.96 | 0.86 |
| HDBSCAN | 0.94 | 0.96 | 1.00 | 0.91 |
| Spectral | 0.85 | 0.86 | 0.91 | 1.00 |

Mean of the six off-diagonal values: 0.92. Pearson gives the same picture (0.91 to 0.99).

**Important caveat before reading this as "shared clustering failure".** MD and FA come from segmentation and are **identical for all four methods on every meeting** (max absolute difference across methods: 0.0 s for both, checked in `closeout1_confusion_only.py`). They make up 76.9% of VBx's total error (4024 s of 5236 s). A large part of the per-meeting DER ranking is therefore the shared segmentation error, which would correlate by construction (VBx total DER against MD+FA rate alone: Spearman 0.92). The table above cannot separate "clustering fails on the same meetings" from "segmentation fails on the same meetings". The cleaner test uses only the part clustering can change, the confusion rate (confusion seconds / reference speech):

| Confusion rate (% of ref speech), Spearman | VBx | Agglo | HDBSCAN | Spectral |
|---|---|---|---|---|
| VBx | 1.00 | 0.97 | 0.90 | 0.86 |
| Agglo | 0.97 | 1.00 | 0.96 | 0.92 |
| HDBSCAN | 0.90 | 0.96 | 1.00 | 0.94 |
| Spectral | 0.86 | 0.92 | 0.94 | 1.00 |

Mean off-diagonal 0.92, essentially unchanged. VBx's confusion rate against its MD+FA rate: Spearman 0.59, so the confusion ranking is only partly the same ranking as the segmentation error. The confusion rates themselves are in `scratchpad/closeout1_confusion_only.py` output (e.g. IS1009a: 5.52 / 5.49 / 9.57 / 9.45; IS1009d: 4.60 / 4.18 / 4.26 / 10.63).

A third view, correlation between the methods' *changes* from VBx (Δ confusion rate, points): Agglo–HDBSCAN 0.95, HDBSCAN–Spectral 0.79, Agglo–Spectral 0.71. The methods also *move in the same direction* relative to VBx on the same meetings.

### Worst quartile and disagreement

Worst quartile = four highest DER meetings per method.

| Method | Worst four (highest first) |
|---|---|
| VBx | EN2002d, EN2002a, TS3003d, EN2002b |
| Agglomerative | EN2002d, EN2002b, EN2002a, TS3003d |
| HDBSCAN | IS1009a, EN2002d, EN2002a, EN2002b |
| Spectral | IS1009a, EN2002d, IS1009d, EN2002a |

- In the worst quartile for **all four**: **EN2002a, EN2002d**. In it for at least three: EN2002a, EN2002b, EN2002d.
- Pairwise overlap of the worst-four sets (out of 4): VBx–Agglo 4, VBx–HDBSCAN 3, Agglo–HDBSCAN 3, HDBSCAN–Spectral 3, VBx–Spectral 2, Agglo–Spectral 2.
- In the **best** four for all four methods: ES2004b, ES2004c, IS1009c, TS3003b.
- **Where the methods disagree most** (range of per-meeting DER across the four methods, and range of rank):

| Meeting | DER range across methods (pts) | Rank range (1 = worst, of 16) | What differs |
|---|---|---|---|
| IS1009d | 6.46 | 7 (rank 3 to 10) | spectral 22.49 vs 16.0 to 16.5 for the other three |
| IS1009a | 4.09 | 4 (rank 1 to 5) | HDBSCAN 23.70 and spectral 23.57 vs 19.6 for VBx and agglo |
| IS1009b | 2.48 | 1 | all four are in the better half; HDBSCAN 12.71 vs VBx 15.19 |
| EN2002b | 1.98 | 3 | agglo 22.52 vs 20.5 to 20.8 for the others |
| TS3003a | 1.81 | 3 | HDBSCAN and spectral 18.95 vs 17.14 to 17.50 |

All large disagreements are single-method excursions on one or two meetings; they come from the IS1009 and TS3003a sessions.

### `IS1009b`

IS1009b is **not** a bad-DER outlier for any method: its rank among the 16 is 11 (VBx), 11 (Agglo), 12 (HDBSCAN), 12 (Spectral), i.e. it is in the easier half everywhere. What is unusual is the **gain** relative to VBx. It is the single largest improvement over VBx for each of the three alternatives (rank 1 of 16 each):

| Method | IS1009b DER | Δ vs VBx (pts) |
|---|---|---|
| VBx | 15.19 | – |
| Agglomerative | 13.85 | −1.34 |
| HDBSCAN | 12.71 | −2.48 |
| Spectral | 12.85 | −2.34 |

So it is an outlier for **every** alternative method (largest gain in each case), though the size of the gain differs (agglomerative −1.34 vs −2.4 for the other two). It is not only HDBSCAN. All three also lose heavily somewhere else (HDBSCAN and spectral +4 on IS1009a; spectral +6 on IS1009d; HDBSCAN and spectral +1.8 on TS3003a; agglo +1.8 on EN2002b). Rank correlations between methods without IS1009b (n=15): VBx–Agglo 0.98, VBx–HDBSCAN 0.93, VBx–Spectral 0.82, Agglo–HDBSCAN 0.95, Agglo–Spectral 0.84, HDBSCAN–Spectral 0.90: essentially unchanged from the n=16 values, so it does not drive the correlations.

### What predicts baseline difficulty

Properties of each meeting, computed from `IHM/test.rttm` and `IHM/test.uem` (script `closeout1_convergent.py`, `load_rttm`/`load_uem` from `pyannote.database.util`; annotation cropped to the UEM):
- **speaker count**: number of distinct speaker labels in the RTTM for that meeting;
- **duration**: UEM extent length in seconds (the scored region), not the audio file length;
- **overlap fraction**: duration of the annotation's overlapped-speech timeline (`Annotation.get_overlap()`, time with two or more active reference speakers) divided by the duration of its speech support (`get_timeline().support()`, union of all speech). It is a fraction of **speech time**, not of file duration;
- **number of turns**: number of RTTM rows (`itertracks()` count) for the meeting, as annotated, without merging adjacent same-speaker segments.

| Meeting | speakers | duration (s) | speech (s) | overlap fraction | turns |
|---|---|---|---|---|---|
| EN2002a | 4 | 2142.7 | 1894.9 | 0.274 | 746 |
| EN2002b | 4 | 1786.8 | 1476.8 | 0.264 | 490 |
| EN2002c | 3 | 2972.3 | 2604.9 | 0.242 | 635 |
| EN2002d | 4 | 2209.9 | 1928.6 | 0.300 | 685 |
| ES2004a | 4 | 1049.4 | 787.3 | 0.158 | 260 |
| ES2004b | 4 | 2345.5 | 2014.2 | 0.101 | 467 |
| ES2004c | 4 | 2334.4 | 2012.7 | 0.107 | 497 |
| ES2004d | 4 | 2222.3 | 1718.9 | 0.151 | 602 |
| IS1009a | 4 | 838.8 | 604.9 | 0.136 | 195 |
| IS1009b | 4 | 2052.3 | 1778.0 | 0.101 | 389 |
| IS1009c | 4 | 1820.8 | 1503.0 | 0.050 | 291 |
| IS1009d | 4 | 1944.5 | 1547.8 | 0.105 | 507 |
| TS3003a | 4 | 1505.6 | 978.1 | 0.046 | 242 |
| TS3003b | 4 | 2210.3 | 1739.3 | 0.043 | 404 |
| TS3003c | 4 | 2570.0 | 1801.8 | 0.050 | 385 |
| TS3003d | 4 | 2618.2 | 1853.6 | 0.107 | 698 |

Spearman correlation of per-meeting DER with each property (n=16; two-sided p in brackets for the baseline only):

| Property | VBx baseline | Agglo | HDBSCAN | Spectral | VBx confusion rate only |
|---|---|---|---|---|---|
| speaker count | −0.03 [0.92] | +0.03 | +0.03 | +0.08 | +0.03 |
| overlap fraction | **+0.82** [<0.01] | +0.80 | +0.70 | +0.61 | +0.78 |
| duration | −0.15 [0.58] | −0.21 | −0.29 | −0.34 | −0.33 |
| number of turns | +0.44 [0.08] | +0.41 | +0.24 | +0.26 | +0.39 |

Speaker count has almost no information here: 15 of the 16 meetings have 4 speakers, one has 3 (EN2002c, DER 17.54 vs mean 16.87 for 4-speaker meetings), so the correlation is not informative (the table's value is essentially one point). Overlap fraction is the one property with a strong correlation with baseline DER (0.82, and it holds for the confusion rate alone, 0.78). Turn count has a weak positive association and duration a weak negative one; neither is distinguishable from zero at n=16.

### Reading

Two readings were fixed in advance: **strongly correlated methods** (shared bottleneck upstream of clustering) or **weakly correlated** (distinct failure modes that average to a similar DER).

**The first reading applies by the numbers**: pairwise Spearman 0.85 to 0.98 (mean 0.92) on total DER; 0.86 to 0.97 (mean 0.92) on the confusion rate alone, which removes the shared MD/FA; two meetings (EN2002a, EN2002d) in the worst quartile of all four, three meetings in at least three; the same four best meetings for all four methods; and the methods' changes relative to VBx are themselves correlated (0.71 to 0.95).

What this does and does not show:
- It is consistent with a shared limit upstream of clustering, and baseline difficulty tracks the overlap fraction of the meeting (0.82; EN2002 meetings with 24 to 30% overlapped speech are the hardest for all four). It does not isolate where upstream. Meetings with more overlap are harder in every component, including segmentation, which is shared by construction.
- With n=16 meetings from only four meeting groups (EN2002, ES2004, IS1009, TS3003, four meetings each), the meetings are not independent draws: sessions share speakers and rooms. The correlation values are descriptive. No p-values are given for the method–method matrix.
- The methods are not interchangeable: they diverge sharply on individual meetings (spectral +6.0 confusion points on IS1009d; HDBSCAN and spectral about +4 on IS1009a; all three alternatives −1.3 to −2.5 on IS1009b). These excursions average out to the near-identical pooled DERs but are not noise at the meeting level. HDBSCAN and spectral move together on two of them (IS1009a and TS3003a); no mechanism was tested here.
- This analysis was not designed to rule out that the alternatives fail on the same meetings for different reasons; it shows only that where they fail is largely the same.

---

## Closeout 3 — IHM gaps

(To be filled by [[closeout-3-ihm-gaps]].)

### §2 VBx hyperparameter sweep

Source: `runs/sweeps/vbx-sweep/` (`points.csv`, `points.jsonl`, `all_points_per_file.csv`, `analysis_output.txt`; driver `harness/vbx_sweep.py`). AMI IHM test, 16 meetings, collar 0, overlap scored, identity refinement, shipped segmentation (shared warm cache). DER in %. Tune half = meetings ending a/c, report half = b/d. Curves only: no setting is selected.

**Grid.** 10 Fb x 6 Fa x 6 threshold = 360 points (shipped values forced into each axis).
- Fb: 0.01, 0.02258, 0.05097, 0.1151, 0.2598, 0.8, 1.324, 2.989, 6.748, 15.0 (log-spaced, 0.5865 replaced by 0.8)
- Fa: 0.01, 0.0219, 0.07, 0.1046, 0.2288, 0.5 (log-spaced, 0.0478 replaced by 0.07)
- threshold: 0.5, 0.56, 0.6, 0.68, 0.74, 0.8 (linear, 0.62 replaced by 0.6)

**Completion.** 279 points scored, 81 crashed inside `src/` (not editable): 78 are `OverflowError ... out of bounds for int8` (VBx produced more than 127 clusters, `clustering.py:138`), 3 are `ValueError: negative dimensions are not allowed` (`speaker_diarization.py:522`, all at Fa=0.5). The crashes sit in the corners of the grid where speaker count explodes or collapses (map in the pivot below, NaN = crashed). Crashed points have no scores.

**Fb axis (Fa=0.07, threshold=0.6)**, figure `runs/sweeps/vbx-sweep/fb_axis.png`:

![[fb_axis.png]]

| Fb | run | DER all | tune | report | MD s | FA s | Conf s | mean hyp speakers | count MAE | signed bias | exact/16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | crashed (int8 overflow, 137 clusters) | | | | | | | | | | |
| 0.02258 | 688567de | 50.95 | 50.56 | 51.28 | 2925.0 | 1099.1 | 11623.2 | 52.500 | 48.562 | +48.562 | 0 |
| 0.05097 | 0c203dc7 | 32.13 | 33.03 | 31.35 | 2925.0 | 1099.1 | 5844.2 | 31.125 | 27.188 | +27.188 | 0 |
| 0.1151 | acc434cd | 23.65 | 22.83 | 24.36 | 2925.0 | 1099.1 | 3240.1 | 16.188 | 12.250 | +12.250 | 0 |
| 0.2598 | 54bcb6ff | 19.94 | 19.21 | 20.57 | 2925.0 | 1099.1 | 2100.1 | 7.625 | 3.688 | +3.688 | 2 |
| 0.8 (shipped) | 1e0eb477 | 17.05 | 16.26 | 17.73 | 2925.1 | 1099.4 | 1212.0 | 4.188 | 0.250 | +0.250 | 12 |
| 1.324 | 7c74ccb4 | 16.94 | 16.38 | 17.43 | 2925.0 | 1099.1 | 1179.7 | 3.875 | 0.188 | -0.062 | 14 |
| 2.989 | cce6f4dc | 16.92 | 16.38 | 17.38 | 2925.0 | 1099.1 | 1171.4 | 3.812 | 0.125 | -0.125 | 15 |
| 6.748 | c8542429 | 17.01 | 16.56 | 17.39 | 2925.0 | 1099.1 | 1199.7 | 3.750 | 0.188 | -0.188 | 14 |
| 15.0 | c16cfc33 | 19.77 | 17.40 | 21.82 | 2925.0 | 1099.1 | 2046.8 | 3.375 | 0.562 | -0.562 | 9 |

Mean reference speakers per meeting 3.9375. Missed detection and false alarm come from shared segmentation and do not move; only confusion does. Region census, JER and overlap DERs per point are in `points.csv`.

**Fa axis (Fb=0.8, threshold=0.6)**

| Fa | DER all | tune | report | mean hyp | count MAE | signed bias | exact/16 |
|---|---|---|---|---|---|---|---|
| 0.01 | 37.05 | 33.05 | 40.50 | 2.750 | 1.188 | -1.188 | 4 |
| 0.0219 | 20.97 | 19.25 | 22.47 | 3.250 | 0.688 | -0.688 | 7 |
| 0.07 (shipped) | 17.05 | 16.26 | 17.73 | 4.188 | 0.250 | +0.250 | 12 |
| 0.1046 | 19.61 | 19.14 | 20.01 | 6.438 | 2.500 | +2.500 | 3 |
| 0.2288 | 31.07 | 29.58 | 32.36 | 30.750 | 26.812 | +26.812 | 0 |
| 0.5 | crashed (ValueError, negative dimensions) | | | | | | |

**Threshold axis (Fa=0.07, Fb=0.8)**

| threshold | DER all | tune | report | mean hyp | count MAE | signed bias | exact/16 |
|---|---|---|---|---|---|---|---|
| 0.50 | 17.86 | 16.26 | 19.25 | 4.312 | 0.375 | +0.375 | 10 |
| 0.56 | 17.05 | 16.26 | 17.73 | 4.188 | 0.250 | +0.250 | 12 |
| 0.60 (shipped) | 17.05 | 16.26 | 17.73 | 4.188 | 0.250 | +0.250 | 12 |
| 0.68 | 17.04 | 16.26 | 17.71 | 4.250 | 0.312 | +0.312 | 12 |
| 0.74 | 17.04 | 16.26 | 17.72 | 4.250 | 0.312 | +0.312 | 12 |
| 0.80 | 18.08 | 16.26 | 19.66 | 4.250 | 0.438 | +0.312 | 10 |

The tune-half DER is 16.26 at every threshold (the whole threshold effect on this axis is in the report half). Thresholds 0.56 and 0.6 give the same DER and counts at two decimals but different final cache keys (their confusion differs at 1212.15 s vs 1211.97 s; the shipped point is the one carrying the cached-RTTM rounding, see checks).

**Grid summary (279 scored points).**
- All-16 DER ranges from 16.89 to 59.90 (median 25.39). 36 points are within 0.3 DER of shipped; 0 are better than shipped by more than 0.3; 243 are worse by more than 0.3.
- Across the 279 points where VBx converged, the largest gain over shipped is 0.16 DER points (Fa=0.1046, Fb=2.989, thr=0.74: all 16.89, tune 16.27, report 17.43). No point gains 1 point or more, so the suspected-bug trigger did not fire.
- **The other 81 of 360 points (22.5%) crashed inside pyannote, not the harness, and are excluded from that claim rather than counted as null results.** All 81 are one mechanism: VBx proposed more than 127 clusters and the result overflowed an `int8` counter (`clustering.py:138`, 78 cases; or the same wraparound reached one line earlier at `speaker_diarization.py:522`, 3 cases, all Fa=0.5) — not two bugs, one bug caught at two points. They sit in a single contiguous band of degenerate, heavily over-splitting settings (low Fb at every Fa, worst at low Fa; mean hypothesis speaker count already 16–50+ at the crash boundary against a true count of 3.94) — not scattered, and not adjacent to the best region: the minimum grid-index (Manhattan) distance from any crashed point to either of the two best-scoring points (16.89 DER, at Fa=0.1046/Fb=2.989/thr=0.74 and Fa=0.07/Fb=1.324/thr=0.74) is 6 steps, with every one of those two points' immediate neighbours scored, not crashed. So the claim is restated, not narrowed by an unmeasured hole next to the optimum: evidence table `runs/sweeps/vbx-sweep/crashed_points.csv` (81 rows), exceptions in `crash_exceptions.txt`.
- Where each half is lowest, stated per half and not as a recommendation: all-16 lowest 16.89 at (Fa 0.1046, Fb 2.989, thr 0.74); tune-half lowest 16.17 at (Fa 0.07, Fb 1.324, thr 0.5), whose report half is 18.19; report-half lowest 17.38 at (Fa 0.1046, Fb 6.748, thr 0.56). These are three different points, and the gaps between them are far below the 0.3 resolution.
- Signed count bias: 170 points over-count, 106 under-count, 3 zero.
- Fa and Fb act together. All-16 DER (%), threshold 0.6, NaN = crashed:

| Fa \ Fb | 0.02258 | 0.05097 | 0.1151 | 0.2598 | 0.8 | 1.324 | 2.989 | 6.748 | 15.0 |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | crash | crash | 34.98 | 32.72 | 37.05 | 40.08 | 48.51 | 48.37 | crash |
| 0.0219 | 39.72 | 38.03 | 21.56 | 17.85 | 20.97 | 22.52 | 27.78 | 41.78 | 56.95 |
| 0.07 | 50.95 | 32.13 | 23.65 | 19.94 | 17.05 | 16.94 | 16.92 | 17.01 | 19.77 |
| 0.1046 | crash | crash | 31.72 | 26.09 | 19.61 | 18.13 | 16.94 | 16.92 | 17.01 |
| 0.2288 | crash | crash | crash | crash | 31.07 | 25.50 | 19.34 | 17.86 | 16.93 |
| 0.5 | crash | crash | crash | crash | crash | 37.45 | 29.46 | 19.71 | 18.03 |

  Mean hypothesis speakers at the same cells (threshold 0.6): Fa=0.07 row: 52.50, 31.12, 16.19, 7.62, 4.19, 3.88, 3.81, 3.75, 3.38. Fa=0.2288 row: crash x4, 30.75, 16.56, 6.06, 4.25, 3.88. Fa=0.0219 row: 35.75, 21.38, 6.44, 3.69, 3.25, 3.12, 2.75, 2.19, 2.00. The low-DER band runs along the diagonal where Fa/Fb is moderate, about 0.01 to 0.1 (Fa=0.07 with Fb 0.8-6.7; Fa=0.1046 with Fb 2.99-15; Fa=0.2288 with Fb 6.7-15), matching the source reading that the two enter mainly as a ratio. Higher Fa raises the count at fixed Fb.
- Not monotone off the shipped Fa: at Fa=0.01 the count falls with Fb (2.75 at 0.8, 2.19 at 2.99) but at Fb=15 with thresholds 0.5, 0.56, 0.6 and 0.68 the run crashed with more than 127 clusters (135 to 295). At thresholds 0.74 and 0.8 it ran (mean 2.81 and 2.56 speakers, DER 44.2 and 46.1). Unexplained and not pursued.

**Falsifiability checks** (`runs/sweeps/vbx-sweep/analyze.py`, output in `analysis_output.txt`):
- Distinct final cache keys: 279 scored points, 279 unique final keys. The intermediate key is identical for all (1 unique), as expected since clustering is not part of it.
- Manifest check: for all 279 scored points, `run_config.clustering_config` in `runs/<run_id>.json` equals the requested Fa, Fb, threshold and the manifest's `pipeline_config_id` equals the recorded final key: 279 of 279 match, 0 mismatches.
- Intermediate cache: 278 of 279 printed `16 full hit(s), 0 partial, 0 miss(es), 0 write(s)`. The exception is the shipped point, served whole from the final-RTTM cache (zero tally); no intermediate miss anywhere.
- Printed config lines (`run.log`, the `clustering:` line):
  - `clustering: VBxClustering(Fa=0.07, Fb=0.01, threshold=0.6) (requested: pyannote-default)` (low extreme; this run then crashed on int8 overflow)
  - `clustering: VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6) (requested: pyannote-default)` (shipped)
  - `clustering: VBxClustering(Fa=0.07, Fb=15, threshold=0.6) (requested: pyannote-default)` (high extreme)
  - Also `Fb=0.02258` and `Fb=6.748` at Fa=0.07, thr=0.6 print the matching values.
- Not a no-op: DER and speaker counts differ at every Fb value.
- Shipped baseline: DER 0.17049342382952826 against the known 0.17048543579940634, a drift of 7.99e-06. The shipped key is served from the cached final RTTMs, which are rounded to milliseconds; the known value came from a run that recomputed them. Supporting evidence: this point's missed detection is 2925.115 s while every freshly computed point has 2925.0495 s, per-file DER differs from the older baseline CSVs by at most 2.1e-05, and all 16 per-meeting speaker counts match. The point was run through a 5e-05 tolerance, not the exact equality asked for, so this is flagged for review. Only this point carries the rounding, about 1e-5 DER.

**Prediction: higher Fb gives fewer speakers, monotonically. It held on the Fb axis.** Mean hypothesis speakers across the nine scored Fb values (Fa=0.07, threshold=0.6): 52.50, 31.12, 16.19, 7.63, 4.19, 3.88, 3.81, 3.75, 3.38 for Fb = 0.02258 up to 15.0. Count MAE: 48.56, 27.19, 12.25, 3.69, 0.25, 0.19, 0.13, 0.19, 0.56. Signed bias: +48.56, +27.19, +12.25, +3.69, +0.25, -0.06, -0.13, -0.19, -0.56. The count fell at every step, and the lower Fb=0.01 point crashed from more than 127 clusters, which is the same direction. DER did move: it falls steeply from 50.95 to about 17 by Fb=0.8, is nearly flat at 16.9 to 17.0 for Fb 0.8 to 6.7 (a spread of about 0.13 points, below the 0.3 resolution), then rises to 19.77 at Fb=15 as the count goes to 3.38 against 3.94 true. The monotone-count result is on this one axis; the Fa=0.01, Fb=15 observation above shows it does not hold everywhere on the grid.

### Ward linkage redo

**Result: ward is no longer untested. The gate passed (threshold above 2.0 is accepted and used), and ward at its own scale is competitive with VBx but not better, by the tune-half selection.**

**Gate.** The harness cannot run one meeting, so all 16 were run at threshold 10 and at 1.4 (ward, `min_cluster_size=12`, the benchmark's only ward setting).
- Printed line in `runs/sweeps/ward-redo/WARD-G10/run.log`: `clustering: AgglomerativeClustering(method='ward', min_cluster_size=12, threshold=10) (requested: agglomerative)`.
- Manifest `20261008T094605Z-4dad900c.json` `clustering_config`: `AgglomerativeClustering(method='ward', min_cluster_size=12, threshold=10)`.
- 1.4 is the benchmark's AGG-1d-25 configuration; it reproduced that point's final key (`8fb70981...`) and DER 91.98 (served from the final-RTTM cache).
- Speaker counts differ at 10 vs 1.4 (mean hypothesis count 3.75 against 68.5; DER 16.79 against 91.98):

| meeting | ref | hyp at 10 | hyp at 1.4 |
|---|---|---|---|
| EN2002a | 4 | 4 | 69 |
| EN2002b | 4 | 4 | 57 |
| EN2002c | 3 | 3 | 103 |
| EN2002d | 4 | 4 | 66 |
| ES2004a | 4 | 4 | 35 |
| ES2004b | 4 | 4 | 89 |
| ES2004c | 4 | 4 | 88 |
| ES2004d | 4 | 4 | 72 |
| IS1009a | 4 | 3 | 29 |
| IS1009b | 4 | 4 | 77 |
| IS1009c | 4 | 4 | 62 |
| IS1009d | 4 | 4 | 68 |
| TS3003a | 4 | 2 | 41 |
| TS3003b | 4 | 4 | 94 |
| TS3003c | 4 | 4 | 77 |
| TS3003d | 4 | 4 | 69 |

pyannote's `Uniform(0.0, 2.0)` bound is not enforced by `instantiate`, and `clustering.py:385` passes the value straight to `fcluster`. Nothing under `src/pyannote/` was edited.

**Grid** (min_cluster_size 12, IHM test, 16 meetings; DER in percent; the shipped VBx baseline is 17.05 all / 16.26 tune / 17.73 report from the fresh baseline `per_file.csv`):

| threshold | run | final key | DER all | tune | report | MD s | FA s | Conf s | DER ovl sys | DER ovl asg | JER | regions T&D / T-D / D-T s | count MAE | bias | exact | mean hyp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | e38c6c02 | 8c816573 | 49.23 | 47.03 | 51.14 | 2925.0 | 1099.1 | 11097.7 | 62.5 | 55.3 | 43.8 | 2433 / 1394 / 383 | 6.062 | +6.062 | 0 | 10.000 |
| 5 | 46615ec9 | cbcc27c9 | 37.98 | 36.74 | 39.04 | 2925.0 | 1099.1 | 7639.5 | 53.4 | 44.9 | 35.3 | 2433 / 1394 / 383 | 2.875 | +2.875 | 2 | 6.812 |
| 6 | 3e6b8b4c | aea9f382 | 28.56 | 27.60 | 29.38 | 2925.0 | 1099.1 | 4746.8 | 43.5 | 33.3 | 28.6 | 2433 / 1394 / 383 | 1.438 | +1.438 | 2 | 5.375 |
| 7 | a649e042 | be0358bf | 19.85 | 18.64 | 20.89 | 2925.0 | 1099.1 | 2071.9 | 37.8 | 27.0 | 24.8 | 2433 / 1394 / 383 | 0.562 | +0.312 | 8 | 4.250 |
| 8 | ef90e3f5 | 54836ccd | 17.47 | 16.37 | 18.42 | 2925.0 | 1099.1 | 1341.2 | 33.6 | 22.2 | 24.0 | 2433 / 1394 / 383 | 0.250 | +0.000 | 12 | 3.938 |
| 10 | 8668f7fc | 4d9dde98 | 16.79 | 16.44 | 17.09 | 2925.1 | 1099.3 | 1132.4 | 32.7 | 21.1 | 24.3 | 2433 / 1394 / 383 | 0.188 | -0.188 | 14 | 3.750 |
| 12 | 7e34a167 | 27a3bea9 | 16.79 | 16.44 | 17.09 | 2925.0 | 1099.1 | 1132.5 | 32.7 | 21.1 | 24.3 | 2433 / 1394 / 383 | 0.188 | -0.188 | 14 | 3.750 |
| 14 | d13505f4 | 7855642e | 17.52 | 17.26 | 17.75 | 2925.0 | 1099.1 | 1357.5 | 33.1 | 21.6 | 28.1 | 2433 / 1394 / 383 | 0.375 | -0.375 | 12 | 3.562 |
| 16 | 831af033 | 21dec885 | 18.84 | 17.26 | 20.21 | 2925.0 | 1099.1 | 1762.1 | 34.9 | 23.9 | 30.7 | 2433 / 1394 / 383 | 0.500 | -0.500 | 10 | 3.438 |
| 20 | 2281c475 | a1fb12e4 | 22.51 | 22.07 | 22.89 | 2925.0 | 1099.1 | 2890.5 | 36.2 | 25.5 | 41.6 | 2433 / 1394 / 383 | 0.875 | -0.875 | 5 | 3.062 |

Threshold 10 (`8668f7fc`) was served whole from the final-RTTM cache (same key as the gate run `4dad900c`), so its numbers differ from 12 by rounding only (MD 2925.093 against 2925.050 s). Thresholds 10 and 12 give the same speaker count on every meeting and the same DER to within cache rounding (the partitions themselves were not compared).

**Checks.**
- The ten grid final keys are distinct (programmatic). Each manifest `clustering_config` equals the requested `AgglomerativeClustering(method='ward', min_cluster_size=12, threshold=T)` (programmatic, all 12 manifests).
- Intermediate tally was 16 full hits, 0 partial, 0 misses in 10 of 12 logs. The other two (WARD-G1.4 and WARD-T10) print 0/0/0 because their final keys repeated earlier recorded configurations (AGG-1d-25 and WARD-G10) and were served from the final-RTTM cache; no GPU work happened in them.
- The curve is not flat: DER 49.2 at threshold 4 falls to a floor near 16.8 at 10-12 and rises to 22.5 at 20; mean count 10.0 falls to 3.06. Not a no-op. No point beats the shipped baseline by more than 1 DER point (largest all-16 gain 0.26).

**Selection.** Made on the **tune half only** (meetings ending a/c): lowest tune DER is threshold **8** (16.37), against 16.44 at 10 and 16.44 at 12.
- Report half, threshold 8: DER 18.42 against baseline 17.73; delta **+0.69 points, 90% paired bootstrap [-0.77, +2.28]** (8 meetings, 2000 draws, seed 0). The interval spans zero. Function: a reimplementation of `harness/clustering_benchmark_report.py:57` `paired_bootstrap_delta` that takes tables as arguments; it reproduces that function exactly on HDB-2b-04 (delta -0.005539267379653795, [-0.010829035359314957, -0.0013743556937805383]; asserted equal before use).
- 16 meetings, threshold 8: DER 17.47 against 17.05; delta **+0.42 points, 90% paired interval [-0.40, +1.36]**, from `harness/assignment_runs.py:150` `bootstrap()` (2000 draws, seed 0), for the cross-condition table.
- Post-hoc, not a selection: thresholds 10-12 score 16.79 on all 16 (-0.26 against baseline) and 17.09 on the report half (-0.64), but they were second on the tune half, so they are not the reported result.

**Does the grid bracket the optimum?** Yes on the tune half: the best (8) is interior, with 7 (18.64 tune) and 10 (16.44) either side; it is not at an edge. It sits near the lower side of a flat basin from 8 to 12 (tune 16.37-16.44), so the choice of 8 against 10 is within noise. The grid was not extended.

### Dev and train confirmation

Settings are the ones used on test, unchanged (VBx shipped parameters; `knn_target_k25`; `nc_target`). Each strategy is compared to the VBx identity run of the same split. Interval: 90% paired bootstrap over that split's meetings (2000 draws, seed 0, total error over total reference; function `harness/assignment_runs.py:150`, first checked to reproduce the test `nc_target` interval [-0.26, -0.06]).

| Split | Strategy | Run | DER | Δ vs VBx [90% CI] | Intermediate tally |
|---|---|---|---|---|---|
| dev (18) | VBx identity | `20261003T055158Z-c9fcfc74` | 16.90 | n/a | cold run (baseline) |
| dev (18) | kNN k=25 | `20261008T173307Z-a316d8e4` | 16.82 | -0.07 [-0.21, +0.05] | 18 / 0 / 0 |
| dev (18) | nearest centroid (`nc_target`) | `20261008T173352Z-a76b4c1c` | 16.84 | -0.06 [-0.19, +0.06] | 18 / 0 / 0 |
| train (136) | VBx identity | `20261003T182220Z-55c55a58` | 15.65 | n/a | cold run (baseline) |
| train (136) | kNN k=25 | `20261008T173928Z-c101e97e` | 15.67 | +0.02 [-0.02, +0.07] | 136 / 0 / 0 |
| train (136) | nearest centroid (`nc_target`) | `20261008T174501Z-30766bde` | 15.70 | +0.05 [+0.01, +0.09] | 136 / 0 / 0 |

Baselines are the existing per-file CSVs of the cold runs (not a cached-RTTM replay), so no rounding drift enters the deltas. Test, for reference: kNN -0.20 [-0.35, -0.06], nc_target -0.16 [-0.26, -0.06].

- **Dev: the ordering kNN < nc_target < VBx holds in the point estimates, but it is a weak confirmation.** Both gains are about a third of the test gain, both intervals include zero, and the kNN/nc_target difference (0.01) is far inside noise.
- **Train: the ordering does not hold.** VBx is lowest; kNN is +0.02 (interval includes zero) and nearest centroid +0.05 (interval excludes zero, but the effect is far below the 0.3 resolution). Reported as found.
- **Limit 1, train is probably seen data.** community-1's models were likely trained on AMI train (assumption A7, unverified; VBx is 15.65 here against 16.90 on dev). Train is a consistency check, not an independent test.
- **Limit 2, dev is partly circular, in the weaker way.** k was chosen among {5, 10, 25} on dev **pair accuracy** (not dev DER): `runs/sweeps/assignment-classifier/knn_dev_selection.txt` lists 70.26 / 70.71 / 70.94% for k = 5 / 10 / 25, and the file (18:57) predates the test run of k=25 (18:58). No kNN DER was ever computed on dev before this run, and k=5 and k=10 were never run on test. `nc_target` has no tunable parameter. The caveat therefore stays on the dev result (weaker circularity) and, because the ordering also fails on train, it is not lifted.
- Net: the test gain of kNN and nearest centroid is not reproduced at its size on dev, and not at all on train. The best-of-about-thirteen selection effect on test remains a live reading; it is not ruled out by these runs.

Reproduce: `python runs/sweeps/dev-train-confirm/confirm.py run` then `report` (outputs in `runs/sweeps/dev-train-confirm/`, `results.csv`).

## Closeout 4 — min_duration_off

Setup: VBx shipped clustering (Fa 0.07, Fb 0.8, threshold 0.6), identity refinement, AMI IHM test, 16 meetings, collar 0, overlap not skipped. `min_duration_off` is the segmentation gap-filling step applied after clustering; shipped value 0.0 (step off). Driver `runs/sweeps/min-duration-off/sweep.py`; outputs in the same folder (`points.csv`, `points.jsonl`, `results.json`, figure `md_fa_der_vs_min_duration_off.png`). Every point ran through `run_harness.py --segmentation-param min_duration_off=<v>`; manifests carry `min_duration_off`.

### Falsifiability checks (run first)

- **Check 1, explicit 0.0.** Final key `241ecd865d15...bced` equals the baseline key (vbx-sweep point `VBX-Fa0.07-Fb0.8-T0.6`). Harness `summary.der` = 0.17049342382952828, exactly the cached shipped value; tally `0 full / 0 partial / 0 miss` (served from the final-RTTM cache), per-file CSV byte-identical to the vbx-sweep baseline point. (My driver's first attempt compared a DER recomputed from the per-file CSV, 0.17049342382952826, 2e-17 away, and halted; the comparison was changed to the harness-reported DER with exact equality. Config `072da1a3` therefore has two manifests.) The fresh-computed baseline is 0.17048544 (8e-6 lower, final-RTTM 3-dp rounding).
- **Check 2, 0.5 s.** The harness cannot run a single meeting, so all 16 were run. Missed detection fell on all 16 meetings versus 0.0 (from -4.6 s on ES2004a to -27.3 s on EN2002c), so the override reaches the pipeline.
- A value above the declared range is **not** rejected: `min_duration_off=1.5` instantiated and scored (run `88118b5e`, DER 18.01%, MD 2111 s, FA 2237 s). `Uniform(0.0, 1.0)` is not enforced by `instantiate`. That run is outside the sweep and not eligible for selection.

### VBx sweep

Seconds are summed over 16 meetings. DER is a percentage. Deltas below are against the **fresh-computed** baseline (`runs/sweeps/assignment-classifier/test_runs/vbx-identity/per_file.csv`: all 17.0485, tune 16.2580, report 17.7321); the 0.0 row is the cache-served one (8e-6 above it).

| value (s) | run | DER | tune DER | report DER | MD | FA | Conf | overlap missed | overlap caught | overlap invented | overlap DER sys / assigned | JER | count MAE / bias / exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 (cache) | 072da1a3 | 17.0493 | 16.2587 | 17.7330 | 2925.1 | 1099.4 | 1212.0 | 1393.9 | 2433.1 | 383.1 | 33.39 / 21.95 | 22.19 | 0.25 / +0.25 / 12 |
| 0.05 | b18cc696 | 17.0512 | 16.2552 | 17.7394 | 2918.1 | 1109.1 | 1209.9 | 1389.6 | 2437.4 | 387.3 | 33.36 / 21.96 | 22.18 | 0.25 / +0.25 / 12 |
| 0.1 | dc4b3b72 | 17.0559 | 16.2602 | 17.7440 | 2903.5 | 1130.8 | 1204.2 | 1381.4 | 2445.7 | 397.5 | 33.30 / 21.98 | 22.16 | 0.25 / +0.25 / 12 |
| 0.2 | 75516003 | 17.0603 | 16.2548 | 17.7567 | 2871.8 | 1170.9 | 1197.2 | 1368.9 | 2458.2 | 413.5 | 33.22 / 22.05 | 22.13 | 0.25 / +0.25 / 12 |
| 0.3 | 6cb87f86 | 17.0554 | 16.2289 | 17.7701 | 2826.7 | 1218.3 | 1193.3 | 1353.1 | 2474.0 | 430.7 | 33.11 / 22.13 | 22.09 | 0.25 / +0.25 / 12 |
| 0.5 | 12829b13 | 16.9516 | 16.1072 | 17.6818 | 2720.2 | 1306.2 | 1180.1 | 1318.7 | 2508.4 | 465.2 | 32.68 / 22.00 | 21.91 | 0.25 / +0.25 / 12 |
| **0.75** | 9b0d9d91 | **16.8918** | **15.9988** | 17.6639 | 2594.3 | 1426.5 | 1167.3 | 1274.2 | 2552.8 | 522.7 | 32.16 / 21.87 | 21.71 | 0.25 / +0.25 / 12 |
| 1.0 | 0d4c3fcd | 16.9725 | 16.0959 | 17.7306 | 2451.9 | 1588.8 | 1172.2 | 1202.9 | 2624.1 | 589.7 | 31.48 / 21.76 | 21.80 | 0.25 / +0.25 / 12 |

Figure: `runs/sweeps/min-duration-off/md_fa_der_vs_min_duration_off.png` (MD, FA, DER against the value).

Reading. Missed detection falls and false alarm rises at every step, almost one for one: at 0.75 s MD is 331 s lower and FA 327 s higher, so nearly every bridged gap was real silence. The DER is not monotone: it rises slightly to 0.2 s (+0.012 points), then falls to a minimum at 0.75 s and rises again at 1.0 s.

**Selection (tune half only).** Lowest tune-half DER is **0.75 s** (15.9988, against 16.2587 at 0.0). It is not the shipped value. Its report-half delta against the fresh shipped baseline is **-0.068 DER points, 90% paired bootstrap [-0.316, +0.153]** (8 report-half meetings, 2000 draws, seed 0), so the interval includes zero. Against the cache-served 0.0 row: -0.069 [-0.317, +0.152]. All-16 delta (descriptive): -0.157 points. Bootstrap function: `harness/clustering_benchmark_report.py:57` `paired_bootstrap_delta` reads that module's fixed benchmark table and cannot take these CSVs, so `sweep.py` `boot_delta` reimplements the same algorithm (same `_der` imported from `:50`, same sampling), and was checked to reproduce `paired_bootstrap_delta("HDB-2b-04")` exactly (-0.005539 [-0.010829, -0.001374]).

Checks: 8 distinct final keys, one shared intermediate key, every manifest `min_duration_off` equals the request, every nonzero point `16 full hit(s), 0 partial, 0 miss(es)`, MD differs across values.

### HDBSCAN check (not selected separately)

Clustering model and params copied from the manifest of `b15caf86` (`HDBSCANClustering(cluster_selection_method='leaf', min_cluster_size=40, min_samples=0)`); runs at 0.0 and at the VBx-selected 0.75.

| value | run | DER | report DER | MD | FA | Conf | tally |
|---|---|---|---|---|---|---|---|
| 0.0 | 805bafef | 16.8558 | 17.1788 | 2925.1 | 1099.4 | 1152.6 | 0/0/0 (final-cache hit) |
| 0.75 | 60a94eaa | 16.6412 | 17.0040 | 2593.9 | 1381.9 | 1135.4 | 16/0/0 |

The 0.0 run has the same final key as `b15caf86` (`ce6cd02f...432c`) and reproduces its DER up to the final-RTTM rounding (16.8558 vs 16.8551, 6.5e-6), which is explained by the 0/0/0 tally. Delta at 0.75 against the fresh `b15caf86` figure: all 16 **-0.214 points**; report half **-0.174 [-0.434, +0.052]**. Against the cache-served 0.0 run: all 16 -0.215, report -0.175 [-0.435, +0.051]. VBx at the same value: all 16 -0.157, report -0.068.

### Predictions

1. MD falls monotonically, FA rises monotonically: **held** (MD 2925.1 to 2451.9; FA 1099.4 to 1588.8; every step in the table).
2. Speaker count unchanged at every value: **held** (MAE 0.25, bias +0.25, 12 exact at all eight values).
3. Missed overlap changes far less, relative to its size, than total MD: **not held**. At 0.75 s missed overlap falls 8.6% against MD 11.3%; at 1.0 s 13.7% against 16.2%. Overlap misses shrink at about 0.76 to 0.85 of MD's rate.
4. Confusion changes by under 5% of 1212 s: **held** (largest change -3.7% at 0.75 s).
5. Report-half delta at the selected value under 0.3 points: **held** (-0.068, interval [-0.316, +0.153]).
6. HDBSCAN delta within 0.1 points of the VBx delta: **not held on the report half** (-0.174 against -0.068, a difference of 0.106), **held on all 16** (-0.214 against -0.157, a difference of 0.057). Both sit inside the interval widths, so this is a borderline miss and reported as such.

Caveats attached to the result: the selected value is the best of eight on the tune half, and 0.75 is not a clean optimum (1.0 is worse); the gain is below the project's 0.3 point resolution and its report-half interval includes zero. No gain exceeded 1 DER point.

## Closeout 2 — SDM

Source: `runs/sweeps/sdm/` (driver `sweep.py`, `points.csv`, `per_meeting.csv`, `results.json`, `points.jsonl`, `report_output.txt`, per-run folders). AMI **SDM** (single distant microphone, `Array1-01`), the same 16 test meetings, `test.rttm` and `test.uem` byte-identical to IHM's. Collar 0, overlap scored (manifest `der_collar` 0.0, `der_skip_overlap` false). DER in %. Tune half = meetings ending a/c, report half = b/d. Settings were carried across from IHM (copied from the IHM manifests and ledger by the driver), **not re-tuned**.

Reproduce, from the repo root, one phase at a time: `diarisation-env/Scripts/python.exe runs/sweeps/sdm/sweep.py --phase gate`, then `--phase runs`, then `--phase report`. Every run is `run_harness.py --data-root C:/Users/Andrew/Code/AMI-diarization-setup/harness-data --condition SDM --split test --cache-dir .harness_cache_sdm`.

### Cache isolation (checked)

`run_harness.py` passes `config.cache_dir` to both `IntermediateCache` (`run_harness.py:169-170`) and `Runner` (`:179-186`; the final RTTM path is `cache_dir / f"{cache_key}.rttm"`, `harness/runner.py:66`), so `--cache-dir` redirects both tiers. The keys contain the uri but not the mic condition, so this isolation is what stops SDM returning IHM data. The driver fingerprints `.harness_cache` (file count, bytes, newest mtime) before and after every run and hard-fails on any change; no run tripped it. The SDM cache holds 16 intermediate entries after the cold run and the tally reads `16 entries` after every later run.

### Gate (ticket section 2): passed, with a caveat on the reference figure

- Cold baseline: run `20261008T182732Z-70c72cb4`, tally **`0 full hit(s), 0 partial, 16 miss(es), 16 write(s)`** (16/16 misses, no hit), 2274 s.
- SDM VBx DER **19.909** (manifest `summary.der` 0.19908762; recomputed independently from the per-file CSV as total error / total reference: 19.90876). Published 19.9. Gap **+0.009**, inside the 0.5 band, so nothing to flag on the gap.
- **Caveat on 19.9 (attached to the gate):** the figure is from the repo `README.md:84-101` benchmark table (row `AMI (SDM) | 22.7 | 19.9 | 15.6`; columns legacy 3.1, community-1, precision-2). That README states **no collar or overlap setting**, and the Hugging Face model card was not fetched. "No collar, overlap not skipped" is an inference from the IHM figure in the same table (17.0) matching this harness's 17.05 under those settings. It is not a quote.
- Not near 17.05, so the IHM cache was not read. The four IS1009 SDM wavs are 0.01–0.16 s longer than IHM's (tails beyond the UEM). No sign of a time-alignment problem: IS1009a–d degrade by +2.13, +1.21, +0.95 and +1.99 points IHM to SDM, all below the 16-meeting median of +2.76.
- Baseline decomposition, SDM against IHM (percent of reference speech): MD 11.16 vs 9.52, FA 3.81 vs 3.58, Conf 4.93 vs 3.95. Of the +2.86 points, +1.64 is missed speech, +0.99 confusion, +0.23 false alarm.

### Results, seven methods (ticket section 3)

All seven runs: every manifest's `clustering_config` and `refinement_strategy` equal the request, `mic_condition` = SDM, `split` = test, `condition` as requested (programmatic checks recorded per run in `points.jsonl`); all six runs after the baseline show **`16 full hit(s), 0 partial, 0 miss(es), 0 write(s)`**. nc_target and kNN share the VBx final key with the baseline but differ in refinement id, and their DER differs from the baseline, so refinement ran rather than a cached identity RTTM being returned.

Intervals: 90% percentile bootstrap over the 16 meetings, 2000 draws, seed 0, `harness/assignment_runs.py:150` `bootstrap()`. Absolute interval: the zeros trick from `point_stats`. Δ vs SDM VBx and SDM − IHM are paired per meeting (same resampled meetings in both arms). IHM per-file sources: the same CSVs as the IHM rows above (ward: `runs/sweeps/ward-redo/WARD-T8/per_file.csv`).

| Method | Run id | SDM DER [90% CI] | tune / report | Δ vs SDM VBx [CI] | IHM DER | SDM − IHM, paired [CI] |
|---|---|---|---|---|---|---|
| VBx baseline | `70c72cb4` | 19.91 [18.07, 21.74] | 19.37 / 20.37 | — | 17.05 | +2.86 [+2.39, +3.40] |
| Agglomerative | `94c45184` | 20.33 [17.99, 22.48] | 20.12 / 20.51 | +0.42 [−0.37, +1.18] | 16.87 | +3.46 [+2.50, +4.37] |
| HDBSCAN | `ffde933d` | 19.55 [17.51, 21.49] | 19.22 / 19.83 | −0.36 [−0.75, −0.00] | 16.86 | +2.69 [+2.24, +3.19] |
| Spectral | `993c6760` | 20.60 [18.02, 23.31] | 19.51 / 21.55 | +0.69 [−0.37, +1.94] | 17.16 | +3.44 [+2.44, +4.57] |
| Nearest centroid (`nc_target`) | `3a04d861` | 19.72 [17.84, 21.59] | 19.25 / 20.12 | −0.19 [−0.32, −0.06] | 16.89 | +2.83 [+2.36, +3.39] |
| kNN k=25 | `00b82017` | 19.66 [17.76, 21.58] | 19.21 / 20.04 | −0.25 [−0.40, −0.11] | 16.85 | +2.81 [+2.34, +3.39] |
| Ward thr 8 (added after ticket) | `32122096` | 21.88 [18.92, 24.66] | 21.05 / 22.59 | **+1.97 [+0.23, +3.75]** | 17.47 | +4.41 [+2.60, +6.43] |

The tune / report columns are descriptive; nothing was selected on either half (the settings are IHM's). Per-meeting IHM to SDM range for VBx: +0.95 (IS1009c) to +8.96 (TS3003a); all 16 values for all seven methods are in `per_meeting.csv`.

Full scorer set. MD and FA are identical across the seven runs because segmentation is shared, as on IHM: MD 3429.0 s (11.16% of reference speech) and FA 1170.9 s (3.81%). Region census (s) is also identical: `t∧d` 2245.8, `t−d` 1581.2, `d−t` 397.3 (IHM: 2433.1 / 1393.9 / 382.9). What differs:

| Method | Conf (s) | Conf (%) | DER overlap, system | DER overlap, assigned | JER | count MAE | signed bias | exact /16 | mean hyp speakers |
|---|---|---|---|---|---|---|---|---|---|
| VBx baseline | 1514.9 | 4.93 | 36.57 | 23.86 | 27.06 | 0.375 | +0.125 | 12 | 4.06 |
| Agglomerative | 1644.0 | 5.35 | 38.08 | 25.83 | 26.86 | 0.562 | +0.312 | 10 | 4.25 |
| HDBSCAN | 1404.0 | 4.57 | 36.15 | 23.42 | 27.62 | 0.188 | −0.188 | 14 | 3.75 |
| Spectral | 1728.0 | 5.63 | 36.89 | 24.49 | 30.29 | 0.312 | −0.312 | 12 | 3.63 |
| Nearest centroid | 1457.0 | 4.74 | 36.41 | 23.73 | 26.78 | 0.375 | +0.125 | 12 | 4.06 |
| kNN k=25 | 1438.0 | 4.68 | 36.25 | 23.45 | 26.85 | 0.375 | +0.125 | 12 | 4.06 |
| Ward | 2120.0 | 6.90 | 39.23 | 26.81 | 28.29 | 0.500 | +0.125 | 9 | 4.06 |

(Conf seconds are rounded to the nearest second here; `points.csv` carries full precision.)

**Readings, with their limits.**
- The largest gain over the SDM baseline is HDBSCAN's −0.36 points, well under the 1-point suspect-bug line; its interval just reaches zero at the upper end (−0.002). Nearest centroid and kNN keep their IHM sign (−0.19, −0.25) with intervals that exclude zero, at gains of the same order as on IHM. The selection caveats for k=25 and for the best-of-about-13 assignment results (closeout 3 section 4) still apply. **SDM is not independent evidence against that selection effect** (coordinator, 9 Oct): it is the same 16 meetings and the same speakers, recorded on a different microphone, and kNN, nearest centroid and the HDBSCAN setting were all chosen with these meetings' IHM test scores in view. The SDM gains replicate a pattern on the same meetings; they do not test it on new ones. Dev (gains about a third the size, intervals spanning zero) and train (no gain) remain the out-of-sample evidence.
- Ward is the one clear loser: +1.97 over the baseline, interval excluding zero, driven by confusion (2120 s against 1515 s).
- Ward's SDM − IHM degradation (+4.41) is above the ticket's 2 to 4 band; the other six sit between +2.69 and +3.46.

### Predictions (ticket section 6), recorded before the gate ran, measures fixed in advance

Written to `closeout-2-sdm.md` (section "§2-§6 SDM runs") before any SDM run. IHM comparators are the 16-meeting DERs in the table above.

| # | Prediction | Measure fixed beforehand | Result | Verdict |
|---|---|---|---|---|
| 1 | Baseline within 1 pt of 19.9 | abs gap <= 1.0 | 19.909, gap +0.009 | **Held** |
| 2 | Every method 2 to 4 points worse on SDM | all seven methods in [2.0, 4.0] | six of seven in range (+2.69 to +3.46); ward +4.41 | **Not held as defined.** Ward, added after the ticket, is 0.41 above the band. On the ticket's own six methods it holds. |
| 3 | Ranking roughly preserved | Spearman over the seven >= 0.7 | rho = **0.857**. IHM ranks (1 = best): kNN 1, HDBSCAN 2, agglomerative 3, nc_target 4, VBx 5, spectral 6, ward 7. SDM ranks: HDBSCAN 1, kNN 2, nc_target 3, VBx 4, agglomerative 5, spectral 6, ward 7 | **Held.** Caveat fixed in advance: six of the IHM DERs span about 0.3 points (inside resolution), so the IHM ranking is close to noise; the informative part is that ward stays last and spectral next to last. |
| 4 | IHM-selected settings transfer imperfectly | any non-baseline method with (SDM Δ vs VBx) − (IHM Δ vs VBx) > +0.5 | agglomerative +0.60, spectral +0.58, ward +1.55; HDBSCAN −0.17, nc_target −0.03, kNN −0.05 | **Held** (also at 0.3). Limit: for agglomerative and spectral the SDM Δ intervals include zero, so these are point estimates; ward's excludes zero. |
| 5 | Counting degrades more than DER | relative count MAE (SDM/IHM − 1) above relative DER (SDM/IHM − 1) for VBx and for at least 4 of 7 methods | VBx: MAE 0.25 to 0.375 (+50%) against DER +16.8%. Count exceeds DER for 6 of 7 (all but HDBSCAN: MAE 0.188 to 0.188, 0%, DER +16.0%). Exact matches IHM to SDM: VBx 12 to 12, agglomerative 13 to 10, HDBSCAN 14 to 14, spectral 13 to 12, nc_target 12 to 12, kNN 12 to 12, ward 12 to 9 | **Held, on a coarse measure.** MAE is a mean of 16 integer errors. SDM VBx miscounts four meetings (TS3003a 2 for 4, ES2004b 6 for 4, ES2004c 5 for 4, IS1009b 5 for 4); IHM VBx also miscounted four, by one each. The exact-match count did not move for VBx, nc_target or kNN, so only the error sizes grew. Not a robust result. |

### Do the IHM conclusions hold in far-field?

Partly, as far as 16 meetings and these intervals allow. The baseline reproduces the published 19.9. Every method is 2.7 to 4.4 points worse on SDM, mostly in missed speech, which all methods share because it comes from the segmenter, so the segmentation-dominated picture from IHM carries over. Among the six ticket methods the spread is again narrow: HDBSCAN, kNN and nearest centroid sit 0.2 to 0.4 below the baseline, and nothing approaches a size that would change the conclusion that no method clearly beats VBx. What transferred less well is the clustering choice: agglomerative and spectral are 0.4 to 0.7 worse than the baseline on SDM (intervals include zero), and ward at threshold 8 is 2.0 worse with an interval that excludes zero, against +0.42 on IHM. Per-meeting degradation is very uneven (+0.95 to +8.96 for VBx), with TS3003a the worst; VBx finds 2 of its 4 speakers there.

### §5 Oracle segmentation on SDM (approved 9 Oct)

One run, approved by Andrew: oracle (ground-truth) segmentation on AMI SDM test, 16 meetings, VBx shipped clustering, identity refinement, `--counts-toward-results true`. Run `20261008T195525Z-259142b4`, cold in `.harness_cache_sdm` (`0 full hit(s), 0 partial, 16 miss(es), 16 write(s)`; the dir grew from 112 to 128 final RTTMs and 16 to 32 intermediate `.npz`). Full detail, prediction and verification in [[closeout-2-sdm]] §5.

**Budget table** (formula from [[Cross-Condition Deltas]]: downstream budget = `baseline_der − oracle_segmentation_der`; share = budget / baseline_der):

| | IHM | SDM |
|---|---|---|
| Baseline DER | 17.0485% (`0fa3b41d`) | 19.9088% (`70c72cb4`) |
| Oracle segmentation DER | 3.9561% (`18016c15`) | 4.0076% (`259142b4`) |
| Segmentation's point contribution | 13.0925 pt | 15.9011 pt |
| Segmentation share of baseline DER | 76.80% | 79.87% |

**Reading.** The prediction (4-5% oracle DER, segmentation share at or above 76.8%) held: SDM oracle DER is 4.0076%, essentially the same as IHM's 3.9561% (+0.0516 pt), while the SDM baseline is +2.8602 pt worse than IHM's. In other words, almost the entire far-field baseline penalty (2.86 of 2.91 points, 98%) disappears once segmentation is forced to ground truth — the segmentation stage accounts for not just most but nearly all of why SDM is harder than IHM for this system. This is the first of the two pre-registered informative outcomes (near 4-5%, "mostly missed speech"), not the second (8%+, which would have meant a large piece of the far-field penalty was not segmentation). MD, FA and the region census for the SDM oracle run are byte-identical to the IHM oracle run's (both inject the same reference content, since `IHM/test.rttm` and `SDM/test.rttm` are byte-identical); the only difference between the two oracle DERs is confusion (1073.35 s SDM vs 1057.51 s IHM), i.e. embeddings/clustering under oracle segmentation are very slightly worse on far-field audio, but this is a small effect next to the baseline's missed-speech-driven gap.
