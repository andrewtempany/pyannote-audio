---
status: in-progress
created: 2026-10-03
type: long-running-agent
aliases: [clustering benchmark run]
---

# Clustering benchmark run: agglomerative, HDBSCAN, spectral

## In short (read this part)

**What we want to find out:** if we swap the pipeline's clustering step for a different method, how much better or worse does diarisation get? And how much does each method's result move when we change its settings?

**Methods tested:**

| Method | What it does, in one line | Already in the code? |
|---|---|---|
| VBx | The current default. Reference point only, not re-run. | Yes |
| Agglomerative | Starts with every voice clip as its own group and keeps merging the two closest groups until they are "too far apart". | Yes |
| HDBSCAN | Finds dense clumps of similar voice clips. Clips that don't fit any clump are marked as outliers. | No, needs adding |
| Spectral + eigengap | Builds a "who sounds like who" graph, works out how many speakers there probably are from the graph's shape, then splits it into that many groups. | No, needs adding |

**Outliers:** any voice clip a method can't place goes to the **nearest speaker group** (nearest centroid). That's the pattern the code already uses. A smarter outlier classifier comes later in its own ticket.

**What we get at the end:**

1. One results table: each method's best setting against the VBx baseline (DER 17.05%).
2. One chart per method showing how DER and speaker counting change as its main setting changes. This shows the range of results each method can give.
3. A short plain-language summary of what each method did well or badly.
4. For HDBSCAN: a count of how many clips were outliers, and whether those clips tend to come from overlapping speech. This feeds the classifier ticket.
5. An **error profile** for each method: which kinds of mistakes it makes, so we can see each method's strengths and weaknesses rather than just one score.

**How we measure it:**

- **DER** (lower is better): the share of speech time the system got wrong.
- **What kind of error it was.** The harness already splits every mistake into types, and every run records all of them:
  - **Missed speech:** someone was talking but the system said nobody was.
  - **False alarm:** the system said someone was talking when nobody was.
  - **Confusion:** speech was found but credited to the wrong speaker. This is the one clustering affects most.
  - **Overlap error:** how wrong the system is during moments when two people talk at once. It's measured twice: over all real overlap, and over only the overlap the system actually noticed. The second one is the project's core measure.
  - **JER:** a second overall score that weighs every speaker equally, so a quiet speaker's mistakes count as much as a talkative one's.
  - **Overlap noticed / missed / invented:** seconds of overlap the system caught, missed, or reported where there wasn't any.
- **Speaker counting:** how far off the number of speakers was per meeting (MAE, lower is better), how many of the 16 meetings got the count exactly right, and whether a method tends to **over-count or under-count**.
- **Fair-test rule:** settings are chosen on 8 meetings (the "tune half") and the result is reported on the other 8 (the "report half"). Choosing and reporting on the same meetings would flatter every method.
- **Small differences don't count:** with only 8 to 16 meetings, a gap under about 0.3 DER points is reported as "no clear difference" rather than a win.

**Done means:** all sweeps below are finished, the table and charts exist, the summary is written, and this ticket is converted into docs.

**Not in this ticket:** no oracle runs, no retraining, no new corpora (SDM and VoxConverse are separate), no KMeans, no outlier classifier, and no changes to VBx or its settings.

---

## For the agent (detailed spec)

Everything below is instructions for the executor. The plain-language section above is the source of truth for goals. If anything here contradicts it, stop and flag it.

### 0. Kickoff protocol (do this before any sweep)

1. Read this ticket and the linked notes: [[clustering-method-comparison]], [[clustering-agglomerative]], [[clustering-hdbscan]], [[clustering-spectral-eigengap]], [[Pre-Clustering Cache]], [[Run Manifest]], [[Evaluation Harness]]. Report any contradictions between them and this ticket. Where there's a conflict, this ticket wins on scope and grids, and the linked notes win on facts about the code.
2. Work through the assumptions register (§3). Mark each row verified or not, citing the file and line or the command output.
3. Create `runs/sweeps/clustering-benchmark/` containing:
   - `TASKS.md`: a checklist of the workstreams and sub-steps. Tick items as you go, and re-read it after any context reset.
   - `SWEEP_INDEX.md`: one row per sweep point (see §8).
4. Reply to Andrew with questions grouped as **blocking now / needed soon / housekeeping**, each with a recommended default, so he can answer "defaults OK". Write them in plain language.
5. Set `status: in-progress` and start WS0.

### 1. Definitions

- **Sweep point:** one full scored run over all 16 AMI IHM test meetings with one clustering configuration.
- **Tune half:** meetings ending in `a` or `c`, which is EN2002a, EN2002c, ES2004a, ES2004c, IS1009a, IS1009c, TS3003a and TS3003c.
- **Report half:** meetings ending in `b` or `d`, the other 8.
  - Each recording series appears twice in each half, so neither half is all from one site.
  - This split is fixed. Write it into [[clustering-vbx]] as well, so later VBx work reuses it.
- **Best setting:** the configuration with the lowest corpus-level DER on the tune half. Ties within 0.1 DER points go to the one with the lower count MAE.
- **Half-level DER:** sum(MD + FA + Conf) / sum(total reference speech) over the meetings in that half, computed from the per-file CSV components. It is **not** the mean of per-file DER.
  - Reference speech per file = (MD + FA + Conf) / der.
  - Check: on all 16 meetings this must reproduce the run's `summary.der` to within 1e-9.
- **Error metrics (record every one at every sweep point).** All of these are already written by `harness/scorer.py` into the per-file CSV and the summary JSON. Don't recompute them a different way.

  | Field | Plain meaning | Half-level aggregation |
  |---|---|---|
  | `der` | overall error rate | sum of components / sum of reference speech (see Half-level DER above) |
  | `missed_detection` | seconds of speech missed | sum |
  | `false_alarm` | seconds of non-speech labelled as speech | sum |
  | `confusion` | seconds credited to the wrong speaker | sum |
  | `der_overlap_system` | DER inside all reference overlap (T) | take from `summary.json` for all 16; for halves report the per-file values only and mark them "per-file, not corpus-level" |
  | `der_overlap_assigned` | DER inside overlap the system detected (T ∩ D), **the thesis metric** | same as above |
  | `jer` | speaker-balanced error rate | same as above |
  | `region_t_and_d`, `region_t_minus_d`, `region_d_minus_t` | seconds of overlap caught, missed, invented | sum |
  | `count_ref`, `count_hyp`, `count_error` | speakers in reference, in hypothesis, absolute difference | MAE, exact matches, plus **signed** mean of `count_hyp - count_ref` and the number of meetings over-counted and under-counted |

  Also derive **error shares**: MD, FA and Conf each as a percentage of total error for that point. These show *what kind* of mistake a method makes, independent of how many.

  **Expected behaviour to check against.** Clustering only relabels speakers, so **Confusion** should move most. MD, FA and the overlap regions can still move, because if a method puts two people who are talking over each other into the same group, that overlap disappears from the output: `region_d_minus_t` and `region_t_and_d` shrink and MD rises. That's a real weakness worth reporting, not a bug. If MD or FA moves by more than about 1% of reference speech between two settings of the **same** method with no change in the overlap regions, investigate before continuing.
- **Baseline reference values:** compute the tune-half, report-half and all-16 values from the existing cold baseline run `runs/20260925T101811Z-db977741.json` and its per-file CSV (`per_file_baseline_batchkey_rebuild.csv`). Don't re-run VBx.
- **Outlier handling (all methods):** an embedding the method leaves unassigned is folded into the nearest large cluster's centroid **inside `cluster()`**, using the same distance metric, before returning. This mirrors the small-cluster reassignment in `AgglomerativeClustering.cluster()` (clustering.py, the block after `small_clusters = ...`). `BaseClustering.assign_embeddings` then assigns every remaining embedding to its nearest centroid as usual. `cluster()` must never return a negative label.

### 2. Current state (verified 2026-10-03 by the planner, re-verify at kickoff)

| Area | What's true | Status |
|---|---|---|
| Default clustering | `VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6)` | verified (manifest db977741) |
| Baseline | DER 0.17048543579940637, count MAE 0.25, 12/16 exact | verified (manifest db977741) |
| Selection flags | `--clustering-model`, repeatable `--clustering-param name=value` | verified (run_harness.py:596-615) |
| Intermediate cache | segmentation + embeddings cached per file, clustering excluded from the key, about 33 s per warm 16-meeting run on the RTX 3060 | verified (Pre-Clustering Cache doc) |
| Final cache key | includes the clustering class + every instantiated hyperparameter, read generically from pyannote `Parameter` descriptors | verified (Pre-Clustering Cache doc) |
| Agglomerative selection | yields `threshold=0.6` inherited from the VBx config, not its own value | verified (clustering-agglomerative ticket) |
| HDBSCAN, Spectral | not implemented anywhere | verified (clustering.py `Clustering` enum lines 759-763) |
| Clustering flow | filter clean embeddings, `cluster()`, centroids, assign all embeddings to the nearest centroid | verified (clustering.py `BaseClustering`) |
| Loose files | per-file CSVs and summaries currently land in the repo root | verified (directory listing) |

### 3. Assumptions register

| # | Claim | Source | Status | How to verify |
|---|---|---|---|---|
| A1 | The harness clustering vocabulary (`CLUSTERING_MODEL_CLASSES` in run_harness.py) maps names to `Clustering` enum names, and adding enum entries plus vocabulary entries is enough to select a new class | planner reading | unverified | read `CLUSTERING_MODEL_CLASSES` and how `speaker_diarization.py` resolves the `clustering` string |
| A2 | Hyperparameters declared as `Uniform` / `Integer` / `Categorical` on a new class reach the final cache key with no change to `cache_id.py` | Pre-Clustering Cache doc | reported | unit test: two configs of the new class give different final keys |
| A3 | Hyperparameters left as `Parameter` descriptors on a class must be instantiated or the pipeline refuses to run; `--clustering-param` sets them | planner reading | unverified | instantiate each new class with and without params in a test |
| A4 | Installed scikit-learn is 1.3 or later, so `sklearn.cluster.HDBSCAN` exists | none | unverified | `python -c "import sklearn; print(sklearn.__version__)"` in `diarisation-env` |
| A5 | Embedding counts per meeting after filtering are small enough (a few thousand) for a dense affinity matrix and eigendecomposition in under 60 s per file | planner estimate | unverified | log `num_embeddings` and `cluster()` wall time per file on the first spectral point |
| A6 | The intermediate cache gives 16/16 full hits for every non-VBx clustering class | Pre-Clustering Cache doc | reported | the cache tally line on the first point of each method |
| A7 | pyannote speaker-diarization-3.1 shipped agglomerative with `method=centroid, min_cluster_size=12, threshold≈0.7046` | planner memory | unverified | if the HF config is reachable, check it; otherwise label that point "3.1-like setting (unverified)" |
| A8 | Results are deterministic for a fixed configuration | expected | unverified | WS0 step 4 |

### 4. Operating conditions

- **Machine:** Windows PC, RTX 3060, `diarisation-env`. Batch size stays at 32, because changing it changes the cache identity.
- **Data:** AMI IHM test, `AMI.SpeakerDiarization.only_words`, under the existing `--data-root`. Nothing new to download.
- **Owner availability:** Andrew may be away. Keep going on anything not blocked. Blockers and questions go in this ticket's `## Blocked on Andrew` section, newest at the top.
- **Branch:** work on `clustering-benchmark` off the current branch. Commit after each workstream with a message naming the workstream. Don't push unless asked.
- **Outputs:** every sweep point writes its per-file CSV and summary into `runs/sweeps/clustering-benchmark/<method>/<point_id>/`, using `--per-file-csv` and `--summary`, never the repo root. Manifests stay in `runs/` as normal.
- **Manifest notes:** every run passes `--notes "clustering-benchmark <WS> <point_id>: <params>"` and `--run-condition clustering_<method>`.

### 5. Workstreams (in order)

#### WS0: Setup and safety checks (no new methods yet)

1. Verify §3 A1, A4 and A8 (A8 by step 4 below).
2. Compute the baseline reference values per §1 and put them at the top of `SWEEP_INDEX.md`.
3. Run one agglomerative point (`method=centroid, min_cluster_size=12, threshold=0.7`). Confirm:
   - the cache tally shows 16 full intermediate hits and 0 writes to the intermediate tier;
   - the manifest's `clustering_config` names `AgglomerativeClustering` with exactly those three values;
   - DER differs from the baseline.
4. Run the same point a second time with the final-RTTM cache bypassed (point `--cache-dir` at a fresh empty folder). DER must match to 1e-6. If it doesn't, stop: results are not deterministic.

#### WS1: Agglomerative (already implemented, settings only)

Set **all three** hyperparameters explicitly at every point. Never rely on the inherited 0.6.

| Step | method | min_cluster_size | threshold values | Points |
|---|---|---|---|---|
| 1a | centroid | 12 | 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00, 1.10, 1.20 | 12 |
| 1b | average | 12 | same 12 values | 12 |
| 1c | best of 1a/1b on tune half | 1, 3, 6, 12, 20 | best threshold from that step | 5 |
| 1d | complete, single, weighted, median, ward | 12 | 0.4, 0.6, 0.8, 1.0, 1.4 | 25 |

Notes:

- `centroid`, `median` and `ward` switch to euclidean distance on normalised embeddings (clustering.py:371-375). Their thresholds aren't on the same scale as `average`, `complete`, `single` and `weighted`. Plot the two families separately.
- If a 1d linkage's best point sits at the edge of its 5 values, add up to 3 more points beyond that edge.
- Record the effective `min_cluster_size` where the runtime adjustment applies (clustering.py:359-362).

#### WS2: HDBSCAN (new class, then settings)

Implement `HDBSCANClustering(BaseClustering)` in `src/pyannote/audio/pipelines/clustering.py`, add it to the `Clustering` enum, and add `hdbscan` to the harness vocabulary.

- `expects_num_clusters = False`.
- Hyperparameters, as descriptors so they reach the cache key:
  - `min_cluster_size = Integer(2, 200)`;
  - `min_samples = Integer(0, 200)`, where 0 means "same as min_cluster_size" (the sklearn `None` behaviour);
  - `cluster_selection_method = Categorical(["eom", "leaf"])`.
- `cluster()`:
  1. Unit-normalise the embeddings.
  2. Run `sklearn.cluster.HDBSCAN(metric="euclidean", ...)`.
  3. Label -1 means outlier: fold those embeddings into the nearest non-noise cluster centroid (cosine), as in §1.
  4. If every point is noise, return all zeros (same as agglomerative's `num_large_clusters == 0` branch).
  5. Renumber the labels 0..K-1.
  6. Respect `min_clusters` / `max_clusters` only if they're given. The harness never gives them, so don't build count-forcing logic.
- **Outlier diagnostic** (needed for the classifier ticket). For each file, log to `runs/sweeps/clustering-benchmark/hdbscan/<point_id>/outliers.csv`, one row per training embedding:
  - `uri, chunk_idx, speaker_idx, is_outlier, assigned_cluster, overlap_fraction`;
  - `overlap_fraction` = the share of that local speaker's active frames in the chunk where 2 or more speakers are active, taken from the binarised segmentation available inside the pipeline.
  - If getting the segmentation into `cluster()` needs more than a small change, log from the harness side instead and describe how in Implementation Notes. Don't change existing classes' behaviour to make it work.
- Tests before the sweep (show them failing first, then passing):
  - routing: selecting `hdbscan` gives a pipeline whose clustering is `HDBSCANClustering`;
  - two different `min_cluster_size` values give different final cache keys;
  - `cluster()` never returns -1 on a fixture with an obvious outlier.

Grid:

| Step | min_cluster_size | min_samples | cluster_selection_method | Points |
|---|---|---|---|---|
| 2a | 5, 10, 20, 40, 80 | 0 | eom | 5 |
| 2b | 5, 10, 20, 40, 80 | 0 | leaf | 5 |
| 2c | best of 2a/2b | 1, 5 | best of 2a/2b | 2 |

Report the **outlier rate** (outliers / training embeddings) at every point, alongside DER.

#### WS3: Spectral with eigengap (new class, then settings)

Implement `SpectralClustering(BaseClustering)` (name it `SpectralEigengapClustering` to avoid clashing with sklearn), add it to the enum, and add `spectral` to the vocabulary.

- `expects_num_clusters = False`.
- Hyperparameters:
  - `pruning = Uniform(0.01, 1.0)`: fraction of strongest similarities kept per row; 1.0 means no pruning;
  - `max_speakers = Integer(2, 20)`: upper limit for the eigengap search.
- `cluster()`:
  1. Unit-normalise and build the cosine similarity matrix.
  2. Set negatives to 0 and keep the top `pruning` fraction of entries per row (at least 1), zeroing the rest.
  3. Symmetrise with `(A + A.T) / 2`.
  4. Build the normalised graph Laplacian and take its smallest `max_speakers + 1` eigenvalues, ascending.
  5. k = 1 + argmax of the consecutive gaps among the first `max_speakers` eigenvalues. Clamp k to `[min_clusters, max_clusters]` if they're given.
  6. Partition with `sklearn.cluster.SpectralClustering(n_clusters=k, affinity="precomputed", random_state=42)` on the pruned matrix.
  7. No outliers are possible, but still guarantee labels 0..K-1.
  8. Log the estimated k per file.
- Tests: routing; different `pruning` values give different final keys; eigengap returns k=3 on a fixture of three well-separated blobs.
- Speed check: if `cluster()` takes over 60 s on any file at the first point, switch to `scipy.sparse.linalg.eigsh` for the eigenvalues and note it.

Grid:

| Step | pruning | max_speakers | Points |
|---|---|---|---|
| 3a | 0.05, 0.10, 0.20, 0.30, 0.50, 1.00 | 10 | 6 |
| 3b | best of 3a | 6, 15 | 2 |

#### WS4: Write-up

1. **Results table:** this goes into `## Results` in this ticket, and its rows are copied into [[clustering-method-comparison]]. It covers baseline VBx, plus each method's best point (selected on the tune half) and its worst swept point. Columns:
   - Method, Setting;
   - DER for the tune half, report half and all 16;
   - MD, FA, Conf;
   - Count MAE, Exact /16;
   - Δ DER vs baseline on the report half, with a 90% paired bootstrap interval (resample report-half meetings with replacement, 2000 draws, seed 0).
2. **Charts** (matplotlib PNG, saved in `runs/sweeps/clustering-benchmark/figures/`), one per method's main setting:
   - DER on the y-axis, with the VBx baseline as a horizontal line;
   - a second panel underneath showing count MAE.
   - The main settings are agglomerative `threshold` (one line per linkage, the two families in separate panels), HDBSCAN `min_cluster_size` (one line per selection method) and spectral `pruning`.
3. **Error profile table and chart** (`## Error profiles` in this ticket; chart `figures/error_profile.png`).
   - Use VBx baseline plus each method's best point, all 16 meetings plus the report half.
   - The table has these columns: Method, MD / FA / Conf in seconds, MD / FA / Conf as a share of total error, `der_overlap_system`, `der_overlap_assigned`, JER, overlap caught / missed / invented in seconds, count MAE, signed count bias, meetings over / under / exact.
   - The chart is a stacked bar per method showing MD, FA and Conf in seconds, with the baseline first.
   - For each method, write one plain line on its **strength** and one on its **weakness**, each pointing at the number that shows it. Example: "Fewer confusion errors than VBx (−180 s) but merges overlapping speakers: overlap caught falls by 300 s and missed speech rises."
   - Also list the 3 meetings where each method did best and worst **relative to VBx**, so patterns by recording site show up.
4. **HDBSCAN outlier summary:** at the best HDBSCAN point, compare the mean `overlap_fraction` of outliers against non-outliers, and the outlier rate per meeting. Two or three plain sentences, no claims beyond what the numbers show.
5. **Plain-language summary:** at most one short paragraph per method. Did it beat, match or lose to VBx on the report half? How wide was the range across its settings? Did it count speakers better or worse, and does it lean to over- or under-counting? Which error type is its main weakness? Use "no clear difference" for gaps under 0.3 DER points or bootstrap intervals that cross 0.

### 6. Predictions (record before each workstream; don't tune toward them)

- **Agglomerative:** the best point lands within ±1.5 DER points of 17.05. Speaker count falls steadily as the threshold rises. The DER curve is flatter than the counting curve.
- **HDBSCAN:** worse than VBx by 1 to 3 DER points at its best. Outlier rate is high (over 10%). Outliers have a higher mean overlap fraction than non-outliers.
- **Spectral:** within ±1 DER point of VBx at its best. Eigengap undercounts on some meetings when pruning is light (high `pruning` values).
- **Any method beating VBx by more than 1 DER point** is treated as a suspected bug until the cache key, the manifest's `clustering_config` and the per-file CSVs have been checked.

### 7. Hard-fails (stop that workstream, write it up under Blocked on Andrew, then continue with the next unblocked workstream)

1. Two different configurations produce the same final cache key.
2. Any sweep point after WS0 reports an intermediate-cache miss or write. That means the GPU is recomputing and the key has drifted.
3. The manifest's `clustering_config` doesn't match the requested class and values.
4. Three or more consecutive sweep points produce byte-identical per-file CSVs. Investigate before continuing; a stuck setting looks exactly like this.
5. `cluster()` returns a negative label, or any embedding ends up unassigned.
6. The `file` dict carries `annotation` or `num_speakers` during any run.
7. A "best setting" is selected using report-half numbers.
8. Any oracle condition is run.

### 8. Run ledger

`runs/sweeps/clustering-benchmark/SWEEP_INDEX.md`: one row per point with these columns:

`point_id | WS | method | params | run_id | DER tune | DER report | DER all | MD s | FA s | Conf s | Conf % of error | der_overlap_system | der_overlap_assigned | JER | overlap caught / missed / invented s | count MAE | signed count bias | exact/16 | outlier rate (HDBSCAN) | est. k range (spectral) | notes`

All values in the main columns are for all 16 meetings. The per-file CSV path in the point's folder holds the per-meeting breakdown.

Also keep `runs/sweeps/clustering-benchmark/all_points_per_file.csv`, the long-format concatenation of every point's per-file CSV with `point_id`, `method` and `half` columns added. Error profiles and the best/worst-meeting lists are computed from this file.

Point IDs: `AGG-1a-01`, `HDB-2a-03`, `SPC-3a-02` and so on. Update the index after every point. Update `TASKS.md` after every step.

### 9. Boundaries

- **Free:**
  - add the two new clustering classes and their enum and vocabulary entries;
  - add tests;
  - add an analysis script under `harness/`;
  - run any sweep point listed in §5;
  - add up to 3 edge points per method where a best point sits at the edge of its grid;
  - commit to the `clustering-benchmark` branch.
- **Park and continue** (write the proposal under Blocked on Andrew, keep working on other workstreams):
  - any grid change beyond the edge-point rule;
  - any change to `harness/cache_id.py`, `runner.py` or `run_manifest.py`;
  - logging the outlier diagnostic needing more than a small change.
- **Hard stop:**
  - anything needing a GPU cold run (an intermediate-cache rebuild);
  - deleting cache entries or past runs;
  - pushing to GitHub.
- **Never:**
  - change `VBxClustering`, `AgglomerativeClustering` or `KMeansClustering`;
  - edit past manifests or the baseline CSV;
  - run oracle conditions;
  - set `counts_toward_results` on a dirty tree.
- **Autonomy rule:** when a step doesn't need Andrew's input, keep going. Put status notes in the same message as your next action. Stop and ask only when you can't continue without him, or before anything destructive.

### 10. Budget and stops

- **Compute:** about 80 sweep points at around 35 s each, roughly 1 hour of GPU-free compute, plus implementation time.
- **Wall-clock cap:** 10 hours of agent time. Report at 8 hours.
- **Stop when:**
  - all of §5 is done; or
  - the cap is reached; or
  - every remaining workstream is blocked.
- Don't use fast or priority model modes.

### 11. Run-end report (and any pause)

Start with these three headings, in plain language first, details after:

- **Blocked on Andrew:** decisions or approvals needed.
- **Changed:** what was added (classes, tests, commits), how many points ran, where the table and charts are.
- **Found:** the headline result per method in one sentence each, surprises, unverified assumptions and caveats.

### 12. On completion

Follow the vault workflow in `CLAUDE.md`:

- Convert this ticket into `Docs/Clustering Benchmark Results.md`, written as reference material (methods, settings tried, results, charts).
- Convert [[clustering-agglomerative]], [[clustering-hdbscan]] and [[clustering-spectral-eigengap]] into `Docs/Clustering Method - <Name>.md`.
- Set `status: done` and move them out of `Tickets/Open/`.

## Blocked on Andrew

_(newest first)_

- **2026-10-03: Ward linkage needs a different threshold scale (grid change, parked per §9).** Ward's 1d grid (0.4-1.4) behaves "backwards": 0.4 collapses to one cluster, 1.4 gives MAE 64 / DER 92%. Cause, checked on IS1009a's cached embeddings (unit-normalised, unfiltered): ward's last merge heights are 6.5, 7.4, 7.7, 12.7, 19.3, 38.5 vs centroid's ~1.1. At 0.4-1.4 ward is cutting deep inside speakers; every cluster is tiny, below min_cluster_size 12, so at 0.4 nothing is "large" and everything becomes one cluster, and as the threshold rises more fragments pass 12 and the count explodes. Its speaker-level range is roughly 5-20. **Proposal:** 6 extra ward points at threshold 4, 6, 8, 12, 16, 20 (~4 min). Caveat: >2.0 is outside `AgglomerativeClustering.threshold`'s declared `Uniform(0.0, 2.0)` search space (instantiate does not enforce it, but it is off the class's own map). **Default if no answer:** report ward as "not tested on its own scale" rather than as a loser.

## Implementation Notes

_(append while working: decisions, rejected options, gotchas, files touched)_

### 2026-10-03 kickoff: decisions from Andrew
- Blanket approval for every harness run in §5; no per-run asks.
- `--run-condition` is a closed vocabulary (`harness/run_manifest.py` `VALID_CONDITIONS`). Adding `clustering_agglomerative`, `clustering_hdbscan`, `clustering_spectral` there (additive, with test) rather than labelling sweep points `baseline`, which would merge them into the baseline row of the cross-condition table.
- Data root: `../AMI-diarization-setup/harness-data` (sibling of the repo).
- HDBSCAN: follow this ticket (fold outliers into nearest centroid). Report BOTH the outlier count (what HDBSCAN left unassigned) and DER after folding.
- Count the embeddings dropped by `BaseClustering.filter_embeddings` (needs >= 20% of the chunk as single-speaker frames) per meeting. Those are not "failed": they are excluded from forming clusters but still assigned to the nearest centroid by `assign_embeddings`, which assigns ALL embeddings.
- Spectral: this ticket's grid (`pruning` x `max_speakers`) wins over the older note's cosine/RBF + refinement-flag design; the per-file estimated k at each point gives the standalone counting result.
- Predictions are the planner's, not Andrew's; record them but don't weight them.
- min/max clusters: the pipeline always passes min=1, max=num_embeddings via `set_num_clusters`. Clamping is a no-op at those values. Andrew OK'd adjusting/removing the clamp if a boundary is ever reached.
- Branch `clustering-benchmark` off `oracle-experiment` (after committing loose files, 92114560).
- Baseline reference run: `db977741` (cold, batch-key rebuild), not `a1f4391c` cited in [[clustering-vbx]]; same DER.

### Assumptions register (kickoff)
- A1 verified: `CLUSTERING_MODEL_CLASSES` maps vocab -> enum name (run_manifest.py:99); pipeline resolves via `Clustering[clustering]` (speaker_diarization.py:284).
- A4 verified: scikit-learn 1.9.0, `sklearn.cluster.HDBSCAN` imports. scipy 1.18.0, numpy 2.5.1.
- A7 unverified (no network check): agglomerative centroid/12/0.70 point labelled "3.1-like setting (unverified)".
- A2, A3, A5, A6, A8: pending tests / first runs.

### WS0 (2026-10-03)
- Tooling: `harness/clustering_benchmark.py` (driver + ledger: `baseline` / `run` / `index`). Each point goes through `run_harness.py`; the driver checks §7.1-7.4 on every point (final-key collision, cache tally must be exactly `16 full hit(s), 0 partial, 0 miss(es), 0 write(s)`, manifest `clustering_config` class + values, 3 consecutive identical CSVs) and regenerates `SWEEP_INDEX.md` from `points.jsonl`.
- Harness hook: `run_harness.py --clustering-diagnostics-dir`. Reads an optional `last_diagnostics` attribute the new clustering classes set on themselves; reset to None before every file so a final-RTTM cache hit can't inherit the previous file's diagnostics. Existing classes untouched. Plain attributes on pyannote `Pipeline` objects are fine and don't enter the cache key (that reads `_flattened_parameters` only).
- `.gitignore`: sweep outputs under `runs/sweeps/` are tracked (results, like manifests); run logs are not.
- Baseline reference (db977741): all 17.05 / tune 16.26 / report 17.73. Half-level DER reproduces summary DER exactly. VBx over-counts on 4/16 meetings (2 per half), never under-counts.
- **Gotcha, ticket §5 WS0.4:** "point `--cache-dir` at a fresh empty folder" would ALSO empty the intermediate tier (it lives at `<cache_dir>/intermediates/`) and force a cold GPU run, which is a §9 hard stop. Used a fresh scratch folder holding a COPY of `.harness_cache/intermediates` instead: RTTM tier empty, intermediate tier warm.
- AGG-0-01 (centroid/12/0.70, "3.1-like setting (unverified)"): DER 17.94 (tune 17.98, report 17.89), MAE 0.50, 11/16. 16 full hits, 0 writes; manifest `AgglomerativeClustering(method='centroid', min_cluster_size=12, threshold=0.7)`. ~37 s per point.
- AGG-0-02 (same, fresh RTTM cache): DER identical to 0.0, per-file CSV byte-identical. **A6 (agglomerative) and A8 verified.**

### WS1 note: final-RTTM repeats (2026-10-03)
- AGG-1a-06 (centroid/12/0.70) is the same configuration as AGG-0-01, so it was served whole from the final-RTTM cache: intermediate tally all zeros. The driver first flagged this as §7.2 hard-fail; it is not one (nothing recomputed). Driver now accepts an all-zero tally only when the final key matches an earlier recorded point, and notes the repeat. Two failed attempts left manifests in `runs/` (kept, never delete runs); ledger has the passing record only.
- Gotcha: the §7.1 collision check compared params as strings (`0.7` vs `0.70`); now numeric.

### WS2 / WS3 implementation (2026-10-03)
- `HDBSCANClustering` and `SpectralEigengapClustering` added to `src/pyannote/audio/pipelines/clustering.py` + `Clustering` enum; `hdbscan`, `spectral` added to `VALID_CLUSTERING_MODELS` / `CLUSTERING_MODEL_CLASSES`. Existing classes untouched. `tests/test_clustering_model_selection.py`: closed-vocabulary test and `RUNNABLE_MODELS` updated for the two new names.
- Tests in `tests/test_clustering_benchmark_methods.py`, shown failing (class missing) before implementation, then passing (108 across the related suites).
- HDBSCAN: sklearn 1.9 `HDBSCAN(metric="euclidean", copy=True)` on unit-normalised embeddings; `min_samples=0` -> None. Noise (any negative label) folded into nearest non-noise centroid by cosine inside `cluster()`; all-noise -> all zeros; labels renumbered 0..K-1. Count bounds are not forced (pipeline only passes 1 and N).
- HDBSCAN diagnostics without touching the base class: the subclass overrides `filter_embeddings`, calls the base one unchanged, and stashes chunk/speaker indices plus per-slot overlap fraction (share of that local speaker's active frames with >= 2 speakers active, from the binarised segmentation `__call__` already receives). `cluster()` turns that into `last_diagnostics` (num_train, num_outliers, num_dropped_by_filter, mean overlap of outliers / inliers / dropped, one row per training embedding). The harness writes the rows; the driver renames them to `outliers.csv`. "Dropped by filter" = active, valid embedding with < 20% solo frames (still assigned later by `assign_embeddings`, just never used to form clusters).
- Fixture gotcha: HDBSCAN EOM absorbs a near-blob point into the blob unless it is very far off (cosine 0.9 from the blob); the fold test asserts the point really was noise first so it can't pass vacuously.
- Spectral: affinity = cosine with negatives zeroed, top ceil(pruning*N) entries kept per row (self-similarity included, it is always a row max), symmetrised. Eigengap over the smallest `max_speakers + 1` eigenvalues of the normalised Laplacian (`scipy.linalg.eigh` subset), k = 1 + argmax(diff) so k in [1, max_speakers] (reading of §5 WS3.5: "first max_speakers eigenvalues" gives only max_speakers-1 gaps, which would make max_speakers unreachable). Partition with sklearn `SpectralClustering(affinity="precomputed", random_state=42)`. Synthetic timing: N=3000 3.1 s, N=6000 12.4 s, so dense eigh is fine (no eigsh switch needed unless the real first point says otherwise).
- A2 verified for both new classes (final keys differ by a hyperparameter, unit tests).

### Finding: clustering can only move Confusion here (2026-10-03)
- Across every WS1 point so far, MD (2925.0 s), FA (1099.1 s) and all three overlap regions are identical to VBx to the second. Only Confusion and speaker count move.
- Mechanism, from source: the frame-level number of active speakers comes from the segmentation alone (`speaker_count(binarized_segmentations)`, speaker_diarization.py:625), and `reconstruct` / `to_diarization` picks the top-`count` clusters per frame. When there are fewer clusters than `count`, extra labels are created (comment at :784). So a method that merges two overlapping speakers cannot erase the overlap: the second slot is still filled, by another cluster or a padded label.
- Consequence: the ticket's §1 expectation ("merging overlapping speakers shrinks region_d_minus_t / region_t_and_d and raises MD") does not apply to this pipeline. All between-method DER differences are Confusion differences. Also explains AGG-1a-11/12: at threshold >= 1.10 every meeting collapses to one cluster but outputs exactly 2 speakers (the padded overlap label), giving byte-identical results that are real saturation, not a stuck setting.
- Tie note: AGG-1a-07 (0.75) and AGG-1a-08 (0.80) tie on tune DER (16.10) but differ by 2 points on report (17.54 vs 19.58): the split is doing its job.

### WS1 1a/1b done (2026-10-03)
- Centroid: minimum at 0.75 (tune 16.10, report 17.54, all 16.87, MAE 0.188, 13/16). Average: minimum at 0.60 (tune 16.16, report 17.77, all 17.03). Both blow up below 0.5 (over-splitting) and above ~0.9 (merging); >= 1.10 every meeting collapses to one cluster (2 output speakers via overlap padding), identical across linkages.
- Count MAE is not monotonic at the low end (centroid 0.30: 3.6, 0.40: 6.1): at very low thresholds most clusters are below min_cluster_size 12 and get absorbed into a few large ones, so the count is set by how many clusters reach 12, not by the threshold.
- 1c selection (tune half only): best of 1a/1b = AGG-1a-07, centroid / 0.75 (tune 16.10, tune MAE 0.125). 1c sweeps min_cluster_size 1/3/6/12/20 there; 1c-04 (=12) repeats 1a-07 and is served from the RTTM cache.

### §7.4 trip on single linkage (2026-10-03)
- AGG-1d-07/08/09 (single, 0.6/0.8/1.0) produced byte-identical per-file CSVs -> §7.4 hard-fail. Investigated: distinct `clustering_config` and distinct final keys for all three; every meeting outputs exactly 2 speakers; the CSV hash equals AGG-1a-11's (centroid 1.10), the verified single-cluster collapse. Single linkage chains everything into one cluster from 0.6 up. Genuine result, not a stuck setting.
- Driver now exempts ONLY an exact match to the AGG-1a-11 collapse signature from §7.4, and labels such points "single-cluster collapse". Backfilled the label on 6 earlier collapse points (1a-12, 1b-11, 1b-12, 1d-05, 1d-07, 1d-08).
- Gotcha: retrying with `--force` after the trip was served from the final-RTTM cache AND overwrote the first attempt's clean-tally record, so the repeat exemption had no evidence. Recomputed 1d-09 in the scratch cache dir (copied intermediates, no RTTMs): 16 full hits. Lesson: retry a hard-failed point in a fresh RTTM cache dir, not with a bare `--force`. Extra manifests from the retries remain in `runs/`.
- 1c result: centroid/0.75 with min_cluster_size 20 (AGG-1c-05): all 16.68, report 17.19, MAE 0.062, 15/16; ties 1c-04 on tune DER (16.10) and wins on tune MAE. min_cluster_size 1/3/6 over-count massively (MAE 43.6/15.9/9.3): small clusters are what keep the count down.

## Results

_(filled in WS4)_

## Error profiles

_(filled in WS4)_
