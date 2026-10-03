---
status: in-progress
created: 2026-10-03
type: long-running-agent
aliases: [assignment classifier, overlap assignment classifier]
---

# Assignment classifier for hard-to-place voice clips

## In short (read this part)

**The problem.** The pipeline cuts each meeting into 10-second chunks and makes one embedding (a voice fingerprint) per voice in each chunk. Clustering groups the clean fingerprints into speakers. But about **38% of fingerprints never take part in clustering**: their speaker talks alone for less than 20% of the chunk, usually because they're talking over someone. These are attached to a speaker afterwards by a simple rule, "nearest centroid": whichever speaker's average fingerprint is closest. With HDBSCAN, the fingerprints it marks as outliers get the same treatment.

**What we're testing.** Can a small trained classifier make that decision better than "nearest centroid"? For each hard-to-place fingerprint, it looks at every speaker in the meeting and scores "is this the right speaker?" using clues such as:
- how similar the fingerprint is to each speaker;
- how much overlap the fingerprint contains;
- which speaker this same voice was given in the neighbouring chunks;
- that two voices in the same chunk can't be the same person.

**Where it runs.** On top of two clustering methods: **VBx** (the default) and **HDBSCAN** (leaf, 40).

**Classifiers tried, simplest first:**

| Option | What it is | Trained? |
|---|---|---|
| Nearest centroid | Today's rule. The floor everything must beat. | No |
| Constrained assignment | Nearest centroid, plus "no two voices in one chunk get the same speaker" | No |
| Nearest neighbours | Compare against several confident examples of each speaker, not one average | No |
| Logistic regression | A weighted sum of the clues | Yes |
| Gradient-boosted trees | Many small decision trees combined. Usually strongest for this kind of data. | Yes |
| Small neural net (MLP) | A small learned model over the same clues | Yes |

**Data.**
- **Training:** the full AMI **train** split, 136 meetings, about 80 hours. It has to be downloaded first.
- **Choosing settings:** the AMI **dev** split, 18 meetings, also downloaded.
- **Final test:** the same **16 test meetings** as every other result, so all numbers stay comparable to VBx's 17.05%. Each final model is tested **once**.
- **One caveat:** community-1's own models were very likely trained on AMI train. The training examples may therefore look "cleaner" than unseen meetings. Dev is used to check how big that effect is, and one extra run trains on dev only for comparison.

**What we get at the end:**

1. A results table: each classifier × each clustering method against nearest centroid and against VBx.
2. **How often each classifier picks the right speaker** for the hard-to-place fingerprints. This is the most direct measure and much more sensitive than DER.
3. The full error breakdown per run (missed, false alarm, confusion, overlap error, JER, speaker count), as in the clustering benchmark.
4. Which clues mattered most.
5. A short plain-language summary.

**How we judge it:**

- **Pair accuracy:** on the hard-to-place fingerprints in the 16 test meetings, the share given the correct speaker. Higher is better.
- **DER, confusion and overlap-assigned DER** on the 16 test meetings. Lower is better.
- **What success could look like:** the earlier oracle experiment showed that **perfect** assignment would lower DER by at most **0.97 points** if every fingerprint were fixed, or **0.47 points** if only the overlap-heavy ones were. A real classifier will land below those. Recovering a meaningful share of the 0.47 would be a good result; recovering none is still a reportable finding.
- **Small differences don't count:** a DER gap under about 0.3 points, or a confidence interval that crosses zero, is reported as "no clear difference". Pair accuracy differences are reported with their intervals too.

**When the agent needs you:**
- **It must stop and wait** before deleting any past runs or cache files, or pushing to GitHub.
- **It leaves you a note and keeps working** if something goes wrong or it wants to change the plan. Notes go in "Blocked on Andrew" at the bottom.
- **It asks you a few questions at the start**, each with a suggested answer. If you don't reply, it uses the suggestions.
- **Downloading data and long GPU runs** (about 7 hours in total) are allowed.

**Not in this ticket:** VoxConverse, SDM, retraining the segmentation or embedding models, new clustering methods, changing how many speakers are active per frame (that belongs to segmentation), and any oracle condition.

---

## For the agent (detailed spec)

Everything below is instructions for the executor. The plain-language section above is the source of truth for goals. If anything here contradicts it, stop and flag it.

### 0. Kickoff protocol (before any work)

1. Read this ticket and: [[Clustering Benchmark Results]], [[Post-Clustering Refinement Hook]], [[Oracle Assignment Strategy]], [[Oracle Ceiling Metrics]], [[Pre-Clustering Cache]], [[AMI Corpus Setup]], [[Evaluation Harness]], [[Run Manifest]]. Report contradictions. This ticket wins on scope; the docs win on facts about the code.
2. Work through the assumptions register (§3), marking each row verified or not with file:line or command output.
3. Create `runs/sweeps/assignment-classifier/` with `TASKS.md` (checklist; re-read after any context reset) and `RUN_INDEX.md` (§8).
4. Check that `.claude/settings.json` allows `python run_harness.py`, `git add` and `git commit` without a prompt. If not, ask Andrew under "blocking now". The run can't be unattended otherwise.
5. Send Andrew prioritised questions in plain language (blocking now / needed soon / housekeeping), each with a recommended default.
6. Set `status: in-progress` and start WS0.

### 1. Definitions

- **Pair:** one `(chunk, local_speaker)` unit with one embedding. This is the unit of assignment ([[Post-Clustering Refinement Hook]]).
- **Active pair:** a pair whose local speaker has any active frame in the binarised segmentation for that chunk.
- **Target pair:** the pairs the classifier decides.
  - **Held-back:** an active pair that `BaseClustering.filter_embeddings` excluded from the training set of clustering (clean single-speaker frames < 20% of the chunk, or NaN embedding).
  - **HDBSCAN outlier:** a pair HDBSCAN labelled -1 before the fold-in, read from the clustering's `last_diagnostics`.
  - Under VBx, target = held-back. Under HDBSCAN, target = held-back ∪ outliers.
  - Non-target pairs keep the clustering's label, untouched.
- **Candidate speakers:** the clusters the clustering produced for that meeting. The classifier may only choose among these. It never creates or deletes a cluster, which mirrors the honest-ceiling constraint in [[Oracle Assignment Strategy]].
- **Correct label for a target pair:** the cluster the oracle strategy would choose for it under `all_pairs`, meaning the dominant reference speaker of the pair's support, mapped to a cluster. Reuse the mapping and dominant-speaker code in `harness/refinement.py` (`make_oracle_strategy`, `_dominant_reference_speaker`). Don't write a second labeller.
  - Pairs with empty support, or whose reference speaker maps to no cluster, are **unlabelled**. They're excluded from training and from pair accuracy, still decided by the classifier at inference, and counted in DER.
- **Splits:**
  - **train** = AMI train (only_words protocol), used for fitting;
  - **dev** = AMI dev (18 meetings), used for choosing classifier settings and for the shift check;
  - **test** = the 16 AMI test meetings, final evaluation only.
- **Pair accuracy:** correct target pairs / labelled target pairs, over a split. Report it overall and in overlap-fraction bins: `0`, `(0, 0.25]`, `(0.25, 0.5]`, `(0.5, 1]`.
- **Overlap fraction of a pair:** the share of that local speaker's active frames in the chunk where 2+ local speakers are active (same definition as the clustering benchmark's `outliers.csv`).
- **Error metrics:** every run records the full scorer set exactly as in [[Clustering Benchmark Results]] §Error profiles: `der`, MD, FA, Conf, `der_overlap_system`, `der_overlap_assigned`, `jer`, the region census, and count MAE / signed bias / exact.
  - Expected: MD, FA and the region census are **identical** to the underlying clustering's run, because refinement only relabels pairs. Any change there is a hard-fail (§7).

### 2. Current state (verified 2026-10-03 by the planner; re-verify)

| Area | What's true | Status |
|---|---|---|
| Data on disk | AMI IHM **test only** (16 meetings) at `~/Code/AMI-diarization-setup/harness-data/IHM/` | verified ([[AMI Corpus Setup]]) |
| Download scripts | `pyannote/download_ami_test_split.sh` (test-only, with https + `curl --fail --location`, CRLF strip, IS1009 WAV repair), `pyannote/convert_for_harness.sh <data-root> [split]` | verified (doc) |
| Refinement hook | `strategy(embeddings, hard_clusters, soft_clusters, centroids, segmentations) -> hard_clusters`, set via `pipeline.refinement`, selected with `--refinement-strategy`; registry in `harness/refinement.py` | verified (doc + refinement.py) |
| Refinement in cache key | `refinement_id` is part of the final-hypothesis key | verified ([[Oracle Assignment Strategy]]) |
| Oracle ceilings (VBx, test) | all_pairs 17.05 → 16.08 (−0.97); overlap_degraded → 16.58 (−0.47) | verified (doc) |
| Clustering benchmark | HDBSCAN leaf/40 = DER 16.86; 18.3% outliers; held-back = 38.1% of active embeddings | verified ([[Clustering Benchmark Results]]) |
| HDBSCAN diagnostics | `run_harness.py --clustering-diagnostics-dir` writes `last_diagnostics` | verified (benchmark doc) |
| Baseline on test | VBx DER 0.17048543579940637 (`runs/20260925T101811Z-db977741.json`) | verified |

### 3. Assumptions register

| # | Claim | Source | Status | How to verify |
|---|---|---|---|---|
| A1 | `AMI-diarization-setup` ships `train` and `dev` meeting lists and only_words RTTM/UEM for both, and the train list has 136 meetings | planner memory | **verified** 2026-10-03: train 136, dev 18, test 16 (`lists/*.meetings.txt`, CRLF-stripped); only_words RTTM and UEM present for all (136/18/16) | count lines in the lists; record the real numbers |
| A2 | Some train/dev WAVs have malformed headers like IS1009 | AMI Corpus Setup gotchas | **verified**: dev 4 repaired (IS1008a–d), train 21 repaired (IS1000a/b/d, IS1002b/c, IS1003b/c/d, IS1004a–d, IS1005a/b/c, IS1006a/c/d, IS1007a/b/c), data chunk ending 74–1,383 bytes before file end. After repair all 154 open with `wave`, are 16 kHz and cover their UEM, and the last second decodes. ES2010d is stereo with byte-identical channels; the pipeline downmixes (speaker_diarization.py:280), so harmless | for every WAV, compare the header-declared length to the file size and load it with torchaudio; repair with the same method as IS1009 |
| A3 | Under VBx, `filter_embeddings` decides the clustering training set the same way as for the `cluster()` classes (VBx overrides `__call__`) | planner reading | **verified with a catch**: same `filter_embeddings` rule (clustering.py:584), but centroids are VBx-responsibility-weighted (:619-620), not plain means, and assignment is a per-chunk Hungarian over all pairs (`constrained_assignment=True`, :559, :658-663). See Implementation Notes C1 | read `VBxClustering.__call__`; record exactly how it picks the training set and builds centroids |
| A4 | `centroids` and `soft_clusters` passed to refinement are non-None and aligned with `hard_clusters` labels for VBx and HDBSCAN | refinement doc | **verified** on IS1009a for both methods (`tests/test_assignment_common.py`, 10 passed): soft shape (c, s, K) with K = centroids rows, every active label in [0, K) and every centroid used; HDBSCAN hard = argmax(soft) exactly; VBx active pairs in a chunk never share a cluster | assert in a test on one cached meeting per method |
| A5 | The refinement hook receives every pair's embedding, including held-back ones (NaN only where extraction failed) | refinement doc | **verified** on one cached test meeting: 1,123 held-back of 3,276 active, 0 NaN. Inactive slots also have non-NaN embeddings, so targets must be filtered by activity | count non-NaN held-back embeddings on one meeting |
| A6 | Intermediates for train/dev are keyed the same way as test, so one cold run per split caches them for both VBx and HDBSCAN | Pre-Clustering Cache | reported | the cache tally on the second clustering method's run over train: all full hits |
| A7 | community-1's segmentation/embedding models were trained on data including AMI train | planner belief | unverified | check the model cards / pyannote papers if reachable; otherwise state it as "likely" in the write-up |
| A8 | The final-key `refinement_id` must change when the classifier's weights change | Oracle Assignment Strategy lesson | required | §5 WS4: include a hash of the saved model file in `refinement_id`; test two models give different keys |
| A9 | Disk: about 10–11 GB for train+dev audio, under 1 GB for the new intermediates | estimate (16 kHz mono) | **verified (space)**: 121 GB free on C: before download; actual size recorded in WS1 | `df` before download; record the actual |

### 4. Operating conditions

- **Machine:** Windows PC, RTX 3060, `diarisation-env`, batch size 32. Git Bash: use `curl --fail --location`, strip CRLF, and use hardlinks rather than symlinks (see [[AMI Corpus Setup]] gotchas).
- **Owner availability:** Andrew may be away. Keep working on anything unblocked. Questions go in `## Blocked on Andrew`, newest first.
- **Branch:** `assignment-classifier`, off the current branch. Commit after each workstream. Don't push.
- **Outputs:** everything under `runs/sweeps/assignment-classifier/`. Manifests stay in `runs/`. Trained models go in `runs/sweeps/assignment-classifier/models/<model_id>/` with `model.joblib`, `config.json` (features, settings, training meeting list, seed) and `sha256.txt`.
- **Seeds:** 0 everywhere.
- **Test discipline:** test-split reference RTTMs are read **only** by the scorer and the pair-accuracy evaluator. Never by feature building, training or selection.

### 5. Workstreams (in order)

#### WS0: Setup and checks

1. Verify A1, A3, A4, A5 and A9.
2. Confirm that `identity` refinement on test reproduces VBx DER 0.17048543579940637 (cold-equivalent; the warm RTTM figure is about 8e-6 off, which is expected) and HDBSCAN leaf/40 DER 16.86% from the benchmark.

#### WS1: Get the data

1. Write `pyannote/download_ami_split.sh <split> <dest>` as a generalised copy of the test-only script, keeping all its fixes. Download **train** and **dev** Mix-Headset audio.
2. Check every WAV (A2). Repair any malformed file and list them in Implementation Notes.
3. Run `convert_for_harness.sh` for `train` and `dev`. Check that the RTTM/UEM meeting counts match the audio counts.
4. Run `run_harness.py` cold over **dev** then **train** with VBx and `identity`. This populates the intermediate cache and gives baseline DER on both splits (record them; they show the shift).
5. Run HDBSCAN leaf/40 with `identity` over dev and train (CPU, cache hits only).

#### WS2: Build the pair dataset

Write `harness/assignment_dataset.py`. For every meeting in train, dev and test, and for each clustering method (VBx, HDBSCAN leaf/40), it rebuilds clustering from cached intermediates (CPU) and writes one row per **(target pair, candidate cluster)** to `runs/sweeps/assignment-classifier/data/<split>_<method>.parquet`, with these columns:

- **ids:** `uri, chunk_idx, speaker_idx, candidate, is_label` (`is_label` is 1 if the candidate is the correct label, 0 otherwise, null if the pair is unlabelled), `is_outlier` (HDBSCAN), `current_label` (what nearest-centroid gave).
- **Similarity:**
  - `cos_centroid`: cosine to the candidate's centroid;
  - `rank`: of the candidate by `cos_centroid` (0 = closest);
  - `margin`: `cos_centroid` minus the best *other* candidate's `cos_centroid`;
  - `soft_score`: `soft_clusters[chunk, speaker, candidate]` if available, else NaN;
  - `knn_sim`: mean cosine to the 10 most similar **non-target** pairs currently in the candidate.
- **Quality:**
  - `overlap_fraction`;
  - `clean_ratio`: clean single-speaker frames / frames in the chunk;
  - `active_ratio`: active frames / frames in the chunk;
  - `emb_norm`: the embedding's L2 norm before normalisation.
- **Temporal context:** chunks overlap by 9 s, so a voice can be followed across chunks.
  - For each neighbour chunk at offsets ±1..±5, find the local speaker whose active frames overlap this pair's active frames most in the shared time span (minimum 0.5 s, else none).
  - `ctx_agree_k1`, `ctx_agree_k5`: the fraction of matched neighbours within ±1 and ±5 chunks whose **current label** equals the candidate. Use only non-target neighbours' labels, so the feature doesn't use other uncertain decisions.
  - `cand_activity_30s`: the share of active non-target pairs within ±30 s labelled with the candidate.
- **Constraint and size:**
  - `cand_taken`: 1 if another non-target local speaker in the same chunk already has this candidate;
  - `cand_size`: the candidate's share of the meeting's non-target pairs.

Report per split × method:
- the number of target pairs (held-back / outliers);
- the share labelled;
- **nearest-centroid pair accuracy** overall and per overlap bin.

That last number is the floor, and how close it is to 100% shows how much room there is. Post it as a plain-language note to Andrew. **Don't stop if it's high; it's informational.**

#### WS3: Non-learned options (score on test)

Add these strategies to `REFINEMENT_STRATEGIES`. Each one changes **target pairs only**:

- `nc_target`: nearest centroid on target pairs. Should reproduce the clustering's own output (check: DER identical to `identity`).
- `constrained_target`: nearest-centroid scores, then a per-chunk Hungarian assignment so no two active local speakers in a chunk share a cluster (non-target labels in the chunk are fixed first).
- `knn_target`: assign to the candidate with the highest `knn_sim` (k = 5, 10, 25; pick the best on **dev** pair accuracy).

Run each on test for VBx and HDBSCAN. Record pair accuracy and all error metrics.

#### WS4: Learned classifiers

Use these training settings:

- Fit on **train** rows with non-null `is_label`.
- The scoring model outputs P(candidate correct).
- At inference, each target pair takes the argmax over its candidates, then the same per-chunk Hungarian constraint as `constrained_target` (report results both with and without the constraint).
- Choose settings by **dev pair accuracy**. Ties within 0.2 points go to the simpler model or settings.

Grids (all sklearn, seed 0; standardise features for LR and MLP; NaN `soft_score` gets imputed with -1 plus an `is_missing` flag):

| Model | Settings |
|---|---|
| Logistic regression | `C` ∈ {0.01, 0.1, 1, 10}, `class_weight` ∈ {None, "balanced"} |
| HistGradientBoosting | `learning_rate` ∈ {0.05, 0.1}, `max_depth` ∈ {3, 6, None}, `max_iter` ∈ {200, 500}, `l2_regularization` ∈ {0, 1} |
| MLP | `hidden_layer_sizes` ∈ {(64,), (128, 64)}, `alpha` ∈ {1e-4, 1e-3}, early stopping on 10% of train |

Train **separate models per clustering method** (VBx rows and HDBSCAN rows). Also train one **pooled** model on both and report it as a variant.

Then:

1. Freeze the best setting of each model type per method. Save it with its hash (A8).
2. Wrap it as refinement strategy `classifier:<model_id>`. `refinement_id` must include the model's sha256.
3. Run on **test once** per frozen model × method. Record everything.

#### WS5: Checks on the result (after test results are frozen)

1. **Training-data shift:** compare nearest-centroid pair accuracy and baseline DER across train, dev and test. Also retrain the best model type on **dev only** (settings fixed, no new selection) and run it once on test. Report it next to the train-trained model.
2. **Which clues matter:** for the best model per method, run permutation importance on **dev** by feature group (similarity / quality / temporal / constraint-size). Then retrain without the temporal group and without the similarity group, and evaluate on dev only.
3. **Bootstrap:** use 90% paired intervals over the 16 test meetings (2000 draws, seed 0) for ΔDER and Δpair accuracy against nearest centroid, for each test run.

#### WS6: Write-up (into this ticket's sections, then convert to docs)

1. **Results table:**
   - **Rows:** clustering method × strategy (identity / nc_target / constrained / knn / LR / HGB / MLP / pooled / dev-only).
   - **Columns:**
     - pair accuracy with its interval, plus accuracy per overlap bin;
     - DER, MD, FA, Conf, `der_overlap_assigned`, JER, count MAE / bias;
     - ΔDER vs nearest centroid [90% CI];
     - the share of the oracle ceiling recovered: ΔDER / 0.97 and ΔDER / 0.47 for VBx (no oracle exists for HDBSCAN; say so).
2. **Charts** (PNG in `figures/`): pair accuracy by overlap bin per strategy; and feature-group importance.
3. **The 3 test meetings** that gained and lost most per method versus nearest centroid.
4. **Plain-language summary:** one short paragraph per clustering method. Did any classifier beat nearest centroid clearly? Where (which overlap levels)? Which clues did it rely on? How big was the train/dev/test shift?

### 6. Predictions (record before WS2 and WS4; don't tune toward them)

- Nearest-centroid pair accuracy on target pairs is clearly lower in the high-overlap bins than in the `0` bin.
- `constrained_target` gives a small but non-zero gain at no training cost.
- The best learned model beats nearest centroid on pair accuracy by a clear margin (several points) but improves DER by at most about 0.3 on VBx, well inside the 0.47 overlap-degraded ceiling. Gains show up in Conf and `der_overlap_assigned` only.
- Temporal context features are the most important group.
- Gains on HDBSCAN are larger than on VBx, because it has more target pairs.
- Training-split nearest-centroid accuracy is higher than dev/test (the shift caveat).
- **Any DER gain above 0.97 on VBx is impossible under the honest-ceiling constraint and is a bug** (likely label leakage or a cache key collision).

### 7. Hard-fails (stop that workstream, write it up under Blocked on Andrew, continue with unblocked work)

1. MD, FA or the region census differ between a refinement run and its clustering's `identity` run.
2. A non-target pair's label changes.
3. Two different strategies or models produce the same final cache key.
4. Any code path outside the scorer or pair evaluator reads a test reference RTTM (add a test that patches the loader and fails if test URIs are requested during feature building or training).
5. A model is selected or tuned using test numbers, or a frozen model is retrained and re-run on test without being recorded as a new, labelled model.
6. `nc_target` doesn't reproduce `identity` DER exactly.
7. Unexplained intermediate-cache misses on a split that was already cached (a GPU rebuild is allowed once the cause is understood; record why).
8. Any oracle condition is run, or `annotation`/`num_speakers` is on the `file` dict during inference.

### 8. Run ledger

`runs/sweeps/assignment-classifier/RUN_INDEX.md`, one row per test run:

`run_point | clustering | strategy | model_id | sha256 (8) | run_id | pair acc [CI] | acc by overlap bin | DER | MD | FA | Conf | der_overlap_assigned | JER | count MAE / bias | ΔDER vs NC [CI] | notes`

Plus `dev_selection.csv` (every grid point × dev pair accuracy) and `dataset_stats.md` (WS2 counts and floors per split).

### 9. Boundaries

- **Free:**
  - download AMI train/dev and repair files;
  - add data scripts and the dataset builder;
  - add refinement strategies and tests;
  - long GPU runs to cache train/dev intermediates;
  - a GPU rebuild once the cause is understood and recorded;
  - training and grid search;
  - committing to the `assignment-classifier` branch.
- **Park and continue:**
  - feature additions beyond §5 WS2;
  - grid changes beyond §5 WS4;
  - changes to `cache_id.py`, `runner.py`, `run_manifest.py` or `speaker_diarization.py` beyond registering strategies;
  - anything that would change the clustering methods themselves.
- **Hard stop (the only two):**
  - deleting anything under `runs/`, `.harness_cache/`, the AMI data folders, or past result files;
  - pushing to GitHub.
- **Never:**
  - read test references outside scoring;
  - select or tune on test;
  - change VBx or HDBSCAN settings;
  - run oracle conditions;
  - edit past manifests.
- **Autonomy rule:** when a step doesn't need Andrew's input, keep going. Put status notes in the same message as your next action. Stop and ask only when you can't continue without him, or before anything destructive.
- **Check-in (non-blocking):** after WS3, post the WS2 floors and WS3 table to Andrew in plain language. Continue unless he objects.

### 10. Budget and stops

- **Download:** about 10–11 GB. **GPU:** about 6–7 h cold caching for train + dev (test took 37 min for 8.9 h of audio). Everything after that is CPU.
- **Wall-clock cap:** 20 hours of agent time, including GPU runs. Report at 16 h.
- **Stop when:**
  - §5 is complete; or
  - the cap is reached; or
  - every remaining workstream is blocked.
- Don't use fast or priority model modes.

### 11. Run-end report (and any pause)

Start with these three headings, plain language first:

- **Blocked on Andrew:** decisions or approvals needed.
- **Changed:** data downloaded, code added, models trained, runs done, where the table and charts are.
- **Found:** the headline per clustering method in one sentence each, surprises, unverified assumptions and caveats (including the training-data shift).

### 12. On completion

Follow the vault workflow in `CLAUDE.md`. Convert this ticket into `Docs/Assignment Classifier Results.md`, written as reference material. Also add an `AMI train/dev` section to [[AMI Corpus Setup]] covering the new download, repairs and paths. Set `status: done` and move it out of `Tickets/Open/`.

## Blocked on Andrew

_(newest first)_

### 2026-10-03 — WS3 check-in (informational; continuing unless you object)

**How often the clustering already picks the right speaker for hard-to-place voices (the floor):**

| | dev | test |
|---|---|---|
| VBx as-is | 51.4% | 44.2% |
| VBx with plain nearest centroid | 70.5% | 67.2% |
| HDBSCAN as-is (= nearest centroid) | 75.6% | 72.7% |

Most target voices (62–65%) are more than half overlap. That's where accuracy is lowest: 31–38% for VBx as-is, about 60% for nearest centroid.

**Non-learned options on the 16 test meetings** (ΔDER vs the clustering's own output, 90% interval):

| | pair acc | DER | ΔDER |
|---|---|---|---|
| VBx + nearest centroid | 67.2% | 16.89 | −0.16 [−0.26, −0.06] |
| VBx + "one speaker per chunk" | 44.2% | 17.05 | 0 (identical to VBx by construction) |
| VBx + nearest neighbours (k=25) | 68.6% | **16.85** | **−0.20 [−0.35, −0.06]** |
| HDBSCAN + "one speaker per chunk" | 53.5% | 16.94 | +0.08 [−0.01, +0.19] |
| HDBSCAN + nearest neighbours (k=25) | 72.9% | 16.93 | +0.07 [0.00, +0.16] |

In plain terms:
- On VBx, simply letting hard voices go to their nearest speaker, without the "one speaker per chunk" rule, already gives a small real gain. Nearest neighbours is a touch better.
- On HDBSCAN nothing beats the default yet.
- All DER changes are under the 0.3-point "clear difference" bar. A 23-point jump in pair accuracy buys only 0.16–0.20 DER, because these voices carry little speaking time.

Dev baselines (for the shift check): VBx 16.90%, HDBSCAN 17.19%. On dev, HDBSCAN is *worse* than VBx, the reverse of test.

### 2026-10-03 — the "one speaker per chunk" rule mostly contradicts the correct labels (non-blocking; continuing)

Early look at two test meetings (IS1009a, EN2002c), done through the pair-evaluator path:
- **About 60% of labelled target voices have a "correct" speaker that is the same as another active voice in the same chunk.** In the most overlapped bin (>50% overlap) it's about 70%. The label is the oracle's: the reference speaker who talks most during that voice's frames. For a voice that's mostly overlap, that is often the *other* person talking.
- VBx enforces "no two voices in a chunk share a speaker". So its own answer is right for only **28–33%** of high-overlap target voices, against **57–68%** for plain nearest centroid. Overall it's 39–43% vs 63–71%. Clean voices are 94–97% either way.
- HDBSCAN doesn't enforce the rule: 68–76% on target voices.

What this means for the plan:
- `constrained_target` and the Hungarian step in WS4 will probably *lower* accuracy against these labels. I'll still run them as the ticket asks, always next to the unconstrained version.
- The §6 prediction "constrained gives a small gain" will likely fail. It stays recorded as written.
- *Suggested:* keep the oracle's label definition, because the 0.97 / 0.47 ceilings were measured with it. A "collision-free" label (dominant speaker after excluding co-speakers) would be a second labeller and a different ceiling, so I haven't built one. Tell me if you want that as an extra analysis.

### 2026-10-03 — kickoff questions (defaults in force until you answer)

**Blocking now:** none. `.claude/settings.json` now allows `run_harness.py`, `git add` and `git commit`, so the run can go unattended.

**Needed soon (before WS3):**
1. **What "nearest centroid" means under VBx.** VBx doesn't use plain nearest centroid. It already stops two voices in one chunk from getting the same speaker. So plain nearest centroid on VBx *changes* the output: the post-fix run `20260917T054753Z-32241285` scored DER 16.79% vs 17.05%. The ticket's check "`nc_target` must reproduce `identity`" would therefore fail on VBx for a real reason, not a bug.
   *Suggested:* the floor everything must beat is the clustering's own output (`identity`). `nc_target` (plain nearest centroid on target pairs only) becomes an extra comparison row. The exact-match check applies to HDBSCAN only, where the two really are the same.
2. **Keeping test answers out of the dataset.** WS2 asks for an `is_label` column on test rows too, but hard-fail 4 says only the scorer and the pair evaluator may read test references.
   *Suggested:* test rows get `is_label = null` in the dataset. A separate pair-evaluator module works out test answers on its own, only for pair accuracy.

**Housekeeping:**
3. The CLAUDE.md and settings.json changes are uncommitted. *Suggested:* commit them as the first commit on `assignment-classifier`.
4. [[Post-Clustering Refinement Hook]] says nearest centroid scored "exactly equal to baseline". That number came from the cache bug fixed in `a4a3101a`. *Suggested:* correct that doc when this ticket closes.
5. Classifier runs reuse the existing run conditions (`baseline` for VBx, `clustering_hdbscan`) and are told apart by `refinement_strategy`. *Suggested:* keep it that way; no new condition values.

## Implementation Notes

_(append while working: decisions, rejected options, gotchas, files touched)_

**Kickoff, 2026-10-03**
- Branch `assignment-classifier` created off `clustering-benchmark` (commit `2eded6fe`).
- Contradictions found reading the docs and code:
  - **C1, VBx assignment is constrained.** `VBxClustering(constrained_assignment=True)` by default. Centroids are responsibility-weighted means over the training set (clustering.py:619-620). Every pair is assigned by `constrained_argmax`, a per-chunk Hungarian (:658-663), with inactive slots forced to a floor score. Plain argmax differs, so `nc_target` ≠ `identity` on VBx. HDBSCAN has `constrained_assignment=False`, and `assign_embeddings` recomputes plain-mean centroids over train (outliers already folded in) and takes argmax for every pair. So on HDBSCAN `nc_target` ≡ `identity`. See question 1.
  - **C2, stale doc result.** Refinement Hook doc: `nearest_centroid` DER 0.17049287 "exactly equal". That run (`e9f25911`, 2026-09-16) predates the cache-key fix `a4a3101a` (2026-09-17). The re-run `32241285` gave 0.16792364.
  - **C3, test labels vs hard-fail 4.** See question 2.
  - **C4, script location.** `download_ami_test_split.sh` / `convert_for_harness.sh` live in `~/Code/AMI-diarization-setup/pyannote/`, not this repo, so the new `download_ami_split.sh` goes there too.
  - **C5, closed vocabularies.** `--refinement-strategy` has argparse `choices` (run_harness.py:621) and `VALID_REFINEMENT_STRATEGIES` (run_manifest.py:67). New strategies, including the `classifier:<id>` form, must be registered in both.
- Gotchas:
  - The refinement hook receives the powerset segmentation, which *is* the binarised one (speaker_diarization.py:614-615), so activity masks can be read straight from it.
  - Inactive slots carry non-NaN embeddings, so target pairs must be selected by activity, not by NaN.
  - `jq` is not installed in Git Bash here; use Python or grep for JSON.
  - `.harness_cache/intermediates/` holds 32 entries: 16 current plus 16 orphaned by the batch-size key change.

## Results

_(filled in WS6)_

**WS0, 2026-10-03**
- Identity reproduction on test, with final RTTMs recomputed from a scratch `--cache-dir` holding a copy of `intermediates/` (16/16 full hits in both):
  - VBx `20261003T044356Z-0fa3b41d`: DER **0.17048543579940637**, bit-identical to the cold baseline `db977741`.
  - HDBSCAN leaf/40 `20261003T044452Z-b96eddd3`: DER **0.16855098234923788**, bit-identical to HDB-2b-04 `b15caf86`.
  - MD 2925.05 / FA 1099.07 / region census are identical across the two. Confusion is 1212.15 (VBx) vs 1152.74 (HDBSCAN).
- `harness/assignment_common.py` rebuilds a meeting's clustering on CPU from the cache, with the same arguments `apply()` passes (num_clusters None, min 1, max inf; `set_num_speakers` defaults). It returns hard/soft/centroids plus the active, held-back and outlier masks. Outliers are read from HDBSCAN `last_diagnostics.rows`. Held-back is recomputed with `filter_embeddings`' rule and asserted equal to it in a test.
- `tests/test_assignment_common.py` covers A3/A4/A5 on IS1009a for both methods. Loading the pipeline on CPU is fast (about 2 s), so the tests are cheap.

**WS1 (in progress)**
- `~/Code/AMI-diarization-setup/pyannote/download_ami_split.sh <split> <dest>` reads `lists/<split>.meetings.txt` (CRLF stripped). It downloads with `curl --fail --location` over https and writes a `.done` marker per file so re-runs resume. It then checks every WAV's RIFF size and data-chunk end against the real file size, and rewrites any mismatching file through `wave` (the IS1009 repair).
  - Before use, it was checked against the 16 test files (0 flagged) and against a deliberately truncated scratch file (flagged, then repaired).
- Dev: 18 meetings downloaded, 4 repaired (IS1008a–d), which matches upstream `download_ami.sh`'s hard-coded list. `convert_for_harness.sh` already took a split argument, so no change was needed. dev.rttm/uem cover 18 URIs each.

**WS2, 2026-10-03**
- `harness/assignment_labels.py` is the only reader of references in this ticket's code. `load_reference(split, uri, purpose)` raises `TestReferenceAccess` for a test split unless `purpose == "pair-eval"`. Labels come from the oracle closure's new `strategy.labels` record, an additive change to `make_oracle_strategy` whose output is unchanged; the 31 existing oracle/refinement tests pass. There is no second labeller.
- `harness/assignment_dataset.py` builds `data/<split>_<method>.csv.gz`. It's gzip CSV because pyarrow isn't installed. It runs on 6 worker processes, each loading the pipeline once.
  - Missing-value handling: NaN → -1 at fit time, with flags `soft_missing`, `ctx_missing_k1`, `ctx_missing_k5`. The ticket asked for this treatment for `soft_score`; the context flags extend it to the other NaN-able features.
  - `knn_sim_k5` / `knn_sim_k25` are stored for `knn_target`'s k choice only and are not classifier features.
- Temporal matching aligns frames by time. Frame f of chunk c lines up with frame f − round(d·step/frame_dur) of chunk c+d, which is about 59 frames per 1 s offset. Per-offset overlap is an einsum `cfs,cfo->cso`, so it's cheap. A synthetic test checks slot-swapping across chunks.
- Test datasets: VBx 21,745 target pairs (all held-back). HDBSCAN 28,205 = 21,745 held-back + 6,460 outliers, which matches the benchmark's outlier count exactly. Each built in 28 s.
- Context is missing for 50–66% of target pairs at ±1 and 29–38% at ±5. These voices are short and overlapped, and a neighbour must be a non-target with ≥0.5 s shared activity to count.
- Test floors (`dataset_stats.md`): VBx identity **44.2%**, plain NC 67.2%. HDBSCAN identity = NC = **72.7%**. 62% of VBx targets are in the >0.5 overlap bin, where identity scores 31.2% and NC 60.3%.

**WS3, 2026-10-03 (code)**
- `harness/assignment_strategies.py`: each strategy is a `decide(ClusteredMeeting)`, shared by the offline evaluator and the in-pipeline hook (`build(name, pipeline)`). The hook rebuilds the outlier mask from `pipeline.clustering.last_diagnostics`, which is safe because refinement runs right after clustering in the same `apply()`. `checked_decision` raises `NonTargetLabelChanged` (hard-fail 2).
- Registration: `get_refinement_strategy(name, pipeline=None)` (an optional second argument), `refinement_id_for` (adds model sha256 for `classifier*:`), CLI validation instead of fixed `choices`, and `VALID_REFINEMENT_STRATEGIES` plus a `classifier:` / `classifier_hungarian:` prefix rule in run_manifest. Nothing in runner/cache_id/speaker_diarization changed.
- Tests: hook path == offline path for three strategies; HDBSCAN `nc_target` == identity exactly; existing harness tests (128) pass.
- `harness/assignment_runs.py` runs one test point through `run_harness.py`. It enforces hard-fails 1, 3 and 7, writes `test_runs/<point>/`, appends `runs.jsonl` and rebuilds `RUN_INDEX.md` with bootstrap intervals (90%, 2000 draws, seed 0, paired over meetings).
- Gotcha: the Windows console is cp1252. Set `PYTHONIOENCODING=utf-8` when printing ≤/Δ.
