---
status: done
created: 2026-10-08
---

# Closeout 3: the remaining IHM experiments

## The goal, plainly

Five things on the original AMI IHM condition. They fill real gaps in the report rather than
chasing a better number.

The first is a ten-minute code reading that decides whether [[closeout-4-min-duration-off]] runs.
The GPU items run in the order the master ticket's queue gives: §5, §2, then (after closeout 4)
§3, §4.

## What we get

1. An answer to whether the biggest error in the system has a retraining-free lever, and the
   facts closeout 4 needs.
2. The hyperparameter sweep of the clustering model the pipeline actually uses, which is the only
   direct test of the speaker-counting hypothesis.
3. One missing row in the method table, or a recorded reason it stays untested.
4. A check of whether the best assignment results hold beyond the test meetings.
5. Confirmation that the September 16.79 still reproduces on the current cache.

## How we measure it

Same protocol as everything else: AMI IHM test, 16 meetings, collar 0, overlap not skipped,
tune/report split of 8 and 8, 90% paired bootstrap intervals.

All runs are **warm** — the intermediate cache already holds IHM segmentation and embeddings, so
a 16-meeting run takes about 33 seconds. Only the segmentation check in §1 could change that, and
only if it finds something.

## Done means

Each of the five has either a result or a recorded "not possible, and here is why", written into
`Obsidian-Diarisation/Docs/Closeout Results.md`.

## Not in this ticket

No new clustering methods. No new classifiers. No SDM — that is closeout 2.

---

## For the agent

### 1. Is there anything tunable in segmentation post-processing? — ten minutes, do first

**Why this is first:** missed detection is 2925 s, which is **56% of the system's total error** —
larger than confusion and false alarm combined. It is speech the pipeline never detected, not
speech it misattributed.

If the pipeline exposes any parameter in how the segmentation model's output becomes timed
segments — a minimum duration, a smoothing or merging threshold, anything of that kind — then
tuning it is a retraining-free attack on the single largest error component, and it instantly
becomes the most valuable experiment available.

Powerset segmentation was partly introduced to remove onset/offset threshold tuning, so this may
come back empty. **Ten minutes to find out.**

- Read the pipeline's instantiated parameters and `SpeakerDiarization`'s source for anything
  between the segmentation model's output and the final timeline, **including steps after
  clustering**. Report what exists, with line numbers, or report plainly that nothing is exposed.
- **Expected:** `segmentation.min_duration_off` (fills short silences within one speaker's
  speech on the final timeline). If it exists, record all four of these for closeout 4:
  1. Its shipped value on community-1, read from the instantiated pipeline, not the docs.
  2. Where in `apply()` it is used, with line numbers, and confirmation that this is after
     clustering (so the intermediate cache stays valid).
  3. Whether it is in either cache key. Expected answer: **no**. Neither
     `_intermediate_components` nor `pipeline_config_id` in `harness/cache_id.py` includes it.
  4. Whether `run_harness.py` can set it. Expected answer: **no**. Only `--clustering-param`
     exists (`run_harness.py:652-660`).
- **Do not sweep it here.** Closeout 4 does that, after the harness change.
- **If anything other than `min_duration_off` is exposed, stop and report.** That one is an
  escalation to Andrew.

### 2. VBx hyperparameter sweep

The clustering model the pipeline actually uses, and the one experiment that was never run — the
3 October benchmark explicitly excluded changes to VBx or its settings.

Shipped values: `Fa=0.07, Fb=0.8, threshold=0.6`.

| Parameter | Range | Notes |
|---|---|---|
| `Fb` | `Uniform(0.01, 15.0)` | **Sweep log-spaced.** Three orders of magnitude, and the shipped 0.8 sits near the bottom. |
| `Fa` | `Uniform(0.01, 0.5)` | |
| `threshold` | `Uniform(0.5, 0.8)` | The AHC initialisation threshold |

One axis at a time is about 22 points and 12 minutes. A full 10×6×6 grid is about 3 hours. Either
is affordable; prefer the full grid if nothing else is queued.

Set values with `--clustering-param Fa=... --clustering-param Fb=... --clustering-param
threshold=...`. Use the full grid; the GPU is free overnight.

**Record DER and speaker counting at every point.** The counting curve matters more here than the
DER curve.

**Show the settings took effect** (a manifest echoing the request is not enough):
- Quote the clustering configuration `run_harness.py` prints for at least three points, including
  the two extremes of `Fb`.
- Every point must have a distinct final cache key.
- If DER **and** speaker count are identical at every `Fb` value, treat it as a suspected silent
  no-op and stop, rather than reporting it as a flat result.

**Report the curve only.** No "best setting" is selected or reported from this sweep. If the
written reading mentions where the curve is lowest, it says which half that came from.

**The prediction, already derived from the source and recorded:** higher `Fb` gives fewer
speakers, monotonically. It appears in the update equations only as the ratio `Fa/Fb`
(`vbx.py:109-112`); raising it drives `invL` toward 1 and `alpha` toward 0, collapsing the speaker
means toward the prior. At `Fa=0.07`, as `Fb` goes 0.01 → 15.0, `alpha` falls 0.199 → 0.038 and
`invL` rises 0.003 → 0.811.

**Caveat on record:** that direction rests on the equations, not on a working simulation. A
synthetic end-to-end check did not converge, with the effective count stuck at its initialisation
of 10 for every `Fb`. **This sweep is therefore also the empirical test of the prediction.**

- If speaker count falls monotonically as `Fb` rises, the counting hypothesis has a mechanism:
  `Fb=0.8` is weak regularisation, consistent with the over-splitting seen under clean
  segmentation.
- If it does not, the prediction is falsified, and that is equally reportable.

A DER gain above 1 point over the shipped settings is a **suspected bug** — check the cache key
and the manifest's `clustering_config` before believing it.

### 3. Ward linkage redo

Five agglomerative points were run with a threshold grid on the wrong scale, so they say nothing
about ward and it currently has to be reported as untested.

Ward operates in euclidean space after conversion (`clustering.py:371-375`), so it needs its own
threshold range rather than the cosine one.

**The scale is already worked out** in [[Clustering Benchmark Results]] ("Ward was not tested on
its own scale"): last merges at 6.5–38.5 on IS1009a, against about 1.1 for centroid. Use a grid
of roughly **4–20**. Rerun, about 30 minutes. Choose on the tune half, report on the report half,
as the benchmark did.

**That range is outside the class's declared `Uniform(0.0, 2.0)`.**
- First run one meeting at threshold 10 and confirm the value was accepted and used: the printed
  configuration shows 10, and the speaker count differs from threshold 1.4's.
- If `instantiate()` rejects it, or the value is silently clamped, **stop**. Ward stays
  "untested", with the reason recorded. Do not edit anything under `src/pyannote/` to allow it.

Turns one hole in the method table into a row, or into a recorded reason.

### 4. Confirm the top assignment results on dev and train

The kNN result (`e2ef8a62`, −0.20) is currently the best of roughly thirteen strategies scored on
the same test meetings. That is a selection effect and it is the kind a marker notices.

Dev and train audio are already on disk from the classifier work, and their intermediates are
cached (VBx dev baseline `20261003T055158Z-c9fcfc74`).

- Run `knn_target_k25` and `nc_target` on dev (18 meetings) and train (136), with 90% paired
  bootstrap intervals over that split's meetings.
- Report each one's Δ against that split's own VBx baseline, and whether the ordering holds.

**Two limits, stated next to the result, not in a separate section:**
1. **Train is probably seen data.** community-1's models were likely trained on AMI train
   (assumption A7, unverified). VBx scores 15.65 there, against 16.90 on dev. A result on train
   is a consistency check, not an independent test.
2. **Dev may not be independent.** Find out how k=25 was chosen (check
   `harness/assignment_strategies.py`, `dev_selection.csv` and the classifier ticket's notes)
   and say which applies:
   - k was chosen on dev **DER**: dev confirmation is circular. Say so.
   - k was chosen on dev **pair accuracy**: weaker circularity. Say so.
   - k was fixed in advance, or chosen on test: dev is independent. If chosen on test, that is
     itself a selection finding to report.

The caveat comes off only if the ordering holds on dev **and** dev is independent. Otherwise
report the outcome with the caveat attached — which is exactly what happened to HDBSCAN, and
that worked out well.

### 5. Rerun the September nearest-centroid variant — 30 seconds, first in the GPU queue

`20260917T054753Z-32241285` scored **0.16792364** with `--refinement-strategy nearest_centroid`.
That strategy relabels **every** voice in every chunk by plain argmax over centroids
(`harness/refinement.py:33-69`), unlike `nc_target`, which relabels held-back voices only. The
run predates the 25 September cache-key change.

- Rerun it once, warm, test split, shipped VBx, same strategy.
- Report the new run id, its DER to 8 decimal places, and whether it matches 0.16792364.
- **Match:** 16.79 is confirmed on the current cache. Closeout 1 bootstraps it as a documented
  variant.
- **No match:** that is a finding. Report both numbers and the difference. Do not adjust
  anything to make them agree. This is an escalation to Andrew, because it would mean the current
  cache differs from the one that produced September's results.

It also serves as the queue's first sanity check: it must show 16/16 intermediate hits.

### 6. Hard-fails — stop and report

1. Two configurations produce the same final cache key.
2. An unexplained intermediate-cache miss on IHM, which is already fully cached. **This one halts
   the whole GPU queue**, not just this ticket.
3. The manifest's `clustering_config` does not match what was requested.
4. A best setting is chosen using report-half numbers.
5. Any oracle condition runs.
6. Something tunable other than `min_duration_off` turns up in §1.
7. The VBx sweep curve is flat in both DER and speaker count (suspected no-op).
8. Ward thresholds above 2.0 are rejected or clamped.

### Deliverable

Per item: the result or a recorded impossibility. For the VBx sweep, the `Fb` curve (DER and
speaker count) as a figure and a plain statement of whether the prediction held.

### Feeds

§5 of [[report-plan]] for item 1, §4 figure and §5 mechanism for item 2, §4 for item 3, §4 and §5
limitations for item 4, §4(c) for item 5.

---

## Implementation Notes

### §1 segmentation post-processing check

Result: **one** tunable parameter exists between segmentation output and the final timeline, `segmentation.min_duration_off`, as expected. Nothing else is exposed, so no escalation. CPU only, no inference run; pipeline loaded via the harness's own `build_pipeline` (offline HF cache, `PYANNOTE_SKIP_DEPENDENCY_CHECK=1` because installed version is a dev build, same as `harness/config.py:66`).

**Declared tunable Pipeline parameters (ParamDict/Uniform)**
- `segmentation.min_duration_off = Uniform(0.0, 1.0)`: `src/pyannote/audio/pipelines/speaker_diarization.py:262-265` (powerset branch; community-1 is powerset: instantiated `specifications.powerset` printed `True`).
- Non-powerset branch (not used by community-1) additionally declares `threshold = Uniform(0.1, 0.9)`: `speaker_diarization.py:267-271`. Absent here.
- Clustering (before the end, not post-processing): VBx `threshold Uniform(0.5,0.8)`, `Fa Uniform(0.01,0.5)`, `Fb Uniform(0.01,15.0)`: `src/pyannote/audio/pipelines/clustering.py:568-570`. Already swept via `--clustering-param`.
- `default_parameters()` (`speaker_diarization.py:305-309`) lists only `segmentation.min_duration_off` and the three VBx params.

**Hard-coded constants on the path (not tunable via instantiate)**
- Powerset segmentation is converted to multilabel and not thresholded: `binarized_segmentations = segmentations` (`speaker_diarization.py:615-616`); threshold only binarizes in the non-powerset else-branch (`:617-622`).
- `speaker_count(..., warm_up=(0.0, 0.0))`: `speaker_diarization.py:625-629`; inside it `np.rint` rounding, `hamming=False`, `missing=0.0` (`src/pyannote/audio/pipelines/utils/diarization.py:175-185`).
- Count capped at `max_speakers` (`speaker_diarization.py:700`), inactive speakers forced to -2 (`:705-709`), exclusive count capped at 1 (`:726`).
- `to_diarization`: `hamming=False`, `missing=0.0`, top-`count` speakers per frame (`utils/diarization.py:~239-268`).
- `to_annotation` call: `min_duration_on=0.0` hard-coded (`speaker_diarization.py:719-722` and `:733-736`).
- Inside `to_annotation`: `Binarize(onset=0.5, offset=0.5, min_duration_on=..., min_duration_off=...)`, `utils/diarization.py:211-216`. No `pad_onset`/`pad_offset` passed (Binarize defaults 0.0, `src/pyannote/audio/utils/signal.py` constructor).
- Constructor args, not Parameters: `segmentation_step=0.1`, `embedding_exclude_overlap`, batch sizes (`speaker_diarization.py:208,213,219-220`); community-1 config sets batch 32/32 and `embedding_exclude_overlap: true`.

**min_duration_off: the four facts for closeout 4**

(a) Shipped value on community-1 is **0.0**.
Local HF cache config (`~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf.../config.yaml`):
```
params:
  clustering: {threshold: 0.6, Fa: 0.07, Fb: 0.8}
  segmentation: {min_duration_off: 0.0}
```
Instantiated pipeline (`build_pipeline("pyannote/speaker-diarization-community-1", device="cpu")`, then `p.parameters(instantiated=True)`), exact output:
```
PARAMS {'segmentation': {'min_duration_off': 0.0}, 'clustering': {'threshold': 0.6, 'Fa': 0.07, 'Fb': 0.8}}
FLAT {'segmentation>min_duration_off': 0.0, 'clustering>threshold': 0.6, 'clustering>Fa': 0.07, 'clustering>Fb': 0.8}
```
Consequence: the grid's low end is the shipped value, 0.0. At 0.0, `Binarize` does not even call `support()`, because the guard is `if self.pad_offset > 0.0 or self.pad_onset > 0.0 or self.min_duration_off > 0.0:` (`signal.py:309`), then `active = active.support(collar=self.min_duration_off)` (`:310`). So 0.0 and "feature off" are the same code path. `support(collar)` merges same-label tracks separated by less than `collar` (pyannote.core `Annotation.support` docstring).

(b) Used in `apply()` at `speaker_diarization.py:718-722` (`self.to_annotation(discrete_diarization, min_duration_on=0.0, min_duration_off=self.segmentation.min_duration_off)`) and again at `:732-736` for the exclusive diarization. Order in `apply()`: segmentation `:609`, embeddings `:647`, clustering `:656`, refinement `:672`, reconstruct `:712`, to_annotation `:718`. So it runs strictly after clustering and refinement, and acts only on the already-built discrete diarization. It does not touch segmentation, embeddings or `hard_clusters`. Intermediate cache stays valid.

(c) **Not in either cache key. Expected answer confirmed.**
`harness/cache_id.py:236-274`: `_intermediate_components` returns `[f"checkpoint={checkpoint}"] + _batch_size_components(pipeline)`.
`harness/cache_id.py:295-309`:
```
components = (
    [_FINAL_PREFIX]
    + _intermediate_components(checkpoint, pipeline)
    + _clustering_components(pipeline)
)
```
`_clustering_components` (`:184-188`) reads only `pipeline.clustering`. Empirical check (CPU, same pipeline object, `p.instantiate({"segmentation": {"min_duration_off": 0.5}})`):
```
after override: {'segmentation': {'min_duration_off': 0.5}, 'clustering': {'threshold': 0.6, 'Fa': 0.07, 'Fb': 0.8}}
final same: True inter same: True
```
So the override is applied to the pipeline but **both keys are unchanged**. Without the closeout-4 harness change, a sweep would return the cached baseline RTTM at every value (flat curve). Baseline keys on this machine right now: final `241ecd865d157bd823106180f9691005b3f6245043a8fadb1c62480b9aedbced`, intermediate `9d14609c921c5d50a850db35c99d6a3d279a924c401e1fb540ccaba30f2fec6f`. (These differ from the `190b62c3.../c636fc56...` quoted in `run_harness.py` comments near line 520 because batch size was added to the key after that comment; the comment is stale, not a problem for closeout 4.)

False-premise-adjacent note: the `harness/cache_id.py` module docstring (`:60-63`) names `_final_only_components`, and `grep -n _final_only` finds only that docstring line. It does not exist as code. The closeout 4 ticket already says this ("does not exist yet; create it").

(d) **run_harness.py cannot set it. Expected answer confirmed.** `grep segmentation-param|segmentation_param` over `run_harness.py` and `harness/*.py` returns nothing. Only the clustering flag exists:
- `run_harness.py:343-370` `_parse_clustering_params` (repeatable `--clustering-param name=value`)
- `run_harness.py:538-542`:
```
if clustering_params:
    ...
    pipeline.instantiate({"clustering": dict(clustering_params)})
```
- `run_harness.py:652-660` the `--clustering-param` argparse definition; `:699` parse, `:704` pass to `build_pipeline`.
Ticket line numbers (652-660, 344-370, 542) are accurate. Note `build_pipeline`'s non-default path (`_from_pretrained_with_clustering`, ends with `pipeline.instantiate(shipped)`) re-applies the shipped `segmentation.min_duration_off`, so it would pick up a config value but there is no CLI route to change it.

**Escalation:** none. No tunable other than `min_duration_off` is exposed on this path (hard-fail 6 not triggered). Nothing was swept; no code edited.

### §5 16.79 rerun

**Command** (warm, shared default `.harness_cache`, no `--no-intermediate-cache`):
```
python run_harness.py --data-root "$HOME/Code/AMI-diarization-setup/harness-data" --condition IHM --split test --device cuda --run-condition nearest_centroid --refinement-strategy nearest_centroid --notes "closeout-3 §5 rerun of 20260917T054753Z-32241285: nearest_centroid refinement (relabels every voice by argmax over centroids), shipped VBx (pyannote-default), IHM test, warm shared cache" --counts-toward-results --per-file-csv <scratchpad>/s5_per_file.csv --summary <scratchpad>/s5_summary.json
```
Reconstructed from the old manifest `run_config`: `split=test`, `condition=nearest_centroid`, `refinement_strategy=nearest_centroid`, `clustering_model=pyannote-default`, `mic_condition=IHM`, `corpus=AMI`, `der_collar=0.0`, `der_skip_overlap=false`, `oracle_scope=all_pairs` (default), `counts_toward_results=true`. Data root is not in the manifest; used the repo's `ac.DATA_ROOT`.

**Printed tally:** `intermediate cache: 16 full hit(s), 0 partial, 0 miss(es), 0 write(s) (186 entries, 682.0 MB on disk)`. Also printed: `clustering: VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6) (requested: pyannote-default)`. Git commit 412a6ac6.

**New run:** `20261008T044106Z-868efc2d` (`runs/20261008T044106Z-868efc2d.json`).
- Manifest `summary.der` = 0.16792364 (full: 0.16792363760650816); old manifest `summary.der` = 0.16792363760650816. Identical to every printed digit; all other summary fields (jer, overlap DERs, MD/FA/confusion, region_*) are also identical.
- Re-derived from the per-file CSV: corpus DER = sum(missed_detection + false_alarm + confusion) / sum(reference); reference per file back-derived as error/der. 5157.593843 / 30713.924000 = 0.16792364.
- **Match: yes**, equals 0.16792364 exactly.
- Per-file: old `per_file_nearest_centroid_t5.csv` vs new, 16 meetings, max abs per-file DER difference 0.0, der/MD/FA/confusion columns identical.

Refreshed `runs/comparison.csv` (row added). Note: reading it with pandas/utf-8 fails (byte 0xa7 at pos 104704, likely a non-UTF-8 character in some notes cell); not investigated.

### §2 VBx sweep

**Driver:** `harness/vbx_sweep.py` (resumable; reuses `half_metrics`/`point_metrics`/`_described_params` from `harness/clustering_benchmark.py`). Outputs in `runs/sweeps/vbx-sweep/`: `points.jsonl` (ledger, includes the failed/crashed attempts), `points.csv`, `all_points_per_file.csv`, one folder per point with `per_file.csv`, `summary.json`, `run.log`. Each point is a normal `run_harness.py` run (default clustering model, `--clustering-param Fa/Fb/threshold`, `--refinement-strategy identity`, shared `.harness_cache`, no `--runs-dir`), so each has a `runs/*.json` manifest. Tune half = meetings ending a/c, report half = b/d.

**Grid (exact values).**
- Fb (10): 0.01, 0.02258, 0.05097, 0.1151, 0.2598, **0.8**, 1.324, 2.989, 6.748, 15.0. This is log-spaced at ratio 1500^(1/9); the point nearest 0.8 (0.5865) was replaced by 0.8.
- Fa (6): 0.01, 0.0219, **0.07**, 0.1046, 0.2288, 0.5. Log-spaced; the 0.0478 point was replaced by 0.07.
- threshold (6): 0.5, 0.56, **0.6**, 0.68, 0.74, 0.8. Linear; the 0.62 point was replaced by 0.6.
- Order: shipped point, Fb axis, Fa axis, threshold axis, then the full 10x6x6 grid (de-duplicated, 360 points in total).

**Deviations and findings so far (recorded, not worked around).**
1. **Shipped-baseline DER is not bit-equal to the known 0.17048543579940634.** The point (run `20261008T044956Z-1e0eb477`) scored 0.17049342382952826, a drift of 7.99e-06. Its intermediate tally was all zeros: it was served from the final-RTTM cache (key `241ecd86...`), which holds millisecond-quantised RTTMs. Supporting evidence: its MD is 2925.115 s, against 2925.0495 s at every freshly computed point; per-file DER differs from `per_file_baseline_batchkey_rebuild.csv` by at most 2.1e-05 (mean 8.6e-06) and the per-meeting speaker counts match on all 16. The known 0.1704854 value came from a run that recomputed final RTTMs. The driver's tolerance was first 1e-12, so the first attempt stopped as a hard-fail (kept in `points.jsonl` as a failed record); I then set the baseline tolerance to 5e-05 and recorded the drift. This is a judgement call, flagged for review. Consequence: the baseline point carries about 1e-5 of rounding that no other point has, which is far below the 0.3 resolution.
2. **Fb=0.01 crashes** (Fa=0.07, threshold=0.6): `OverflowError: Python integer 137 out of bounds for int8` at `src/pyannote/audio/pipelines/clustering.py:138` (`hard_clusters[c, s] = k`; `hard_clusters` is int8). VBx at that setting yields more than 127 clusters in at least one meeting. Not editable (boundary), so the point is a recorded crash with no scores. The printed config for it was `clustering: VBxClustering(Fa=0.07, Fb=0.01, threshold=0.6) (requested: pyannote-default)`. The driver records only this specific overflow as a crash; any other non-zero exit or any intermediate miss is a hard stop.

**Fb axis (Fa=0.07, threshold=0.6)**, from `points.csv`:

| Fb | run | DER all | tune | report | MD s | FA s | Conf s | mean hyp speakers | count MAE | signed bias | exact/16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | crashed | | | | | | | | | | |
| 0.02258 | 688567de | 50.95 | 50.56 | 51.28 | 2925.0 | 1099.1 | 11623.2 | 52.500 | 48.562 | +48.562 | 0 |
| 0.05097 | 0c203dc7 | 32.13 | 33.03 | 31.35 | 2925.0 | 1099.1 | 5844.2 | 31.125 | 27.188 | +27.188 | 0 |
| 0.1151 | acc434cd | 23.65 | 22.83 | 24.36 | 2925.0 | 1099.1 | 3240.1 | 16.188 | 12.250 | +12.250 | 0 |
| 0.2598 | 54bcb6ff | 19.94 | 19.21 | 20.57 | 2925.0 | 1099.1 | 2100.1 | 7.625 | 3.688 | +3.688 | 2 |
| 0.8 (shipped) | 1e0eb477 | 17.05 | 16.26 | 17.73 | 2925.1 | 1099.4 | 1212.0 | 4.188 | 0.250 | +0.250 | 12 |
| 1.324 | 7c74ccb4 | 16.94 | 16.38 | 17.43 | 2925.0 | 1099.1 | 1179.7 | 3.875 | 0.188 | -0.062 | 14 |
| 2.989 | cce6f4dc | 16.92 | 16.38 | 17.38 | 2925.0 | 1099.1 | 1171.4 | 3.812 | 0.125 | -0.125 | 15 |
| 6.748 | c8542429 | 17.01 | 16.56 | 17.39 | 2925.0 | 1099.1 | 1199.7 | 3.750 | 0.188 | -0.188 | 14 |
| 15.0 | c16cfc33 | 19.77 | 17.40 | 21.82 | 2925.0 | 1099.1 | 2046.8 | 3.375 | 0.562 | -0.562 | 9 |

Mean reference speakers per meeting = 3.9375. MD and FA do not move with Fb (shared segmentation); only confusion does. Figure: `runs/sweeps/vbx-sweep/fb_axis.png` (script `make_fb_figure.py`).

**Prediction (higher Fb gives fewer speakers, monotonically): held on the nine completed points.** Mean hypothesis count falls at every step: 52.5, 31.1, 16.2, 7.6, 4.19, 3.88, 3.81, 3.75, 3.375. The tenth point (Fb=0.01) crashed from over-splitting, which is the same direction. DER moved a lot, not flat: 50.9% at Fb=0.023, falling to a shallow floor around 16.9-17.0% for Fb between 0.8 and 6.7, then rising to 19.8% at Fb=15 (count under-estimated, bias -0.56). That floor is about 0.1 points below shipped on all-16 DER; the tune/report halves are in the table and the 0.8-6.7 values sit within a few hundredths of a point of each other. No point is more than 1 DER point below shipped (suspect threshold not triggered). Not a no-op: the counts and DER differ at every Fb. The curve only is reported; no setting is selected.

**Full grid finished (second entry, appended after the Fb-axis note above).**
- 360 grid points attempted: 279 scored, 81 crashed in `src/` (78 `OverflowError` int8 at `clustering.py:138`, 3 `ValueError: negative dimensions` at `speaker_diarization.py:522`, all at Fa=0.5). The second crash type appeared at Fa=0.5, Fb=0.8, thr=0.6, which stopped the driver once as a hard-fail (recorded in `points.jsonl`); I then taught the driver to record that exact message as a crash too and resumed. Any other non-zero exit, any intermediate miss, any key collision or manifest mismatch still stops it. Ledger has 362 records = 279 scored + 81 crashed + 2 earlier hard-failed attempts (the first Fb=0.01 and Fa=0.5 tries, before the crash handling existed; both were rerun). The shipped baseline was run once: its first record already held its metrics when the drift check stopped the driver, so after loosening the tolerance the resumed driver kept it and did not rerun it. That record therefore still carries a `failure` note about the 1e-12 tolerance; its numbers are the ones used.
- Checks (`runs/sweeps/vbx-sweep/analyze.py`, output `analysis_output.txt`): 279 scored points, 279 unique final keys; manifest `clustering_config` and `pipeline_config_id` match the request for 279 of 279; 278 of 279 had the 16/16 full-hit tally, the other is the shipped point served from the final-RTTM cache; 1 unique intermediate key. No point gains 1 DER point over shipped (largest gain 0.16). Hard-fail conditions 1, 2, 3, 4, 7 not triggered; no oracle runs.
- New observation: at Fa=0.01, Fb=15 the run crashes on over-splitting (135-295 clusters) for thresholds 0.5-0.68 though the same Fa at Fb=6.748 gives about 2.2 speakers; the count is not monotone in Fb off the Fa=0.07 slice. Not pursued.
- The grid reading (cross-table of DER and count over Fa and Fb) is in `Docs/Closeout Results.md` under Closeout 3. Fa and Fb act roughly as a ratio, as the source reading predicted.
- Files: `harness/vbx_sweep.py`, `runs/sweeps/vbx-sweep/{points.jsonl,points.csv,all_points_per_file.csv,analyze.py,analysis_output.txt,make_fb_figure.py,fb_axis.png}`, per-point folders `runs/sweeps/vbx-sweep/VBX-*/`, plus 279 manifests in `runs/` (one per scored point; crashed runs write no manifest).
- `runs/comparison.csv` refreshed after the sweep (left uncommitted, as it already had uncommitted changes).


### §3 Ward linkage redo

- Driver: `runs/sweeps/ward-redo/sweep.py` (`--phase gate|grid|report`), outputs `points.jsonl`, `points.csv`, `results.json`, per-point folders `WARD-G10`, `WARD-G1.4`, `WARD-T<thr>`, 12 manifests in `runs/`. Mirrors the benchmark's agglomerative points: `--run-condition clustering_agglomerative --clustering-model agglomerative`, params `method=ward`, `min_cluster_size=12` (the benchmark's only ward size, AGG-1d-21..25), `threshold=T`; identity refinement; shared `.harness_cache`.
- Gate passed. Printed config shows threshold=10, manifest `clustering_config` shows threshold=10, counts differ from threshold 1.4 on 16 of 16 meetings. `instantiate` does not enforce Uniform bounds and `clustering.py:385` feeds the value to `fcluster`, so no src edit was needed.
- Grid 4, 5, 6, 7, 8, 10, 12, 14, 16, 20. Tune-best 8 (interior, not an edge). Report-half delta vs fresh VBx baseline +0.69 [-0.77, +2.28]; 16-meeting delta +0.42 [-0.40, +1.36].
- The bootstrap helper in `sweep.py` was asserted equal to `paired_bootstrap_delta` on HDB-2b-04 before use; the 16-meeting interval uses `assignment_runs.bootstrap()`.
- Two of 12 runs (WARD-G1.4, WARD-T10) show 0/0/0 intermediate tallies because their final keys repeat AGG-1d-25 and WARD-G10 (final-RTTM cache hits, ms-quantised, ~1e-5 DER). The other 10 are 16/0/0.
- Surprise: thresholds 10 and 12 give identical per-meeting counts, DER equal to within cache rounding, and the best all-16 DER (16.79), but second on the tune half; tune-half selection picked 8 and I reported that. No GPU rerun was needed.
- Reference text is in `Docs/Closeout Results.md`, subsection "Ward linkage redo".

### §4 dev/train confirmation

- Driver `runs/sweeps/dev-train-confirm/confirm.py` (`run`, `report`); outputs in the same folder (`results.csv`, `report_output.txt`, per-run `command.txt`, `run.log`, `per_file.csv`, `summary.json`). Four sequential harness runs, mirroring the WS1 dev/train baselines: `--split dev|train --condition IHM --run-condition baseline --clustering-model pyannote-default --refinement-strategy <s>`, data root `~/Code/AMI-diarization-setup/harness-data`, no `--runs-dir`, no `--counts-toward-results` (dev/train are not results runs).
- Runs: dev kNN `20261008T173307Z-a316d8e4`, dev nc `20261008T173352Z-a76b4c1c`, train kNN `20261008T173928Z-c101e97e`, train nc `20261008T174501Z-30766bde`. All tallies 18/0/0 (dev) and 136/0/0 (train); no misses. Manifest `clustering_config` = VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6) on all four.
- Baselines: per-file CSVs of the cold WS1 runs (`assignment-classifier/ws1/{dev,train}_vbx_identity/per_file.csv`; manifests `c9fcfc74`, `55c55a58`), not a cached-RTTM replay. Their per-file CSVs recompute to the manifest DERs (16.896, 15.648).
- Bootstrap: `harness.assignment_runs.bootstrap` takes arrays so it works for any split; `report` first asserts it reproduces the test nc_target interval (-0.26, -0.06).
- Results (DER; delta, 90% CI): dev VBx 16.896; kNN 16.825 (-0.071 [-0.205, +0.049]); nc 16.836 (-0.061 [-0.187, +0.061]). Train VBx 15.648; kNN 15.669 (+0.021 [-0.024, +0.068]); nc 15.698 (+0.050 [+0.011, +0.092]). No gain over 1 point.
- Gotcha: the manifest's `pipeline_config_id` is 241ecd86... for identity, kNN and nc_target alike (it excludes the refinement id, as for the test runs). The final key also holds the refinement id; the per-strategy DERs differ, so the runs did not collide.
- k=25 provenance: chosen among {5,10,25} on dev pair accuracy (`knn_dev_selection.txt`: 70.26/70.71/70.94), not dev DER; the file's mtime precedes the test run's. `STATIC_NAMES` in `harness/assignment_strategies.py:40` fixes the candidate set. `nc_target` has no parameter. So: weaker circularity applies to dev; caveat stays. Ordering holds on dev (weakly, intervals include 0), fails on train (VBx lowest). Train is probably seen data (A7).

### Crash accounting (addendum)

Follow-up task: account for the 81 crashed points of the 360-point VBx grid, to check the report's "best gain anywhere in the 360-point grid is 0.16" claim. Code/data reading only, no GPU, no re-runs (not needed — see below).

**Source of truth used:** `runs/sweeps/vbx-sweep/points.jsonl`, de-duplicated by `point_id` (two ids, `VBX-Fa0.07-Fb0.01-T0.6` and `VBX-Fa0.5-Fb0.8-T0.6`, each appear twice — once from an earlier hard-failed attempt before the driver learned to recognise that crash string, once as the final recorded crash; this matches the Implementation Notes above, "two earlier hard-failed attempts... both were rerun"). After de-duplication: 360 unique `point_id`s, 279 with `metrics` (scored), 81 without (crashed). Grid is exactly 10×6×6=360 with none missing.

**1. Which 81 crashed.** Full list with Fa/Fb/threshold extracted into `runs/sweeps/vbx-sweep/crashed_points.csv` (81 rows, written during this check). Crosstab, count of crashed thresholds out of 6, by (Fa, Fb) — this accounts for all 81:

| Fa \ Fb | 0.01 | 0.02258 | 0.05097 | 0.1151 | 0.2598 | 0.8 | 1.324 | 2.989 | 6.748 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 4/6 | 4/6 | 4/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 1/6 | 4/6 |
| 0.0219 | 4/6 | 2/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| 0.07 | 3/6 | 1/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| 0.1046 | 4/6 | 4/6 | 3/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| 0.2288 | 4/6 | 4/6 | 4/6 | 4/6 | 3/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| 0.5 | 4/6 | 4/6 | 4/6 | 4/6 | 4/6 | 3/6 | 1/6 | 0/6 | 0/6 | 0/6 |

Row/column sums check out: 81 total.

**2. What the exception actually was** (quoted from `run.log` under `runs/sweeps/vbx-sweep/VBX-*/`, all 81 directories and logs present on disk, no re-run needed):
- 78 of 81: `OverflowError: Python integer <N> out of bounds for int8`, raised at `src/pyannote/audio/pipelines/clustering.py:138` (`hard_clusters[c, s] = k`), reached via `speaker_diarization.py:656` -> `clustering.py:661` `constrained_argmax`. `<N>` ranges from 128 to 295 across the 78 (sampled: 137, 254, 188, 142, 128, 223, 155, 129, 138, 147, 205, 151, 143, 218, 295 — all the recorded cluster counts that overflowed int8's [-128,127] range). Full traceback for `VBX-Fa0.07-Fb0.01-T0.6` quoted verbatim:
  ```
  File "...\src\pyannote\audio\pipelines\speaker_diarization.py", line 656, in apply
      hard_clusters, soft_clusters, centroids = self.clustering(
  File "...\src\pyannote\audio\pipelines\clustering.py", line 661, in __call__
      hard_clusters = self.constrained_argmax(
  File "...\src\pyannote\audio\pipelines\clustering.py", line 138, in constrained_argmax
      hard_clusters[c, s] = k
  OverflowError: Python integer 137 out of bounds for int8
  ```
- 3 of 81, all at Fa=0.5 (`VBX-Fa0.5-Fb0.8-T0.6`, `VBX-Fa0.5-Fb0.2598-T0.68`, `VBX-Fa0.5-Fb1.324-T0.5`): `ValueError: negative dimensions are not allowed` at `speaker_diarization.py:522` (`clustered_segmentations = np.nan * np.zeros(...)`). Reading the full log (not just the final line) shows this is the **same root cause one step later in the pipeline**, not a separate failure mode: the log for all three shows a `RuntimeWarning: overflow encountered in scalar add` at `speaker_diarization.py:521` (`num_clusters = np.max(hard_clusters) + 1`), immediately preceded by a UserWarning `"The detected number of speakers (-128) for <meeting> is outside the given bounds [1, inf]"`. So `hard_clusters` (int8) already wrapped around to -128 before this line ran; `num_clusters` becomes a small/negative number via the wraparound, which goes negative-dimension into `np.zeros`. In other words, **all 81 crashes are the same mechanism**: VBx proposes more than 127 effective clusters for at least one meeting, and `hard_clusters` is a fixed int8 array (`clustering.py`, not editable here). The prior summary's category split (78/3) is confirmed correct by reading the logs, but the two categories are not independent causes — they are the same overflow caught at two different lines depending on which code path hits the wrapped-around value first.

**3. Where in parameter space — clustered at degenerate settings, not scattered, not adjacent to the best points.**
- The crosstab above shows crashes occupy a contiguous low-Fb band at every Fa (plus a high-Fb edge at Fa=0.01, Fb=15, and the Fa=0.5 band pushed to higher Fb because large Fa also inflates cluster count). This is the region where VBx over-splits into dozens to hundreds of clusters (true speaker count is 3.94 on average) — e.g. the Fa=0.07 row shows mean hypothesis counts of 52.5, 31.1, 16.2 at the three lowest surviving Fb values right next to the crash boundary, and the crash values themselves are int8-overflow counts of 128-295. This is a degenerate operating regime, not sampling noise.
- **Adjacency check (grid-index, not value):** the two lowest-DER scored points are `VBX-Fa0.1046-Fb2.989-T0.74` and `VBX-Fa0.07-Fb1.324-T0.74` (both 16.89 all-16 DER). Every one of their 6 immediate one-step grid neighbours (±1 step on Fa, Fb, or threshold alone) is a **scored, non-crashed** point (DERs 16.92-21.49). The minimum grid-index Manhattan distance from any crashed point to either of these two best points is **6** (axis steps combined), not 0 or 1; the full distance distribution (5 points at distance 6, 11 at distance 7, up to 1 at distance 13) shows no crash close to the optimum. So reading (c), "a contiguous region adjacent to the best points", does **not** apply — the crash region and the best-scoring region are well separated in the grid.
- This supports reading **(a)**: crashes are clustered at degenerate/extreme settings outside VBx's effective operating range for this data, not scattered (reading b) and not adjacent to the optimum (reading c).

**Corrected sentence for §4 of the report**, replacing "the best gain anywhere in the 360-point grid is 0.16":
> "Across the 279 points where VBx converged (81 of 360 crashed from the same int8 cluster-count overflow in `clustering.py:138`/`speaker_diarization.py:522`, all in a contiguous low-Fb/high-Fa region where VBx proposes well over 127 clusters — none adjacent to the best-scoring points), the best gain anywhere in the scored grid is 0.16 DER points."

**Not re-run:** no GPU re-runs were needed or performed. The crash region is explained mechanistically from logs already on disk (int8 overflow, confirmed by exact traceback text and the preceding overflow warnings), so there is no indication of environmental flakiness (reading b) that would call for a repeat-run check. `harness/vbx_sweep.py:184-194` was also read to confirm the driver only ever records a point as "crashed" when stderr/stdout contains one of exactly these two literal strings, and any other non-zero exit raises `HardFail` and would have stopped the whole sweep — so there is no third, silently-dropped failure category hiding in the 81.

**Correction to this ticket's own §2 count:** `points.jsonl` has 362 lines, not 360 or 361; the 2 extra lines are pre-crash-detection hard-fail attempts for the same two point_ids that were later resumed and recorded as crashes (matches the "Ledger has 362 records = 279 scored + 81 crashed + 2 earlier hard-failed attempts" note above — confirmed by direct read, not just trusted).
