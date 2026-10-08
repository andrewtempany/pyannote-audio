---
status: done
created: 2026-10-08
---

# Closeout 4: `min_duration_off`, the one retraining-free lever on missed speech

## The goal, plainly

Missed speech is 2925 s, **56% of the system's error**, and clustering can never touch it: every
clustering method and setting so far left missed speech and false alarm exactly unchanged.

`min_duration_off` is a clean-up step at the very end of the pipeline. Once each speaker's
timeline is built, any silence within one speaker's speech shorter than this value is filled in.
It changes neither the segmentation model's output nor the clustering. It only edits the final
timeline.

This ticket measures what that lever is worth. **Expect it to be small.** It can only bridge gaps
inside speech the segmenter already found, it cannot recover speech the segmenter never flagged,
and every bridged gap that was real silence becomes false alarm.

**Precondition:** closeout 3 §1 has confirmed the parameter exists, recorded its shipped value
and where `apply()` uses it, and confirmed it runs after clustering. If §1 found it does not
exist, set this ticket to `blocked` with that reason and stop.

## What we get

1. A curve of missed speech, false alarm and DER against `min_duration_off` on the VBx baseline.
2. A tune-half-selected value with its report-half Δ and interval.
3. One HDBSCAN check that the effect does not depend on the clustering method.
4. One row for §4 and a paragraph for §5: the size of the only retraining-free lever on the
   largest error component.

## How we measure it

- AMI IHM test, 16 meetings, collar 0, overlap not skipped. Tune half (meetings ending a/c) to
  choose, report half (b/d) to report. 90% paired bootstrap over the report-half meetings, 2000
  draws, seed 0.
- All runs warm. The parameter acts after clustering, so the intermediate cache stays valid.

## Done means

The harness change is in with its tests individually passing, the sweep is complete, the curve
and the selected value are in `Obsidian-Diarisation/Docs/Closeout Results.md`, and every
prediction below is marked held or not held.

## Not in this ticket

No other segmentation parameters, no retraining, no SDM, no crossing with every clustering
method. No edits under `src/pyannote/`.

---

## For the agent

### 1. The harness change — tests first

Today the harness cannot set this parameter, and neither cache key contains it. A sweep run as
things stand would either fail or, worse, return the cached baseline RTTM at every value and
draw a flat curve that reads as a real negative result.

**a. A CLI flag.** Add a repeatable `--segmentation-param NAME=VALUE` to `run_harness.py`,
mirroring `--clustering-param` (`run_harness.py:652-660`, parsed at `:344-370`, applied at
`:542`). Apply it through the pipeline's own `instantiate({"segmentation": {...}})`, so an
unknown name fails loudly. Record the instantiated value in the run manifest.

**b. The final cache key.** The parameter changes the final RTTM but not the intermediates, so by
the rule in `harness/cache_id.py`'s module docstring it belongs in the **final key only**. That
docstring names a `_final_only_components` list, which does not exist yet; create it and call it
from `pipeline_config_id`. Do not touch `_intermediate_components`.

**Decision, recorded so it can be revisited:** add the component **only when the value is
overridden** from the shipped value. That keeps every recorded final key unchanged, so the
baseline and all existing runs stay reachable. Rejected alternative: always including it, which
orphans every final RTTM in `.harness_cache`. That is harmless but would make this closeout's
other runs compute against fresh final keys mid-queue.

**Tests, written and shown failing before the change:**
1. Two different overridden values give two different final keys. (Fails today: the key ignores
   the parameter.)
2. No override gives exactly today's final key for the baseline configuration.
3. An unknown `--segmentation-param` name raises before the models load.
4. The intermediate key is identical with and without an override.

Report each test's individual pass or fail state, before and after, and the command run. An
aggregate count is not a report.

Commit the change on its own, with the tests, before any sweep run.

### 2. Falsifiability checks — run before the sweep

1. **Shipped value, explicitly passed:** final cache key identical to the current baseline key,
   and DER exactly 0.17048… (the baseline). If not, stop.
2. **One clearly different value** (for example 0.5 s), one meeting: missed detection must
   differ from the baseline's for that meeting. If it does not, the override is not reaching the
   pipeline. Stop.

### 3. The sweep

Grid on VBx (shipped clustering), from the shipped value upward: for example 0.0, 0.05, 0.1, 0.2,
0.3, 0.5, 0.75, 1.0 s. Adjust the low end if the shipped value is not 0. About 30 s per point.

Record the full scorer set at every point: `der`, MD, FA, Conf, `der_overlap_system`,
`der_overlap_assigned`, `jer`, region census (overlap caught / missed / invented), count MAE,
signed bias, exact matches. Tune-half and report-half DER separately.

**Selection:** the value with the lowest **tune-half** DER. Report its report-half Δ against the
shipped value, with interval. State plainly if the selected value is the shipped one.

**HDBSCAN check:** two runs with `--clustering-model hdbscan` and the exact `--clustering-param`
settings of the benchmark's best point (`b15caf86`: leaf, min cluster size 40; copy them from its
manifest): the shipped value and the VBx-selected value. Report the Δ. Do not select separately for HDBSCAN.

### 4. Predictions — record before running, do not tune toward them

1. Missed detection falls monotonically as the value rises; false alarm rises monotonically.
2. Speaker count (MAE, bias, exact matches) is **unchanged at every value**: filling gaps cannot
   create or remove a speaker.
3. Missed overlap (1394 s in the region census) changes far less, relative to its size, than
   total missed detection does.
4. Confusion changes by under 5% of its baseline (1212 s).
5. The report-half Δ at the selected value is under 0.3 DER points, inside the resolution floor.
6. The HDBSCAN Δ at the same value is within 0.1 points of the VBx Δ.

### 5. Hard-fails — stop and report

1. Either check in §2 fails.
2. Any intermediate-cache miss. The intermediates must not change.
3. Two values produce the same final cache key.
4. Missed detection is identical at every value (suspected silent no-op).
5. A gain above 1 DER point: suspected bug. Check the cache key and the manifest before believing
   it.
6. A value is chosen using report-half numbers.
7. Any edit under `src/pyannote/`, to `harness/refinement.py`, or to the intermediate key.

### Deliverable

The MD / FA / DER curve as one figure, the selected value with its interval, the HDBSCAN Δ, each
prediction marked held or not held, and the commit hash of the harness change.

### Feeds

§4(a) of [[report-plan]] (one line under the budget: what the only retraining-free lever on
missed speech recovers) and §5 (the discussion of why the ceiling is narrow, and future work).

---

## Implementation Notes

### §1 harness change (2026-10-08)

Files: `harness/cache_id.py` (`SHIPPED_MIN_DURATION_OFF`, `min_duration_off_of`, `_final_only_components`, called from `pipeline_config_id` only), `run_harness.py` (`_parse_segmentation_params`, `build_pipeline(segmentation_params=)`, `--segmentation-param`, `run_harness(min_duration_off=)` -> manifest field `min_duration_off`), `tests/test_segmentation_param.py` (new, 10 tests).

- **Shipped value is a hard-coded constant (0.0), not read from the checkpoint.** Its job is to define which runs keep the key they already have: every existing final RTTM was produced at 0.0 under a key without the component. Reading the checkpoint dynamically would let a changed shipped value silently change what an unadorned key means. Component is `min_duration_off=<repr>`, included only when the instantiated value != 0.0. No override and explicit 0.0 both reproduce today's final key exactly. `_intermediate_components` untouched.
- Value is read off the live pipeline (`pipeline.segmentation.min_duration_off`) so the key and manifest describe what ran. A pipeline with no `segmentation` attribute (the SimpleNamespace stand-ins in older tests) contributes no component.
- **Premise nuance:** "unknown name raises before the models load" cannot hold literally for names: validity is only knowable via `pipeline.instantiate`, which needs the loaded pipeline (same as `--clustering-param`; a hand-kept allowlist was rejected as drift-prone). What holds: a malformed item (no `=`) raises before `from_pretrained`; an unknown name raises ValueError from instantiate inside `build_pipeline`, before `run_harness` (any file scoring). Tests assert exactly those.
- Baseline final key for the test fake (VBx 0.6/0.07/0.8, batch 32): `241ecd86...bced`, computed by the pre-change code, hard-coded in the test and also recomputed by an independent formula. Mutation check (condition changed to always add the component): the three no-override/explicit-0.0 tests failed; reverted.
- Pre-existing failures unrelated to this change: `tests/test_run_harness_integration.py` (2, summary lacks `overlap_der`; key renamed to `der_overlap_system`), upstream `tests/test_train.py`, `tests/inference_test.py`, `tests/tasks/test_reproducibility.py`, and `tests/test_run_notebooks.py` (no papermill).

### §2-§4 sweep, HDBSCAN check, predictions (2026-10-08)

Files: `runs/sweeps/min-duration-off/sweep.py` (resumable driver, phases `vbx`, `hdbscan`, `report`), outputs beside it (`points.jsonl`, `points.csv`, `results.json`, `*.log`, per-point folders, figure). Numbers and the prediction table are in `Docs/Closeout Results.md`, section "Closeout 4".

- **Check 1 stop and fix.** The driver's first run halted at check 1: final key matched, but the DER I recomputed from the per-file CSV was 0.17049342382952826, not ...828 (2e-17). The harness's own `summary.json` der is exactly ...828 and the per-file CSV is byte-identical (`cmp`) to the vbx-sweep baseline point, so my comparison was against the wrong quantity (recomputed, not reported). Changed to exact equality on the harness-reported DER; no tolerance added. The 0.0 point was rerun (cache hit), leaving two manifests for that config (`...093254Z` and `...093352Z`).
- **Single-meeting run is not possible** in `run_harness.py`, so check 2 used the 16-meeting 0.5 s run (also a sweep point). MD fell on all 16 meetings.
- **Baseline for deltas:** the fresh-computed `vbx-identity` per-file CSV (17.0485), not the cache-served 0.0 row (17.0493). Both deltas are in `results.json`; they differ by about 1e-5 points.
- **Bootstrap function:** `paired_bootstrap_delta` (`harness/clustering_benchmark_report.py:57`) reads a fixed table, so it cannot take these CSVs. `boot_delta` in `sweep.py` is the same algorithm with the tables passed in, asserted equal to `paired_bootstrap_delta("HDB-2b-04")` before use.
- **Selection** used tune-half DER only: 0.75 s (not the shipped value).
- **Surprise recorded:** a value above the declared `Uniform(0.0, 1.0)` is not rejected by `instantiate`. 1.5 ran and scored (run `88118b5e`, DER 18.01%). Extra run, outside the sweep, never a candidate.
- **Prediction 3 did not hold** (missed overlap falls at roughly 0.76-0.85 of MD's relative rate), and prediction 6 missed on the report half by 0.006 points (held on all 16). Bridged gaps were almost all real silence: MD -331 s and FA +327 s at 0.75 s.
- HDBSCAN 0.0 run has b15caf86's final key and a 0/0/0 tally (final cache hit), so its DER 16.8558 differs from 16.8551 only by the `:.3f` RTTM rounding.
