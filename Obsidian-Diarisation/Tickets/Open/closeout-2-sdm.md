---
status: in-progress   # §2–§6 done including §5 oracle (run 259142b4); §8 re-tune still optional/not started
created: 2026-10-08
---

# Closeout 2: SDM, the second microphone condition

## The goal, plainly

Everything so far has been measured on AMI IHM — headset microphones, one per person, clean
audio. SDM is the same sixteen meetings recorded on a **single distant microphone** in the middle
of the table. Harder audio, same speakers, same words, same reference.

This is the only cross-condition evidence the project can get in the time available, and it
addresses the biggest stated limitation: one corpus, one condition.

**Start the download first.** It is the only item with real lead time.

## What we get

1. The cross-condition results table filled in: every method measured on both IHM and SDM, in the
   shape of the pyannote model card.
2. A check that the harness is sound on a condition it has never seen, against a published figure
   of **19.9%**.
3. Optionally, and only with Andrew's sign-off, a second data point for the error budget — see §5.

## How we measure it

- Same sixteen meetings, so every comparison is **paired per meeting**. Same tune/report split of
  8 and 8 as IHM. This is a real strength: nothing about the content changes, only the microphone.
- **Settings are carried across from IHM, not re-tuned on SDM.** The question is whether the
  settings transfer, which is the honest generalisation test and also the cheap one. Re-tuning on
  SDM would answer a different question and is a secondary step if time allows.
- Published reference: community-1 scores **19.9%** on AMI SDM, no collar, overlap not skipped —
  the same settings this harness uses.

## Done means

The baseline reproduces close to 19.9, the three clustering bests and the two assignment
strategies have SDM numbers with intervals, and the cross-condition table is filled.

## Not in this ticket

No new methods, no re-tuning as the primary result, no VoxConverse.

---

## For the agent

### 1. Data

Sixteen SDM wavs for the same test meetings, converted into the existing harness layout at
`<data-root>/SDM/` with the same `test.rttm`, `test.uem` and `audio/` structure IHM uses. The
harness already reads that layout: `run_harness.py --condition SDM` reads audio from
`{data_root}/{condition}/audio/{uri}.wav` (`run_harness.py:65-66`).

**Unverified, check before relying on it:** the download and convert scripts live in
`~/Code/AMI-diarization-setup/pyannote/`. `download_ami_split.sh` and `convert_for_harness.sh`
take a **split** argument. Nobody has confirmed they handle a **mic condition** (SDM is the
`Array1-01` recording). Read both scripts, then record in Implementation Notes the exact commands
that fetch and convert SDM. If they need changing, edits inside `AMI-diarization-setup` are
pre-approved; record what changed.

The reference and UEM are the same as IHM — the recording changed, the transcript did not. Verify
that with a byte-level `diff` rather than assuming it.

**The 19.9 figure:** quote the community-1 model card line it comes from, including its scoring
settings (collar, overlap handling). Do not inherit it from this ticket.

### 1a. Cache isolation — required

**SDM runs must use their own cache directory**, for example
`--cache-dir .harness_cache_sdm`. Both cache keys contain the meeting `uri` but **not** the mic
condition or the data root (`harness/cache_id.py:236-275`). SDM meetings have the same uris as
IHM (`ES2004a` and so on), so a run against the shared `.harness_cache` would silently return
IHM's cached segmentation, embeddings or final RTTMs, and report roughly 17.05.

- Pass the same `--cache-dir` to **every** SDM run in this ticket.
- Do not modify `cache_id.py` to add the condition. That is a separate change and is out of scope.

### 2. Baseline first, and it is a gate

Run VBx at the shipped settings, cold, on SDM, in the SDM cache dir.

**First check the run was genuinely cold:** the cache tally `run_harness.py` prints must show
**16/16 intermediate misses**. Any hit means the run read cached data from somewhere: stop and
report.

**Expected: close to 19.9.** The published figure is from the model card under the same scoring
settings.

- Within 0.5 points: proceed.
- Between 0.5 and about 1 point: proceed, and flag the gap next to the result.
- Further off: **stop and report**. Something in the data, the conversion or the cache is wrong,
  and every number after it would be wrong too. A result near 17.05 points at the cache, not the
  conversion. Do not proceed and explain the gap afterwards.

This mirrors how the IHM harness was validated against 17.0, and it is the only external check
available on a new condition.

### 3. The runs

After the baseline, the intermediate cache is warm for SDM and everything else is cheap.

| Run | Settings |
|---|---|
| VBx baseline | shipped defaults |
| Agglomerative | centroid, threshold 0.75, min cluster size 12 — **from IHM** |
| HDBSCAN | leaf, min cluster size 40 — **from IHM** |
| Spectral + eigengap | pruning 0.20 — **from IHM** |
| Nearest centroid on held-back pairs | `--refinement-strategy nc_target`, as IHM (16.89) |
| kNN k=25 on held-back pairs | `--refinement-strategy knn_target_k25`, as IHM |

After the cold baseline, every later run must show 16/16 intermediate **hits** in the SDM cache
dir.

Record the full scorer set for every run, exactly as the clustering benchmark did: `der`, MD, FA,
Conf, `der_overlap_system`, `der_overlap_assigned`, `jer`, region census, count MAE, signed bias,
exact matches.

90% paired bootstrap intervals over the 16 meetings, 2000 draws, seed 0.

### 4. Fill the cross-condition table

Write the SDM row into the table closeout 1 set up. Published 19.9 goes in the reference row,
cited to the model card, clearly marked as not something we ran.

### 5. Optional: one oracle segmentation run — NEEDS ANDREW'S SIGN-OFF

**Do not run this without explicit approval.** Oracle experiments were closed in September and
that decision stands unless Andrew reopens it for this one case.

The argument for it: the report's central claim is the error budget — segmentation owns 13.09 of
17.05 points on IHM. That number comes from a single oracle segmentation run. Without one on SDM,
the headline finding stays a single-condition observation.

One run would give the far-field budget split. The mechanism predicts the **segmentation share
should grow**, since the segmenter has a harder job and embeddings degrade further in the
far-field. That is a pre-registerable prediction on the paper's main claim.

Cost: one run, roughly 35 minutes.

If approved, record the prediction before running, and report whether the segmentation share grew.

### 6. Predictions — record before running, do not tune toward them

- The SDM baseline lands within 1 point of 19.9.
- Every method is worse on SDM than on IHM, by 2 to 4 DER points.
- The **ranking** of methods is roughly preserved, because they are all competing inside the same
  narrow band. If the ranking inverts, that is a finding worth reporting carefully.
- IHM-selected settings transfer imperfectly: at least one method lands notably worse than its
  IHM result relative to the baseline, because its setting was tuned for cleaner embeddings.
- Speaker counting degrades more than DER does.

### 7. Hard-fails — stop and report

1. The SDM baseline is more than about 1 point from 19.9.
2. The SDM reference or UEM differs from IHM's.
3. Any SDM run uses the shared `.harness_cache`, or the cold baseline shows any intermediate hit.
4. An oracle condition runs without sign-off.
5. The download or conversion cannot be done with the existing scripts plus edits inside
   `AMI-diarization-setup`. Report what is missing; do not install packages.

If the download stalls overnight, record it and let the coordinator move on. SDM is last in the
GPU queue for this reason.

### 8. Secondary, only if everything above lands with time to spare

Re-tune each method on SDM's tune half and report on its report half. This answers "how good is
each method in far-field at its own best setting", which is a different and fairer question than
transfer. Report it alongside, never instead of, the transferred-settings result.

### Deliverable

The filled cross-condition table, a note on whether the baseline reproduced, and one short
paragraph on whether the IHM conclusions hold in far-field.

### Feeds

§4 of [[report-plan]] (the second column) and §5 (cross-condition discussion, and the limitation
that currently says "one microphone condition").


---

## Implementation Notes

### §1 Data (done 2026-10-08)

**Premise check: the scripts could NOT do SDM as written.** `download_ami_split.sh` hard-coded `$uri.Mix-Headset.wav` (download URL, output path, and the WAV-header check), and `convert_for_harness.sh` hard-coded the `IHM` output dir and `Mix-Headset.wav` source. `download_ami_sdm.sh` and `download_ami_sdm_mini.sh` exist but use `wget` (not installed here: `which wget` finds nothing) and fetch lists of meetings that are not the test split. So edits were needed (pre-approved under the ticket's hard-fail 5; no packages installed).

**Edits in `~/Code/AMI-diarization-setup/pyannote/` (both untracked files in that repo; backups of the originals are in the session scratchpad):**
- `download_ami_split.sh`: new optional 3rd arg `MIC="${3:-Mix-Headset}"`; used in the output path, the URL, and passed to the embedded Python header check (`sys.argv` now `root, mic, uris`). Default behaviour unchanged.
- `convert_for_harness.sh`: new optional 3rd arg `CONDITION="${3:-IHM}"` (`IHM` -> `Mix-Headset`, `SDM` -> `Array1-01`, anything else errors); `OUT_DIR="$DATA_ROOT/$CONDITION"`. Default behaviour unchanged. It never touches the IHM dir when called with SDM.

**Exact commands run:**
```
cd ~/Code/AMI-diarization-setup/pyannote
bash download_ami_split.sh test "$PWD" Array1-01
cd ~/Code/AMI-diarization-setup
bash pyannote/convert_for_harness.sh "$PWD/harness-data" test SDM
```
(`curl --show-error --silent --fail --location --continue-at -` inside the script, from `https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<uri>/audio/<uri>.Array1-01.wav`.)

**Data root:** `C:\Users\Andrew\Code\AMI-diarization-setup\harness-data` (`--data-root`), so SDM is `harness-data/SDM/{test.rttm,test.uem,audio/<uri>.wav}`. Raw downloads sit beside the IHM ones in `pyannote/amicorpus/<uri>/audio/<uri>.Array1-01.wav`. SDM audio is hardlinked (same as IHM). Download took about 1 hour 20 minutes total, no stall, no failures; slow at first (~140 KB/s) then sped up.

**WAV repair:** the script's header check repaired 6 files (RIFF/data-chunk size mismatch, trailing bytes): IS1009a/b/c/d and TS3003a/b. Same mechanism as the IHM repair (IS1009 was in the original hard-coded list). Repair only rewrites the header via Python `wave`; samples are unchanged.

**Reference/UEM check (hard-fail 2): PASSED.**
```
cmp harness-data/IHM/test.rttm harness-data/SDM/test.rttm && echo identical   -> rttm cmp identical
cmp harness-data/IHM/test.uem  harness-data/SDM/test.uem  && echo identical   -> uem cmp identical
diff of both -> no output, exit 0
sha256 test.rttm (IHM and SDM): 3d50ff1d36dba8d951bb1f537e4aa86d0b4a4fed58a186ad4b1944b343d29e79
sha256 test.uem  (IHM and SDM): 7beaf9092651fdd633f25da9b06c9047e0a4494247ab9d3e8e9e9d990fad1d85
wc -l: test.rttm 7493 lines, test.uem 16 lines
```
Caveat: the SDM files are regenerated by the convert script from the same `only_words/rttms/test/*.rttm` and `uems/test/*.uem` sources, so identity is expected by construction; the IHM copies are the pre-existing files, not regenerated, so the comparison is against what the IHM runs actually used.

**WAV check (ffprobe):** all 16 SDM wavs exist, all pcm_s16le, 16000 Hz, 1 channel, same as IHM. Durations (s), IHM vs SDM:
- 12 meetings identical to the microsecond (ES2004a-d, TS3003a-d, EN2002a-d).
- IS1009a 838.833 vs 838.997; IS1009b 2052.333 vs 2052.437; IS1009c 1820.833 vs 1820.843; IS1009d 1944.500 vs 1944.661. SDM is 0.01 to 0.16 s LONGER on the four IS1009 meetings. The UEM end for these equals the IHM duration (e.g. IS1009a 838.833313), so scoring is clipped to the UEM and the extra tail is unscored. Last reference segment ends at 805.72 (IS1009a) and 1925.75 (IS1009d), well inside both. Not a problem, just noted: the harness will process 0.16 s more audio on those files.
- Unexpected but harmless: EN2002a-d SDM file sizes are byte-identical to the IHM sizes in the first listing, which fits the equal durations.

**The 19.9 source: model card NOT found locally.**
- HF cache `~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf.../` holds only `config.yaml`, `embedding/`, `plda/`, `segmentation/`: no README / model card.
- The repo `README.md` (pyannote-audio, lines 84-101, "Benchmark (last updated in 2025-09)") has the table row, quoted verbatim:
  `| [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) (SDM) | 22.7 | 19.9 | 15.6 |`
  with columns `legacy (3.1)` | `community-1` | `precision-2`, and the footnote `Diarization error rate (in %, the lower, the better)`. The IHM row in the same table is `18.8 | 17.0 | 12.9`.
- That README says NOTHING about collar or overlap handling (`grep -n -i "collar\|skip_overlap" README.md` returns nothing). So the ticket's claim "no collar, overlap not skipped" is NOT confirmed by any local source. It is plausible because the IHM figure 17.0 in the same table matches this harness's 17.05 under no collar / overlap scored, but that is inference, not a quote. The HF model card page itself (huggingface.co/pyannote/speaker-diarization-community-1) was not fetched.
- Cite the 19.9 to the pyannote-audio README benchmark table (2025-09), not to the HF card, unless someone fetches the card.

**Reminder for the GPU queue:** SDM runs need `--data-root C:/Users/Andrew/Code/AMI-diarization-setup/harness-data --condition SDM --cache-dir .harness_cache_sdm` (see §1a). No harness run, no GPU use in §1.

### §2-§6 SDM runs (2026-10-09, sub-agent) -- predictions, recorded BEFORE any SDM run

Written before the cold baseline ran, so before the gate and before every later run. The ticket's five predictions, each with a measure fixed now so it cannot be bent to the result. "Methods" = seven rows: VBx baseline, agglomerative (AGG-1a-07), HDBSCAN (HDB-2b-04), spectral (SPC-3a-03), nc_target, knn_target_k25, and ward (threshold 8, added after the ticket was written). IHM comparators are the 16-meeting DERs already in Closeout Results.md (VBx 17.0485 fresh, AGG 16.8678, HDB 16.8551, SPC 17.1600, nc_target 16.8927, kNN 16.8499, ward-T8 17.4688).

1. **Baseline within 1 point of 19.9** -- held iff |SDM VBx DER - 19.9| <= 1.0 (the gate itself is 0.5 proceed / ~1.0 stop).
2. **Every method worse on SDM by 2 to 4 DER points** -- held iff for ALL seven methods 2.0 <= (SDM DER - IHM DER) <= 4.0 (percentage points, all-16 aggregate). The baseline prediction implies about +2.85 (19.9 - 17.05), so this is expected to be borderline at the low end.
3. **Ranking roughly preserved** -- held iff Spearman rho between the IHM and SDM DER ranking over the seven methods is >= 0.7. Caveat fixed now: the IHM spread is only ~0.3 points across six of the methods (inside the project's resolution), so the IHM ranking itself is close to noise; a low rho would be weak evidence of inversion.
4. **IHM-selected settings transfer imperfectly** -- held iff at least one non-baseline method has (SDM delta vs SDM VBx) - (IHM delta vs IHM VBx) > +0.5 DER points (i.e. is more than 0.5 worse relative to the baseline than it was on IHM). Also reported at 0.3.
5. **Speaker counting degrades more than DER** -- relative measure: relative DER degradation = SDM DER / IHM DER - 1; relative count degradation = SDM count MAE / IHM count MAE - 1 (count MAE = mean over 16 meetings of |hyp speakers - ref speakers|). Held iff the count measure exceeds the DER measure for the VBx baseline AND for at least 4 of the 7 methods. Exact-match percentage change is reported alongside. Caveat: IHM count MAE is small (0.125-0.25, i.e. 2-4 meetings off by one), so the ratio is coarse.

Cache note: `run_harness.py --cache-dir` feeds both `IntermediateCache(config.cache_dir, ...)` (run_harness.py:169-170) and `Runner(..., config.cache_dir, ...)` (:179-186), so one flag redirects both tiers. All SDM runs pass `--cache-dir .harness_cache_sdm`.

### §2-§4, §6 results (2026-10-09, sub-agent) -- no §5 oracle, no §8 re-tune

- **Driver and outputs:** `runs/sweeps/sdm/sweep.py` (phases gate / runs / report); outputs `points.csv`, `per_meeting.csv`, `results.json`, `points.jsonl`, `report_output.txt`, per-run folders. The driver builds each request from the IHM manifests and the clustering-benchmark ledger (not retyped), checks every manifest against it (clustering_config, refinement_strategy, mic_condition SDM, split, run condition, collar 0 / overlap scored), fingerprints the shared `.harness_cache` before/after each run (hard-fail on change), and hard-fails a cold run with any hit or a warm run without 16/16 hits.
- **Cache isolation verified in code first:** `--cache-dir` goes to both `IntermediateCache` and `Runner` (`run_harness.py:169-186`). `.harness_cache_sdm` did not exist before the gate; it is not gitignored and is NOT committed.
- **Gate:** run `20261008T182732Z-70c72cb4`, tally `0 full / 0 partial / 16 miss / 16 write`, DER 19.9088 vs 19.9 (gap +0.009). 19.9 provenance caveat (README only, scoring settings not stated) is attached to the gate result in Closeout Results.md. IS1009 meetings not outliers (+0.95 to +2.13 vs median +2.76).
- **Runs:** six more, each 16/16 intermediate hits: agglomerative `94c45184` 20.33, HDBSCAN `ffde933d` 19.55, spectral `993c6760` 20.60, nc_target `3a04d861` 19.72, kNN k25 `00b82017` 19.66, ward thr 8 `32122096` 21.88 (ward added after the ticket). Max gain vs baseline -0.36 (HDBSCAN); no gain over 1 point.
- **Predictions:** 1 held, 2 not held as defined (ward +4.41; six ticket methods all in 2.69-3.46), 3 held (rho 0.857), 4 held (agglomerative +0.60, spectral +0.58, ward +1.55 beyond their IHM delta), 5 held on a coarse measure (6/7 methods; VBx exact matches unchanged at 12).
- **Decisions:** predictions written to this file before the gate ran (stricter than "before runs after the gate"). Ward gets its own column in the cross-condition table (IHM cells filled from closeout 3 section 3 / the ward-redo CSV). IHM comparators for the per-meeting pairing are the same per-file CSVs as the IHM rows; reference durations asserted equal per method between SDM and IHM before pairing.
- **Gotcha:** manifest run ids carry the date stamp 20261008 (the harness clock), not 9 Oct. A first attempt to edit the results doc through an inline heredoc failed on quoting; the doc edit script and section text were written as files instead.

### §5 SDM oracle segmentation (approved 9 Oct)

**Sign-off.** Andrew approved exactly one oracle condition: oracle segmentation on AMI SDM test, 16 meetings, VBx clustering (shipped settings), `--counts-toward-results true`. No oracle assignment, no combined 2x2 cell, no IHM oracle re-run — explicitly out of scope even if convenient.

**Prediction, recorded 2026-10-09 before the run, timestamped here, not to be tuned toward:**
- Predicted oracle SDM DER: 4-5%, close to oracle IHM's ~3.96% (run `18016c15`).
- Predicted segmentation share on SDM: at or above IHM's 76.8% (13.09 / 17.05).
- **A result near or below IHM's 3.96% would be implausible** (SDM is harder, far-field audio, and the SDM baseline is already +2.86 points worse than IHM's) **and must be treated as a suspected bug** — check the cache key and the `--oracle-rttm` path before believing it.
- Two informative outcomes distinguished in advance: (a) oracle SDM DER lands near 4-5%, confirming the "mostly missed speech, segmentation-dominated" reading also holds far-field; or (b) oracle SDM DER lands notably higher, e.g. 8%+, meaning a large piece of the +2.86 far-field penalty is NOT attributable to segmentation and the SDM paragraph in Closeout Results.md needs revising.

**Facts verified before launching (not assumed):**
- **IHM oracle run's exact reference.** Run `18016c15` (2026-09-18) itself predates the `oracle_rttm` manifest field (added later by the run-manifest-provenance ticket; `run_harness.py:277-282` marks it explicitly additive, "every manifest written before this change lacks the key entirely"), so its own manifest does not carry the path. A later oracle run on the same IHM test split, `20260920T082056Z-2aada13d.json` (oracle segmentation + oracle assignment combined cell, same `segmentation_source_id: "oracle-v2"`, same `pipeline_config_id`, same `mic_condition: IHM`), records `"oracle_rttm": "C:/Users/Andrew/Code/AMI-diarization-setup/harness-data/IHM/test.rttm"`. So the IHM oracle condition uses the **standard collapsed `test.rttm`** as its own oracle reference — not a separate `only_words` RTTM (the Oracle Segmentation Provider doc discusses an `only_words` conversion as a non-goal/follow-on that was never built; "until that lands, `--oracle-rttm` must point at a manually prepared or fixture-scale RTTM" did not block the real run, which used the standard RTTM instead). This matches the Oracle Findings Report's own stated limitation: "The oracle reference is the same file used for scoring." The SDM run below uses the SDM-side analogue of that exact same file.
- **SDM test.rttm / test.uem re-verified byte-identical to IHM's, today, not assumed from closeout 2 §1:** `cmp harness-data/IHM/test.rttm harness-data/SDM/test.rttm` -> identical; `cmp` on the two `test.uem` -> identical. SHA-256 re-hashed and matches §1's recorded values exactly: `test.rttm` `3d50ff1d36db...d29e79` (both), `test.uem` `7beaf909...fad1d85` (both).
- **Cache key separation, checked in code before launching.** Both the final-hypothesis key (`harness/runner.py:59-63`, `Runner.cache_key`) and the intermediate key (`harness/intermediate_cache.py:248-256`, `IntermediateCache.key`) are:
  ```python
  # Runner.cache_key (final RTTM)
  raw = f"{self._pipeline_config_id}|{self._segmentation_source.id}|{self._refinement_id}|{uri}"
  # IntermediateCache.key
  raw = f"{self._config_id}|{self._segmentation_source_id}|{uri}"
  ```
  Neither key contains the oracle RTTM's path or content, and neither contains the mic condition or data root — `segmentation_source.id` is the fixed string `"oracle-v2"` for *any* `OracleSegmentation` instance regardless of which `reference_lookup` it was built from (`harness/segmentation.py:107`). This means the **hash itself is identical** between an SDM oracle run and the IHM oracle run (same pipeline_config_id, same `"oracle-v2"`, same refinement id, same uri). **The only thing separating them is the cache directory** — exactly the same isolation mechanism already relied on and verified for every other SDM run in this ticket (mic condition was never in the key either; see §1a and the "Cache isolation (checked)" note in Closeout Results.md). This is not a new risk, but it is worth stating plainly: this run MUST pass `--cache-dir .harness_cache_sdm`, or it will read/write directly into the shared `.harness_cache` and silently collide with the IHM oracle run's cached RTTMs and intermediates (same hash, same files). Checked before running: `.harness_cache_sdm` holds 112 final RTTMs + 16 intermediate `.npz` (the 7 prior SDM method runs; no `"oracle-v2"` intermediate exists there yet), and the shared `.harness_cache`'s newest file predates this session (2026-10-08), confirming it was untouched going in.
  - **Consequently, a cold run is still expected.** The SDM cache has no `"oracle-v2"` segmentation-source entries yet (only `"baseline"` ones from the 7 earlier SDM runs), so this run must show 16/16 intermediate misses, same as the original SDM VBx gate.
- **VBx `expects_num_clusters` and the oracle num_speakers guard, read not assumed:**
  - `src/pyannote/audio/pipelines/clustering.py:550-552`: `class VBxClustering(BaseClustering): ... expects_num_clusters: bool = False`.
  - `src/pyannote/audio/pipelines/speaker_diarization.py:295`: `self._expects_num_speakers = self.clustering.expects_num_clusters` — so for a VBx pipeline this is `False`.
  - `harness/segmentation.py:122-124`: `if getattr(pipeline, "_expects_num_speakers", False): raise RuntimeError(...)` — the condition is `False` for VBx, so **the guard does not fire**. `OracleSegmentation.populate()` proceeds normally with VBx clustering, as it did for the IHM oracle run (same pipeline default).

**Command run** (one invocation, all 16 meetings, cold in the SDM cache):
```
diarisation-env/Scripts/python.exe run_harness.py \
  --data-root C:/Users/Andrew/Code/AMI-diarization-setup/harness-data \
  --condition SDM --split test --cache-dir .harness_cache_sdm \
  --run-condition oracle_segmentation \
  --oracle-rttm C:/Users/Andrew/Code/AMI-diarization-setup/harness-data/SDM/test.rttm \
  --counts-toward-results \
  --notes "SDM oracle segmentation (approved 9 Oct, closeout-2 sec 5): ground-truth segmentation on AMI SDM test, 16 meetings, VBx shipped clustering, identity refinement. Oracle reference is SDM/test.rttm (byte-identical to IHM/test.rttm, re-verified same day). Companion to IHM oracle run 18016c15."
```

**Result: run `20261008T195525Z-259142b4`, DER 4.0076%. Prediction held; outcome (a), not (b).**

- **Cold-run check, as required before trusting the number.** Cache tally printed: `0 full hit(s), 0 partial, 16 miss(es), 16 write(s) (32 entries, 122.2 MB on disk)` — genuinely cold, no hit of any kind. `.harness_cache_sdm` grew from 112 to 128 final RTTMs (+16, the new `"oracle-v2"` entries) and from 16 to 32 intermediate `.npz` (+16). The shared `.harness_cache` has no file newer than the last IHM run (`runs/20261008T183201Z-32122096.json`, the last closeout-2 IHM-side read) — checked with `find .harness_cache -newer <that manifest>`, empty output. No collision occurred.
- **Manifest fields** (`runs/20261008T195525Z-259142b4.json`): `mic_condition: "SDM"`, `condition: "oracle_segmentation"`, `segmentation_source_id: "oracle-v2"`, `refinement_strategy: "identity"`, `oracle_scope: "all_pairs"`, `oracle_rttm: "C:/Users/Andrew/Code/AMI-diarization-setup/harness-data/SDM/test.rttm"`, `clustering_config: "VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6)"` (shipped defaults, matching the IHM oracle run), `counts_toward_results: true`, `git_commit: "d01ae8bc"`.
- **Full scorer set** (manifest `summary`, independently re-summed from `runs/sweeps/sdm-oracle/per_file.csv`'s 16 rows and matching to every digit): DER 0.04007633170710202 (4.0076%); MD 16.42709375038809 s; FA 141.12684375038467 s; Conf 1073.3474687499488 s; `der_overlap_system` 0.09970517737876688; `der_overlap_assigned` 0.09911323308227928; JER 0.09138778146806291; region census `t_and_d` 3821.2587187498343 s, `t_minus_d` 5.7972812501686946 s, `d_minus_t` 73.25503125016866 s; counting MAE 0.3125, signed bias +0.3125 (over-counts only, same direction as the IHM oracle finding), exact speaker-count matches 12/16.
- **MD, FA and the region census are byte-identical to the IHM oracle run's** (`18016c15`: MD 16.42709375038809, FA 141.12684375038467, regions 3821.2587187498343 / 5.7972812501686946 / 73.25503125016866). This is expected and a strong internal consistency check: both oracle conditions inject segmentation from the same reference content (IHM and SDM `test.rttm` are byte-identical), so the ground-truth segmentation tensor fed to the pipeline is identical in both conditions; only the audio (and therefore the embeddings and clustering/assignment downstream of segmentation) differs. The entire difference between SDM oracle (4.0076%) and IHM oracle (3.9561%) is in confusion: 1073.35 s vs 1057.51 s.
- **Not a suspected bug.** SDM oracle DER (4.0076%) is slightly *above* IHM oracle's 3.9561% (not near or below it), consistent with SDM being harder far-field audio. No cache or path issue to chase.

**Budget table** (formula reused exactly from `Obsidian-Diarisation/Docs/Cross-Condition Deltas.md`: downstream budget = `baseline_der − oracle_segmentation_der`, sign convention baseline minus oracle, positive = oracle lowered DER; share = budget / baseline_der):

| | IHM | SDM |
|---|---|---|
| Baseline DER | 17.0485% (`0fa3b41d`, fresh) | 19.9088% (`70c72cb4`) |
| Oracle segmentation DER | 3.9561% (`18016c15`) | 4.0076% (`259142b4`) |
| Segmentation's point contribution (baseline − oracle) | 13.0925 pt | 15.9011 pt |
| Segmentation share of baseline DER | 76.80% | 79.87% |

- IHM figure reproduces the cited 13.09/17.05 = 76.8% exactly (13.0925, 76.795%, matches to the precision quoted elsewhere).
- SDM segmentation budget is 15.90 points, share 79.87% — **above** IHM's 76.8%, as predicted ("at or above").
- IHM-to-SDM baseline degradation is +2.8602 points; IHM-to-SDM oracle degradation is only **+0.0516 points** — almost all of the +2.86 far-field baseline penalty disappears once segmentation is forced to ground truth. This is the same reading stated two ways: the segmentation share grew (76.8% -> 79.9%), and in absolute terms the oracle condition erases 98% of the far-field gap (2.8602 -> 0.0516 pt residual).

**Which informative outcome, and whether the prediction held.** This matches outcome (a): oracle SDM DER landed at 4.01%, inside the predicted 4-5% band and close to IHM's 3.96%, confirming the "mostly missed speech, segmentation-dominated" reading also holds far-field. It is nowhere near outcome (b) (8%+). All parts of the prediction held: DER in the 4-5% band (4.0076% measured), share at or above 76.8% (79.87% measured), and the result is well above IHM's 3.96% so the suspected-bug trigger did not fire.

**Driver.** No separate driver script was written — a single `run_harness.py` invocation (printed above) covers the one approved run; this matches how other single-run checks in this project (e.g. the min-duration-off falsifiability checks) were done without a dedicated sweep script. Outputs were moved from the default root-level `per_file.csv`/`summary.json` paths into `runs/sweeps/sdm-oracle/` to keep the repo root clean, consistent with every other sweep's output location.
