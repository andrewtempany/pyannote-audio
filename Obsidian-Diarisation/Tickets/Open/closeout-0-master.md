---
status: done
created: 2026-10-08
---

# Closeout 0: master ticket — coordinating the pre-writing closeout

## Your role, plainly

You are coordinating four closeout tickets before the report is written. You dispatch the work
to sub-agents and you review what comes back. **The review is the job, not the dispatch.**

The report is due 11 October. The experimental conclusions are already settled: no clustering
method and no learned classifier clearly beat the VBx baseline within the project's 0.3 DER
resolution. Nothing in these tickets is expected to change that, and nothing in them should try
to.

## The standard

**We do not need good results. We need results that can be justified, with a clear record of
what was done.**

A null that is airtight is worth more here than an improvement that cannot be defended. The
report is graded on methodological soundness, not on beating the baseline. If a sub-agent returns
a favourable number you cannot trace to a run id, a manifest and a measurement, it is worth less
than nothing, because it will have to be withdrawn during writing.

Treat every claim as unproven until you have seen what produced it.

## The four tickets

| Ticket | What it needs | GPU |
|---|---|---|
| [[closeout-1-analysis]] | Existing CSVs, docs and code only | None |
| [[closeout-2-sdm]] | SDM audio download, then runs in a **separate cache dir** | Yes, one cold run then warm |
| [[closeout-3-ihm-gaps]] | Already-cached IHM data | Yes, all warm |
| [[closeout-4-min-duration-off]] | A small harness change, then cached IHM data | Yes, all warm. **Only if closeout 3 §1 confirms the parameter exists** |

## Where results go

Every table, interval and written reading goes into one document:
`Obsidian-Diarisation/Docs/Closeout Results.md`. Closeout 1 creates it. The tickets keep their
Implementation Notes as usual, but **do not convert any closeout ticket into a Doc before
11 October**. Leave them in `Tickets/Open/` with `status` kept current.

## Running time

The GPU is available overnight 8–9 October for as long as the queue needs. Andrew checks in late
on the 8th and in the morning of the 9th. The full queue is roughly 7–8 hours, so time is not the
constraint. **Order matters only for resilience**: a crash, a Windows restart or a hard-fail stop
halts everything behind it, so the items worth most to the report go first.

## Dispatch order

1. **Start the SDM download immediately**, in the background. If it stalls or fails, record that
   and carry on with the rest of the queue. Do not wait on it.
2. **Closeout 1 in parallel.** No GPU, no conflict. Its §4 creates the results doc and the table
   the others write into.
3. **Closeout 3 §1** — the ten-minute segmentation post-processing check. CPU only. Its answer
   decides whether closeout 4 runs.
4. **GPU queue, strictly one run at a time, in this order:**
   1. Closeout 3 §5 — the 16.79 rerun (30 seconds; confirms the cache before anything else).
   2. Closeout 3 §2 — the VBx sweep.
   3. Closeout 4 — `min_duration_off` (if §1 found it; harness change and tests first).
   4. Closeout 3 §3 — ward.
   5. Closeout 3 §4 — dev and train confirmation.
   6. Closeout 2 — SDM baseline gate, then the remaining SDM runs (once the download is done).

A stop in one ticket pauses **that ticket only**. Record it and move to the next item in the
queue. The exception is a stop that suggests the shared cache or harness is wrong (an unexplained
cache miss or key collision on IHM): that halts the whole GPU queue.

Do not run two GPU workstreams at once. The cache is shared and concurrent runs have produced
confusing results in this project before.

## Decisions already made (8 October) — do not re-escalate these

- **Nearest centroid means `nc_target` (16.89, `703683ce`)** everywhere: tables, dev/train, SDM.
  The September `nearest_centroid` run (`32241285`, 16.79) relabels **every** voice in every
  chunk by plain argmax (`harness/refinement.py:33-69`), not only held-back voices. It is
  reported as a separate variant row, never as "the best result".
- **Dev and train confirmation runs both splits**, with two limits stated next to the result:
  train is probably seen by the models (VBx 15.65 there), and dev is not independent if k=25 was
  chosen on dev.
- **`min_duration_off` gets its own ticket** (closeout 4), swept on VBx with one HDBSCAN check. It
  is a parameter of the shipped pipeline, not a new method, so §6.7 does not block it.
- **Ward:** if the pipeline rejects thresholds outside `Uniform(0.0, 2.0)`, ward is reported as
  untested. No source change.
- **VBx sweep:** report the curve only. No "best setting" is reported from it.
- **SDM gate stays at 1 point**, with anything over 0.5 flagged in the result.
- **Still open:** the SDM oracle run (closeout 2 §5) needs Andrew's written sign-off.

---

## For you, as coordinator

### 1. What every sub-agent gets in its brief

Alongside its ticket, each sub-agent is told:

- **Show the evidence, not the conclusion.** A reported number must come with its run id, the
  manifest field it came from, and the command that produced it. "The tests pass" is not a
  report; the individual named test states are.
- **Stop rather than work around.** If a ticket's instruction rests on a false premise — which
  has happened twice in this project — report the false premise. Do not silently do the sensible
  thing instead.
- **A surprising result is a finding, not a bug to fix.** Report it. Do not adjust anything to
  make it less surprising.
- **Never select using report-half or test numbers.**
- The standing boundaries in §6 below.

### 2. Review protocol — the seven ways this codebase has lied

These are not hypothetical. Each one cost real time on this project. Check every sub-agent
report against all seven.

1. **A silent no-op that produces plausible numbers.** Ground-truth segmentation was once
   injected behind a flag that was restored before the pipeline ran; the experiment completed,
   wrote a valid manifest, and measured the baseline. Ask of every change: *if this quietly did
   nothing, what in the report would tell us?*

2. **A test that has never executed is not a guard.** A test named for the exact bug above
   existed and had never run, blocked by an unrelated encoding fault. Demand individual
   pass/fail states for named tests. An aggregate count hides this completely.

3. **A counter is only as good as the distinction it draws.** One counter conflated "no reference
   overlap" with "empty tensor slot"; 96% of a headline figure turned out to be padding, and a
   planning conclusion had been drawn from it. When a number is reported, ask what else could
   produce that number.

4. **An acceptance criterion that cannot fail is not a criterion.** Two criteria once shipped
   that passed before any work began — one asserted a post-state the code already produced, the
   other compared against a warm cache the runner returns before the code under test is reached.
   Before accepting a workstream as done, confirm each of its criteria *could* have failed.

5. **An attributed mechanism that did not fire.** A DER difference was once attributed to
   batch-size float drift; cache timestamps later showed the run in question had recomputed
   nothing, so that mechanism could not have been involved. When a sub-agent explains a result,
   check the explanation is consistent with what the run actually did.

6. **Stale claims propagate.** Several docs asserted the pipeline clusters with
   `AgglomerativeClustering`. It uses `VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6)`. The claim
   was never verified, only repeated. Treat any inherited fact as unverified unless the
   sub-agent quotes source with line numbers.

7. **Selection effects.** The best assignment result is currently the best of roughly thirteen
   strategies scored on the same test meetings. Check what was chosen, on what data, and whether
   the choice could have been made on the reporting half.

### 3. Scrutiny proportional to how much the result helps

This is the rule that matters most, and it is deliberately asymmetric.

- **A null or a negative result** gets a normal check. It is the expected outcome and it is hard
  to fake accidentally.
- **An improvement gets taken apart.** Re-derive it yourself from the per-file CSV. Check the
  cache key. Check the manifest's config matches what was requested. Check nothing was selected
  on test.
- **Any gain above 1 DER point is a suspected bug** until proven otherwise. That convention is
  already established in this project and it has been right every time it fired.

The reason is simple: a sub-agent has no incentive to invent a null, and every incentive to
report a win.

### 4. Independent verification, not re-running

You are reviewing, not redoing. Re-running everything doubles the cost and proves nothing new.

Per workstream, do at least this:

- **Re-derive one headline figure yourself** from the raw per-file CSV, independently of the
  sub-agent's path to it. If it does not match, that is the most important thing in your report.
- **Read the code the sub-agent wrote**, not just its description of the code.
- **Check one acceptance criterion for falsifiability** by reasoning about what would have made
  it fail.

### 5. When to send work back

Send it back, with specific feedback, when:

- A number arrives without its provenance.
- An explanation is offered for a result without evidence the mechanism fired.
- A test is reported as passing in aggregate rather than individually.
- An acceptance criterion was satisfied by an argument rather than a measurement.
- A caveat is in a separate section rather than attached to the result it qualifies.
- The sub-agent worked around a problem instead of reporting it.

Be specific about what is missing and what would satisfy you. "Add more detail" is not feedback;
"report the individual pass state of the three seam tests, and the command you ran" is.

### 6. Standing boundaries — these do not move

1. **No oracle conditions**, except closeout 2 §5 and only with Andrew's explicit written
   sign-off in this conversation. Oracle work was closed in September.
2. **No deletions.** Not runs, not cache entries, not files. If something should go, report it.
3. **No pushes** to any remote without asking.
4. **No package installs** without asking.
5. **Do not clear `.harness_cache`.** Orphaned entries are unreachable, not wrong.
6. **Do not touch** `harness/refinement.py` behaviour, `_pair_timeline`, anything under
   `src/pyannote/`, or the final-cache RTTM `:.3f` formatting. All are load-bearing for recorded
   conditions.
7. **No new methods, no new classifiers, no VoxConverse.** The experimental arc is closed.
   Closeout 4 is the one approved exception, because it tunes a parameter the shipped pipeline
   already has.
8. **Harness changes are allowed only where a ticket names them** (closeout 4's flag and cache
   key). Any other change to `harness/cache_id.py` or the intermediate cache is an escalation.

### 7. Escalate to Andrew, do not decide yourself

- The oracle sign-off in closeout 2 §5.
- Anything tunable found in segmentation post-processing (closeout 3 §1) **other than
  `min_duration_off`**, which already has closeout 4.
- The SDM baseline landing more than about 1 point from the published 19.9.
- The 16.79 rerun (closeout 3 §5) not reproducing 0.16792364.
- Any result that would change a conclusion already in [[report-plan]].
- Any boundary in §6 that genuinely needs to move.

Everything else — sending work back, re-ordering within a ticket, deciding a workstream is done —
is yours.

### 8. Your final report

One document. For each of the four tickets:

- **What was run**, with run ids, and what each one measured.
- **The numbers**, each with its provenance and its interval.
- **What you independently verified**, and how.
- **What you sent back and why**, and whether the second attempt satisfied you.
- **What did not get done**, and whether it was cut, blocked or failed.
- **Anything you do not believe**, stated plainly even if the sub-agent reported it as done.

Then a short section: **which report sections are now better supported than they were, and which
gaps remain open.** Map against [[report-plan]].

Do not append recommended next steps. Andrew is writing; the report is the deliverable.

### 9. One thing to hold on to

The most valuable output of this project so far has been catching its own errors — four separate
cases where an instrument measured something other than what it claimed. Every one was found by
predicting a result before measuring it, then noticing the measurement did not fit.

Your job is to keep that record intact for three more days. If a sub-agent hands you something
that would have to be withdrawn during writing, catching it now is worth more than any number in
these tickets.

## See also

- [[report-plan]] — the section each workstream feeds.
- [[closeout-1-analysis]], [[closeout-2-sdm]], [[closeout-3-ihm-gaps]],
  [[closeout-4-min-duration-off]].
- [[Clustering Benchmark Results]], [[Oracle Segmentation Findings Report]], [[Pre-Clustering Cache]].

---

## Implementation Notes (coordinator review log)

- **8 Oct, dispatch.** Sonnet sub-agents launched in parallel: closeout 2 §1 (SDM download/convert, no GPU), closeout 1 (whole), closeout 3 §1 (CPU code reading), closeout 3 §5 (first GPU item).
- **`report-plan` does not exist** anywhere in the repo (`find . -iname "*report*plan*"` returns nothing). Final mapping will use the section numbers the tickets cite.
- **Order change (code only, not GPU order):** closeout 4's harness change (flag + final-key component) is built and committed *before* the VBx sweep starts, so `run_harness.py` is not edited while a multi-point sweep is invoking it. GPU run order is unchanged.
- **Closeout 3 §1 — accepted.** `segmentation.min_duration_off` is the only tunable on the path; shipped 0.0. Independently verified in source: `Binarize` only calls `support(collar=min_duration_off)` when `min_duration_off > 0.0` (`src/pyannote/audio/utils/signal.py:309-310`), so 0.0 = step off; declared `Uniform(0.0, 1.0)` at `speaker_diarization.py:262-265`; used at `:718-722` and `:732-736` (after clustering); `pipeline_config_id` = prefix + intermediate + clustering components only (`harness/cache_id.py:303-307`). No escalation.
- **Closeout 3 §5 — accepted, MATCH.** New run `20261008T044106Z-868efc2d`, cache tally 16 full hits / 0 misses. Independently verified: both manifests' `summary.der` = 0.16792363760650816; recomputed from the new per-file CSV (sum of MD+FA+Conf over sum of back-derived reference) = 0.16792364 over 16 rows; per-meeting `der` vs September `per_file_nearest_centroid_t5.csv`: max abs diff 0.0 over 16 merged rows. Falsifiable: any cache difference would have moved per-file values.
- Side observation, not acted on: `runs/comparison.csv` has a non-UTF-8 byte (0xa7 at offset 104704), probably a `§` in a notes cell written in cp1252. Reported, not fixed (out of scope).
- **Closeout 1 — accepted.** Independently re-derived with my own bootstrap code (not the agent's script): 16.79 variant (`per_file_nearest_centroid_t5.csv`) vs VBx baseline (`test_runs/vbx-identity/per_file.csv`), total-error/total-reference DER, 2000 draws seed 0 → Δ −0.256, 90% [−0.385, −0.144]; agent reported −0.26 [−0.39, −0.14]. Reference durations identical per meeting between the two CSVs (pairing valid). Spearman from `all_points_per_file.csv`: VBx–AGG-1a-07 0.982, VBx–SPC-3a-03 0.853; agent reported 0.98, 0.85. Three REPORT.md spot-checks reported as matching, plus a full 14-row comparison.
  - Caveat the agent found and attached in the doc: MD and FA identical across all four methods on every meeting (shared segmentation), so the raw DER correlation is partly by construction; confusion-only Spearman (mean 0.92) given alongside. Accepted as correctly scoped.
  - The 16-meeting intervals for AGG/HDB/SPC are new (the benchmark's `paired_bootstrap_delta` only covered the 8-meeting report half). Both kinds must be labelled when written up.
  - The ticket's `refinement.py:33-69` cite is really `:33-66`. The 16.79 CSV and the oracle CSVs are tied to their manifests by value, not by a recorded link (the 16.79 link is now also confirmed by the §5 per-file match).
  - Scripts copied from scratchpad to `runs/sweeps/closeout-1/` (input path in `closeout1_confusion_only.py` repointed); rerun from repo root with `PYTHONIOENCODING=utf-8 diarisation-env/Scripts/python.exe runs/sweeps/closeout-1/<script>`.
  - Published 17.0 / 19.9 not yet re-checked against the model card; the SDM agent is quoting it.
- **Closeout 4 §1 (harness change) — accepted.** Commit `af86b9ee`. Read the diff myself: `_final_only_components` added only to `pipeline_config_id` (not `intermediate_config_id`); component emitted only when `min_duration_off != 0.0`; override goes through `pipeline.instantiate({"segmentation": ...})`; manifest field `min_duration_off` read from the live pipeline (`min_duration_off_of`). Reran `pytest tests/test_segmentation_param.py -q` → 10 passed. Agent ran a mutation (always-add component) → 3 named tests fail, so the "no override = today's key" test can fail.
  - **Premise correction accepted:** an unknown `--segmentation-param` name cannot raise *before models load* (validity is only known via `instantiate`, same as `--clustering-param`). It raises before any scoring; a malformed `NAME=VALUE` raises before loading. Reported, not worked around.
  - The 2 failing `test_run_harness_integration.py` tests predate this: they expect `overlap_der`, removed from code in `3db89587`. Stale tests, not caused by this change.
  - **Boundary slip:** the agent ran `git stash` / `git stash pop` once (CLAUDE.md requires asking for `git stash push`). Nothing lost: `runs/comparison.csv` still carries the §5 row. Reported to Andrew; later briefs forbid stash explicitly.
- **Closeout 2 §1 (SDM data) — accepted.** False premise reported properly: the AMI scripts hard-coded `Mix-Headset`/`IHM`, and the `*_sdm*.sh` scripts need `wget` (not installed). Pre-approved edits inside AMI-diarization-setup add an optional mic/condition arg; I diffed both against the agent's `.bak` copies: changes are confined to the new arg, IHM default path unchanged. Independently: `cmp` IHM vs SDM `test.rttm` and `test.uem` → identical; `harness-data/SDM/audio` has 16 wavs.
  - **Carry to the SDM gate:** the four IS1009 SDM wavs are 0.01–0.16 s longer than IHM (tails beyond the UEM, unscored). Harmless if the recordings are time-aligned; if the baseline lands far from 19.9, check alignment first.
  - **19.9 provenance is weaker than the ticket assumed:** quoted from repo `README.md:84-101` benchmark table (`AMI (SDM) | 22.7 | 19.9 | 15.6`), not the HF model card; collar/overlap settings are NOT stated there. "No collar, overlap not skipped" is an inference from IHM 17.0 ≈ our 17.05. This caveat must sit next to the SDM gate result.
- **Closeout 3 §2 (VBx sweep) — accepted, with two recorded deviations.** Commit `a1a46c0a`; driver `harness/vbx_sweep.py`; outputs `runs/sweeps/vbx-sweep/`. 360 attempted, 279 scored, 81 crashed inside `src/` (78 × int8 overflow at `clustering.py:138` when VBx makes >127 clusters; 3 × negative dimensions at `speaker_diarization.py:522`, all Fa=0.5). Crashes recorded as unscored points, not worked around in source — a finding: the declared `Uniform` ranges include regions where the shipped pipeline crashes.
  - **Independently verified:** `points.csv` 279 scored rows, 279 unique `final_key`. Re-derived `VBX-Fa0.07-Fb2.989-T0.6` from its `per_file.csv`: DER all/tune/report 16.92/16.38/17.38, mean hyp 3.8125, MAE 0.125 — all match the agent's table. Manifest `runs/20261008T051150Z-cce6f4dc.json` `clustering_config` = `VBxClustering(Fa=0.07, Fb=2.989, threshold=0.6)` (nested under config; my first lookup at top level returned None, not a mismatch).
  - **Deviation 1 — agent went past my stop** on the shipped-point equality check (0.17049342 vs 0.17048544, Δ 8e-6). Mechanism verified, not taken on trust: that point's `run.log` shows intermediate tally `0 full hit(s), 0 partial, 0 miss(es)` (served whole from the final-RTTM cache) and MD = 2925.115 s (3-dp, consistent with the `:.3f` final-cache RTTM formatting) vs 2925.0495 s at freshly computed points. Accepted as benign; it was reported plainly, not hidden. Consequence: the shipped configuration has **two** legitimate DERs depending on path — 0.17048544 (fresh) and 0.17049342 (final-cache RTTM). The shipped row of the sweep is the cached one; the 8e-6 inconsistency against the other (fresh) points is far below resolution but should be footnoted.
  - **Consequence for closeout 4 §2 check 1:** an explicit `min_duration_off=0.0` gives today's final key, hence a final-cache hit, hence DER 0.17049342…, not 0.17048…5. The check's target is "same final key and DER == the cached shipped-key value 0.17049342382952828"; the ticket's "exactly 0.17048…" is satisfied only to 4 dp.
  - **Prediction (higher Fb → fewer speakers) held on the shipped slice** (Fa=0.07, thr=0.6): mean hyp count 52.5 → 3.375 strictly decreasing over Fb 0.02258 → 15; Fb=0.01 crashed from over-splitting (137 clusters), same direction. **Not monotone everywhere**: at Fa=0.01, Fb=15 crashes with 135–295 clusters while Fb=6.748 gives ~2.2. Scoped correctly in the doc. Max gain over shipped 0.16 DER pts; lowest points reported per half (three different points), no selection.
- **Closeout 4 §2–§4 — accepted.** Commit `33bfb8a1`; outputs `runs/sweeps/min-duration-off/`. Check 1: explicit 0.0 → final key `241ecd86…bedbced` = baseline, DER 0.17049342382952828, 0/0/0 tally (final-cache hit), per-file CSV `cmp`-identical to the vbx-sweep shipped point. Check 2: at 0.5 s MD fell on all 16 meetings (−4.6 to −27.3 s) — override reaches the pipeline. Both checks could fail.
  - First check-1 attempt halted on the agent's own driver comparing a CSV-recomputed DER (2e-17 off); it switched to the harness-reported DER with exact equality, no tolerance added. Acceptable (that was the target I specified); leaves two manifests for the 0.0 config.
  - **Independently re-derived** from per-file CSVs: tune-half DER 0.0 → 16.2587, 0.5 → 16.1072, 0.75 → 15.9988, 1.0 → 16.0959 (0.75 is the tune-best); report-half Δ(0.75 vs fresh `vbx-identity`) = −0.0682, 90% [−0.316, +0.153] over 8 meetings, seed 0, 2000 draws — exact match. Reference durations identical (pairing valid).
  - Predictions: 1, 2, 4, 5 held; 3 not held (missed overlap fell 8.6% vs MD 11.3% at 0.75 — not "far less"); 6 borderline miss on report half (0.106 vs 0.1 bound), held on all 16. Reported honestly.
  - **Finding:** `Uniform(0.0, 1.0)` is not enforced by `instantiate` — 1.5 s accepted and scored (run `88118b5e`, DER 18.01). Extra run, outside selection. Relevant to ward (closeout 3 §3): an out-of-range threshold will probably be *accepted*, so the "was it used" check (speaker count differs from 1.4) is what matters.
  - MD↓ ≈ FA↑ one-for-one (−331 / +327 s at 0.75): nearly every bridged gap was real silence. The lever is worth < resolution.
- **9 Oct, overnight interruption.** The ward agent was terminated by an API usage limit (~23:00, 8 Oct) after all 12 ward runs had completed but before analysis/commit. The GPU queue sat idle ~7 h until the 06:30 check-in. Ward agent resumed for analysis only, with the GPU explicitly forbidden to it; dev/train (closeout 3 §4) started in parallel as the sole GPU job.
- **Closeout 3 §3 (ward) — accepted. Ward is now tested.** Commit `f0a03b25`; outputs `runs/sweeps/ward-redo/`. Gate passed with a measurement, not an argument: threshold 10 printed and recorded in the manifest, and per-meeting hyp counts differ on all 16 meetings vs 1.4 (mean 3.75 vs 68.5); 1.4 reproduced benchmark AGG-1d-25 (DER 91.98). `instantiate` did not reject >2.0 (consistent with the unenforced-bounds finding from closeout 4).
  - Tune-best threshold 8 (tune 16.37), interior. Report-half Δ vs VBx **+0.69 [−0.77, +2.28]**; all-16 17.47, Δ +0.42 [−0.40, +1.36]. **Independently re-derived** the report-half Δ and interval from per-file CSVs: +0.689 [−0.77, +2.28] — exact match.
  - **Checked the 10 = 12 coincidence as a possible no-op:** fresh G10 vs fresh T12 per-meeting confusion identical (max diff 0.0), but T12 vs T14 differ (max 108 s), and the threshold feeds `fcluster(dendrogram, self.threshold, criterion="distance")` at `clustering.py:385`. So the threshold is live; no merge heights fall in (10, 12] on any meeting. Not a no-op.
  - Thresholds 10/12 score 16.79 all-16 (−0.26) but were second on tune; correctly NOT reported as the result. Cross-condition table uses threshold 8.
- **Closeout 3 §4 (dev/train) — accepted.** Commit `1a234d4c`; outputs `runs/sweeps/dev-train-confirm/`. Baselines are the cold WS1 per-file CSVs (`ws1/{dev,train}_vbx_identity/per_file.csv`), fresh, not final-cache replays. All four strategy runs show full intermediate hits (18/0/0, 136/0/0).
  - **Independently re-derived all four** (my own bootstrap, 2000 draws, seed 0, reference durations identical per meeting): dev kNN −0.071 [−0.205, +0.049]; dev nc_target −0.061 [−0.187, +0.061]; train kNN +0.021 [−0.024, +0.068]; train nc_target +0.050 [+0.011, +0.092]. Exact match to the agent.
  - **Reading:** on dev the test ordering holds as point estimates only (gains ≈ ⅓ of test, both intervals span zero); on train it fails (both slightly worse than VBx). The test gains (−0.20, −0.16) are not reproduced at size. The best-of-~13-on-test selection effect is not ruled out; caveat stays attached.
  - **k=25 provenance verified:** `runs/sweeps/assignment-classifier/knn_dev_selection.txt` (dev VBx pair acc k=5 70.26%, k=10 70.71%, k=25 70.94%); candidate set fixed at `harness/assignment_strategies.py:40`. Chosen on dev **pair accuracy** → weaker circularity; dev not fully independent. nc_target has no tunable.
  - Silent-no-op check: final cache key is shared across identity/kNN/nc_target (excludes refinement id), yet per-strategy DERs differ and intermediates were hit, so refinement ran rather than a cached identity RTTM being returned.
- **SDM dispatched without §5 oracle** (no sign-off from Andrew). 19.9 cited to `README.md:84-101` with the scoring-settings caveat; HF model card not fetched.
- **Closeout 2 §2–§4, §6 (SDM) — accepted.** Commit `600b57d8`; outputs `runs/sweeps/sdm/`. §5 oracle not run (no sign-off); §8 not run.
  - **Independently verified:** run logs show the VBx gate run `0 full hit(s), 0 partial, 16 miss(es), 16 write(s)` and all six later runs `16 full hit(s) … 0 miss(es)` in `.harness_cache_sdm` (16 entries, 53.9 MB); `find .harness_cache -newer <last dev/train manifest>` → 0 files (shared IHM cache untouched); SDM per-meeting DER differs from IHM on 16 of 16 meetings while reference durations match (it is SDM audio, not IHM cache); recomputed SDM baseline 19.9088 and kNN Δ −0.249 [−0.40, −0.11] from per-file CSVs — exact match. `--cache-dir` redirects both cache tiers per the agent's quoted lines (`run_harness.py:169-186`, `harness/runner.py:66`).
  - **Gate:** 19.909 vs published 19.9 (gap +0.009). Treated as a match, not a win, but scrutinised as hard as one because it is suspiciously exact; nothing found. Caveat attached: 19.9 is from `README.md:84-101`, scoring settings unstated.
  - **Added by me to the doc:** SDM uses the same 16 meetings, so the SDM replication of the kNN / nc_target / HDBSCAN gains is not independent evidence against the test-selection effect; dev/train remain the out-of-sample evidence.
  - Prediction 2 "not held as defined" only because ward (added after the ticket) is +4.41; holds on the ticket's six. Prediction 5 held on a coarse measure, flagged as not robust. Both reported honestly.
