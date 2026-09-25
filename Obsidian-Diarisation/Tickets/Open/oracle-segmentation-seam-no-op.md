---
status: done
created: 2026-09-18
---

# Oracle segmentation seam is a no-op at run time

## Goal

Make the `oracle_segmentation` condition actually use injected ground-truth segmentation,
re-run it, and record that the number it produces isolates segmentation and nothing else.

Blocks the segmentation-budget figure, which in turn gates the clustering-sweep vs.
classifier decision.

## Symptom

Run `3c8deb71` (`oracle_segmentation`) recorded seven metrics identical to run `aa8fbfa4`
(`baseline`) to every digit:

```
der                  0.17048543579940637
der_overlap_system   0.3339417838704696
der_overlap_assigned 0.219454688210064
jer                  0.22186961360102092
region_t_and_d       2433.11687500005
region_t_minus_d     1393.9391249999528
region_d_minus_t     382.94937499995154
```

The output is **semantically identical to baseline with different speaker labels** — not
byte-identical. Measured across all 16 meetings from the cached RTTMs (2026-09-18):

| evidence | result |
|---|---|
| identical segment timings (oracle vs baseline) | **16/16** |
| identical speaker cardinality | **16/16** |
| identical labels | **0/16** |

Segment counts match exactly too (EN2002a 1000/1000, EN2002b 624/624, …). The sole
difference is naming: `SPEAKER_00..03` in baseline vs. real AMI IDs (`MEE073`, `FEO072`, …)
in oracle. Since DER is label-invariant under optimal mapping, the renaming moves no metric
— hence seven identical numbers.

> **Superseded claim.** An earlier version of this brief argued identical region-census
> geometry proved the hypothesis was *byte-identical*. That overreached: identical geometry
> proves identical **timings**, and labels do not affect overlap geometry, so files can
> differ by name alone — which is what happened. The no-op conclusion held, but that
> evidence did not support it as stated. The 16/16 timing-and-cardinality table above is the
> actual proof.

Magnitude confirms it independently: baseline
components are 2925.05 s missed detection, 1099.07 s false alarm, 1212.15 s confusion over
~30,714 s of reference speech. A working oracle run should collapse missed detection and
false alarm toward zero, leaving ~1212 / 30714 ≈ **4% DER**. It landed at 17.0% — the whole
error budget still present.

Not a cache collision (the failure mode behind the invalid `e9f25911` `nearest_centroid`
run): `segmentation_source.id` is in the cache key at `harness/runner.py:38-41` and differs
between conditions. The oracle run genuinely recomputed, and recomputed the baseline answer.

## Root cause (confirmed by code read)

`SpeakerDiarization.get_segmentations()` at `speaker_diarization.py:337-344` gates its only
read of `CACHED_SEGMENTATION` on `self.training`, and that is the sole guard:

```python
if self.training:
    if self.CACHED_SEGMENTATION in file:
        segmentations = file[self.CACHED_SEGMENTATION]
    else:
        segmentations = self._segmentation(file, hook=hook)
        file[self.CACHED_SEGMENTATION] = segmentations
else:
    segmentations = self._segmentation(file, hook=hook)
```

`OracleSegmentation.populate()` at `harness/segmentation.py:86-93` restores `training` in a
`finally` **before returning**, and `harness/runner.py:57-59` only calls
`self._pipeline(file)` afterwards. At the moment the gate is evaluated, `training` is False.
The `else` branch runs and the injected array is never read.

The seam was implemented correctly per [[Segmentation Injection Seam]] and then closed one
line too early. Note that doc's framing — "the fix is a calling convention" — is what led
here: the convention is right, but its *scope* was set to `populate()` rather than to the
pipeline call it was meant to affect.

Ruled out by the same read:

- **Not a 4.x gate relocation.** `Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")`
  resolves to this same `SpeakerDiarization` class. (There is a separate
  `pipelines/pyannoteai/local.py` that also returns `output.speaker_diarization`, so the
  runner's use of that attribute does not by itself disambiguate — but the checkpoint lands
  on `SpeakerDiarization`.)
- **Not a rebuilt file dict.** `apply()` at `speaker_diarization.py:609` passes `file`
  straight to `get_segmentations` with no `validate_file` rebuild in between.
- **Not a wrong RTTM.** `populate()` raises `KeyError` on a URI miss and the run completed
  over all 16 meetings. (Still worth recording the RTTM path — see
  [[run-manifest-provenance]].)

## Two confounds, not one

The run is meant to isolate segmentation. It currently differs from baseline in three ways:
the injected array (which does nothing), the `training` flag (which gates embedding caching
too), and `file["annotation"]` (which may leak true speaker count).

### Confound A: the annotation leak — DORMANT, not dead

`populate()` sets `file["annotation"] = reference` at `harness/segmentation.py:84` purely as
scaffolding, because `oracle_segmentation(file, window, frames)` reads it to build the array.
That key then reaches `apply()`, where it has **two** possible effects.

**Effect 1 — cosmetic relabelling (live today; fully explains the symptom).**
`speaker_diarization.py:742-764`, at the very end of `apply()`:

```python
if "annotation" in file and file["annotation"]:
    # when reference is available, use it to map hypothesized speakers
    # to reference speakers (this makes later error analysis easier
    # but does not modify the actual output of the diarization pipeline)
    _, mapping = self.optimal_mapping(file["annotation"], diarization, return_mapping=True)
    mapping = {key: mapping.get(key, key) for key in diarization.labels()}
else:
    mapping = {label: expected for label, expected in zip(diarization.labels(), self.classes())}
diarization = diarization.rename_labels(mapping=mapping)
```

This is the `MEE073`/`FEO072` relabelling, and upstream's own comment confirms it is
cosmetic by construction — which is why timings and cardinality were untouched and no metric
moved. The mechanism is the **pipeline**, not the harness or the scorer.

**Effect 2 — speaker-count leak (dormant on this config).**
`speaker_diarization.py:600-602`:

```python
if self._expects_num_speakers and num_speakers is None:
    if isinstance(file, Mapping) and "annotation" in file:
        num_speakers = len(file["annotation"].labels())
```

`_expects_num_speakers` comes from `self.clustering.expects_num_clusters`
(`speaker_diarization.py:295`). Community-1 uses `pyannote-default` =
**`VBxClustering`**, which sets `expects_num_clusters = False`
(`clustering.py:552`). **So this branch does not fire today** and there is no live
speaker-count confound.

> **Attribution corrected 2026-09-25.** This paragraph previously credited
> `AgglomerativeClustering` (`clustering.py:310`). The shipped default is VBx; agglomerative
> has never been run here. **The conclusion is unchanged** — both classes declare
> `expects_num_clusters = False`, so the branch is dormant either way. Only the reason is
> corrected. See [[Pre-Clustering Cache]].

**But it is one config change away from firing.** The only two classes that set it True are
`KMeansClustering` (`clustering.py:496`) and `OracleClustering` (`clustering.py:675`). The
planned clustering sweep includes KMeans. The moment it runs, the *same already-present*
`file["annotation"]` key additionally feeds true speaker count in, and
`oracle_segmentation` silently becomes oracle-segmentation-**plus-oracle-count** — an
invalid condition that would look better than it should and would not announce itself.

Effect 1 proves the key reaches `apply()`. Only the guard on Effect 2 is currently false.
That is why **Fix 1 is a landmine defusal, not hygiene** — it is free to do now and three
weeks ahead of the run that would otherwise step on it.

### Confound B: embedding caching

`training` also gates embedding cache read (`speaker_diarization.py:377-386`) and write
(`483-487`). Holding it True for only the oracle condition makes the two arms behave
differently in a run whose entire purpose is single-variable attribution.

Worth noting the read/write asymmetry, since it constrains any future reuse of a file dict:
the read at line 384 does a bare `cache["segmentation.threshold"]` subscript, while the
powerset write at 484-487 stores only `{"embeddings": ...}` with no threshold key. A
powerset write followed by a non-powerset read is a `KeyError`. Within a single file that
cannot happen (one read, one write, same model), so this is latent, not live.

## Scope

### Fix 1 — close the annotation leak

Restore `file["annotation"]` to its prior state once the array is built, so the pipeline
never sees it. Unambiguous and independent of how Confound B is resolved.

**Justification is the forward risk, not a live confound.** On today's config
(`VBxClustering`) the only effect is cosmetic relabelling. Retain this fix because
the planned KMeans clustering sweep flips `expects_num_clusters` to True and turns the same
key into a real speaker-count leak — see Confound A above. Doing it now costs nothing; not
doing it means a future run is silently invalid.

Side benefit: output labels revert to `SPEAKER_00..` in both conditions, so a future
oracle/baseline RTTM diff is a straight byte comparison rather than needing
timing-and-cardinality analysis to see past the names.

```python
original_annotation = file.get("annotation")
file["annotation"] = reference
try:
    file[pipeline.CACHED_SEGMENTATION] = oracle_segmentation(file, window, frames)
finally:
    if original_annotation is None:
        file.pop("annotation", None)
    else:
        file["annotation"] = original_annotation
```

`_expects_num_speakers` has been resolved for this checkpoint: **False**
(`VBxClustering`, `clustering.py:552`; attribution corrected 2026-09-25 from
`AgglomerativeClustering`, the finding itself unchanged). Verified 2026-09-18, re-verified
2026-09-25. The write-up can state
that the oracle condition does not leak speaker count on this configuration, and must note
the KMeans caveat alongside it.

### Fix 2 — apply `training` uniformly across both conditions

Move flag management out of `populate()` and into the runner, wrapping the pipeline call:

```python
original_training = self._pipeline.training
self._pipeline.training = True
try:
    self._segmentation_source.populate(self._pipeline, file)
    output = self._pipeline(file)
finally:
    self._pipeline.training = original_training
```

Rationale for *uniform* rather than oracle-only: under this arrangement the sole difference
between conditions is whether `CACHED_SEGMENTATION` was pre-populated. Baseline takes the
inner `else` branch, computes its own segmentation and caches it — behaviourally the old
path plus one dict write. Embedding caching now happens in both arms, so Confound B stops
being a confound instead of being argued harmless.

Update the `OracleSegmentation` and `SegmentationSource` docstrings, which currently state
the flag is flipped inside `populate()`.

**Fallback (only if Fix 2's equivalence test fails):** wrap `pipeline._segmentation` in a
proxy returning the precomputed array and forwarding every other attribute (`.step`,
`.duration`, `.model.receptive_field`) to the real `Inference`. Leaves `training` at False
and changes nothing but the segmentation return value. More code, needs the same
runner-scoped lifetime.

### Fix 3 — bump the oracle source id

After the fix the cache key is unchanged, because `OracleSegmentation.id` is still
`"oracle"`. The 16 stale RTTMs from the broken run would be served as cache hits
(`harness/runner.py:51-52`) and the same wrong numbers re-scored.

Set `OracleSegmentation.id = "oracle-v2"`. New key, old entries orphaned and harmless, no
cache surgery, and `segmentation_source_id` in the manifest now records which implementation
produced each result — better provenance than deleting files.

Do **not** clear `.harness_cache` wholesale; that pays for baseline recomputation too.

## Acceptance criteria

1. **Baseline equivalence (explicit criterion, not just the oracle result).** Under Fix 2,
   a baseline run produces a byte-identical hypothesis RTTM to the committed baseline cache
   entry for the same URI. This is what retires Confound B by measurement rather than
   argument.

   **This must run against a throwaway `cache_dir`.** `BaselineSegmentation.id` stays
   `"baseline"`, so a normal re-run hits the existing cache and returns the old RTTM without
   invoking the pipeline at all — the diff would compare a file against itself and pass
   trivially, greenlighting an untested change. Point the runner at a scratch cache for this
   one run and diff its output against the committed entry. One meeting suffices.

2. `tests/test_segmentation_seam.py` runs to a genuine pass (see dependency below), including
   `test_reproduces_bug_cached_segmentation_ignored_outside_training`, which is named for
   this exact bug and has never executed.

3. Full 16-meeting oracle run lands near **4% DER**, dominated by confusion, with missed
   detection and false alarm near zero and `region_t_minus_d` collapsed.

   The ~4% figure is now on solid ground: with the speaker-count leak confirmed dormant
   (Confound A) and embedding caching made uniform (Confound B), a post-fix oracle run has
   exactly one variable changed against baseline.

4. **Config guard.** Assert `pipeline._expects_num_speakers is False` (or that
   `file["annotation"]` is absent at pipeline entry) in the oracle path, so the KMeans sweep
   fails loudly rather than silently producing an oracle-count-contaminated run. Cheap
   insurance given Fix 1 already removes the key.

5. If DER stays near 17%, Fix 2 did not take and the gate is elsewhere.

6. Post-fix, the oracle hypothesis RTTM should differ from baseline in **timings**, not just
   labels — the inverse of the 16/16 table in the Symptom section. That table is the
   "before" artifact; regenerating it after the fix is the clearest single check that the
   injection took effect.

## Ordering

1. ~~Diff the cached oracle vs. baseline RTTM.~~ **Done 2026-09-18** — result is the 16/16
   table in the Symptom section. All 80 `.harness_cache` entries left intact; the 16 stale
   oracle RTTMs are preserved as the "before" artifact (another reason to bump the id rather
   than delete them).
2. Resolve [[non-ascii-filename-decoding]] (see dependency). **Must land and be verified
   green before step 3** — the seam tests are how the fix gets confirmed, rather than
   inferred from a DER number.
3. Apply Fixes 1, 2 and 3.
4. Baseline equivalence test, on a scratch cache dir.
5. Single-meeting oracle run, check the shape.
6. Full 16-meeting oracle run.
7. Regenerate the timing/cardinality table; expect timings to now differ (criterion 6).

## Dependency: RESOLVED 2026-09-18 — [[Non-ASCII Filename Decoding Fix]]

**Cleared. The seam tests now pass 3/3.** Verified independently, not just reported:

```
tests/test_segmentation_seam.py::test_reproduces_bug_cached_segmentation_ignored_outside_training PASSED
tests/test_segmentation_seam.py::test_cached_segmentation_is_used_via_training_flag PASSED
tests/test_segmentation_seam.py::test_fix_has_no_side_effect_on_normal_output PASSED
3 passed in 1.81s
```

Two fixes were needed in `tests/conftest.py`, both confined to test setup (no `src/`,
`diarisation-env/` or `harness/` changes):

1. **UTF-8 decode.** `pyannote/database/loader.py`'s `load_lst()` used
   `open(file_lst, mode="r")` with no `encoding=`, so Windows decoded the deliberately
   non-ASCII `trñ00` entry as cp1252. Monkeypatched onto **both**
   `pyannote.database.loader.load_lst` and `pyannote.database.custom.load_lst` — `custom.py`
   does `from .loader import load_lst`, binding its own reference at import time, so patching
   the definition site alone would miss the actual call.
2. **Windows `spawn` pickling.** Once decoding was fixed, fixture setup reached
   `trainer.fit()` and hit
   `PicklingError: Can't pickle <class 'pyannote.database.registry.Debug'>`.
   `SpeakerDiarizationTask` defaults `num_workers` to `cpu_count() // 2`
   (`src/pyannote/audio/core/task.py:288-289`); Windows has no `fork()`, and the `Debug`
   protocol class is created dynamically at runtime so it cannot be resolved by attribute
   lookup in a spawned child. Fixed by passing `num_workers=0` in the two conftest task
   constructors — the same carve-out upstream already makes for macOS at
   `task.py:291-301`. Confirmed pre-existing: with the fix stashed, `tests/test_train.py`
   still failed 18/18.

Full suite moved 199 → 211 passed, errors 10 → 6, failures unchanged at 20. Nothing
regressed. Also unblocked `test_run_manifest_written_alongside_report` and
`test_run_manifest_not_written_on_mid_run_failure`, which had never run to a pass on this
machine.

**Significance for this ticket:** criterion 2's verification path is live.
`test_reproduces_bug_cached_segmentation_ignored_outside_training` passing confirms the
`if self.training:` gate still behaves as the Root Cause section describes, so the fix below
rests on tested behaviour rather than inference from a DER number.

### Known-unrelated red tests (do not fix here)

`tests/test_run_harness_integration.py` has 2 remaining failures:
`test_per_file_csv_has_one_row_per_file_with_all_four_metrics` and
`test_corpus_summary_has_all_required_fields`. Both assert a summary field named
`overlap_der`, which T1/T2 renamed to `der_overlap_system`/`der_overlap_assigned`. Stale
assertions, not a defect: `overlap_der` survives only as a *parameter* name in
`harness/scorer.py:44` and `harness/reporter.py:63`, while the emitted key is
`der_overlap_system` (`reporter.py:91`). The value is unchanged across the rename — run
`a9c41646` records `overlap_der = 0.3339467146571433` and every later run records
`der_overlap_system = 0.3339467146571433`. The tests only read, so they cannot affect
results.

One reading trap worth knowing: `runs/comparison.csv` carries **both** columns, each
half-populated (`a9c41646` in the old one, the other seven in the new one). Charting by a
single column name silently drops runs. Deferred to its own ticket after this work lands —
deliberately out of scope here.

Correction to a premise that was circulating: the three seam tests do **not** error at
collection. `pytest --collect-only` collects all three cleanly. They error at *fixture
setup*, on the `pipeline` / `full_pipeline` fixtures, with
`FileNotFoundError: Could not find file "trñ00"` from
`pyannote/database/file_finder.py:126`. Same root cause, different stage — which matters
because it means the test bodies themselves are untested, not merely unreached.

Consequence for [[Segmentation Injection Seam]]: that doc states all three tests are
"(all passing)". They cannot have passed on this machine in their current state. The doc
also names the second test `test_cached_segmentation_is_used_via_chosen_fix` while the file
defines `test_cached_segmentation_is_used_via_training_flag`, suggesting the doc's
verification section was written from intent rather than from a run. Worth correcting when
this ticket converts to documentation.

Note `test_fix_has_no_side_effect_on_normal_output` already attempts Fix 2's equivalence
check at unit scale (runs `full_pipeline` with `training` False then True, no cache
populated, asserts identical `speaker_diarization`). Once the fixture bug is fixed, that
test plus acceptance criterion 1 cover the confound at both scales.

## Invalidated runs

Full triage of every run in `runs/comparison.csv`, so it is not read at face value. Note
`oracle_assignment` uses a **different mechanism** (the post-clustering refinement hook,
`refinement_strategy: oracle`, `segmentation_source_id: baseline`) and never touches the
`CACHED_SEGMENTATION` seam — so this bug does not reach it.

| run | condition | verdict |
|---|---|---|
| `a9c41646` | baseline | valid (pre-manifest, sparse fields) |
| `d3ef4437` | baseline | valid |
| `e9f25911` | nearest_centroid | **DEAD** — cache-key collision; identical to `d3ef4437` in every metric. Superseded by `32241285`. |
| `1db2ba34` | oracle_assignment, all_pairs | valid mechanism, unverifiable provenance (see below) |
| `c93fb9f9` | oracle_assignment, overlap_degraded | valid mechanism, unverifiable provenance (see below) |
| `3c8deb71` | oracle_segmentation | **DEAD** — this ticket. Segmentation injection never applied. |
| `aa8fbfa4` | baseline | valid; the DER-component source (2925.05 MD / 1099.07 FA / 1212.15 conf) |
| `32241285` | nearest_centroid | valid; confusion 1212.15 → 1133.47 with MD/FA pinned, i.e. the mechanism fired |

**Why the two `oracle_assignment` runs are unaffected:** their DER moved (17.05% → 16.08%
all_pairs, → 16.58% overlap_degraded) and `der_overlap_system`/`der_overlap_assigned` both
shifted. A broken injection reproduces baseline exactly, as `3c8deb71` and `e9f25911` did.
These did not. The +0.97pt / +0.47pt ceilings stand, and [[Oracle Assignment Strategy]] /
[[Oracle Ceiling Metrics]] do not need revisiting.

**Provenance caveat on those two:** both record `git_commit = 3d3d9fa6` while their notes say
"post cache-key fix", and `a4a3101a` (that fix) is newer in the log — the fix was almost
certainly uncommitted at run time. Numbers are self-consistent and trusted, but the exact
code state is not recoverable from the manifest. Re-run under a clean committed tree once
[[run-manifest-provenance]] lands, which also exercises the new dirty-tree flag.

## Non-goals

- Manifest provenance gaps (`--oracle-rttm` untracked, unreliable `git_commit`). Real, but
  they touch the manifest writer rather than the seam, and bundling them would prevent this
  ticket closing on unrelated work. Split to [[run-manifest-provenance]].
- Patching `src/pyannote/audio`. The seam remains a calling convention; only its scope
  changes.

## Implementation Notes

(append as work proceeds)

**2026-09-18 — code changes found already applied; ran acceptance checks 1-3, found an unflagged regression.**

On starting this task, `harness/segmentation.py` and `harness/runner.py` were read
before any edit was made (per instructions) and turned out to **already contain all
three fixes**, matching the ticket's specified code verbatim:

- Fix 1 (annotation leak): `original_annotation = file.get("annotation")` /
  try-finally restore, present in `OracleSegmentation.populate()`.
- Fix 2 (uniform training flag): removed from `populate()` entirely; `Runner.run()`
  now wraps both `populate()` and `self._pipeline(file)` in a
  `has_training_attr`/`original_training` try-finally, with a comment explaining the
  `hasattr` guard is for lightweight test fakes.
- Fix 3 (cache id bump): `OracleSegmentation.id = "oracle-v2"`.
- Docstrings (module-level, `SegmentationSource.populate`, `OracleSegmentation` class)
  all updated to describe the runner-scoped flag instead of `populate()`-scoped.
- **Config guard (acceptance criterion 4) was also already present**: `populate()`
  raises `RuntimeError` if `getattr(pipeline, "_expects_num_speakers", False)` is
  True, before setting `file["annotation"]`, with a message pointing at Confound A.
  This fit cleanly — it did not need to be skipped.

`git diff` against HEAD confirms these are uncommitted working-tree changes (not
something merged separately). Confirmed with the coordinator: an earlier launch of
this same task was interrupted mid-flight and had almost certainly applied the
edits before being cut off — not a separate/mystery session. I verified the applied
code against the ticket spec (see diffs quoted above) and it matches exactly. No
further code edits were made to either file at that point; this session's job
became verification.

Ran the three acceptance checks as instructed:

1. `python -m pytest tests/test_segmentation_seam.py -q` → **3 passed** (1.87s).
   `test_reproduces_bug_cached_segmentation_ignored_outside_training` still passes,
   confirming it still documents current upstream `src/` behaviour untouched by
   this work.
2. `python -m pytest tests/test_segmentation.py tests/test_runner.py -q` →
   **19 passed, 1 skipped**. No failures.
3. `python -m pytest tests/ -q --ignore=tests/test_run_notebooks.py` →
   **24 failed, 209 passed, 1 skipped, 6 errors** — this is **below the required
   floor**. Baseline to beat was 22 failed / 211 passed; this is 2 worse on both
   counts.

**Root cause of the regression, fully isolated:** two pre-existing, committed test
files not mentioned anywhere in this ticket's Scope —
`tests/test_run_harness_integration.py::test_oracle_segmentation_run_completes_scores_and_tagged_in_manifest`
and
`tests/test_run_harness_oracle_segmentation.py::test_oracle_segmentation_run_appears_in_comparison_csv`
(both from commit `287ed25f`, T3) — each hardcode
`assert manifest["run_config"]["segmentation_source_id"] == "oracle"`. Fix 3 changes
the actual value to `"oracle-v2"`, so both now fail with
`AssertionError: assert 'oracle-v2' == 'oracle'`. Confirmed these are the entire
delta: 22 (baseline) + 2 (these) = 24 failed; 211 − 2 = 209 passed. Nothing else
moved. Neither file was touched by me — `git status` shows them clean/untracked-diff
before and after.

This is a real gap in the ticket, not a flaw in Fix 3: the ticket's Fix 3 rationale
(orphan the 16 stale cached RTTMs by changing the cache-key id) is sound and the
id-bump is doing exactly what it was designed to do. But two tests elsewhere in the
suite asserted the literal old id as their pass condition, and the ticket's Scope
never surveyed for that. My task instructions authorized edits only to
`harness/segmentation.py` and `harness/runner.py` (test edits were scoped to
`tests/test_segmentation.py`/`tests/test_runner.py` only, and only if they encoded
the old flag-in-`populate()` behaviour — neither of these two files is one of those,
and their assertion is about the id string, not the flag placement). Editing them
was outside what I was told I could do, and the acceptance bar (passed must not drop
below 211, failed must not rise above 22) is explicitly violated as the tree stands.
Stopping here and reporting rather than silently patching those two files or
declaring the ticket done against a failing gate.

**Not done in this session:** acceptance criteria 1 (baseline-equivalence byte-diff
on a scratch cache dir), 3 (full 16-meeting oracle run), 5, and 6 (regenerated
timing/cardinality table) — these require running the harness, which this task's
instructions explicitly excluded ("no run_harness.py, no scoring — that is a later
agent's job"). Only the code-level fixes and the three specified pytest checks were
in scope here.

---

**2026-09-18 (new session) — running the verification experiment. Path resolution
and a scoping constraint found before any run started.**

This session's job is measurement only — no further code changes to `harness/` or
`src/`. Confirmed the prior session's applied fixes are still in place by re-reading
`harness/runner.py` and `harness/segmentation.py` in full before running anything:
`Runner.run()` wraps both `populate()` and `self._pipeline(file)` in the
`has_training_attr`/`original_training` try-finally; `OracleSegmentation.id =
"oracle-v2"`; the `_expects_num_speakers` `RuntimeError` guard is present in
`populate()`. Matches the previous note exactly, nothing to redo.

**Data paths resolved with confidence, no guessing needed:**
- `--data-root` = `~/Code/AMI-diarization-setup/harness-data` — this is
  `run_experiment.sh`'s own default (`DATA_ROOT="${DATA_ROOT:-$HOME/Code/AMI-diarization-setup/harness-data}"`),
  confirmed present on disk (`IHM/test.rttm`, `IHM/test.uem`, `IHM/audio/*.wav` for
  all 16 URIs).
- `--oracle-rttm` = `~/Code/AMI-diarization-setup/harness-data/IHM/test.rttm` — the
  *same* file the adapter already uses as the scoring reference. Not a guess:
  [[run-manifest-provenance]] Gap 1 states this explicitly ("the scoring reference
  and the oracle reference are currently the same file
  (`harness-data/IHM/test.rttm`)... that is what makes the ~4% DER prediction...
  valid"). No second RTTM exists anywhere in either repo.
- Cache-key formula in `harness/runner.py:37-42` matches the ticket's Step 1
  instruction verbatim: `f"{pipeline_config_id}|{segmentation_source.id}|{refinement_id}|{uri}"`.
  Computed the EN2002a baseline key
  (`pyannote/speaker-diarization-community-1|baseline|identity|EN2002a` →
  `eeae371d0aa23cf9d7b3e75a9a94e4ceb726752ef4958604615f0f6682050133`) and confirmed
  `.harness_cache/eeae371d0aa23cf9d7b3e75a9a94e4ceb726752ef4958604615f0f6682050133.rttm`
  exists (64540 bytes) before touching anything.
- Found the run launcher at repo-root `run_experiment.sh` (not `launch_run.py` as
  the task description guessed — the actual filename from `f62549ac`). Did not use
  it directly for step 1 because it has no scratch-cache-dir override wired to a
  single invocation cleanly alongside the other required flags; called
  `run_harness.py` directly instead, matching every flag `run_experiment.sh` would
  have passed.

**Scoping constraint hit before running anything (reporting, not silently working
around):** the ticket's Steps 2 and Step 1's "one meeting" instruction assume some
way to restrict a run to a single URI. No such flag exists —
`run_harness.py`/`harness/datasets.py`'s `AMIDatasetAdapter` always iterates every
URI in the split's combined `test.rttm`/`test.uem` (confirmed by reading both files
in full; no `--uri`/`--limit` argument anywhere in `run_harness.py`). Adding one
would be a `harness/` code change, which this task explicitly disallows, and writing
a separate ad hoc single-file script is also explicitly disallowed. So "one meeting"
for steps 1 and 2 is implemented as: run the full 16-meeting harness (unavoidably
full cost each time, no way to shortcut it within the stated constraints), then read
out only the EN2002a row/RTTM from the result. Flagging this plainly rather than
quietly reinterpreting "single-meeting" as something cheaper than it is — step 1's
scratch-cache run is paying the full ~4300s-class cost, not a quick spot check.

Step 1 launched: full baseline run, `--cache-dir` pointed at a fresh temp directory
(outside `.harness_cache` entirely) so every URI is forced to recompute rather than
serving a cache hit. Command and result to follow once it completes.
