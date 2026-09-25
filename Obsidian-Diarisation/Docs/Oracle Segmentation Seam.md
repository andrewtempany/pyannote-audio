---
status: done
created: 2026-09-18
completed: 2026-09-25
aliases: [oracle-segmentation-seam-no-op]
---

# Oracle Segmentation Seam

How the harness injects ground-truth segmentation into
`pyannote/speaker-diarization-community-1` without patching the library.

This is the **mechanism** document. For the experimental result (3.96% DER, run `18016c15`)
and what it means for planning, see [[Oracle Segmentation Findings Report]]. Nothing here
repeats those numbers beyond what is needed to explain the mechanism.

All line numbers below were verified against the working tree on 2026-09-25.

---

## 1. What the seam is

`SpeakerDiarization` already contains a private cache slot used when the pipeline's
hyperparameters are being optimised. The harness reuses it as an injection point: write a
precomputed segmentation array into that slot and the pipeline will consume it instead of
running its own segmentation model.

The slot is a plain dictionary key on the `file` mapping
(`src/pyannote/audio/pipelines/speaker_diarization.py:317-319`):

```python
@property
def CACHED_SEGMENTATION(self):
    return "training_cache/segmentation"
```

The read of that slot is gated, and the gate is the whole story
(`speaker_diarization.py:337-344`):

```python
if self.training:
    if self.CACHED_SEGMENTATION in file:
        segmentations = file[self.CACHED_SEGMENTATION]
    else:
        segmentations = self._segmentation(file, hook=hook)
        file[self.CACHED_SEGMENTATION] = segmentations
else:
    segmentations: SlidingWindowFeature = self._segmentation(file, hook=hook)
```

`training` is a plain boolean on the base `pyannote.pipeline.Pipeline`, defaulting to False.
It is unrelated to `nn.Module.train()/eval()` — it does not touch dropout or batchnorm.
During ordinary inference the `else` branch runs and `CACHED_SEGMENTATION` is never
consulted at all.

So the seam has two halves, and **both are required**:

1. `file[pipeline.CACHED_SEGMENTATION]` holds the injected array.
2. `pipeline.training` is True **at the moment `get_segmentations()` runs**.

The seam is a calling convention, not a patch. No file under `src/pyannote/audio/` is
modified to make oracle segmentation work.

---

## 2. Where it lives: the flag is runner-scoped

Flag lifetime is owned by `Runner.run()` (`harness/runner.py:69-83`), not by the
segmentation source:

```python
has_training_attr = hasattr(self._pipeline, "training")
original_training = getattr(self._pipeline, "training", None)
if has_training_attr:
    self._pipeline.training = True
try:
    self._segmentation_source.populate(self._pipeline, file)
    output = self._pipeline(file)
finally:
    if has_training_attr:
        self._pipeline.training = original_training
```

Three properties of this arrangement matter.

**The flag spans the pipeline call, not just `populate()`.** This is the single most
important fact about the seam, because getting it wrong is the original bug (§5). The gate at
`speaker_diarization.py:337` is evaluated *inside* `self._pipeline(file)`. A flag that is set
and restored around `populate()` alone is already back to False by the time the gate is
reached, and the injected array is silently ignored.

**It is restored in a `finally`.** A failed or aborted run never leaves the pipeline
permanently in training mode.

**`populate()` runs before the pipeline is touched.** `OracleSegmentation.populate()` raises
`KeyError` on a URI with no reference (`harness/segmentation.py:114`), so a missing reference
fails before any pipeline work and cannot leave a partial or wrong cache entry on disk.

The `hasattr`/`getattr` guard exists only for lightweight test fakes that never defined a
`training` attribute. Every real pyannote `Pipeline` sets `self.training = False` in
`__init__`, so in production the guard always passes.

---

## 3. Why the flag is applied uniformly to both conditions

`Runner.run()` sets `training = True` for **every** segmentation source, baseline included —
not only for oracle runs. This is deliberate.

`training` gates more than the segmentation read. It also gates the embedding cache, both the
read (`speaker_diarization.py:377-386`) and the write (`speaker_diarization.py:483-492`).
Setting the flag for the oracle arm only would have made the two arms differ in *two* ways —
injected segmentation and embedding caching — in an experiment whose entire purpose is
single-variable attribution.

Under uniform application, the only difference between conditions is whether
`CACHED_SEGMENTATION` was pre-populated. Baseline takes the inner `else` branch at
`speaker_diarization.py:341-342`: it computes its own segmentation and additionally caches it,
which is the old behaviour plus one dictionary write.

**This confound was removed by measurement, not by argument.** A baseline control run with the
flag on, forced to fully recompute against a throwaway cache directory, produced **16/16
byte-identical** RTTMs against the pre-fix baseline output. The flag is inert with respect to
output. (The throwaway cache was essential: re-running against the normal cache would have
served the existing files without invoking the pipeline, and the comparison would have passed
trivially while testing nothing.) `tests/test_segmentation_seam.py::test_fix_has_no_side_effect_on_normal_output`
asserts the same property at unit scale.

A latent asymmetry in the embedding cache is worth knowing, though it cannot currently fire:
the read at `speaker_diarization.py:384` does a bare `cache["segmentation.threshold"]`
subscript, while the powerset write at `speaker_diarization.py:484-487` stores only
`{"embeddings": ...}` with no threshold key. A powerset write followed by a non-powerset read
would be a `KeyError`. Within a single file that cannot happen — one read, one write, same
model — so this constrains any future reuse of a `file` dict across pipelines, nothing more.

---

## 4. `OracleSegmentation.populate()`

`harness/segmentation.py` defines the source interface: each source has a stable `id` feeding
the runner's cache key and a `populate(pipeline, file)` hook.

`OracleSegmentation.populate()` (`harness/segmentation.py:112-148`) does three things.

**It derives the discretisation geometry from the pipeline itself**
(`harness/segmentation.py:116-120`), reading `pipeline._segmentation.step`, `.duration` and
`.model.receptive_field`. This mirrors the convention `SpeakerDiarization.apply()` uses when
it calls `oracle_segmentation` for oracle clustering, so the
`(num_chunks, num_frames, num_speakers)` shape contract is known-good rather than guessed.

**It refuses speaker-count-driven clustering** (`harness/segmentation.py:122-136`) — see §6.

**It builds the array on a shallow copy of `file`** (`harness/segmentation.py:144-148`):

```python
scaffold = dict(file)
scaffold["annotation"] = reference
file[pipeline.CACHED_SEGMENTATION] = oracle_segmentation(
    scaffold, window, frames
)
```

The upstream helper `oracle_segmentation()` needs a reference `Annotation` under the
`annotation` key in order to discretise it. It reads only `duration` and `annotation` and
never writes to the mapping it is given, so a shallow copy carries everything it needs. The
caller's `file` therefore **never holds an `annotation` key at any point** — there is no
window in which the key exists and no restore to depend on.

---

## 5. Why the original implementation was a no-op

The seam was implemented correctly and then closed one line too early.

The first version managed `training` inside `OracleSegmentation.populate()`, flipping it True
to write the array and restoring it in a `finally` **before returning**. `Runner.run()` then
called the pipeline afterwards, with `training` already back to False. The gate at
`speaker_diarization.py:337` took its `else` branch, the model recomputed its own
segmentation, and the injected array was never read.

The convention was right; its *scope* was set to `populate()` rather than to the pipeline call
it was meant to affect.

The failure was silent and nearly invisible. The run completed over all 16 meetings, wrote a
manifest, and produced a plausible number identical to baseline to eleven decimal places
across seven metrics. The only visible difference from baseline output was speaker *names* —
real AMI IDs instead of `SPEAKER_00..03` — and DER is label-invariant under optimal mapping,
so no metric moved.

That relabelling came from `apply()`'s own end-of-run cosmetic remapping
(`speaker_diarization.py:742-748`), triggered by an `annotation` key on `file`. Upstream's
comment states it "does not modify the actual output of the diarization pipeline", which is
why timings and cardinality were untouched. Under the current shallow-copy `populate()` the
key never reaches `apply()`, so output labels are `SPEAKER_00..` in both conditions and an
oracle/baseline RTTM comparison is a straight byte diff.

**The generalisable lesson**: identical geometry proves identical timings, not identical
files. An earlier version of the diagnosis argued from region-census geometry that the output
was byte-identical to baseline; that overreached, because labels do not affect overlap
geometry. The conclusion held but the evidence did not support it as stated. The actual proof
was a 16/16 identical-timings-and-cardinality / 0/16 identical-labels table.

---

## 6. The KMeans landmine

An `annotation` key on `file` has a second possible effect, far more serious than
relabelling (`speaker_diarization.py:600-607`):

```python
if self._expects_num_speakers and num_speakers is None:
    if isinstance(file, Mapping) and "annotation" in file:
        num_speakers = len(file["annotation"].labels())
    else:
        raise ValueError(
            f"num_speakers must be provided when using {self.klustering} clustering"
        )
```

`_expects_num_speakers` is taken from `self.clustering.expects_num_clusters`
(`speaker_diarization.py:295`).

**Today this branch does not fire.** Community-1 uses **`VBxClustering`**, which sets
`expects_num_clusters = False` (`clustering.py:552`).

> **Premise correction, 2026-09-25 (coordinator).** Earlier notes in this project —
> including [[oracle-segmentation-seam-no-op]] and the first draft of this doc — stated that
> community-1 uses `pyannote-default` = `AgglomerativeClustering` (`clustering.py:310`, also
> False). **That is wrong.** The shipped `config.yaml` sets `clustering: VBxClustering` with
> `threshold: 0.6`, `Fa: 0.07`, `Fb: 0.8`, and the live instantiated pipeline reports
> `klustering = VBxClustering`. The *conclusion* that the speaker-count branch is dormant is
> unaffected, since both classes set `expects_num_clusters = False` — but it holds for VBx's
> reason, not agglomerative's. Anything that reasons about the default condition's clustering
> behaviour (a hyperparameter sweep, a cache key over clustering config) must use VBx and its
> `threshold`/`Fa`/`Fb`, not agglomerative's parameter set.

**Two classes set it True**: `KMeansClustering` (`clustering.py:496`) and `OracleClustering`
(`clustering.py:675`). Under either of those, an `annotation` key on `file` would additionally
feed the *true speaker count* into the pipeline, and an oracle-segmentation run would silently
become oracle-segmentation-**plus-oracle-count** — an invalid condition that looks better than
it should and does not announce itself.

Two things defuse this. First, `populate()` keeps `annotation` off the caller's `file`
entirely (§4), so the key is not there to be read. Second, `populate()` fails loudly rather
than proceeding (`harness/segmentation.py:122-136`):

```python
if getattr(pipeline, "_expects_num_speakers", False):
    raise RuntimeError(...)
```

The `RuntimeError` is a **refusal of an unsupported combination, not leak prevention**.
Speaker-count-driven clustering needs a count and the harness supplies none: `run_harness.py`
builds `file` as `{"uri", "audio"}` and `Runner.run()` calls the pipeline with no
`num_speakers`, so `apply()` would fall through to its own `ValueError` from inside the
library. Failing at the harness level gives an error that names the actual problem.

**This is forward-relevant: clustering model selection work is starting now.** Anyone wiring
KMeans (or any `expects_num_clusters = True` clustering) into the harness will hit this
`RuntimeError`. That is the intended behaviour. Deciding where `k` comes from — oracle count,
fixed, or estimated — is a prerequisite for the clustering sweep, with a different experiment
behind each option, and is not something `OracleSegmentation` should decide implicitly. If the
oracle count is chosen, the condition must be recorded as
oracle-segmentation-plus-oracle-count.

---

## 7. The `oracle-v2` source id and the orphaned cache entries

`OracleSegmentation.id = "oracle-v2"` (`harness/segmentation.py:107`).

The id feeds the runner's cache key (`harness/runner.py:37-42`):

```python
raw = (
    f"{self._pipeline_config_id}|{self._segmentation_source.id}|"
    f"{self._refinement_id}|{uri}"
)
```

Fixing the flag scope does not change the cache key. Left as `"oracle"`, the 16 RTTMs written
by the broken run would have been served as cache hits (`harness/runner.py:51-52`) and the
same wrong numbers re-scored, with no pipeline invocation and no indication anything was
stale. Bumping the id produces a new key.

**The 16 stale pre-fix RTTMs remain in `.harness_cache` deliberately.** They are orphaned —
unreachable under any current cache key — rather than deleted, for three reasons: they are the
"before" artifact behind the 16/16 timings table that diagnosed the bug; `segmentation_source_id`
in each manifest now records which implementation produced each result, which is better
provenance than deletion; and no cache surgery was needed. `.harness_cache` was not cleared
wholesale, which would have paid for baseline recomputation too.

Do not delete them. Do not clear `.harness_cache` to "tidy up".

---

## 8. How to use it

Oracle segmentation is selected through the normal harness entry point. Nothing special is
required of the caller — the flag handling is internal to `Runner.run()`.

1. Build an `OracleSegmentation` with a `{uri: Annotation}` reference lookup, typically parsed
   from an RTTM via `pyannote.database.util.load_rttm`. On AMI IHM this is
   `harness-data/IHM/test.rttm`, the same file used as the scoring reference — which is
   correct and necessary for the error-budget prediction to hold, but means the result is an
   upper bound on what perfect segmentation *of this reference* buys.
2. Hand it to `Runner` as `segmentation_source`. The runner does the rest: flag on, populate,
   pipeline call, flag restored.
3. Results land under cache key component `oracle-v2`, and `segmentation_source_id` in the run
   manifest records it.

To write a new segmentation source, subclass `SegmentationSource`, give it a stable `id`, and
implement `populate()`. **Do not manage `pipeline.training` inside `populate()`** — the runner
owns it, and a source that flips it locally reintroduces the §5 bug.

Note the runner has no single-URI option: `AMIDatasetAdapter` always iterates every URI in the
split, so any run — including a spot check — pays the full 16-meeting cost.

The seam's guards are `tests/test_segmentation_seam.py`, three tests covering the gate's
behaviour outside `training`, the injection working under `training`, and the flag having no
side effect on normal output. All three pass (verified 2026-09-25, 1.76 s, no GPU).

---

## 9. Corrections to earlier docs

Recorded here rather than by editing the other notes.

**[[Segmentation Injection Seam]] overstates its test evidence.** Line 44 of that doc says the
seam tests are "(all passing)". They cannot have passed on this machine when that was written:
all three errored at *fixture setup* on the `pipeline`/`full_pipeline` fixtures with
`FileNotFoundError: Could not find file "trñ00"`, a Windows filename-encoding bug in a
dependency, resolved separately in [[Non-ASCII Filename Decoding Fix]]. (They collected
cleanly — the failure was at setup, not collection, which matters because it means the test
*bodies* were untested rather than merely unreached.) The same doc, line 47, names the second
test `test_cached_segmentation_is_used_via_chosen_fix`, while
`tests/test_segmentation_seam.py:42` defines `test_cached_segmentation_is_used_via_training_flag`.
Both claims verified against the current doc and test file on 2026-09-25. The verification
section appears to have been written from intent rather than from a run — which is the same
class of error as §5 itself.

**`harness/segmentation.py` docstrings are now correct.** Checked on 2026-09-25: the module
docstring (lines 1-15), `SegmentationSource.populate` (lines 38-42) and the
`OracleSegmentation` class docstring (lines 72-79) all describe the flag as runner-scoped and
state explicitly that `populate()` does not manage it. The `OracleSegmentation` docstring also
carries its own correction note (lines 100-105) retracting an earlier claim that `apply()`
would read `file["annotation"]` "before it's restored".

**One stale description in the closing ticket.** The superseded ticket
`Tickets/Open/oracle-segmentation-seam-no-op.md` specifies Fix 1 as a
`file["annotation"] = reference` write with a try/finally restore. The implementation that
shipped is strictly stronger: the shallow-copy scaffold of §4, under which the key never
touches the caller's `file` at all. Descriptions of the annotation leak as "restored in a
finally" describe the ticket's plan, not the code.
