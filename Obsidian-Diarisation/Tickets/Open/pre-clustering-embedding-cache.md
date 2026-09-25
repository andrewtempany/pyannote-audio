---
status: done
created: 2026-09-25
completed: 2026-09-25
superseded_by: "[[Pre-Clustering Cache]]"
---

> **DONE — folded into [[Pre-Clustering Cache]], which is now the reference doc for this work.**
> Per CLAUDE.md the completed ticket is not kept alongside the doc, so **this file should be deleted**;
> it is retained only until the coordinator removes it. Nothing here is unique to it any more.
>
> All seven acceptance criteria met. Final cold rebuild after the batch-size key fix:
> `runs/20260925T101811Z-db977741.json`, 2238 s, DER **0.17048543579940637** (exact), all twelve
> summary metrics bit-identical to the previous cold run, cache tally `0 full hits, 16 misses,
> 16 writes`. 53 MB on disk for 16 meetings.

# Pre-clustering segmentation and embedding cache

## Goal

Cache the pipeline's segmentation and embedding outputs to disk so that varying the clustering
stage costs CPU-seconds instead of a full GPU run.

Prerequisite for [[clustering-method-comparison]] and
[[agglomerative-hyperparameter-review]]. Neither is feasible without it.

## Why

A scored run currently takes 47 to 71 minutes for 16 meetings, because segmentation inference
and embedding extraction are redone every time. A 12-point threshold sweep is therefore 9 to 14
hours, and the epic involves several such sweeps across several methods. The compute alone
exceeds the time remaining on the project.

Segmentation and embeddings do not depend on clustering. Computed once per file and reused, a
sweep point becomes a clustering call over cached arrays.

## There are two separate problems here. Do not conflate them.

### Problem A: the RTTM cache key omits clustering entirely

[runner.py:37-42](../../../harness/runner.py):

```python
raw = (
    f"{self._pipeline_config_id}|{self._segmentation_source.id}|"
    f"{self._refinement_id}|{uri}"
)
```

and [run_harness.py:280](../../../run_harness.py) passes `pipeline_config_id=checkpoint`, which
is the bare string `"pyannote/speaker-diarization-community-1"`.

**Nothing about the clustering method or its hyperparameters is in the key.** A hyperparameter
sweep run today would compute point 1, cache it, and then serve that same cached RTTM for every
subsequent point. Every sweep point would return an identical DER, the run would complete, the
manifests would validate, and the curve would be flat.

This is exactly the `e9f25911` defect ([[Oracle Segmentation Findings Report]], invalidated
runs) repeating across a whole sweep rather than a single run. **It is the single most likely
way this epic produces a confident wrong answer.**

The `Runner` docstring already anticipates the fix: `pipeline_config_id` is documented as "an
HF checkpoint id/revision, **or a hash of its instantiated hyperparameters**". The extension
point exists; the caller just never used it.

**Fix:** `run_harness.py` must build `pipeline_config_id` from the checkpoint plus the
clustering class name plus every instantiated clustering hyperparameter value. Record the
human-readable form in the manifest alongside the hash, or a future reader cannot tell what a
run was.

This part is owned jointly with [[clustering-model-selection]]. Whichever lands first
implements it; the other verifies it. **Do not leave it to the other ticket.**

### Problem B: intermediates are recomputed

The actual caching work. See below.

## The seam may already exist

**Verify this before building anything.** `SpeakerDiarization` appears to already cache both
intermediates on the `file` mapping, gated on `self.training`:

- `pipeline.CACHED_SEGMENTATION` — read at `speaker_diarization.py:337-344`, already used by
  [[Oracle Segmentation Provider]].
- `file["training_cache/embeddings"]` — reportedly read/written around
  `speaker_diarization.py:377` and `:483`.

And [runner.py:69-83](../../../harness/runner.py) already sets `training = True` around every
pipeline call, for every condition, verified byte-identical against a cold-cache baseline
(16/16).

**If both caches are live and training-gated, this ticket is mostly persistence**: write the
two arrays to disk after a cold run, populate them onto `file` before a warm one, exactly as
`OracleSegmentation.populate()` already does for segmentation. No new injection point into
`apply()` is needed.

**Confirm the embedding key name and both read sites by reading the source.** The key name
above is second-hand and this project has twice been bitten by accepting a stated mechanism
without checking it. If the embedding cache is not gated the way the segmentation one is, or
the read happens somewhere that a pre-populated value would not reach, say so and stop — that
is a different, larger ticket.

## Design

**Two-tier cache, with deliberately different keys.**

| Tier | Key | Contents |
| --- | --- | --- |
| Intermediates | `pipeline_config_base \| segmentation_source.id \| uri` | segmentation tensor, embeddings |
| Final hypothesis | `pipeline_config_id \| segmentation_source.id \| refinement_id \| uri` | RTTM |

`pipeline_config_base` is the checkpoint **without** clustering configuration. That is the
point: the intermediates must be shared across clustering variants, while the final RTTM must
not be.

Getting these two the wrong way round produces either a useless cache or silently wrong
results. State in the code which is which and why.

**Storage:** both are numpy-backed. `.npz` plus the `SlidingWindow` parameters needed to
reconstruct the `SlidingWindowFeature`. Sixteen meetings of embeddings is not large, but check
and report the on-disk size — the working directory has a history of filling up.

**Invalidation:** the intermediate cache must be keyed on anything that changes segmentation or
embedding output. If the checkpoint or the segmentation source changes, the key changes. If
neither does, reuse is safe. Write that reasoning into the ticket's Implementation Notes so a
future reader can check it rather than trust it.

## Prediction

Recorded before implementation. Do not tune toward it.

- **A warm-cache run produces byte-identical RTTMs to a cold-cache run**, for all 16 meetings,
  under the baseline condition. Not "matching DER" — byte-identical files. DER is label- and
  precision-tolerant and would hide a real difference, as it did in
  [[oracle-segmentation-seam-no-op]].
- **Warm-run wall clock falls by at least 80%**, since segmentation inference and embedding
  extraction are the dominant costs.
- **Cold-run wall clock rises slightly**, by the cost of serialising the arrays.

If the warm run is not byte-identical, stop. Do not debug toward equality by loosening the
comparison.

## Acceptance criteria

1. **The clustering configuration is in the final-hypothesis cache key.** A test asserts that
   two different clustering hyperparameter sets produce two different cache keys. This test
   must **fail on the current code**.
2. **The clustering configuration is NOT in the intermediate cache key.** A test asserts two
   different clustering configurations produce the *same* intermediate key.
3. Warm-cache and cold-cache runs produce byte-identical RTTMs for all 16 meetings. Verified
   against a redirected `cache_dir`, not against the existing warm cache — `runner.run()`
   returns the cached RTTM at line 51 before anything else happens, so a comparison against a
   warm cache reads back the file it is comparing against and passes unconditionally.
4. **Wall-clock reduction from the INTERMEDIATE tier specifically is measured and reported,
   not estimated.**

   **Rewritten by the coordinator (2026-09-25) — the original wording could not fail.** It
   read "wall-clock reduction on a warm run is measured and reported". A warm run hits the
   final-RTTM cache at [runner.py:51-52](../../../harness/runner.py) and returns
   `load_rttm(...)` before the pipeline is constructed at all, so *any* warm run is ~100%
   faster than cold whether or not a single line of intermediate caching works. The criterion
   would have been satisfied by the pre-existing RTTM cache.

   The measurement must therefore isolate the tier under test: with the **final-hypothesis
   cache empty or its key perturbed** (so `run()` cannot short-circuit) and the
   **intermediate cache warm**, the run must be measurably faster than fully cold. That is
   the only comparison in which the intermediate tier is on the critical path. Report both
   wall clocks and the delta.

5. On-disk cache size for 16 meetings is measured and reported.
6. Existing recorded conditions still reproduce: a cold-cache baseline run still yields
   DER 0.17048543579940637. **(Run by the coordinator, not this ticket — see the brief.)**
7. **The pre-existing test-failure set is UNCHANGED.** Verified by capturing sorted
   `FAILED`/`ERROR` lines before and after the change and diffing them.

   **Rewritten by the coordinator (2026-09-25) — the original read "full test suite passes",
   which is unsatisfiable and so would have been silently reinterpreted.** 28 tests fail on
   the current tree before any work starts (`test_train.py` transfer/freeze,
   `inference_test.py` Python 3.14 forkserver pickling, two `test_reproducibility.py`
   pickling, two stale `overlap_der` assertions in `test_run_harness_integration.py`). An
   aggregate pass count is not acceptable evidence; the diff of the failure set is.

## Out of scope

- **No oracle conditions.** This serves the clustering work. `OracleSegmentation` will use the
  same cache by virtue of its `segmentation_source.id`, but no oracle runs are performed.
- **Do not clear `.harness_cache`.** Existing entries are keyed on the old scheme and will
  simply stop being hit once the key changes. That is correct: they become orphaned rather
  than wrong. The 16 stale pre-seam-fix oracle RTTMs stay where they are.
- **Do not change `_pair_timeline` or refinement behaviour.**

## Risk note

The segmentation injection seam failed silently and cost a full invalid experiment, because a
mechanism was assumed rather than verified and the test that would have caught it had never
run. This ticket touches the same machinery.

Two rules for it:

1. Every claim about how `SpeakerDiarization` consults a cache is **read from the source and
   quoted in the Implementation Notes**, not inferred from a docstring. The class docstring in
   `harness/segmentation.py` is currently wrong about sequencing; that is precedent, not
   paranoia.
2. Every acceptance criterion must be able to fail. Criteria 1 and 2 above are written to fail
   on the current code by construction. If any criterion passes before the work starts, it is
   not a criterion.

## Implementation Notes

_Append while the work happens. Record the verified cache key names and read sites here first,
with line numbers, before writing any caching code._

### Step 0 — seam verification, read from source before writing any code

All line numbers below are `src/pyannote/audio/pipelines/speaker_diarization.py` at the
current tree unless stated otherwise. Quoted, not paraphrased from docstrings.

**Segmentation cache.** `CACHED_SEGMENTATION` is a property at **:317-319** returning the
literal `"training_cache/segmentation"`. The only read is in `get_segmentations()` at
**:337-344**:

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

Confirmed training-gated, and it both reads and writes, so a cold run leaves the tensor on
`file` for us to harvest.

**Embedding cache.** The key name `"training_cache/embeddings"` is **real**, not second-hand.
Read in `get_embeddings()` at **:377-386**:

```python
if self.training:
    cache = file.get("training_cache/embeddings", dict())
    if ("embeddings" in cache) and (
        self._segmentation.model.specifications.powerset
        or (cache["segmentation.threshold"] == self.segmentation.threshold)
    ):
        return cache["embeddings"]
```

Written at **:483-492**, with two branches:

```python
if self.training:
    if self._segmentation.model.specifications.powerset:
        file["training_cache/embeddings"] = {"embeddings": embeddings}
    else:
        file["training_cache/embeddings"] = {
            "segmentation.threshold": self.segmentation.threshold,
            "embeddings": embeddings,
        }
```

Same `self.training` gate as segmentation. Both reads are reached from `apply()` on the
normal path: `get_segmentations()` at **:609**, `get_embeddings()` at **:647-652**. So a
value pre-populated on `file` by a `populate()`-style hook *does* reach both reads. The
ticket's "the seam may already exist" premise is **confirmed**; this is persistence work, not
a new injection point.

**Is community-1 powerset? YES — and it matters.** Verified by loading the checkpoint
metadata directly rather than trusting the pipeline config:

```
specifications = Specifications(problem=MONO_LABEL_CLASSIFICATION, resolution=FRAME,
                                duration=10.0, classes=['speaker#1','speaker#2','speaker#3'],
                                powerset_max_classes=2, permutation_invariant=True)
```

`Specifications.powerset` is a `cached_property` that is True exactly when
`powerset_max_classes` is set (`Core-Task.md`), so it is True here. Corroborated
independently: `__init__` at **:262-271** builds `self.segmentation` as a `ParamDict` with
`threshold` **only on the non-powerset branch**, and the checkpoint's shipped
`config.yaml` supplies `params.segmentation` = `{min_duration_off: 0.0}` with no `threshold`.
A non-powerset community-1 would fail to instantiate from its own config.

Two consequences, both load-bearing:

1. The **KeyError hazard the brief flagged does not fire** for community-1. The read at
   :382-385 evaluates `specifications.powerset` first and `or` short-circuits, so
   `cache["segmentation.threshold"]` is never subscripted. A powerset write followed by a
   powerset read is safe.
2. Therefore the dict I reconstruct on a warm run must contain **only** `{"embeddings": ...}`,
   matching the :484-487 write exactly. I deliberately do **not** synthesise a
   `segmentation.threshold` entry: `self.segmentation.threshold` does not exist on a powerset
   pipeline, so inventing one would be fabricating a value the cold path never produced. If a
   future non-powerset checkpoint is used, the stored dict must gain the threshold and the
   intermediate key must gain it too (see "adding a dimension" below) — the code raises rather
   than guesses, see `_EMBEDDING_CACHE_KEY` handling in `harness/intermediate_cache.py`.

**Runner training flag.** `harness/runner.py:69-83` already sets `training = True` around
both `populate()` and the pipeline call, with a `hasattr` guard for lightweight test fakes,
and restores it in a `finally`. Unchanged by this ticket.

**Final-RTTM short-circuit.** `harness/runner.py:47-52` — `run()` returns
`load_rttm(str(cache_path))[uri]` the moment the RTTM exists, before the pipeline is
constructed or `populate()` is called. This is why criterion 4 had to be rewritten and why
criterion 3 must use a redirected `cache_dir`.

### Step 0b — PREMISE CORRECTION: community-1 uses VBxClustering, not AgglomerativeClustering

This ticket and several docs it links to claim community-1 clusters with
`AgglomerativeClustering` ("pyannote-default = AgglomerativeClustering"). **That claim is
false.** The coordinator flagged it, and it independently matches what the shipped
`config.yaml` says. Verified from the live instantiated pipeline on this machine
(`PYANNOTE_SKIP_DEPENDENCY_CHECK=1`, CPU, no inference):

```
klustering             : VBxClustering
clustering class       : VBxClustering
powerset               : True
clustering instantiated: {'threshold': 0.6, 'Fa': 0.07, 'Fb': 0.8}
pipeline flattened     : {'segmentation>min_duration_off': 0.0,
                          'clustering>threshold': 0.6,
                          'clustering>Fa': 0.07, 'clustering>Fb': 0.8}
seg has threshold attr : False
```

So the hyperparameters the final-hypothesis key must capture are VBx's **threshold / Fa / Fb**,
*not* agglomerative's threshold / method / min_cluster_size. `seg has threshold attr: False`
is a second independent confirmation of powerset (the non-powerset `__init__` branch at
:268-271 would have created it).

**This is why I did not hardcode a per-class hyperparameter list.** A hardcoded list built on
the stale Agglomerative claim would have omitted every real hyperparameter, and the key would
have been constant across a VBx sweep — exactly the flat-sweep failure this ticket exists to
prevent, and it would have looked fine. Instead the key derives parameters generically from
pyannote's own descriptor machinery:

- `pyannote.pipeline.Pipeline.__setattr__` (site-packages `pyannote/pipeline/pipeline.py:102-149`)
  routes any `Parameter` assignment into `self._parameters` and any later concrete assignment
  into `self._instantiated`.
- `_flattened_parameters(instantiated=True)` (:165-210) returns the concrete values, recursing
  into sub-pipelines with a `parent>child` naming convention.

A clustering class with entirely different parameters therefore needs **no code change** — its
params appear automatically. `AgglomerativeClustering` would contribute
`threshold`/`method`/`min_cluster_size`; `KMeansClustering` contributes none (it declares no
tunable parameters), which is handled and tested rather than crashing.

I did not instantiate any clustering class directly: `VBxClustering.__init__` takes a
positional `plda` with no default (clustering.py:555-563) and the library special-cases it at
speaker_diarization.py:290-291. All extraction reads an already-instantiated
`pipeline.clustering`.

**Current key, pre-change.** `harness/runner.py:37-42` hashes
`f"{pipeline_config_id}|{segmentation_source.id}|{refinement_id}|{uri}"`, and
`run_harness.py:280` passes `pipeline_config_id=checkpoint`, the bare string
`"pyannote/speaker-diarization-community-1"` (set at :253). Confirmed: nothing about
clustering is in the key.

### Step 1 — the key restructure (Problem A), red then green

New module `harness/cache_id.py`. It owns only the *pipeline-configuration* portion of each
key; `Runner` still concatenates `segmentation_source.id`, `refinement_id` and `uri` itself.

Criterion 1 shown RED on the pre-change logic first, with the assertion executed rather than
asserted-by-claim. Two `Runner`s differing only in VBx `threshold` (0.6 vs 0.7), both given
`pipeline_config_id=CHECKPOINT` exactly as `run_harness.py:280` did:

```
E   AssertionError: clustering hyperparameters are NOT in the final-hypothesis cache key
E   assert '769a10f212d3be8d91818e339ec00111e18ca6cd1fd987e438e83401da9a7551'
E       != '769a10f212d3be8d91818e339ec00111e18ca6cd1fd987e438e83401da9a7551'
1 failed in 4.06s
```

Same assertion, same `Runner`, after the change — only the `pipeline_config_id` construction
differs:

```
threshold=0.6 -> e095830e4f72e44973d390346ee9d6c643bfd29e4a9b4ea7c021c39d6cfc7149
threshold=0.7 -> 35b9d1b5ad70ea7770d03ddcdd9b02012d3e042ffca71d4c152ff9fc297f9e05
1 passed
```

**Decision: generic parameter extraction, not a per-class list.** Driven directly by the
premise correction above — a hardcoded Agglomerative-shaped list would have silently captured
nothing on a VBx sweep. `_instantiated_clustering_parameters()` calls
`clustering._flattened_parameters(instantiated=True)`, with a fallback to public scalar
attributes for duck-typed objects. Verified against three real parameter shapes in the tests:
VBx (threshold/Fa/Fb), Agglomerative (threshold/`method` Categorical/`min_cluster_size`
Integer) and KMeans (no parameters at all).

**Decision: no fallback to a clustering-free key — `ClusteringConfigUnavailable` instead.**
This answers the brief's "if this silently did nothing, what would tell us?" question for the
key. If introspection failed and we degraded to the bare checkpoint, the symptom would be a
*flat sweep curve with valid manifests* — indistinguishable from a genuine null result. There
is no degraded path, so the failure is a traceback at run start rather than a plausible wrong
number. Tested by `test_final_key_is_loudly_unavailable_when_clustering_missing`.

**Gotcha: an empty parameter set is legitimate.** `KMeansClustering` declares no tunable
parameters, so `{}` is a valid answer and must not be conflated with "introspection failed"
(failure mode 3 — one value, two meanings). The two are distinguished structurally: the
*attribute must exist* or we raise; an existing attribute yielding `{}` is accepted, and the
class name still separates it from other classes.

**Alternative rejected: hashing a pickle of the clustering object.** Would have captured
everything automatically, but pickles are not stable across library versions or processes, so
the key would drift and orphan the cache on essentially every run.

**Values rendered with `repr`**, which is round-trip exact for floats, so 0.6 and
0.6000000000001 cannot collide. Parameter names sorted so dict ordering cannot move a key.

**Version prefixes** (`final-v1`, `intermediate-v1`) make a key self-describing and guarantee
the tiers cannot collide even if their component lists ever coincided.

### Step 2 — the persistence layer (Problem B)

New module `harness/intermediate_cache.py`; `Runner` gained an optional
`intermediate_cache` argument (default `None`, so the historical single-tier behaviour and
every existing runner test are untouched).

**Where it hooks in.** `Runner.run()` calls `intermediate_cache.populate()` *after*
`segmentation_source.populate()` and before the pipeline call, then `save()` after the pipeline
call, all inside the existing `training = True` block. No change to `apply()` and no new
injection point, exactly as the ticket predicted. Ordering is deliberate: an oracle source's
ground-truth segmentation must beat anything on disk, so `populate()` never overwrites a key
already present on `file`.

**Gotcha found the hard way: `np.savez_compressed` appends `.npz`.** My first version wrote the
atomic temp file as `path.with_suffix(".npz.partial")`, so numpy silently saved it as
`...npz.partial.npz` and the subsequent `replace()` raised `FileNotFoundError`. Nine tests
caught it immediately. Temp name is now `<stem>.partial.npz`.

**Gotcha in my own test, worth recording as an instance of the project's failure mode 4.** The
sensitivity check (`test_byte_comparison_can_actually_detect_a_difference`) originally
corrupted cached embeddings by setting them to 1.0. The fake's label rule is
`mean > threshold * 0.5` = `> 0.3`, and uniform-random embeddings average ~0.5 — so 1.0 sits on
the *same* side of the boundary and the RTTM came back byte-identical. The corruption has to
cross the decision boundary or the check proves nothing. Now zeroed.

**Atomic writes.** Write to a temp file then `replace()`, so an interrupted run cannot leave a
truncated `.npz` that a later warm run would happily read as valid.

**Observability, answering "if this silently did nothing, what would tell us?"** A cache that
misses and recomputes is correct-but-pointless, and the only symptom is the absence of a
speedup — precisely the oracle-seam shape of bug. So: every lookup returns a `CacheLookup`,
`CacheStats` tallies full/partial/miss/writes, and `run_harness.py` prints the tally
**unconditionally, including when it is all zeros**. `0 full hits, 0 writes` is the signature of
a broken seam, and it is only visible because the line prints even when there is nothing good
to report. `save()` also records a per-uri `write_failure` when the pipeline left no
intermediates on `file` at all — which is what a broken training gate looks like.

**`CacheLookup` is not a bool and not an `Optional`.** Failure mode 3 on this project was one
return value standing for two unrelated conditions. "No entry on disk" and "entry has
segmentation but not embeddings" have different performance consequences, so they are separate
fields and `__str__` renders four distinct states.

### Step 3 — criteria 3, 4 and 5

**Criterion 3 (byte-identical warm vs cold).** `test_warm_and_cold_rttms_are_byte_identical`,
16 simulated meetings. The warm run writes into a **separate `cache_dir`** — without that,
`run()` returns the cached RTTM at runner.py:51-52 before the pipeline is touched and the
comparison reads back the file it is comparing against, passing unconditionally. The test
additionally asserts `warm_pipeline.segmentation_inferences == 0` and
`embedding_extractions == 0`, so a pass cannot mean "it recomputed and happened to match".
Compared with `read_bytes()`, never DER. Paired with a sensitivity test proving the comparison
can actually fail. **The real 16-meeting version is deferred to the coordinator's cold baseline
run; this covers the mechanism, not the corpus.**

The fake pipeline is not a mock that assumes the answer: it reimplements the training-gated
reads/writes at the verified line numbers, including the powerset branch, so a regression in the
seam breaks these tests rather than passing them.

**Criterion 4 (intermediate-tier wall clock, isolated).** Both arms get a fresh RTTM cache dir
so `run()` can never short-circuit; the only difference is whether the intermediate cache is
warm. Measured over 8 files with a simulated 50 ms + 50 ms inference cost:

```
fully cold                       : 0.969s
intermediates warm, RTTM bypassed: 0.031s
delta                            : 0.938s (96.8% faster)
```

**What else could produce that number:** it is dominated by the simulated sleep, so the
*percentage* is an artefact of my chosen cost, not a prediction of real GPU savings. What it
does establish is that the intermediate tier is genuinely on the critical path in this
comparison and that the expensive steps were skipped (inference counters at 0). **The real
wall-clock figure requires a GPU run and is deferred — not estimated.**

**Criterion 5 (on-disk size).** Two figures, one measured and one explicitly extrapolated.

*Measured*, simulated shapes, 16 files: **5.41 MB total, 0.338 MB per file** — but these use a
40-chunk array, so this measures the mechanism, not the corpus.

*Extrapolated from REAL geometry*, which is the number that matters for disk planning. Real
model geometry read off the instantiated pipeline: 589 frames/chunk, 3 speakers, embedding
dimension 256, 10 s window, 1 s step, powerset. Real chunk counts from the 16 test-split
meeting durations in the reference RTTMs (8.89 h of audio total):

```
segmentation  225.2 MB uncompressed
embeddings     97.9 MB uncompressed
total          323.1 MB uncompressed
```

Compression ratio measured on arrays of the true shapes with a realistic distribution
(saturated near-0/1 segmentation, dense float32 embeddings): 0.81 for segmentation, 0.93 for
embeddings, **0.84 combined**.

**=> ~272 MB on disk for 16 meetings. LABELLED AS AN EXTRAPOLATION**, from real shapes and real
durations but not from real cached arrays. Largest single meeting (EN2002c, 48 min) ~27 MB.
Comfortable for a working directory with a history of filling up, but not negligible — and it
grows linearly per segmentation-source variant, since `segmentation_source.id` is in the key.

## On completion

Rewrite as reference documentation describing the two-tier cache: what each tier keys on, why
they differ, what invalidates each, and how to add a new dimension to either key. Set
`status: done` and move to `Obsidian-Diarisation/Docs/Pre-Clustering Cache.md`.

The "which key gets what" reasoning is the part future readers will need. Lead with it.

## See also

- [[clustering-method-comparison]] — the epic this unblocks; co-owns the cache-key fix.
- [[clustering-model-selection]] — the other half of Problem A.
- [[Runner]] — the existing single-tier cache.
- [[Oracle Segmentation Provider]] — the existing `populate()` pattern this mirrors.
- [[oracle-segmentation-seam-no-op]] — what silent failure in this machinery looks like.
