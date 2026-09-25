---
status: done
created: 2026-09-25
---

# Pre-Clustering Cache

A two-tier disk cache that lets the clustering stage be varied without repeating
segmentation inference and embedding extraction. Varying clustering costs CPU-seconds instead
of a full GPU run, which is what makes [[clustering-method-comparison]] and
[[agglomerative-hyperparameter-review]] feasible at all.

## Which key gets what, and why

This is the part to read before touching anything here. There are two keys, and they differ in
exactly one respect: whether clustering configuration is part of them.

| Tier | Key | Contents | Clustering in key? |
| --- | --- | --- | --- |
| Intermediates | `intermediate_config_id \| segmentation_source.id \| uri` | segmentation tensor, embeddings | **No** |
| Final hypothesis | `pipeline_config_id \| segmentation_source.id \| refinement_id \| uri` | RTTM | **Yes** |

`intermediate_config_id` is the checkpoint alone. `pipeline_config_id` is the checkpoint plus
the clustering class name plus every instantiated clustering hyperparameter.

**Why clustering is excluded from the intermediate key.** In
`SpeakerDiarization.apply()`, segmentation runs at `speaker_diarization.py:609` and embeddings
at `:647`, while clustering is not consulted until `:656`. The intermediates are produced
strictly before clustering and cannot depend on it. That is the entire premise of the tier: one
cached copy per (checkpoint, segmentation source, file) serves *every* point of a clustering
sweep. If clustering configuration leaked into this key, every sweep point would miss and pay
for GPU inference again — the cache would be useless, though still correct.

**Why clustering is required in the final key.** Clustering is what produces the speaker
labels, so the RTTM depends on it. If clustering is missing from this key, a sweep computes
point 1, caches it, and then serves that same RTTM for every subsequent point. Every point
reports an identical DER, the run completes, the manifest validates, and the curve is flat.

Note the asymmetry in consequence: an over-specified intermediate key is merely slow, while an
under-specified final key is **silently wrong**, and much harder to notice. That is why the
final key errs toward including things and the intermediate key errs toward excluding them.
Getting the two the wrong way round yields one failure or the other, so the code states which
is which at both definitions (`harness/cache_id.py`, and `Runner.cache_key`'s docstring).

This was not a hypothetical. Before this work, `run_harness.py` passed
`pipeline_config_id=checkpoint` — the bare string
`"pyannote/speaker-diarization-community-1"` — so nothing about clustering was in the key at
all. Two `Runner`s differing only in VBx `threshold` (0.6 vs 0.7) produced the identical hash
`769a10f212d3be8d91818e339ec00111e18ca6cd1fd987e438e83401da9a7551`. It is the same defect as
`e9f25911` ([[Oracle Segmentation Findings Report]]) but spread across a whole sweep rather
than a single run.

## How to add a new dimension to either key

Ask one question: **does this thing change segmentation or embedding output?**

- **Yes** → it belongs in `intermediate_config_id`, and therefore in both keys (the final key is
  built on the same base). Add it to `_intermediate_components()` in `harness/cache_id.py`.
  Adding a dimension here orphans the existing intermediate entries, which is the correct
  outcome: they become unreachable, not wrong.
- **No, but it changes the final RTTM** → it belongs in `pipeline_config_id` only. Add it to the
  clustering/final-only components.

Clustering hyperparameters need **no code change**. They are enumerated generically from
pyannote's own descriptor machinery, so a clustering class with an entirely different parameter
set is picked up automatically (see below).

`segmentation_source.id` and `refinement_id` are not handled in `cache_id.py` — `Runner` and
`IntermediateCache` concatenate those themselves. `cache_id.py` owns only the
pipeline-configuration portion.

## What invalidates each tier

**Intermediate tier.** Written so a reader can check the reasoning rather than trust it:

- `checkpoint` selects **both** the segmentation model and the embedding model
  (`speaker_diarization.py:204-217` — `segmentation`, `embedding` and `plda` all default to
  subfolders of the same checkpoint). Change it and both cached arrays are wrong. In the key.
- `segmentation_source.id` changes the segmentation tensor, and changes the embeddings
  downstream of it, since embeddings are extracted from the binarized segmentation
  (`:647-652`). In the key, contributed by the caller.
- `uri` identifies the audio. In the key.
- Clustering configuration is deliberately absent, per the reasoning above.

Not currently in the key, and safe **only because the harness holds them fixed**:
`segmentation_step` (0.1), `embedding_exclude_overlap` (True), and — on a non-powerset
checkpoint — `segmentation.threshold`, which would change the binarization the embeddings are
extracted from. If the harness ever varies any of these, they must be added. A reader auditing
this list should confirm those values are still fixed in `run_harness.py`.

**The inference batch sizes ARE in the key** (`segmentation_batch_size`,
`embedding_batch_size`, both 32 from the shipped `config.yaml`). They were originally left out
on the reasoning that batching "affects speed, not values". That reasoning was wrong:
`get_embeddings()` stacks waveforms into one tensor and runs the embedding model on the batch
(`speaker_diarization.py:460-468`), and segmentation is batched the same way (`:259`), so batch
shape changes float reduction order and therefore the arrays themselves.

> **The batch-size collision is a PROVEN HAZARD, not an observed effect.** Established by
> computing both key hashes at batch 32 and batch 8 and finding them **identical** — so two
> runs at different batch sizes would have shared one cache entry while producing different
> embeddings, the second being served arrays it never computed.
>
> **It did not fire in any recorded run.** It is tempting to cite run
> `20260925T080105Z-7ddbf6ae` (batch 8, DER 0.17049342382952828) as a measurement of batch-size
> drift. **It is not one.** All 16 RTTM entries and all 16 intermediates for the baseline
> condition carry timestamps between 05:00:30 and 07:58:46 UTC — the window of the cold run
> `a1be3514`, which ended 07:58:47. Nothing in either tier is stamped at or after 08:01:05,
> when `7ddbf6ae` started. That run therefore wrote nothing, read all 16 cached RTTMs, and
> **never invoked the embedding model at all**. Batch size could not have affected any number
> in it.
>
> The entire 8e-6 difference in `7ddbf6ae` is RTTM quantisation — see the tier distinction
> below.

## The two tiers differ in exactness, and this is the key fact about them

**The intermediate cache is EXACT.** Arrays round-trip byte-identically: verified with
`np.array_equal` on the real cached entries, dtype preserved (`float32` in, `float32` out —
that is what the model emits, so there is no `float64` to lose), and `.npz` compression is
lossless. A warm intermediate run and a cold run produce the same hypotheses.

**The final-RTTM cache is LOSSY BY DESIGN, and needs no fix.** `Annotation._iter_rttm` formats
every boundary with `:.3f`, so an RTTM stores milliseconds. Anything scored from a *cached
RTTM* is therefore millisecond-truncated, while a fresh run scores full-precision in-memory
`Annotation` objects.

Consequences, which explain several otherwise-puzzling numbers in this project:

- A cold run and a warm (RTTM-cached) run of **one identical configuration** always differ by
  roughly **8e-6 DER**. That is the format, not nondeterminism and not a caching defect.
- **The figure to cite against the published benchmark is the COLD one**, because it is scored
  at full precision: `DER 0.17048543579940637`.
- Evidence, if needed: on the 16 baseline meetings the warm run's per-file `missed_detection`
  is an exact multiple of 0.001 in **16/16** files, the cold run's in **0/16**.

**Final tier.** The checkpoint, the clustering class, every instantiated clustering
hyperparameter, the segmentation source, the refinement id, and the uri.

## The seam it is built on

The pipeline already cached both intermediates on the `file` mapping, gated on
`self.training`, so this feature is persistence only — no new injection point into `apply()`.
All line numbers are `src/pyannote/audio/pipelines/speaker_diarization.py`, read from source
rather than from docstrings (the `harness/segmentation.py` docstring was once wrong about
sequencing and cost a round of misdirected planning):

- `CACHED_SEGMENTATION` is a property at **:317-319** returning the literal
  `"training_cache/segmentation"`; read and written in `get_segmentations()` at **:337-344**
  under `if self.training:`.
- `"training_cache/embeddings"` is read in `get_embeddings()` at **:377-386** under the same
  gate, and written at **:483-492**.
- Both reads are on the normal `apply()` path (**:609**, **:647-652**), so a value
  pre-populated on `file` does reach them.
- `harness/runner.py:69-83` already sets `pipeline.training = True` around both `populate()`
  and the pipeline call, for every condition, restoring it in a `finally`.

Because the pipeline both reads *and* writes these keys, a cold run leaves the arrays on `file`
for the cache to harvest afterwards. Nothing has to be intercepted mid-`apply()`.

## The powerset coupling

The embedding cache value is a dict whose **shape depends on whether the segmentation model is
powerset** (`:483-492`):

```
powerset      -> {"embeddings": ...}                          (no threshold)
non-powerset  -> {"segmentation.threshold": ..., "embeddings": ...}
```

while the read at `:382-385` is:

```python
if ("embeddings" in cache) and (
    self._segmentation.model.specifications.powerset
    or (cache["segmentation.threshold"] == self.segmentation.threshold)
):
```

A powerset *write* followed by a non-powerset *read* would raise `KeyError` on the subscript.

**community-1 is powerset**, so that hazard is unreachable here — `or` short-circuits on
`specifications.powerset` and the subscript is never evaluated. Verified two independent ways:
the checkpoint metadata carries `powerset_max_classes=2` (and `Specifications.powerset` is a
`cached_property` true exactly when that is set), and `pipeline.segmentation` has no
`threshold` key at all, which only happens on the powerset branch of `__init__` at `:262-271`.

`IntermediateCache` therefore reconstructs the powerset shape only, and deliberately does **not**
synthesise a `segmentation.threshold`: on a powerset pipeline that value does not exist, and
inventing one would populate a cache state the cold path never produces.

On a non-powerset checkpoint two things must change together, and `save()` raises
`NonPowersetCheckpointUnsupported` rather than guessing: the stored dict needs the threshold,
**and** `segmentation.threshold` must join the intermediate key, because embeddings cached at
one threshold are simply wrong at another.

## Storage

`.npz` (numpy-native, compressed) under `<cache_dir>/intermediates/`, one file per cache key,
holding the segmentation array, the embeddings array, and the `SlidingWindow` parameters
(duration / step / start / end, plus an explicit unbounded flag since `end` is `inf` for an
unbounded window) needed to rebuild the `SlidingWindowFeature`.

Writes are atomic: written to a temp file then `replace()`, so an interrupted run cannot leave a
truncated `.npz` that a later warm run would read as valid. A `_FORMAT_VERSION` constant exists
to orphan every entry deliberately if the layout changes.

**Gotcha:** `np.savez_compressed` appends `.npz` unless the filename already ends in it, so the
temp name must be `<stem>.partial.npz`, not `<name>.npz.partial`. Getting this wrong makes numpy
write the file somewhere other than where the subsequent rename looks for it.

### On-disk size

Real model geometry, read off the instantiated pipeline: 589 frames per chunk, 3 speakers,
embedding dimension 256, 10 s window, 1 s step. Real chunk counts from the 16 test-split
meeting durations (8.89 h of audio):

```
segmentation  225.2 MB uncompressed
embeddings     97.9 MB uncompressed
total         323.1 MB uncompressed
```

Measured compression ratio on arrays of the true shapes: 0.81 segmentation, 0.93 embeddings,
0.84 combined.

**≈272 MB on disk for 16 meetings** — an *extrapolation* from real shapes and real durations,
not a measurement of real cached arrays (that requires a GPU run). Largest single meeting
(EN2002c, 48 min) ≈27 MB. It grows linearly per segmentation-source variant, since
`segmentation_source.id` is part of the key.

## Generic clustering-hyperparameter extraction

The key does not hardcode a per-class hyperparameter list. It reads
`clustering._flattened_parameters(instantiated=True)`, pyannote's own machinery:
`Pipeline.__setattr__` routes a `Parameter` assignment into `self._parameters` and a later
concrete assignment into `self._instantiated`, and `_flattened_parameters` returns the concrete
values, recursing into sub-pipelines with a `parent>child` naming convention. A fallback reads
public scalar attributes for duck-typed objects.

This is a correctness measure, not a style choice. This ticket and several docs asserted that
community-1 clusters with `AgglomerativeClustering`; **it actually uses `VBxClustering`**
(threshold / Fa / Fb, defaults 0.6 / 0.07 / 0.8 — verified from the shipped `config.yaml` and
from the live instantiated pipeline). A hardcoded Agglomerative-shaped list would have captured
*no* real hyperparameter, producing a constant key across a VBx sweep — the exact flat-curve
failure this cache exists to prevent, while looking perfectly healthy.

Verified against three real parameter shapes: VBx (three floats), Agglomerative
(`threshold` float, `method` Categorical, `min_cluster_size` Integer) and KMeans (no tunable
parameters at all).

An **empty** parameter set is a legitimate answer and is not conflated with "introspection
failed": the attribute must exist or the code raises, while an existing attribute yielding `{}`
is accepted and the class name still separates it from other classes.

Values are rendered with `repr` (round-trip exact for floats, so 0.6 and 0.6000000000001 cannot
collide) and parameter names are sorted, so dict ordering cannot move a key. Hashing a pickle of
the clustering object was rejected: pickles are not stable across library versions or processes,
so the key would drift and orphan the cache on essentially every run.

Nothing here instantiates a clustering class directly. `VBxClustering.__init__` takes a
positional `plda` with no default (`clustering.py:555-563`) and the library special-cases it at
`speaker_diarization.py:290-291`; the extractor only ever reads an already-instantiated
`pipeline.clustering`.

## Loud failure, by design

Two silent-failure paths are closed deliberately, because a no-op here produces plausible
numbers rather than a crash — the shape of bug that already cost this project a full invalid
experiment ([[oracle-segmentation-seam-no-op]]).

**The key never degrades.** If the clustering configuration cannot be read,
`ClusteringConfigUnavailable` is raised. There is no fallback to a clustering-free key, because
the symptom of that fallback would be a flat sweep curve with valid manifests —
indistinguishable from a genuine null result.

**A cache miss is observable.** A cache that silently misses and recomputes is correct but
pointless, and its only symptom is the absence of a speedup. So every lookup returns a
`CacheLookup`, `CacheStats` tallies full / partial / miss / writes, and `run_harness.py` prints
the tally **unconditionally, including when it is all zeros**. `0 full hits, 0 writes` is the
signature of a broken seam, and it is only visible because the line prints even when there is
nothing good to report. `save()` additionally records a per-uri write failure when the pipeline
left no intermediates on `file` — which is what a broken training gate looks like.

`CacheLookup` is not a bool and not an `Optional`, because "no entry on disk" and "entry has
segmentation but not embeddings" are different states with different performance consequences.
Conflating two conditions into one return value is how 96.3% of a headline figure once turned
out to be empty tensor slots.

## Usage

The cache is on by default. `--no-intermediate-cache` disables it entirely (neither read nor
written) for a genuinely cold reference run; it does not affect the final-hypothesis RTTM cache,
which is bypassed instead by pointing `--cache-dir` at an empty directory.

`run_harness.py` prints the clustering configuration, both key ids, and the cache tally on every
run. The manifest additionally records two **additive** fields, `clustering_config` (human
readable, e.g. `VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6)`) and `intermediate_config_id`.
They are deliberately **not** in `REQUIRED_RUN_CONFIG_FIELDS`, because the 12 manifests written
before this change lack both keys and must stay readable — the same precedent as `oracle_rttm`.
`clustering_model` previously recorded the literal `"pyannote-default"`, which said nothing; it
now carries the real instantiated configuration. Without the human-readable form the hash is
write-only: it distinguishes two runs but never says what they were.

## Existing cache entries

Changing `pipeline_config_id` orphans all ~128 pre-existing `.harness_cache` entries. This is
correct and expected: they were keyed on the old scheme and simply stop being hit. They become
**unreachable, not wrong**, and are deliberately left in place — including the 16 stale
pre-seam-fix oracle RTTMs.

## Files

- `harness/cache_id.py` — both keys, the generic hyperparameter extraction, the human-readable
  description. Leads with the which-key-gets-what reasoning.
- `harness/intermediate_cache.py` — `IntermediateCache` (`populate` / `save` / `key` / `path` /
  `disk_usage_bytes`), `CacheLookup`, `CacheStats`, `NonPowersetCheckpointUnsupported`.
- `harness/runner.py` — optional `intermediate_cache` argument (default `None`, preserving the
  historical single-tier behaviour); `populate()` after the segmentation source's, `save()` after
  the pipeline call, both inside the existing `training = True` block. Ordering matters: an
  oracle source's ground-truth segmentation must beat anything on disk, so `populate()` never
  overwrites a key already on `file`.
- `run_harness.py` — builds both ids from the live pipeline, wires the cache, prints the tally,
  records the manifest fields, adds `--no-intermediate-cache`.
- `tests/test_pipeline_config_id.py` (13 tests) — both key directions.
- `tests/test_intermediate_cache.py` (15 tests) — persistence, the seam, byte-identical warm vs
  cold, tier isolation.

The test fake in `test_intermediate_cache.py` is not a mock that assumes the answer: it
reimplements the training-gated reads and writes at the verified line numbers, including the
powerset branch, so a regression in the seam (wrong key name, wrong gate, populated value not
reaching the read) makes those tests fail rather than pass.

## Verification status

- **Clustering config in the final key** — shown failing on the pre-change code (identical hash
  for two thresholds), passing after.
- **Clustering config absent from the intermediate key** — two different clustering configs, and
  three different clustering *classes*, produce the same intermediate key.
- **Byte-identical warm vs cold RTTMs** — 16 simulated meetings, warm run writing to a separate
  `cache_dir` so `Runner.run()` cannot short-circuit at `runner.py:51-52` and read back the file
  it is comparing against. Asserts zero inference calls on the warm side, so a pass cannot mean
  "recomputed and happened to match". Paired with a sensitivity test proving the byte comparison
  can actually fail. **The real 16-meeting corpus run is separate from this work.**
- **Intermediate-tier wall clock** — isolated by giving both arms a fresh RTTM cache dir; 0.969 s
  cold vs 0.031 s warm over 8 files. The percentage is an artefact of the simulated inference
  cost, not a prediction of real GPU savings; the real figure needs a GPU run.
- **On-disk size** — ≈272 MB for 16 meetings, extrapolated as described above.

## See also

- [[clustering-method-comparison]] — the epic this unblocks.
- [[clustering-model-selection]] — adds clustering fields into the key shape established here.
- [[Runner]] — the final-hypothesis tier.
- [[Oracle Segmentation Provider]] — the `populate()` pattern this mirrors.
- [[oracle-segmentation-seam-no-op]] — what silent failure in this machinery looks like.
