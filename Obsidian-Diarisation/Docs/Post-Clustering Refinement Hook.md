---
status: done
created: 2026-09-13
---

# Post-Clustering Refinement Hook

> Part of the [[Evaluation Harness]]. **Two things are contributed here, of different provenance.** `harness/refinement.py` is project-only code (like [[Scorer]]/[[Harness Reporter]]). The hook itself is a small, deliberate addition to `src/pyannote/audio/pipelines/speaker_diarization.py`'s `SpeakerDiarization` class — a real source change to the pipeline (unlike [[Segmentation Injection Seam]], which needed no source change), scoped to be a pure extension point: with the default strategy, pipeline output is bit-for-bit unchanged.

## Why this exists

The oracle/ceiling-analysis batch's thesis is that some diarisation error lives downstream of the frozen neural network, in post-clustering speaker assignment — fixable without retraining. Testing that thesis means being able to swap in different assignment strategies (a naive nearest-centroid baseline, an oracle ceiling strategy, eventually a learned classifier) without touching clustering, embedding extraction, or anything else in the pipeline.

## The unit of assignment: pairs, not frames

`SpeakerDiarization.get_embeddings()` pools within each chunk and local speaker before returning. `embeddings` has shape `(num_chunks, local_num_speakers, dimension)`; `hard_clusters` (from `self.clustering(...)`) has shape `(num_chunks, local_num_speakers)`. **The unit of assignment is the `(chunk, local_speaker)` pair** — frame-level speaker identity is a downstream consequence, produced by `reconstruct()` from the pair labels. Every refinement strategy operates on pairs, not frames.

## The seam

Inside `SpeakerDiarization.apply()`, between the clustering call and inactive-speaker masking:

```python
hard_clusters, soft_clusters, centroids = self.clustering(...)
hard_clusters = self.refinement(embeddings, hard_clusters, soft_clusters, centroids, segmentations)
hard_clusters[inactive_speakers] = -2
```

Everything downstream — `reconstruct()`, the relabeling, and the centroid reordering — is untouched; they only ever see `hard_clusters` after refinement, indistinguishable from clustering having produced that array directly.

### `soft_clusters` is now recovered, not discarded

Before this change, the clustering call's second return value was discarded (`hard_clusters, _, centroids = self.clustering(...)`). It is the per-cluster posterior for every pair — already computed by clustering, and the most information-dense feature available to a future strategy — so it is now captured and threaded through the refinement call.

### `self.refinement` is a plain settable attribute

`SpeakerDiarization.__init__` gained an optional `refinement: Callable = None` parameter, following the same pattern as `self.clustering`/`self.klustering`. When `None` (the default), it falls back to an inline no-op:

```python
self.refinement = refinement if refinement is not None else (
    lambda embeddings, hard_clusters, soft_clusters, centroids, segmentations: hard_clusters
)
```

Because it's a plain instance attribute rather than something consumed only during construction, a caller can also set it after the fact — `pipeline.refinement = get_refinement_strategy(name)` — without reconstructing the pipeline. This is how `run_harness.py` wires it (see below).

### Why `harness/refinement.py` isn't imported from inside `src/pyannote/audio/...`

`harness/` is a project-only script directory, not part of the installed `pyannote-audio` package. Importing it from core library code would make the installed package depend on files that don't ship with it, breaking the package for any consumer outside this repo. The pipeline only knows about a `Callable` with the right signature; the harness is responsible for handing it one.

## Strategy interface

```python
def strategy(embeddings, hard_clusters, soft_clusters, centroids, segmentations) -> hard_clusters
```

Returns a `hard_clusters` array of identical shape to its input. `harness/refinement.py` provides:

- **`identity(...)`** — true no-op. Returns `hard_clusters.copy()` (not the same object — the interface guarantees the input is never mutated in place).
- **`nearest_centroid(...)`** — reassigns each `(chunk, local_speaker)` pair to whichever centroid is closest by cosine similarity. Flattens `embeddings` to `(num_pairs, dimension)`, L2-normalizes both embeddings and centroids, computes a batched cosine-similarity matrix, and takes `argmax` per pair.
- **`REFINEMENT_STRATEGIES`** — a plain `dict[str, Callable]` registry (`"identity"`, `"nearest_centroid"`).
- **`get_refinement_strategy(name: str) -> Callable`** — plain dict lookup; raises `KeyError` on an unknown name (config-driven selection needs a clear failure for typos, and a bare `KeyError` is sufficient — no custom exception was needed).

### The `centroids=None` case

`OracleClustering` returns `centroids=None` when no embeddings are supplied to it. `nearest_centroid` has nothing to compare against in that case, so it delegates straight to `identity` rather than raising:

```python
if centroids is None:
    return identity(embeddings, hard_clusters, soft_clusters, centroids, segmentations)
```

This is a documented choice, not an inferred one — a raising `ValueError` would be equally defensible, but the no-op fallback means a config-selected `nearest_centroid` run doesn't hard-crash an entire harness run just because it happens to hit an `OracleClustering` condition partway through.

## Harness wiring

`run_harness.py`'s `run_harness()` takes a `refinement_strategy: str = "identity"` parameter and, before constructing the `Runner`:

```python
pipeline.refinement = get_refinement_strategy(refinement_strategy)
```

The CLI exposes `--refinement-strategy` (choices: `identity`, `oracle`, `nearest_centroid`; default `identity`). `"oracle"` is accepted by the CLI/manifest vocabulary (see [[Oracle Ceiling Metrics]]) but will raise `KeyError` from `get_refinement_strategy` if actually selected — the oracle strategy is implemented by a separate, later ticket; this is expected, not a bug.

## The `legacy=True` concern (checked, not an issue)

`SpeakerDiarization.apply()` has a `legacy` mode where it returns a bare `Annotation` instead of the full `DiarizeOutput`, and centroids never leave the method. If the harness ever constructed the pipeline with `legacy=True`, centroids/soft_clusters would be unavailable downstream. Checked: neither `harness/` nor `run_harness.py` references `legacy` anywhere — the harness always gets the default `legacy=False` path. Not an issue for this harness's usage, but worth re-checking if a future caller ever constructs the pipeline differently.

## Verification (real GPU runs, not just unit tests)

Ran both implemented strategies on an RTX 3060 against the real AMI IHM test split:

- **`identity`** (`--run-condition baseline`): `der = 0.17049287033463784` — bit-for-bit identical to the pre-hook baseline. (This is a **warm, RTTM-cached** figure; the cold full-precision equivalent is `0.17048543579940637`. The ~8e-6 gap is RTTM millisecond truncation, not nondeterminism — resolved 2026-09-25, see [[Pre-Clustering Cache]] and [[Oracle Ceiling Metrics]].) Confirms the hook's `identity` path is a true no-op end to end, not just in isolated unit tests.
- **`nearest_centroid`** (`--run-condition nearest_centroid`): `der = 0.17049287033463784` — **exactly** equal to baseline, not merely close. This is the expected result, not a red flag: clustering already assigns each pair to its own nearest centroid by construction (that's what clustering is), so re-assigning by nearest-centroid on top of an already-converged clustering result is close to a no-op. The batch's own design notes frame `nearest_centroid` as "not a headline comparator... expected to land close to baseline" — a plumbing validation and a citable naive-method result, not evidence the refinement mechanism does nothing. A *large* swing from baseline would have been the actual red flag (wrong shape, wrong axis, mismatched chunk ordering, etc.).

Both runs appear correctly in `runs/comparison.csv` with matching `condition`/`refinement_strategy` values.

## Non-goals

- The oracle refinement strategy — a separate ticket owns implementing it; this hook only guarantees the extension point and vocabulary are ready for it.
- Any learned/classifier-based strategy — contingent on the oracle-ceiling go/no-go decision, out of scope for this batch entirely.
- Frame-level refinement — the interface operates on `(chunk, local_speaker)` pairs by design (see "The unit of assignment" above); a strategy that wanted frame-level information would need to derive it itself from `segmentations`.

## Key files

- `harness/refinement.py` — `identity`, `nearest_centroid`, `REFINEMENT_STRATEGIES`, `get_refinement_strategy`
- `src/pyannote/audio/pipelines/speaker_diarization.py` — `SpeakerDiarization.__init__`'s `refinement` parameter, `apply()`'s hook insertion and `soft_clusters` recovery
- `run_harness.py` — `refinement_strategy` parameter, `--refinement-strategy` CLI flag
- `tests/test_refinement.py` — 12 unit tests covering the interface, both strategies, the `centroids=None` case, shape preservation, and registry lookup
