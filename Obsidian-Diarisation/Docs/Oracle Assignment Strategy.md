---
status: done
created: 2026-09-13
---

# Oracle Assignment Strategy

Part of the [[Evaluation Harness]]'s oracle/ceiling-analysis batch. Built on
[[Post-Clustering Refinement Hook]]; its results feed [[Cross-Condition Deltas]].
**All code here is project-contributed** (`harness/refinement.py`, `run_harness.py`) — no
changes to upstream `pyannote-audio`.

## What it measures

The ceiling on post-clustering speaker assignment: how much diarisation error would
disappear if every `(chunk, local speaker)` pair were assigned to the best possible
cluster, holding the real segmentation and the real clusters fixed.

This is the decision-relevant number in the batch. Baseline minus this result is the
error budget any real assignment intervention competes for — if the ceiling is small,
a perfect intervention can't win much, and the effort belongs elsewhere.

**The honest-ceiling constraint:** the strategy may only choose among clusters the
pipeline actually produced. It never invents a speaker the pipeline missed. Without that
constraint the number would be a ceiling for the whole pipeline (including its clustering
and speaker-count decisions), not for an assignment intervention specifically.

## How it works

`make_oracle_strategy(reference, oracle_scope) -> Callable` in `harness/refinement.py`.

1. Materialise `hard_clusters` + `segmentations` into a hypothesis `Annotation`
   (`_reconstruct_hypothesis`).
2. Compute the optimal hypothesis-cluster → reference-speaker mapping with
   `pyannote.metrics.diarization.DiarizationErrorRate().optimal_mapping()` — Hungarian
   matching on total temporal overlap. Reused rather than hand-rolling an overlap matrix.
3. For each pair, build its temporal support from its active frames
   (`_pair_timeline`, via `segmentations.sliding_window`).
4. Find the reference speaker with greatest overlap over that support
   (`_dominant_reference_speaker`).
5. Assign the cluster that maps to that speaker. If no cluster maps to it, leave the pair
   unchanged — this is where the honest-ceiling constraint is enforced.

### Why a factory, not a plain strategy

[[Post-Clustering Refinement Hook]]'s interface is fixed at
`strategy(embeddings, hard_clusters, soft_clusters, centroids, segmentations) -> hard_clusters`
— no slot for ground truth, and `run_harness.py` sets `pipeline.refinement` once, before
the per-file loop. Oracle assignment needs the *current file's* reference.

The factory closes over `reference`, so it still matches the 5-arg interface exactly.
`run_harness.py`'s loop rebuilds `pipeline.refinement` per file, but only when
`refinement_strategy == "oracle"`; identity and nearest_centroid keep their single
up-front assignment. This left the T2 interface and `speaker_diarization.py` untouched.

Rejected: smuggling the reference through `segmentations` or a module-level global —
fragile, and breaks the "operates only on pairs and what clustering produced" contract.

## `oracle_scope`

| Value | Meaning |
|---|---|
| `overlap_degraded` (primary) | Refine only pairs whose support is predominantly overlapping speech in the reference. Isolates the ceiling to the population an overlap-focused intervention would target. |
| `all_pairs` (secondary) | Refine every pair. The full post-clustering ceiling, useful as context. |

"Predominantly coincident with another speaker" is made precise as: within the pair's own
support, the reference shows ≥2 simultaneous speakers for more than half that duration.
Checkable purely from `reference.get_overlap()` cropped to the support.

Selected with `--oracle-scope`; recorded in the run manifest. Note the flag has a default
(`all_pairs`) that is written to *every* manifest even when the run never used it — see
[[Cross-Condition Deltas]], which hides the value for non-assignment conditions.

## Results (full AMI IHM test split, 16 meetings, RTX 3060)

| Run | DER | `der_overlap_system` | `der_overlap_assigned` |
|---|---|---|---|
| baseline (identity) | 0.1705 | 0.3339 | 0.2195 |
| oracle, `all_pairs` | 0.1608 | 0.3137 | 0.1920 |
| oracle, `overlap_degraded` | 0.1658 | 0.3189 | 0.1993 |

Assignment budget (`baseline - oracle`): **+0.0097** for `all_pairs`, **+0.0047** for
`overlap_degraded`. Both positive — the oracle lowered DER in both scopes. The narrower
scope recovering less than the full one is the expected shape, since it refines a strict
subset of pairs.

Interpretation of whether that budget justifies building a real intervention is
deliberately out of scope here; see [[Cross-Condition Deltas]].

## A caching bug these numbers depend on

The first attempt at these runs produced `der_overlap_assigned` **identical to baseline
to 16 decimal places** for both scopes. That was not a result — it was
`harness/runner.py`'s cache key omitting the refinement strategy, so both oracle runs
loaded the identity baseline's cached hypothesis and never executed any oracle logic.

Fixed by threading a `refinement_id` (e.g. `oracle:all_pairs`) into the cache key. The
numbers above are from runs after that fix, with the cache cleared. See [[Runner]].

The lesson worth keeping: a suspiciously *identical* number across conditions is a
stronger signal of a plumbing bug than a suspiciously large difference.

## Verification

- `tests/test_oracle_refinement.py` — 9 tests on real `pyannote.core` fixtures, no mocks.
  Covers the 5-arg interface contract, correct reassignment on hand-built overlap
  scenarios, both scopes, and the never-invent-a-cluster constraint (a reference speaker
  with no matching hypothesis cluster leaves its pair unchanged).
- `tests/test_run_harness_oracle_wiring.py` — 2 tests proving `run_harness.py` calls
  `make_oracle_strategy` once per file with that file's own reference, and that
  identity/nearest_centroid keep their single up-front assignment.
- Real end-to-end GPU runs for both scopes, recorded in `runs/comparison.csv`.

The originally-planned fixture-based integration test was replaced with the mocked wiring
test above, because `tests/test_run_harness_integration.py`'s shared `full_pipeline`
fixture is broken in this environment for unrelated reasons — see
[[non-ascii-filename-decoding]].

An earlier verification attempt on the 2-file `tests/fixtures/ami/basic/IHM` fixture could
not demonstrate the ceiling at all: the reference contains real overlap, but the pipeline
never predicted simultaneous speakers there, so `region_t_and_d` was 0 and
`der_overlap_assigned` was 0.0 for every strategy including baseline. A fixture with
reference overlap is not enough — the *pipeline* must also detect overlap somewhere, which
in practice meant using the full test split.

## Key files

- `harness/refinement.py` — `make_oracle_strategy`, `_reconstruct_hypothesis`,
  `_pair_timeline`, `_dominant_reference_speaker`, `_is_overlap_degraded`
- `run_harness.py` — `oracle_scope` param, `--oracle-scope` flag, the per-file
  `pipeline.refinement` branch, `oracle_scope` in the manifest
- `tests/test_oracle_refinement.py`, `tests/test_run_harness_oracle_wiring.py`
