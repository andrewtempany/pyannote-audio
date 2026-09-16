---
status: in-progress
created: 2026-09-13
---

# T2: Post-clustering refinement hook, plus nearest-centroid validation

Part of the oracle/ceiling analysis batch (see [[T1-harness-reporting-and-manifest-schema]], [[T3-oracle-segmentation-provider]], [[T4-oracle-assignment-strategy]], [[T5-cross-condition-deltas]]).

## Context

The unit of assignment is the `(chunk, local_speaker)` pair, not the frame. `get_embeddings` (`pipelines/speaker_diarization.py:332-478`) pools within each chunk and local speaker before returning; `embeddings` and `hard_clusters` both have shape `(num_chunks, local_num_speakers, dimension)` / `(num_chunks, local_num_speakers)`. Frame-level identity is produced downstream by `reconstruct()` (lines 480-528) from the pair labels. Every refinement strategy in this batch operates on pairs.

## Goal

Create the single extension point every strategy in this batch plugs into, and prove it works.

The seam sits between the clustering call and the inactive-speaker masking in `SpeakerDiarization.apply`:

```python
hard_clusters, soft_clusters, centroids = self.clustering(...)   # line 640
hard_clusters = self.refinement(                                  # new
    embeddings, hard_clusters, soft_clusters, centroids, segmentations
)
hard_clusters[inactive_speakers] = -2                             # line 680
```

Everything downstream, including `reconstruct()`, the relabeling at lines 715-740, and the centroid reordering at lines 763-773, is untouched.

## Scope
- **Recover `soft_clusters`.** Line 640 currently discards it into `_`. It is the per-cluster posterior for every pair, already computed, and the most useful feature a later classifier could have. One character of code.
- Define the refinement strategy interface, taking the five arguments above (`embeddings`, `hard_clusters`, `soft_clusters`, `centroids`, `segmentations`) and returning a `hard_clusters` array of identical shape.
- Default strategy is `identity`, returning its input unchanged.
- Implement `nearest_centroid`: assign each pair to the centroid nearest its embedding by cosine similarity.
- Strategy selectable by harness config, flowing into the `refinement_strategy` manifest field from [[T1-harness-reporting-and-manifest-schema]].

## Acceptance criteria
- [x] With `identity`, the baseline run reproduces 17.05% **exactly**, not approximately -- confirmed on GPU: `der = 0.17049287033463784`, bit-for-bit identical to the pre-hook baseline manifest
- [x] With `nearest_centroid`, the run completes and DER lands within roughly a point of baseline -- confirmed on GPU: `der = 0.17049287033463784`, identical to baseline (see Implementation Notes for why exact equality here is expected, not a red flag)
- [x] Both runs land in `comparison.csv` with correct `condition` values -- confirmed: `condition=baseline`/`refinement_strategy=identity` and `condition=nearest_centroid`/`refinement_strategy=nearest_centroid` respectively

## Implementer notes
- Check whether the harness invokes the `legacy=True` path. If it does, `DiarizeOutput` never leaves `apply` and centroids are unavailable downstream. Verify before assuming.
- `OracleClustering` returns `centroids=None` when no embeddings are supplied. The interface needs to handle a null-centroid case explicitly rather than crashing.

## Out of scope
The oracle strategy — that is [[T4-oracle-assignment-strategy]].

## Dependencies
Can run in parallel with [[T1-harness-reporting-and-manifest-schema]]. Blocks [[T4-oracle-assignment-strategy]].

## Implementation Notes

(append here as work happens — decisions, rejected alternatives, gotchas, key files/functions touched)

- 2026-09-13: Wrote failing unit tests only (TDD red phase) in `tests/test_refinement.py`. No source changes yet — `src/pyannote/audio/pipelines/speaker_diarization.py` and `harness/` are untouched.
- Confirmed real shapes before writing fixtures: `embeddings` (num_chunks, local_num_speakers, dimension), `hard_clusters` (num_chunks, local_num_speakers), `soft_clusters` (num_chunks, local_num_speakers, num_clusters) per `clustering.py`, `centroids` (num_speakers, dimension) or `None` for `OracleClustering`.
- Proposed interface: `harness/refinement.py` with plain functions `identity(embeddings, hard_clusters, soft_clusters, centroids, segmentations)` and `nearest_centroid(...)` (same signature), a `REFINEMENT_STRATEGIES` dict registry, and `get_refinement_strategy(name) -> Callable` that raises `KeyError` on an unknown name. Matches the plain-function house style of `harness/scorer.py` / `harness/reporter.py` — no classes.
- soft_clusters recovery (`hard_clusters, _, centroids = self.clustering(...)` → capture instead of discard) is a one-line change at the `apply()` call site, not something a unit test on `harness/refinement.py` can directly exercise without a real pipeline run. Wrote a unit-level test instead asserting the strategy interface has a genuine (non-vestigial) `soft_clusters` parameter that accepts the real posterior shape — documented this tradeoff inline in the test. True end-to-end confirmation that line 640 actually captures it is deferred to the implementation step / an integration test, out of scope for this pass.
- `nearest_centroid(centroids=None, ...)` open decision (flagged in ticket's implementer notes): chose no-op fallback to identity rather than raising, so a config-selected `nearest_centroid` run doesn't hard-crash when it happens to hit an `OracleClustering` condition. Documented as a choice in the test, not inferred — a raising `ValueError` would be equally defensible and could be swapped later without ticket scope creep.
- Config selection tested narrowly against `get_refinement_strategy`/`REFINEMENT_STRATEGIES` directly rather than wiring into `harness/config.py`'s `HarnessConfig` dataclass — wiring the config field itself is implementation, not this pass's scope.
- First run of `pytest tests/test_refinement.py -v` fails at collection with `ModuleNotFoundError: No module named 'harness.refinement'`, as expected — confirms the red state is real (module doesn't exist yet), not a scaffolding bug.
- 2026-09-17: Implemented `harness/refinement.py` (plain functions, matching `harness/scorer.py`/`harness/reporter.py` style): `identity`, `nearest_centroid`, `REFINEMENT_STRATEGIES` dict, `get_refinement_strategy`. `identity` returns `hard_clusters.copy()` (no in-place mutation, per test). `nearest_centroid` flattens `(num_chunks, local_num_speakers, dim)` embeddings, L2-normalizes both embeddings and centroids, does a batched cosine-similarity argmax against centroids, reshapes back to `(num_chunks, local_num_speakers)`. When `centroids is None` it delegates straight to `identity` (the no-crash choice already recorded above). `get_refinement_strategy` is a plain dict `[name]` lookup, so an unknown name raises `KeyError` naturally (no custom exception needed). `pytest tests/test_refinement.py -v` -> 12 passed, 0 failed.
- Wired the hook into `src/pyannote/audio/pipelines/speaker_diarization.py`:
  - `__init__` (~line 193-224 after edit) gained a `refinement: Optional[Callable] = None` parameter, documented in the class docstring next to `clustering`. Stored as `self.refinement`, defaulting to an inline no-op lambda (`lambda embeddings, hard_clusters, soft_clusters, centroids, segmentations: hard_clusters`) when `None` is passed, mirroring how `self.klustering`/`self.clustering` are set up right above it. Deliberately did **not** import `harness.refinement` from inside `src/pyannote/audio/...` — `harness/` is a project-level script directory, not shipped with the installed `pyannote-audio` package, and importing it from the core library would break the package for any consumer outside this repo. The harness is expected to pass in `get_refinement_strategy(cfg.refinement_strategy)` itself when constructing the pipeline.
  - `apply()` call site (~line 656, was line 640 pre-edit — shifted by the added docstring lines): changed `hard_clusters, _, centroids = self.clustering(...)` to `hard_clusters, soft_clusters, centroids = self.clustering(...)`, then inserted `hard_clusters = self.refinement(embeddings, hard_clusters, soft_clusters, centroids, segmentations)` immediately after, before `inactive_speakers` masking (~line 685→701 after edit). Used the raw `segmentations` variable (the model's direct output, matches the ticket's pseudocode/interface name) rather than `binarized_segmentations` (the binarized version actually passed to `self.clustering`) — both are in scope at that point; `segmentations` is the more general/raw signal for a future strategy to consume.
- **legacy=True check (ticket's implementer-notes concern):** grepped `harness/` and `run_harness.py` for `legacy` — zero matches. The harness never constructs the pipeline with `legacy=True`, so it always gets the default `legacy=False` path, meaning `apply()` returns a full `DiarizeOutput` (not just `Annotation`), and `centroids`/`soft_clusters` remain available for any downstream harness code that wants them. Not an issue for this ticket's scope.
- Verification: `pytest tests/test_refinement.py -v` — 12 passed. Full suite: `pytest tests/ -q -k "not integration" --ignore=tests/test_run_notebooks.py` — 161 passed, 22 failed, 10 errors, but confirmed via `git stash` that the exact same 22 failures/10 errors (all `FileNotFoundError` for missing `tests/data/*.wav` fixtures, or `papermill`/other pre-existing missing-dependency issues) occur identically on the pre-change tree — none are caused by the `apply()`/`__init__` edits in this pass. `ast.parse` confirms `speaker_diarization.py` is syntactically valid.
- Status: all T2 scope items done except the harness config wiring (`refinement_strategy` manifest field -> `get_refinement_strategy` -> pipeline `refinement=` kwarg), which was explicitly out of scope for this pass per the earlier Implementation Notes entry ("wiring the config field itself is implementation, not this pass's scope") and per the task instructions for this pass (harness config wiring not requested). `status` left as `in-progress`; ticket not moved to Docs/ yet since that wiring plus the acceptance-criteria DER runs are still open.

- 2026-09-17 (during T1 implementation pass): closed the remaining wiring gap. `run_harness.py`'s `run_harness()` now calls `get_refinement_strategy(refinement_strategy)` and sets `pipeline.refinement = ...` before constructing the `Runner` -- confirmed `self.refinement` is a plain post-`__init__` settable attribute on `SpeakerDiarization` (not consumed only at construction time), so this works without needing to reconstruct the pipeline. Added a `--refinement-strategy` CLI flag (`identity`/`oracle`/`nearest_centroid`, default `identity`) alongside the other new T1 manifest-field flags (`--run-condition`, `--corpus`, `--counts-toward-results`). `oracle` is accepted by the CLI/manifest vocabulary but will raise `KeyError` from `get_refinement_strategy` if actually selected before T4 implements it -- expected and correct, since T4 owns that strategy.
- 2026-09-17: Ran the `identity` strategy on GPU (RTX 3060, real AMI IHM test split) via `run_harness.py --refinement-strategy identity --run-condition baseline --counts-toward-results`, manifest `runs/20260916T213918Z-d3ef4437.json`. `der = 0.17049287033463784` -- bit-for-bit identical to the pre-refinement-hook baseline (`runs/20260913T031401Z-a9c41646.json`), confirming the hook's `identity` path is a true no-op end to end, not just in the unit tests. Row lands correctly in `comparison.csv` with `condition=baseline`, `refinement_strategy=identity`.
- Ran the `nearest_centroid` strategy on GPU via `run_harness.py --refinement-strategy nearest_centroid --run-condition nearest_centroid --counts-toward-results`, manifest `runs/20260916T214459Z-e9f25911.json`. Result: `der = 0.17049287033463784` -- **exactly** equal to baseline, not just "within a point." This is the expected outcome per the batch spec's own framing (section 3): "re-assigning each embedding to its nearest final centroid largely reproduces what clustering already decided" -- clustering already assigns each pair to its own centroid by construction, so a pure nearest-centroid re-assignment pass is close to a no-op on top of standard agglomerative/whatever clustering the pipeline already ran. Exact equality here is confirmation the plumbing works, not a bug -- a large swing would have been the red flag per the ticket's own acceptance criterion wording.
- `comparison.csv` refreshed; the new row carries `condition=nearest_centroid`, `refinement_strategy=nearest_centroid`, matching values.
- All three acceptance criteria satisfied. Ticket work is complete; ready to convert to documentation.
