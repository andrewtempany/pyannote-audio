---
status: in-progress
created: 2026-09-13
---

# T3: Oracle segmentation provider

Part of the oracle/ceiling analysis batch (see [[T1-harness-reporting-and-manifest-schema]], [[T2-post-clustering-refinement-hook]], [[T4-oracle-assignment-strategy]], [[T5-cross-condition-deltas]]).

## Goal

Supply ground-truth speech and overlap regions through the harness's existing injectable segmentation interface (see [[Segmentation Injection Seam]] and [[Segmentation Source Interface]]), so a run can be scored with the network's segmentation stage replaced by ground truth.

## Scope
- Build segmentation from the `only_words` RTTMs in the shape the pipeline expects
- Conform to the same interface contract as the model-based source, so the cache key picks up a distinct `segmentation_source.id` automatically (see `harness/runner.py:35-40`)
- Timeboxed first step: assess whether `pipelines/utils/oracle.py` can be used directly before writing anything custom

## Acceptance criteria
- [ ] Provider passes the same interface contract as the model source
- [ ] A fixture file produces segmentation matching its RTTM
- [ ] Run completes, scores, and appears in `comparison.csv` as `oracle_segmentation`

## Dependencies
Needs the manifest fields from [[T1-harness-reporting-and-manifest-schema]] (`condition`, `segmentation_source_id`). Can start once T1 lands.

## Implementation Notes

### Timeboxed research: can `oracle_segmentation()` be used directly?

Yes, confirmed usable as-is, no source patch needed. Read
`src/pyannote/audio/pipelines/utils/oracle.py`: `oracle_segmentation(file, window,
frames, num_speakers=None)` takes `file["annotation"]` (a reference `Annotation`,
optionally `file["duration"]`), a `window` (chunk duration/step), and `frames`
(resolution), and returns exactly the `(num_chunks, num_frames, num_speakers)`
`SlidingWindowFeature` that `CACHED_SEGMENTATION` expects. It's already exercised by
existing oracle-clustering code (`src/pyannote/audio/pipelines/clustering.py:712`),
so the shape contract is known-good and battle-tested.

Confirmed the `window`/`frames` convention from that same call site
(`clustering.py:710`, `speaker_diarization.py:627/663`): `window` should match the
pipeline's own segmentation inference window
(`SlidingWindow(step=pipeline._segmentation.step, duration=pipeline._segmentation.duration)`),
and `frames=pipeline._segmentation.model.receptive_field`. Both are available off any
`SpeakerDiarization` pipeline instance without needing to know its internals beyond
what `clustering.py` already relies on.

### Design decision: where does the reference Annotation come from?

The runner's `file` dict (`run_harness.py`'s `run_harness()`) currently only carries
`uri`/`audio` — the reference used for scoring is threaded separately through
`score()`, never placed on `file`. So `OracleSegmentation.populate()` cannot assume
`file["annotation"]` is already present; it must supply it itself.

Decision: `OracleSegmentation` takes a `reference_lookup: Mapping[str, Annotation]`
at construction time (uri -> Annotation, exactly matching the shape
`pyannote.database.util.load_rttm()` already returns). `populate()` looks up
`file["uri"]`, sets `file["annotation"]` to that reference, then computes and injects
`file[pipeline.CACHED_SEGMENTATION]` via `oracle_segmentation()`, wrapped in the
`pipeline.training = True` seam from [[Segmentation Injection Seam]] (restored in a
`try/finally` so a failure never leaves the pipeline in training mode).

Rejected alternative: having `OracleSegmentation` read an RTTM path directly and
parse it lazily per-file. Rejected because `AMIDatasetAdapter` already parses the
combined per-split RTTM once via `load_rttm()` — reparsing per-file would duplicate
that work and diverge from the existing adapter convention. Building the
`reference_lookup` from the `only_words` RTTM is the caller's job (mirroring how
`pipeline_config_id` ownership was deferred to the caller in `harness/runner.py`),
kept outside this module.

### `only_words` RTTM

The ticket calls for building segmentation from the `only_words` RTTMs specifically
(a stricter/word-level ground truth than the collapsed RTTM already used elsewhere,
distinct from the `AMI-SDM.SpeakerDiarization.only_words` protocol referenced in
[[AMI Corpus Setup]]'s follow-ons). No `only_words` RTTM file exists yet in this
repo's `tests/data/` or in the converted AMI harness data directory — only the
standard collapsed RTTM/UEM pairs used by `AMIDatasetAdapter`. Since
`OracleSegmentation` is decoupled from any specific RTTM's provenance (it just takes
a `reference_lookup` dict), wiring an actual `only_words` RTTM source into a real AMI
run is left as a follow-on: it needs either a converted `only_words.rttm` file
produced the same way `convert_for_harness.sh` produces the existing combined RTTM,
or a small loader alongside `AMIDatasetAdapter`. Out of scope for this ticket's
fixture-level and unit-level acceptance criteria, which build the `reference_lookup`
directly from a small synthetic RTTM fixture instead.

### Files/functions touched

- `harness/segmentation.py` — `OracleSegmentation.__init__(reference_lookup)`,
  `OracleSegmentation.populate()` real implementation.
- `tests/test_segmentation.py` — extended with oracle-specific tests (fixture-based
  segmentation-matches-RTTM check, interface contract, training-seam
  restore-on-failure).
- `run_harness.py` — `main()` wiring so `--run-condition oracle_segmentation` selects
  `OracleSegmentation` (built from a `--oracle-rttm` fixture path) instead of
  `BaselineSegmentation`, so a real run appears in `comparison.csv` tagged
  `oracle_segmentation` via the existing `run_condition` manifest field (not
  `segmentation_source.id`, which stays `"oracle"` for cache-key purposes per
  [[Segmentation Source Interface]]).
