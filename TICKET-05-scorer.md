# 05 — Scorer

Part of the [eval harness epic](TICKET-eval-harness.md). Depends on: [00](TICKET-00-environment-setup.md).

## Scope

`harness/scorer.py` — scoring logic, no I/O, no model loading. One subtlety the epic's phrasing glossed over and this ticket must resolve explicitly: the epic describes the scorer as "a pure function `(reference, hypothesis, uem) -> dict`" but *also* requires that DER/overlap-DER/JER be accumulated across the whole corpus by exactly one metric instance each, constructed once at the start of the run. A function that both stays pure/stateless and accumulates state across calls is a contradiction unless the metric accumulator objects are themselves passed in as parameters, owned by the caller (the orchestrator, ticket 08). Resolve it that way: `score(reference, hypothesis, uem, der, overlap_der, jer) -> dict`, where `der`, `overlap_der`, `jer` are pre-constructed `DiarizationErrorRate`/`JaccardErrorRate` instances passed in by whoever is looping over files. This keeps `scorer.py` itself free of model loading and file I/O (the actual meaning of "pure" here) while making the shared-accumulator requirement explicit in the function signature instead of implicit and easy to get wrong.

Metrics and settings (all confirmed against installed `pyannote.metrics`):
- `DiarizationErrorRate(collar=0, skip_overlap=False)` — overall DER.
- A **separate** `DiarizationErrorRate(collar=0, skip_overlap=False)` instance — overlap-region DER, called with `uem=reference.get_overlap().crop(uem, mode="intersection")` (both methods confirmed to exist with this exact signature on the installed `pyannote.core==5.0.0`).
- `JaccardErrorRate(collar=0, skip_overlap=False)` — confirmed `JaccardErrorRate.__bases__ == (DiarizationErrorRate,)`, same defaults, pass explicitly anyway per the epic's reasoning.
- Speaker counting: `len(reference.labels())` vs `len(hypothesis.labels())`, plus absolute difference — per-file, no accumulator object involved.

## Non-goals

- No corpus-level aggregation logic beyond calling the passed-in accumulators (that belongs to the orchestrator/reporter, tickets 07/08).
- Do not reimplement DER/JER math.

## Tests first

Add `tests/test_scorer.py`:

1. `test_der_known_value_on_toy_example` — construct a small hand-built reference/hypothesis pair with a known, hand-computed DER (e.g., one speaker fully missed), assert the scorer's `der` value matches within floating-point tolerance.
2. `test_der_configured_with_collar_0_skip_overlap_false` — inspect or exercise the constructed metric instances to confirm these settings (e.g., a case that would score differently under a nonzero collar, asserting the zero-collar result).
3. `test_overlap_der_restricted_to_reference_overlap_regions` — build a reference with one 2-speaker overlap region and one single-speaker region, a hypothesis that's wrong only outside the overlap region, assert overlap-DER is 0 (or near it) despite overall DER being nonzero — proves the region restriction actually narrows scoring.
4. `test_overlap_region_intersected_with_uem` — same as above but with a UEM that excludes part of the overlap region, assert only the intersected portion is scored (construct the case so excluding vs including that portion changes the result).
5. `test_jer_uses_same_collar_and_skip_overlap` — analogous to test 2 for JER.
6. `test_three_metric_instances_each_called_exactly_once_per_file` — pass in spy-wrapped `der`/`overlap_der`/`jer` objects, call `score()` once, assert each spy's `__call__` was invoked exactly once, and that `der` was never invoked with the overlap-restricted uem (catches the epic's flagged "one instance called twice" bug class directly).
7. `test_counting_reports_ref_hyp_counts_and_abs_diff` — reference with 3 labels, hypothesis with 5, assert returned dict has `count_ref=3, count_hyp=5, count_error=2`.
8. `test_scorer_does_not_construct_its_own_metric_instances` — assert calling `score()` twice with the *same* passed-in accumulators results in the accumulators reflecting two files' worth of state (i.e., the function isn't silently creating fresh instances internally) — this is the direct regression test for the accumulation-contract decision made above.

## Acceptance criteria

- All tests pass using only `pyannote.core`/`pyannote.metrics` toy fixtures — no model, no audio, no disk I/O anywhere in `scorer.py` or its tests.
- The `score()` signature and the pure/stateless-vs-accumulator resolution is documented in a short module-level comment, since it's a nonobvious design call a future reader could easily "fix" back into the epic's literal (contradictory) phrasing.

## Implementation notes (resolved during build — 2026-09-04)

- Implemented exactly as this ticket's Scope resolved it: `score(reference, hypothesis, uem, der, overlap_der, jer) -> dict`, with `der`/`overlap_der`/`jer` owned and constructed once by the caller (TICKET-08's orchestrator). `harness/scorer.py`'s module docstring repeats the rationale so a future reader doesn't "simplify" it back into the epic's literal (self-contradictory) stateless-function phrasing.
- All toy-example expected values (DER `0.5`, `0.05`, `0.2`/`0.0`, etc.) were independently computed by calling real `pyannote.metrics.diarization.DiarizationErrorRate`/`JaccardErrorRate` instances directly while designing the fixtures, then hardcoded into the tests as the expected values — not derived from `harness/scorer.py` itself, so the tests aren't just checking "does the code agree with itself."
- **Construction gotcha (companion to TICKET-03's comparison gotcha):** building a reference `Annotation` with two genuinely overlapping speakers requires explicit, distinct track keys — `annotation[segment, "track_a"] = "A"` — because assigning to the *same* segment with no explicit track (or the same default track) silently **overwrites** the previous label rather than adding a second concurrent track. `annotation[Segment(3,5)] = "A"; annotation[Segment(3,5)] = "B"` leaves only `"B"` in the annotation, not both. Every overlap fixture in `tests/test_scorer.py` uses explicit track ids for this reason.
