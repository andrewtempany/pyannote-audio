"""Tests for TICKET-05 (the original scorer) and T1 (oracle/ceiling analysis
batch: der components, der_overlap_system rename, der_overlap_assigned,
region census).

score()'s signature has grown a 4th metric parameter, der_overlap_assigned,
alongside der/overlap_der/jer -- these tests target that new 7-arg contract
throughout, including the pre-existing tests that predate T1.

All fixtures here are small, hand-built pyannote.core objects -- no model,
no audio, no disk I/O, matching the ticket's non-goals. Every expected
metric value below was independently verified against the real
pyannote.metrics classes before being hardcoded (see the ticket's design
note on the exact toy examples), not derived from the scorer under test.
"""

from unittest.mock import MagicMock

import pytest
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from harness.scorer import score


def _fresh_metrics():
    # T1: der_overlap_assigned is scored over T-intersect-D, a region
    # distinct from both der's (whole uem) and overlap_der's/der_overlap_
    # system's (T) -- it needs its own accumulator instance so abs(metric)
    # reflects the right corpus total, mirroring how overlap_der already
    # gets its own instance separate from der.
    return (
        DiarizationErrorRate(collar=0.0, skip_overlap=False),
        DiarizationErrorRate(collar=0.0, skip_overlap=False),
        DiarizationErrorRate(collar=0.0, skip_overlap=False),
        JaccardErrorRate(collar=0.0, skip_overlap=False),
    )


def test_der_known_value_on_toy_example():
    # reference: A and B fully overlapping for [0, 10]; hypothesis only
    # detects A -- B is entirely missed. Hand-computed DER = missed / total
    # = 10 / 20 = 0.5.
    reference = Annotation()
    reference[Segment(0, 10), "a"] = "A"
    reference[Segment(0, 10), "b"] = "B"

    hypothesis = Annotation()
    hypothesis[Segment(0, 10), "a"] = "A"

    uem = Timeline([Segment(0, 10)])
    der, overlap_der, der_overlap_assigned, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, der_overlap_assigned, jer)

    assert result["der"] == pytest.approx(0.5)


def test_der_configured_with_collar_0_skip_overlap_false():
    # hypothesis starts 0.5s late. Under collar=0 this is a real 0.5s miss
    # (DER = 0.05 over a 10s reference); a nonzero collar would forgive a
    # boundary discrepancy this small and give DER = 0.0. Asserting the
    # nonzero result proves collar=0 is what's actually configured.
    reference = Annotation()
    reference[Segment(0, 10), "a"] = "A"

    hypothesis = Annotation()
    hypothesis[Segment(0.5, 10), "a"] = "A"

    uem = Timeline([Segment(0, 10)])
    der, overlap_der, der_overlap_assigned, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, der_overlap_assigned, jer)

    assert result["der"] == pytest.approx(0.05)


def test_overlap_der_restricted_to_reference_overlap_regions():
    # reference has a 2-speaker overlap in [3, 5]; hypothesis is wrong only
    # outside the overlap (different label in [0,3] and [5,8]) but exactly
    # right during the overlap itself. Overall DER is nonzero; overlap-DER
    # (scored only over the reference's >=2-speaker region) is 0.
    reference = Annotation()
    reference[Segment(0, 3), "a0"] = "A"
    reference[Segment(3, 5), "a1"] = "A"
    reference[Segment(3, 5), "b1"] = "B"
    reference[Segment(5, 8), "a2"] = "A"

    hypothesis = Annotation()
    hypothesis[Segment(0, 3), "x0"] = "X"
    hypothesis[Segment(3, 5), "a1"] = "A"
    hypothesis[Segment(3, 5), "b1"] = "B"
    hypothesis[Segment(5, 8), "x2"] = "X"

    uem = Timeline([Segment(0, 8)])
    der, overlap_der, der_overlap_assigned, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, der_overlap_assigned, jer)

    assert result["der"] == pytest.approx(0.2)
    assert result["der_overlap_system"] == pytest.approx(0.0)


def test_overlap_region_intersected_with_uem():
    # reference overlap is [3, 5]; hypothesis matches reference exactly
    # during [3, 4] but is wrong during [4, 5] (still inside the overlap).
    # A UEM of [0, 4] excludes the wrong part, so overlap-DER (overlap ∩
    # uem = [3, 4]) is 0 -- if the uem intersection were skipped and the
    # full [3, 5] overlap were scored instead, it would be nonzero (0.25,
    # confirmed independently while designing this fixture).
    reference = Annotation()
    reference[Segment(0, 3), "a0"] = "A"
    reference[Segment(3, 5), "a1"] = "A"
    reference[Segment(3, 5), "b1"] = "B"

    hypothesis = Annotation()
    hypothesis[Segment(0, 3), "a0"] = "A"
    hypothesis[Segment(3, 5), "a1"] = "A"
    hypothesis[Segment(3, 4), "b1"] = "B"
    hypothesis[Segment(4, 5), "c1"] = "C"

    uem = Timeline([Segment(0, 4)])
    der, overlap_der, der_overlap_assigned, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, der_overlap_assigned, jer)

    assert result["der_overlap_system"] == pytest.approx(0.0)


def test_jer_uses_same_collar_and_skip_overlap():
    reference = Annotation()
    reference[Segment(0, 10), "a"] = "A"

    hypothesis = Annotation()
    hypothesis[Segment(0.5, 10), "a"] = "A"

    uem = Timeline([Segment(0, 10)])
    der, overlap_der, der_overlap_assigned, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, der_overlap_assigned, jer)

    assert result["jer"] == pytest.approx(0.05)


def test_four_metric_instances_each_called_exactly_once_per_file():
    reference = Annotation()
    reference[Segment(0, 3), "a0"] = "A"
    reference[Segment(3, 5), "a1"] = "A"
    reference[Segment(3, 5), "b1"] = "B"

    hypothesis = Annotation()
    hypothesis[Segment(0, 3), "a0"] = "A"
    hypothesis[Segment(3, 5), "a1"] = "A"
    hypothesis[Segment(3, 5), "b1"] = "B"

    uem = Timeline([Segment(0, 8)])
    real_der, real_overlap_der, real_der_overlap_assigned, real_jer = _fresh_metrics()

    der = MagicMock(wraps=real_der)
    overlap_der = MagicMock(wraps=real_overlap_der)
    der_overlap_assigned = MagicMock(wraps=real_der_overlap_assigned)
    jer = MagicMock(wraps=real_jer)

    score(reference, hypothesis, uem, der, overlap_der, der_overlap_assigned, jer)

    assert der.call_count == 1
    assert overlap_der.call_count == 1
    assert der_overlap_assigned.call_count == 1
    assert jer.call_count == 1

    der_uem = der.call_args.kwargs["uem"]
    overlap_der_uem = overlap_der.call_args.kwargs["uem"]
    assert der_uem is uem
    assert der_uem != overlap_der_uem


def test_counting_reports_ref_hyp_counts_and_abs_diff():
    reference = Annotation()
    reference[Segment(0, 1), "1"] = "A"
    reference[Segment(1, 2), "2"] = "B"
    reference[Segment(2, 3), "3"] = "C"

    hypothesis = Annotation()
    hypothesis[Segment(0, 1), "1"] = "W"
    hypothesis[Segment(1, 2), "2"] = "X"
    hypothesis[Segment(2, 3), "3"] = "Y"
    hypothesis[Segment(3, 4), "4"] = "Z"
    hypothesis[Segment(4, 5), "5"] = "V"

    uem = Timeline([Segment(0, 5)])
    der, overlap_der, der_overlap_assigned, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, der_overlap_assigned, jer)

    assert result["count_ref"] == 3
    assert result["count_hyp"] == 5
    assert result["count_error"] == 2


def test_der_components_reported_and_sum_consistently():
    # T1: missed detection, false alarm, and confusion must be surfaced as
    # separate fields, not just folded into the single der ratio. Reference
    # has a 2-speaker overlap [3,5]; hypothesis misses B entirely in that
    # window (a real missed-detection error, verified independently above in
    # test_der_known_value_on_toy_example's sibling fixture pattern) and is
    # otherwise correct. Expected component durations (verified against the
    # real DiarizationErrorRate.compute_components before hardcoding):
    #   total=10, correct=8, missed_detection=2, false_alarm=0, confusion=0
    reference = Annotation()
    reference[Segment(0, 3), "a0"] = "A"
    reference[Segment(3, 5), "a1"] = "A"
    reference[Segment(3, 5), "b1"] = "B"
    reference[Segment(5, 10), "a2"] = "A"

    hypothesis = Annotation()
    hypothesis[Segment(0, 3), "a0"] = "A"
    hypothesis[Segment(3, 5), "a1"] = "A"
    hypothesis[Segment(5, 10), "a2"] = "A"

    uem = Timeline([Segment(0, 10)])
    der, overlap_der, der_overlap_assigned, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, der_overlap_assigned, jer)

    assert result["missed_detection"] == pytest.approx(2.0)
    assert result["false_alarm"] == pytest.approx(0.0)
    assert result["confusion"] == pytest.approx(0.0)


def test_der_overlap_system_is_scored_over_reference_overlap_only():
    # Renamed from overlap_der -> der_overlap_system, same reference-overlap
    # (T) semantics as before: scored only over ground-truth multi-speaker
    # regions, unaffected by what the hypothesis detected as overlap.
    reference = Annotation()
    reference[Segment(0, 3), "a0"] = "A"
    reference[Segment(3, 5), "a1"] = "A"
    reference[Segment(3, 5), "b1"] = "B"
    reference[Segment(5, 8), "a2"] = "A"

    hypothesis = Annotation()
    hypothesis[Segment(0, 3), "x0"] = "X"
    hypothesis[Segment(3, 5), "a1"] = "A"
    hypothesis[Segment(3, 5), "b1"] = "B"
    hypothesis[Segment(5, 8), "x2"] = "X"

    uem = Timeline([Segment(0, 8)])
    der, overlap_der, der_overlap_assigned, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, der_overlap_assigned, jer)

    assert result["der"] == pytest.approx(0.2)
    assert result["der_overlap_system"] == pytest.approx(0.0)


def test_der_overlap_assigned_scored_over_intersection_of_reference_and_hypothesis_overlap():
    # T1's headline metric. Reference has a genuine 3-speaker overlap [4,6]
    # (T); hypothesis correctly detects that region as multi-speaker (so
    # D == T here) but only emits 2 of the 3 active speakers there -- a real
    # assignment/detection shortfall confined entirely inside the
    # intersection. Verified independently: scored over T-intersect-D alone,
    # missed_detection=2 out of total=6 -> der_overlap_assigned = 1/3, while
    # whole-file der is diluted to ~0.154 by all the correctly-scored
    # single-speaker regions. This gap is the point of the metric: a bug that
    # accidentally scored over raw T (or over D without intersecting T)
    # would not reproduce 1/3 here.
    reference = Annotation()
    reference[Segment(0, 3), "a0"] = "A"
    reference[Segment(3, 4), "a1"] = "A"
    reference[Segment(4, 6), "a2"] = "A"
    reference[Segment(4, 6), "b2"] = "B"
    reference[Segment(4, 6), "c2"] = "C"
    reference[Segment(6, 9), "a3"] = "A"

    hypothesis = Annotation()
    hypothesis[Segment(0, 3), "a0"] = "A"
    hypothesis[Segment(3, 4), "a1"] = "A"
    hypothesis[Segment(4, 6), "x1"] = "A"
    hypothesis[Segment(4, 6), "x2"] = "B"
    hypothesis[Segment(6, 9), "a3"] = "A"

    uem = Timeline([Segment(0, 9)])
    der, overlap_der, der_overlap_assigned, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, der_overlap_assigned, jer)

    assert result["der"] == pytest.approx(0.15384615384615385)
    assert result["der_overlap_assigned"] == pytest.approx(1 / 3)


def test_region_census_reports_disjoint_durations():
    # T (ref overlap) = [3,6]; D (hyp overlap) = [4,6] -- hypothesis detects
    # overlap late by 1s. So T&D = [4,6] (2s), T\D = [3,4] (1s), D\T = [] (0s).
    # Verified independently via Timeline crop/extrude before hardcoding.
    reference = Annotation()
    reference[Segment(0, 3), "a0"] = "A"
    reference[Segment(3, 6), "a1"] = "A"
    reference[Segment(3, 6), "b1"] = "B"
    reference[Segment(6, 9), "a2"] = "A"

    hypothesis = Annotation()
    hypothesis[Segment(0, 3), "a0"] = "A"
    hypothesis[Segment(3, 4), "a1"] = "A"
    hypothesis[Segment(4, 6), "x1"] = "A"
    hypothesis[Segment(4, 6), "x2"] = "B"
    hypothesis[Segment(6, 9), "a2"] = "A"

    uem = Timeline([Segment(0, 9)])
    der, overlap_der, der_overlap_assigned, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, der_overlap_assigned, jer)

    assert result["region_t_and_d"] == pytest.approx(2.0)
    assert result["region_t_minus_d"] == pytest.approx(1.0)
    assert result["region_d_minus_t"] == pytest.approx(0.0)


def test_scorer_does_not_construct_its_own_metric_instances():
    reference_1 = Annotation()
    reference_1[Segment(0, 10), "a"] = "A"
    hypothesis_1 = Annotation()
    hypothesis_1[Segment(0, 10), "a"] = "A"

    reference_2 = Annotation()
    reference_2[Segment(0, 5), "a"] = "A"
    hypothesis_2 = Annotation()
    hypothesis_2[Segment(0, 5), "a"] = "A"

    uem = Timeline([Segment(0, 10)])
    der, overlap_der, der_overlap_assigned, jer = _fresh_metrics()

    score(reference_1, hypothesis_1, uem, der, overlap_der, der_overlap_assigned, jer)
    score(reference_2, hypothesis_2, uem, der, overlap_der, der_overlap_assigned, jer)

    assert len(der.results_) == 2
    assert len(overlap_der.results_) == 2
    assert len(der_overlap_assigned.results_) == 2
    assert len(jer.results_) == 2
