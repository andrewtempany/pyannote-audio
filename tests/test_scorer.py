"""Tests for TICKET-05: the scorer.

harness/scorer.py doesn't exist yet at the time these tests are written --
the first run of this suite is expected to fail on import.

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
    return (
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
    der, overlap_der, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, jer)

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
    der, overlap_der, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, jer)

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
    der, overlap_der, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, jer)

    assert result["der"] == pytest.approx(0.2)
    assert result["overlap_der"] == pytest.approx(0.0)


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
    der, overlap_der, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, jer)

    assert result["overlap_der"] == pytest.approx(0.0)


def test_jer_uses_same_collar_and_skip_overlap():
    reference = Annotation()
    reference[Segment(0, 10), "a"] = "A"

    hypothesis = Annotation()
    hypothesis[Segment(0.5, 10), "a"] = "A"

    uem = Timeline([Segment(0, 10)])
    der, overlap_der, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, jer)

    assert result["jer"] == pytest.approx(0.05)


def test_three_metric_instances_each_called_exactly_once_per_file():
    reference = Annotation()
    reference[Segment(0, 3), "a0"] = "A"
    reference[Segment(3, 5), "a1"] = "A"
    reference[Segment(3, 5), "b1"] = "B"

    hypothesis = Annotation()
    hypothesis[Segment(0, 3), "a0"] = "A"
    hypothesis[Segment(3, 5), "a1"] = "A"
    hypothesis[Segment(3, 5), "b1"] = "B"

    uem = Timeline([Segment(0, 8)])
    real_der, real_overlap_der, real_jer = _fresh_metrics()

    der = MagicMock(wraps=real_der)
    overlap_der = MagicMock(wraps=real_overlap_der)
    jer = MagicMock(wraps=real_jer)

    score(reference, hypothesis, uem, der, overlap_der, jer)

    assert der.call_count == 1
    assert overlap_der.call_count == 1
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
    der, overlap_der, jer = _fresh_metrics()

    result = score(reference, hypothesis, uem, der, overlap_der, jer)

    assert result["count_ref"] == 3
    assert result["count_hyp"] == 5
    assert result["count_error"] == 2


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
    der, overlap_der, jer = _fresh_metrics()

    score(reference_1, hypothesis_1, uem, der, overlap_der, jer)
    score(reference_2, hypothesis_2, uem, der, overlap_der, jer)

    assert len(der.results_) == 2
    assert len(overlap_der.results_) == 2
    assert len(jer.results_) == 2
