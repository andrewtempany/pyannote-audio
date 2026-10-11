"""Tests for TICKET-04: the segmentation source interface.

harness/segmentation.py doesn't exist yet at the time these tests are
written -- the first run of this suite is expected to fail on import.

The `pipeline`/`one_file` fixtures used here live in tests/conftest.py
(shared with TICKET-01 and TICKET-06, which need the same real, fully
offline SpeakerDiarization pipeline).
"""

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pyannote.core import SlidingWindow
from pyannote.database.util import load_rttm

from harness.segmentation import BaselineSegmentation, OracleSegmentation, SegmentationSource

_ORACLE_FIXTURE_RTTM = Path(__file__).parent / "data" / "oracle_only_words.rttm"


class _FakeSegmentationInference:
    """Stand-in for the pipeline's `_segmentation` Inference object -- only
    exposes what OracleSegmentation.populate() needs (step/duration/
    model.receptive_field), avoiding the real, slow, session-scoped
    SpeakerDiarization pipeline fixture (which requires the shared
    tests/conftest.py protocol fixtures -- unrelated infra not needed here,
    since populate() never calls the segmentation model itself)."""

    def __init__(self):
        self.step = 1.0
        self.duration = 2.0
        self.model = SimpleNamespace(receptive_field=SlidingWindow(start=0.0, step=0.5, duration=0.5))


class _FakePipelineForOracle:
    """Minimal fake pipeline exposing exactly the surface OracleSegmentation
    touches: CACHED_SEGMENTATION (real property behavior copied from
    SpeakerDiarization), training (plain bool, per the injection-seam
    convention), and _segmentation (see _FakeSegmentationInference)."""

    def __init__(self):
        self.training = False
        self._segmentation = _FakeSegmentationInference()

    @property
    def CACHED_SEGMENTATION(self):
        return "training_cache/segmentation"


def test_baseline_id_is_stable_string():
    a = BaselineSegmentation().id
    b = BaselineSegmentation().id

    assert isinstance(a, str)
    assert a == b


def test_baseline_populate_hook_is_noop():
    file = {"uri": "ES2002a", "some_other_key": 123}
    before = dict(file)

    BaselineSegmentation().populate(pipeline=None, file=file)

    assert file == before


def test_baseline_end_to_end_runs_pipeline_normally(pipeline, one_file):
    """Inverse of TICKET-01's seam-bug test: baseline must NOT trigger the
    injection path, so the segmentation model runs normally."""
    spy = MagicMock(wraps=pipeline._segmentation)
    pipeline._segmentation = spy

    BaselineSegmentation().populate(pipeline, one_file)
    pipeline.get_segmentations(one_file)

    assert spy.call_count == 1


def test_oracle_docstring_references_oracle_segmentation_util():
    text = (OracleSegmentation.__doc__ or "") + (OracleSegmentation.populate.__doc__ or "")
    assert "pyannote.audio.pipelines.utils.oracle.oracle_segmentation" in text


def test_oracle_populate_sets_cached_segmentation_matching_fixture_rttm():
    """Acceptance criterion: a fixture file produces segmentation matching its
    RTTM. Builds OracleSegmentation from a small synthetic only_words-style
    RTTM (tests/data/oracle_only_words.rttm: speaker A 0-2s, B 2-4s, A 4-6s)
    and checks the resulting CACHED_SEGMENTATION agrees with that RTTM at the
    frame level -- not just that *some* array got set."""
    pipeline = _FakePipelineForOracle()
    reference_lookup = load_rttm(str(_ORACLE_FIXTURE_RTTM))
    uri = "oracle_fixture"
    file = {"uri": uri, "duration": 6.0}

    OracleSegmentation(reference_lookup=reference_lookup).populate(pipeline, file)

    assert pipeline.CACHED_SEGMENTATION in file
    result = file[pipeline.CACHED_SEGMENTATION]

    window = SlidingWindow(
        step=pipeline._segmentation.step, duration=pipeline._segmentation.duration
    )
    frames = pipeline._segmentation.model.receptive_field

    from pyannote.audio.pipelines.utils.oracle import oracle_segmentation

    expected = oracle_segmentation(
        {"uri": uri, "duration": 6.0, "annotation": reference_lookup[uri]},
        window,
        frames,
    )

    assert result.data.shape == expected.data.shape
    assert (result.data == expected.data).all()


def test_oracle_populate_restores_training_flag_after_success():
    pipeline = _FakePipelineForOracle()
    reference_lookup = load_rttm(str(_ORACLE_FIXTURE_RTTM))
    file = {"uri": "oracle_fixture", "duration": 6.0}

    assert pipeline.training is False
    OracleSegmentation(reference_lookup=reference_lookup).populate(pipeline, file)
    assert pipeline.training is False


def test_oracle_populate_restores_training_flag_after_failure():
    """Uri not present in the reference lookup should raise (fail fast, per
    the seam doc's fail-fast requirement for OracleSegmentation), but must
    not leave pipeline.training permanently flipped."""
    pipeline = _FakePipelineForOracle()
    file = {"uri": "not_in_lookup", "duration": 6.0}
    assert pipeline.training is False

    with pytest.raises(KeyError):
        OracleSegmentation(reference_lookup={}).populate(pipeline, file)

    assert pipeline.training is False


def test_interface_contract():
    assert issubclass(BaselineSegmentation, SegmentationSource)
    assert issubclass(OracleSegmentation, SegmentationSource)

    baseline_sig = inspect.signature(BaselineSegmentation.populate)
    oracle_sig = inspect.signature(OracleSegmentation.populate)
    assert baseline_sig == oracle_sig

    assert isinstance(BaselineSegmentation().id, str)
    assert isinstance(OracleSegmentation(reference_lookup={}).id, str)
    assert BaselineSegmentation().id != OracleSegmentation(reference_lookup={}).id
