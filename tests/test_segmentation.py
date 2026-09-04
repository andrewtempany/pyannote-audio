"""Tests for TICKET-04: the segmentation source interface.

harness/segmentation.py doesn't exist yet at the time these tests are
written -- the first run of this suite is expected to fail on import.

The `pipeline`/`one_file` fixtures used here live in tests/conftest.py
(shared with TICKET-01 and TICKET-06, which need the same real, fully
offline SpeakerDiarization pipeline).
"""

import inspect
from unittest.mock import MagicMock

import pytest

from harness.segmentation import BaselineSegmentation, OracleSegmentation, SegmentationSource


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


def test_oracle_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        OracleSegmentation().populate(pipeline=None, file={})


def test_oracle_docstring_references_oracle_segmentation_util():
    text = (OracleSegmentation.__doc__ or "") + (OracleSegmentation.populate.__doc__ or "")
    assert "pyannote.audio.pipelines.utils.oracle.oracle_segmentation" in text


def test_interface_contract():
    assert issubclass(BaselineSegmentation, SegmentationSource)
    assert issubclass(OracleSegmentation, SegmentationSource)

    baseline_sig = inspect.signature(BaselineSegmentation.populate)
    oracle_sig = inspect.signature(OracleSegmentation.populate)
    assert baseline_sig == oracle_sig

    assert isinstance(BaselineSegmentation().id, str)
    assert isinstance(OracleSegmentation().id, str)
    assert BaselineSegmentation().id != OracleSegmentation().id
