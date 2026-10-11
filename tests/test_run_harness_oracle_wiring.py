"""Unit-level test for T4's run_harness.py wiring, independent of the
full_pipeline integration fixture (which is currently broken in this
environment for unrelated reasons -- see T4's Implementation Notes: every
existing test in test_run_harness_integration.py also fails with the same
FileNotFoundError, before any T4 code runs).

This test drives run_harness() with a mocked pipeline/Runner so it can prove
the actual behavior this ticket adds -- that selecting
refinement_strategy="oracle" sets pipeline.refinement to a fresh, per-file
make_oracle_strategy(...) closure (keyed to each file's own reference
Annotation) rather than the single shared callable identity/nearest_centroid
use -- without needing a real trained pipeline.

run_harness.py doesn't yet branch on refinement_strategy == "oracle" at the
time this test is written -- the first run of this suite is expected to
fail (pipeline.refinement stays a bare get_refinement_strategy("oracle")
lookup, which raises KeyError, since "oracle" is a placeholder in T2's
registry per its own docs).
"""

from unittest.mock import MagicMock, patch

import pytest
from pyannote.core import Annotation, Segment, Timeline

from harness.config import HarnessConfig
from harness.segmentation import BaselineSegmentation
from run_harness import run_harness


def _config(tmp_path, data_root):
    return HarnessConfig.load(
        data_root=data_root,
        condition="IHM",
        cache_dir=tmp_path / "cache",
    )


def _fake_adapter_row():
    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    reference[Segment(1.0, 2.0)] = "B"
    uem = Timeline([Segment(0.0, 2.0)])
    return "fake-uri", reference, uem


def test_oracle_strategy_selected_per_file_with_correct_reference_and_scope(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = _config(tmp_path, data_root)

    uri, reference, uem = _fake_adapter_row()
    fake_pipeline = MagicMock()

    captured_refinement_calls = []

    class _RefinementRecorder:
        """Stands in for pipeline.refinement: records every closure it's set
        to, so the test can inspect what run_harness assigned without
        needing a real SpeakerDiarization pipeline to invoke it."""

    def _set_refinement(value):
        captured_refinement_calls.append(value)

    # pipeline.refinement is a plain settable attribute (per T2) -- track
    # every value it's set to across the run.
    type(fake_pipeline).refinement = property(
        lambda self: None, lambda self, value: _set_refinement(value)
    )

    with patch("run_harness.AMIDatasetAdapter") as mock_adapter_cls, \
         patch("run_harness.Runner") as mock_runner_cls, \
         patch("run_harness.make_oracle_strategy") as mock_make_oracle:
        mock_adapter_cls.return_value = [(uri, reference, uem)]
        mock_runner = MagicMock()
        mock_runner.run.return_value = Annotation()
        mock_runner_cls.return_value = mock_runner
        mock_make_oracle.return_value = "the-oracle-closure"

        run_harness(
            config, fake_pipeline, "test-pipeline", BaselineSegmentation(),
            tmp_path / "per_file.csv", tmp_path / "summary.json",
            refinement_strategy="oracle",
            oracle_scope="overlap_degraded",
        )

    # make_oracle_strategy must have been called with THIS file's reference
    # and the configured scope -- not some global/shared reference.
    mock_make_oracle.assert_called_once_with(reference, oracle_scope="overlap_degraded")
    # and pipeline.refinement must actually have been set to that closure
    # before the file was run.
    assert "the-oracle-closure" in captured_refinement_calls


def test_non_oracle_strategy_still_sets_refinement_once_up_front(tmp_path):
    # identity/nearest_centroid must keep working exactly as before this
    # ticket: pipeline.refinement set once, to the plain registry lookup,
    # not per-file and not via make_oracle_strategy.
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = _config(tmp_path, data_root)

    uri, reference, uem = _fake_adapter_row()
    fake_pipeline = MagicMock()
    captured_refinement_calls = []
    type(fake_pipeline).refinement = property(
        lambda self: None, lambda self, value: captured_refinement_calls.append(value)
    )

    with patch("run_harness.AMIDatasetAdapter") as mock_adapter_cls, \
         patch("run_harness.Runner") as mock_runner_cls, \
         patch("run_harness.make_oracle_strategy") as mock_make_oracle:
        mock_adapter_cls.return_value = [(uri, reference, uem)]
        mock_runner = MagicMock()
        mock_runner.run.return_value = Annotation()
        mock_runner_cls.return_value = mock_runner

        run_harness(
            config, fake_pipeline, "test-pipeline", BaselineSegmentation(),
            tmp_path / "per_file.csv", tmp_path / "summary.json",
            refinement_strategy="identity",
        )

    mock_make_oracle.assert_not_called()
    assert len(captured_refinement_calls) == 1
