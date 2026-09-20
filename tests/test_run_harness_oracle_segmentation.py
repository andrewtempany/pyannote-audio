"""Unit-level test for T3's third acceptance criterion: a run using
OracleSegmentation completes, scores, and appears in comparison.csv as
oracle_segmentation.

Independent of the full_pipeline integration fixture, which is currently
broken in this environment for unrelated reasons (see
tests/test_run_harness_oracle_wiring.py's header and T3's Implementation
Notes: every existing test in test_run_harness_integration.py that depends
on full_pipeline fails with the same pre-existing FileNotFoundError, before
any T3 code runs). This test mocks Runner/AMIDatasetAdapter (not
OracleSegmentation itself, and not run_harness()/aggregate_runs(), which are
the actual things under test here) so it can prove real end-to-end behavior
-- manifest written with run_condition/segmentation_source_id recorded
correctly, and aggregate_runs() picking that up into comparison.csv -- on
any machine, without needing a real trained pipeline.
"""

import csv
import json
from unittest.mock import MagicMock, patch

from pyannote.core import Annotation, Segment, Timeline

from harness.aggregate_runs import aggregate_runs
from harness.config import HarnessConfig
from harness.segmentation import OracleSegmentation
from run_harness import run_harness


def _config(tmp_path, data_root):
    return HarnessConfig.load(
        data_root=data_root,
        condition="IHM",
        cache_dir=tmp_path / "cache",
    )


def test_oracle_segmentation_run_appears_in_comparison_csv(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = _config(tmp_path, data_root)

    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    reference[Segment(1.0, 2.0)] = "B"
    uem = Timeline([Segment(0.0, 2.0)])

    hypothesis = Annotation()
    hypothesis[Segment(0.0, 1.0)] = "A"
    hypothesis[Segment(1.0, 2.0)] = "B"

    oracle_source = OracleSegmentation(reference_lookup={"fake-uri": reference})

    with patch("run_harness.AMIDatasetAdapter") as mock_adapter_cls, \
         patch("run_harness.Runner") as mock_runner_cls:
        mock_adapter_cls.return_value = [("fake-uri", reference, uem)]
        mock_runner = MagicMock()
        mock_runner.run.return_value = hypothesis
        mock_runner_cls.return_value = mock_runner

        runs_dir = tmp_path / "runs"
        summary = run_harness(
            config,
            pipeline=MagicMock(),
            pipeline_config_id="test-pipeline",
            segmentation_source=oracle_source,
            per_file_csv_path=tmp_path / "per_file.csv",
            summary_path=tmp_path / "summary.json",
            clustering_model="agglomerative-v2",
            runs_dir=runs_dir,
            run_condition="oracle_segmentation",
        )

    # run completes and scores
    assert "der" in summary
    assert (tmp_path / "per_file.csv").exists()

    # Runner was constructed with the real OracleSegmentation instance,
    # so the cache key (harness/runner.py) picks up its distinct
    # segmentation_source.id automatically -- confirms this run went
    # through the real injectable seam, not a bypass.
    _, kwargs = mock_runner_cls.call_args
    args = mock_runner_cls.call_args.args
    assert oracle_source in args or oracle_source in kwargs.values()

    manifest_files = list(runs_dir.glob("*.json"))
    assert len(manifest_files) == 1
    manifest = json.loads(manifest_files[0].read_text())
    # id bumped "oracle" -> "oracle-v2" (oracle-segmentation-seam-no-op fix) to
    # orphan the pre-fix .harness_cache entries via the cache key -- do not revert.
    assert manifest["run_config"]["segmentation_source_id"] == "oracle-v2"
    assert manifest["run_config"]["condition"] == "oracle_segmentation"

    # appears in comparison.csv as oracle_segmentation
    comparison_csv = tmp_path / "comparison.csv"
    aggregate_runs(runs_dir, comparison_csv)

    with open(comparison_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["condition"] == "oracle_segmentation"
    # id bumped "oracle" -> "oracle-v2" (oracle-segmentation-seam-no-op fix) to
    # orphan the pre-fix .harness_cache entries via the cache key -- do not revert.
    assert rows[0]["segmentation_source_id"] == "oracle-v2"
