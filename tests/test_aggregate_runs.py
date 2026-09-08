"""Tests for the run-manifest ticket: harness/aggregate_runs.py.

harness/aggregate_runs.py doesn't exist yet at the time these tests are
written -- the first run of this suite is expected to fail on import.
"""

import csv
import json

from harness.aggregate_runs import aggregate_runs


def _write_manifest(runs_dir, run_id, run_config, summary):
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "created_at": "2026-09-07T12:00:00Z",
        "run_config": run_config,
        "summary": summary,
    }
    (runs_dir / f"{run_id}.json").write_text(json.dumps(manifest))


def test_aggregate_builds_one_row_per_manifest(tmp_path):
    runs_dir = tmp_path / "runs"
    _write_manifest(
        runs_dir, "run-1",
        {"pipeline_config_id": "p1", "segmentation_source_id": "baseline", "clustering_model": "cm1"},
        {"der": 0.1, "counting_mae": 0.5},
    )
    _write_manifest(
        runs_dir, "run-2",
        {"pipeline_config_id": "p2", "segmentation_source_id": "oracle", "clustering_model": "cm2"},
        {"der": 0.2, "counting_mae": 0.7},
    )

    output_csv = tmp_path / "comparison.csv"
    aggregate_runs(runs_dir=runs_dir, output_csv_path=output_csv)

    with open(output_csv, newline="") as f:
        rows = {row["run_id"]: row for row in csv.DictReader(f)}

    assert len(rows) == 2
    assert rows["run-1"]["clustering_model"] == "cm1"
    assert rows["run-1"]["segmentation_source_id"] == "baseline"
    assert float(rows["run-1"]["der"]) == 0.1
    assert rows["run-2"]["clustering_model"] == "cm2"
    assert float(rows["run-2"]["counting_mae"]) == 0.7


def test_aggregate_handles_empty_runs_dir(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    output_csv = tmp_path / "comparison.csv"
    aggregate_runs(runs_dir=runs_dir, output_csv_path=output_csv)

    with open(output_csv, newline="") as f:
        rows = list(csv.reader(f))

    assert rows == [] or rows == [["run_id"]]
