"""Tests for the run-manifest ticket: harness/run_manifest.py.

harness/run_manifest.py doesn't exist yet at the time these tests are
written -- the first run of this suite is expected to fail on import.

All inputs are fabricated run_config/summary dicts -- no model, no audio,
no real dataset needed (same bar as tests/test_reporter.py).
"""

import json

import pytest

from harness.run_manifest import RunManifestError, write_manifest


def _run_config(**overrides):
    config = {
        "pipeline_config_id": "pyannote/speaker-diarization-community-1",
        "segmentation_source_id": "baseline",
        "clustering_model": "pyannote-default",
        "extra_pipeline_steps": [],
        "split": "test",
        "condition": "IHM",
        "der_collar": 0.0,
        "der_skip_overlap": False,
    }
    config.update(overrides)
    return config


def _summary(**overrides):
    summary = {
        "der": 0.17,
        "overlap_der": 0.22,
        "jer": 0.3,
        "counting_mae": 0.5,
        "counting_exact_match_percent": 75.0,
    }
    summary.update(overrides)
    return summary


def test_manifest_contains_full_run_config_and_summary(tmp_path):
    run_config = _run_config(clustering_model="agglomerative-v2", extra_pipeline_steps=["multispeaker-flag-v1"])
    summary = _summary()

    manifest_path = write_manifest(run_config, summary, runs_dir=tmp_path)

    written = json.loads(manifest_path.read_text())
    assert written["run_config"] == run_config
    assert written["summary"] == summary
    assert "run_id" in written
    assert "created_at" in written


def test_run_id_is_unique_per_config(tmp_path):
    config_a = _run_config(clustering_model="agglomerative-v2")
    config_b = _run_config(clustering_model="agglomerative-v3")

    path_a = write_manifest(config_a, _summary(), runs_dir=tmp_path)
    path_b = write_manifest(config_b, _summary(), runs_dir=tmp_path)

    assert path_a != path_b
    assert path_a.exists()
    assert path_b.exists()

    manifest_a = json.loads(path_a.read_text())
    manifest_b = json.loads(path_b.read_text())
    assert manifest_a["run_id"] != manifest_b["run_id"]


def test_identical_config_rerun_raises_on_collision(tmp_path, monkeypatch):
    frozen_timestamp = "20260907T120000Z"
    monkeypatch.setattr("harness.run_manifest._utc_timestamp", lambda: frozen_timestamp)

    run_config = _run_config()

    first_path = write_manifest(run_config, _summary(), runs_dir=tmp_path)
    original_contents = first_path.read_text()

    with pytest.raises(RunManifestError):
        write_manifest(run_config, _summary(der=0.99), runs_dir=tmp_path)

    assert first_path.read_text() == original_contents


def test_runs_dir_created_if_missing(tmp_path):
    runs_dir = tmp_path / "nested" / "runs"
    assert not runs_dir.exists()

    manifest_path = write_manifest(_run_config(), _summary(), runs_dir=runs_dir)

    assert runs_dir.exists()
    assert manifest_path.exists()


@pytest.mark.parametrize("missing_field", ["clustering_model", "segmentation_source_id"])
def test_missing_required_run_config_field_raises(tmp_path, missing_field):
    run_config = _run_config()
    del run_config[missing_field]

    with pytest.raises(RunManifestError):
        write_manifest(run_config, _summary(), runs_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []
