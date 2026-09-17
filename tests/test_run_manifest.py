"""Tests for the run-manifest ticket: harness/run_manifest.py, plus T1
(oracle/ceiling analysis batch): condition/refinement_strategy controlled
vocabularies, new required fields (corpus, mic_condition, git_commit,
refinement_strategy), and counts_toward_results defaulting to False.

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
        "condition": "baseline",
        "der_collar": 0.0,
        "der_skip_overlap": False,
        "refinement_strategy": "identity",
        "corpus": "AMI",
        "mic_condition": "IHM",
        "git_commit": "abc1234",
    }
    config.update(overrides)
    return config


def _summary(**overrides):
    summary = {
        "der": 0.17,
        "der_overlap_system": 0.22,
        "der_overlap_assigned": 0.2,
        "jer": 0.3,
        "counting_mae": 0.5,
        "counting_exact_match_percent": 75.0,
    }
    summary.update(overrides)
    return summary


def test_manifest_records_duration_seconds_when_given(tmp_path):
    manifest_path = write_manifest(
        _run_config(), _summary(), runs_dir=tmp_path, duration_seconds=842.5
    )

    written = json.loads(manifest_path.read_text())
    assert written["duration_seconds"] == 842.5


def test_manifest_duration_seconds_is_null_when_omitted(tmp_path):
    """Old callers that don't pass duration_seconds still work -- the field
    is present but null, not a KeyError waiting to happen for readers."""
    manifest_path = write_manifest(_run_config(), _summary(), runs_dir=tmp_path)

    written = json.loads(manifest_path.read_text())
    assert written["duration_seconds"] is None


def test_manifest_round_trips_notes_in_run_config(tmp_path):
    run_config = _run_config(notes="oracle segmentation experiment")
    summary = _summary()

    manifest_path = write_manifest(run_config, summary, runs_dir=tmp_path)

    written = json.loads(manifest_path.read_text())
    assert written["run_config"]["notes"] == "oracle segmentation experiment"


def test_manifest_contains_full_run_config_and_summary(tmp_path):
    # T1: counts_toward_results defaults to False when omitted, so the
    # written run_config is the caller's dict plus that one added field --
    # not byte-identical to what was passed in.
    run_config = _run_config(clustering_model="agglomerative-v2", extra_pipeline_steps=["multispeaker-flag-v1"])
    summary = _summary()

    manifest_path = write_manifest(run_config, summary, runs_dir=tmp_path)

    written = json.loads(manifest_path.read_text())
    assert written["run_config"] == {**run_config, "counts_toward_results": False}
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


@pytest.mark.parametrize(
    "missing_field",
    [
        "clustering_model",
        "segmentation_source_id",
        "refinement_strategy",
        "corpus",
        "mic_condition",
        "git_commit",
    ],
)
def test_missing_required_run_config_field_raises(tmp_path, missing_field):
    run_config = _run_config()
    del run_config[missing_field]

    with pytest.raises(RunManifestError):
        write_manifest(run_config, _summary(), runs_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "condition",
    ["baseline", "oracle_segmentation", "oracle_assignment", "nearest_centroid"],
)
def test_condition_accepts_each_controlled_vocabulary_value(tmp_path, condition):
    run_config = _run_config(condition=condition)

    manifest_path = write_manifest(run_config, _summary(), runs_dir=tmp_path)

    written = json.loads(manifest_path.read_text())
    assert written["run_config"]["condition"] == condition


def test_condition_rejects_value_outside_controlled_vocabulary(tmp_path):
    run_config = _run_config(condition="not_a_real_condition")

    with pytest.raises(RunManifestError):
        write_manifest(run_config, _summary(), runs_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "strategy", ["identity", "oracle", "nearest_centroid"],
)
def test_refinement_strategy_accepts_each_controlled_vocabulary_value(tmp_path, strategy):
    run_config = _run_config(refinement_strategy=strategy)

    manifest_path = write_manifest(run_config, _summary(), runs_dir=tmp_path)

    written = json.loads(manifest_path.read_text())
    assert written["run_config"]["refinement_strategy"] == strategy


def test_refinement_strategy_rejects_value_outside_controlled_vocabulary(tmp_path):
    run_config = _run_config(refinement_strategy="not_a_real_strategy")

    with pytest.raises(RunManifestError):
        write_manifest(run_config, _summary(), runs_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_counts_toward_results_defaults_to_false_when_omitted(tmp_path):
    run_config = _run_config()
    assert "counts_toward_results" not in run_config

    manifest_path = write_manifest(run_config, _summary(), runs_dir=tmp_path)

    written = json.loads(manifest_path.read_text())
    assert written["run_config"]["counts_toward_results"] is False


def test_counts_toward_results_explicit_true_is_preserved(tmp_path):
    run_config = _run_config(counts_toward_results=True)

    manifest_path = write_manifest(run_config, _summary(), runs_dir=tmp_path)

    written = json.loads(manifest_path.read_text())
    assert written["run_config"]["counts_toward_results"] is True
