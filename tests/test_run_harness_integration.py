"""Tests for TICKET-08: the orchestrator (integration).

run_harness.py doesn't exist yet at the time these tests are written -- the
first run of this suite is expected to fail on import.

Deviation from the ticket's literal test description, called out explicitly:
the ticket frames this suite as requiring a valid HF_TOKEN (i.e. running the
real community-1 pipeline). This environment doesn't have one. Instead these
tests drive the orchestrator with the same real, fully offline
SpeakerDiarization pipeline built in tests/conftest.py (the `full_pipeline`
fixture, shared with TICKET-01/04/06) against real (short, reused) audio
fixtures under tests/fixtures/ami/basic/IHM/audio/. This still exercises
every real component (HarnessConfig, AMIDatasetAdapter, Runner, score(),
write_report()) and a real Pipeline object -- only the model weights are
tiny/untrained rather than community-1's. TICKET-06's own
test_runner_real_pipeline_one_file remains the place a real HF_TOKEN run
gets exercised.
"""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pyannote.metrics.diarization import DiarizationErrorRate

from harness.config import HarnessConfig
from harness.datasets import AMIDatasetAdapter
from harness.segmentation import BaselineSegmentation, OracleSegmentation
from run_harness import run_harness

FIXTURES = Path(__file__).parent / "fixtures" / "ami" / "basic"


def _config(tmp_path):
    return HarnessConfig.load(
        data_root=FIXTURES,
        condition="IHM",
        cache_dir=tmp_path / "cache",
    )


def test_end_to_end_produces_csv_and_summary_files(tmp_path, full_pipeline):
    config = _config(tmp_path)
    per_file = tmp_path / "per_file.csv"
    summary = tmp_path / "summary.json"

    run_harness(config, full_pipeline, "test-pipeline", BaselineSegmentation(), per_file, summary)

    assert per_file.exists() and per_file.stat().st_size > 0
    assert summary.exists() and summary.stat().st_size > 0


def test_per_file_csv_has_one_row_per_file_with_all_four_metrics(tmp_path, full_pipeline):
    config = _config(tmp_path)
    per_file = tmp_path / "per_file.csv"
    summary = tmp_path / "summary.json"

    run_harness(config, full_pipeline, "test-pipeline", BaselineSegmentation(), per_file, summary)

    with open(per_file, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2  # ES2002a, ES2002b
    for row in rows:
        for field in ("der", "overlap_der", "jer", "count_ref", "count_hyp", "count_error"):
            assert row[field] not in ("", None)
            float(row[field])  # raises if missing/garbage; also rejects "nan" text later


def test_corpus_summary_has_all_required_fields(tmp_path, full_pipeline):
    config = _config(tmp_path)
    per_file = tmp_path / "per_file.csv"
    summary_path = tmp_path / "summary.json"

    summary = run_harness(
        config, full_pipeline, "test-pipeline", BaselineSegmentation(), per_file, summary_path
    )

    for field in ("der", "overlap_der", "jer", "counting_mae", "counting_exact_match_percent"):
        assert field in summary
    assert summary == json.loads(summary_path.read_text())


def test_metrics_configured_correctly_end_to_end(tmp_path, full_pipeline):
    config = _config(tmp_path)
    per_file = tmp_path / "per_file.csv"
    summary_path = tmp_path / "summary.json"

    run_harness(config, full_pipeline, "test-pipeline", BaselineSegmentation(), per_file, summary_path)

    # independently recompute DER for one file directly from the reference
    # and a *freshly re-run* hypothesis -- if the wiring silently used a
    # different collar/skip_overlap than the config specifies, this fresh
    # collar=0/skip_overlap=False computation would disagree with what the
    # orchestrator reported. Deliberately not reloaded from the RTTM cache:
    # write_rttm() formats timestamps to millisecond precision (%.3f), so a
    # round-tripped hypothesis differs by sub-millisecond amounts from the
    # in-memory one the orchestrator actually scored -- re-running the same
    # (deterministic, eval-mode) pipeline call avoids that entirely.
    adapter_items = {uri: (reference, uem) for uri, reference, uem in AMIDatasetAdapter(config)}
    reference, uem = adapter_items["ES2002a"]

    audio_path = FIXTURES / "IHM" / "audio" / "ES2002a.wav"
    fresh_output = full_pipeline({"uri": "ES2002a", "audio": str(audio_path)})

    independent_der = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    expected = independent_der(reference, fresh_output.speaker_diarization, uem=uem)

    with open(per_file, newline="") as f:
        rows = {row["uri"]: row for row in csv.DictReader(f)}

    assert float(rows["ES2002a"]["der"]) == pytest.approx(expected)


def test_rerun_is_cache_accelerated(tmp_path, full_pipeline):
    config = _config(tmp_path)
    spy = MagicMock(wraps=full_pipeline)

    run_harness(
        config, spy, "test-pipeline", BaselineSegmentation(),
        tmp_path / "per_file.csv", tmp_path / "summary.json",
    )
    assert spy.call_count == 2

    spy.reset_mock()

    run_harness(
        config, spy, "test-pipeline", BaselineSegmentation(),
        tmp_path / "per_file_2.csv", tmp_path / "summary_2.json",
    )
    assert spy.call_count == 0


def test_run_manifest_written_alongside_report(tmp_path, full_pipeline):
    config = _config(tmp_path)
    per_file = tmp_path / "per_file.csv"
    summary_path = tmp_path / "summary.json"
    runs_dir = tmp_path / "runs"

    run_harness(
        config, full_pipeline, "test-pipeline", BaselineSegmentation(), per_file, summary_path,
        clustering_model="agglomerative-v2", runs_dir=runs_dir,
    )

    manifest_files = list(runs_dir.glob("*.json"))
    assert len(manifest_files) == 1

    manifest = json.loads(manifest_files[0].read_text())
    assert manifest["run_config"]["clustering_model"] == "agglomerative-v2"
    assert manifest["run_config"]["segmentation_source_id"] == "baseline"
    assert manifest["run_config"]["pipeline_config_id"] == "test-pipeline"
    assert manifest["summary"] == json.loads(summary_path.read_text())


def test_run_manifest_not_written_on_mid_run_failure(tmp_path, full_pipeline):
    config = _config(tmp_path)
    runs_dir = tmp_path / "runs"

    with pytest.raises(NotImplementedError):
        run_harness(
            config, full_pipeline, "test-pipeline", OracleSegmentation(),
            tmp_path / "per_file.csv", tmp_path / "summary.json",
            clustering_model="agglomerative-v2", runs_dir=runs_dir,
        )

    assert not runs_dir.exists() or list(runs_dir.glob("*.json")) == []


def test_segmentation_source_is_swappable_via_config(tmp_path, full_pipeline):
    config = _config(tmp_path)

    per_file = tmp_path / "per_file.csv"
    summary = tmp_path / "summary.json"
    run_harness(config, full_pipeline, "test-pipeline", BaselineSegmentation(), per_file, summary)
    assert per_file.exists()

    oracle_per_file = tmp_path / "oracle_per_file.csv"
    oracle_summary = tmp_path / "oracle_summary.json"

    with pytest.raises(NotImplementedError):
        run_harness(
            config, full_pipeline, "test-pipeline", OracleSegmentation(),
            oracle_per_file, oracle_summary,
        )

    assert not oracle_per_file.exists()
    assert not oracle_summary.exists()
