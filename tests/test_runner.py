"""Tests for TICKET-06: the runner.

harness/runner.py doesn't exist yet at the time these tests are written --
the first run of this suite is expected to fail on import.

Tests 1-6 use fake/mocked pipelines -- no model, no network -- per the
ticket's acceptance criteria that these run fast on every commit. Test 7 is
a real, marked-slow integration test requiring a valid HF_TOKEN with
community-1 access accepted; it's designed to skip with a clear, explicit
reason (not silently) when credentials aren't available, which is the case
in this environment.
"""

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pyannote.core import Annotation, Segment

from harness.runner import Runner
from harness.segmentation import BaselineSegmentation, OracleSegmentation


class _FakePipeline:
    """A pipeline stand-in returning a fixed hypothesis Annotation directly
    (not wrapped in a DiarizeOutput) -- used by tests that only care about
    call counting/caching, not the full/exclusive field distinction (that's
    test_returns_full_annotation_not_exclusive's job)."""

    def __init__(self, hypothesis: Annotation):
        self.calls = 0
        self._hypothesis = hypothesis

    def __call__(self, file):
        self.calls += 1
        return SimpleNamespace(
            speaker_diarization=self._hypothesis,
            exclusive_speaker_diarization=Annotation(uri=file.get("uri")),
        )


def _tracks(annotation):
    return sorted((seg, label) for seg, _, label in annotation.itertracks(yield_label=True))


def test_cache_key_includes_pipeline_config_source_id_and_uri(tmp_path):
    pipeline = _FakePipeline(Annotation(uri="ES2002a"))

    runner_a = Runner(
        pipeline,
        pipeline_config_id="config1",
        segmentation_source=BaselineSegmentation(),
        cache_dir=tmp_path,
    )
    runner_b = Runner(
        pipeline,
        pipeline_config_id="config1",
        segmentation_source=OracleSegmentation(),
        cache_dir=tmp_path,
    )
    assert runner_a.cache_key("ES2002a") != runner_b.cache_key("ES2002a")

    runner_c = Runner(
        pipeline,
        pipeline_config_id="config2",
        segmentation_source=BaselineSegmentation(),
        cache_dir=tmp_path,
    )
    assert runner_a.cache_key("ES2002a") != runner_c.cache_key("ES2002a")

    assert runner_a.cache_key("ES2002a") != runner_a.cache_key("ES2002b")


def test_cache_miss_invokes_pipeline_and_writes_rttm(tmp_path):
    hypothesis = Annotation(uri="ES2002a")
    hypothesis[Segment(0, 1)] = "A"
    pipeline = _FakePipeline(hypothesis)
    runner = Runner(
        pipeline,
        pipeline_config_id="config1",
        segmentation_source=BaselineSegmentation(),
        cache_dir=tmp_path,
    )

    result = runner.run({"uri": "ES2002a"})

    assert pipeline.calls == 1
    assert result is hypothesis
    assert runner.cache_path("ES2002a").exists()


def test_cache_hit_skips_pipeline_and_loads_rttm(tmp_path):
    hypothesis = Annotation(uri="ES2002a")
    hypothesis[Segment(0, 1)] = "A"
    hypothesis[Segment(1, 2)] = "B"
    pipeline = _FakePipeline(hypothesis)
    runner = Runner(
        pipeline,
        pipeline_config_id="config1",
        segmentation_source=BaselineSegmentation(),
        cache_dir=tmp_path,
    )

    runner.run({"uri": "ES2002a"})
    assert pipeline.calls == 1

    result = runner.run({"uri": "ES2002a"})

    assert pipeline.calls == 1  # unchanged -- not invoked a second time
    assert _tracks(result) == _tracks(hypothesis)


def test_returns_full_annotation_not_exclusive(tmp_path):
    full = Annotation(uri="ES2002a")
    full[Segment(0, 1)] = "A"
    full[Segment(1, 2)] = "B"

    exclusive = Annotation(uri="ES2002a")
    exclusive[Segment(5, 6)] = "Z"  # deliberately different from `full`

    class _FixedOutputPipeline:
        def __init__(self):
            self.calls = 0

        def __call__(self, file):
            self.calls += 1
            return SimpleNamespace(
                speaker_diarization=full, exclusive_speaker_diarization=exclusive
            )

    pipeline = _FixedOutputPipeline()
    runner = Runner(
        pipeline,
        pipeline_config_id="config1",
        segmentation_source=BaselineSegmentation(),
        cache_dir=tmp_path,
    )

    result = runner.run({"uri": "ES2002a"})

    assert _tracks(result) == _tracks(full)
    assert _tracks(result) != _tracks(exclusive)


def test_runner_uses_segmentation_source_populate_hook(tmp_path):
    call_order = []

    class _SpySource:
        id = "spy"

        def populate(self, pipeline, file):
            call_order.append("populate")

    hypothesis = Annotation(uri="ES2002a")

    class _RecordingPipeline:
        def __init__(self):
            self.calls = 0

        def __call__(self, file):
            call_order.append("pipeline")
            self.calls += 1
            return SimpleNamespace(
                speaker_diarization=hypothesis,
                exclusive_speaker_diarization=Annotation(uri="ES2002a"),
            )

    pipeline = _RecordingPipeline()
    runner = Runner(
        pipeline, pipeline_config_id="config1", segmentation_source=_SpySource(), cache_dir=tmp_path
    )

    runner.run({"uri": "ES2002a"})

    assert call_order == ["populate", "pipeline"]


def test_oracle_source_fails_fast_not_silently(tmp_path):
    pipeline = _FakePipeline(Annotation(uri="ES2002a"))
    runner = Runner(
        pipeline,
        pipeline_config_id="config1",
        segmentation_source=OracleSegmentation(),
        cache_dir=tmp_path,
    )

    with pytest.raises(NotImplementedError):
        runner.run({"uri": "ES2002a"})

    assert pipeline.calls == 0
    assert not runner.cache_path("ES2002a").exists()


@pytest.mark.integration
def test_runner_real_pipeline_one_file(tmp_path):
    token = os.environ.get("HF_TOKEN")
    if not token:
        pytest.skip(
            "requires a real HF_TOKEN with community-1 access accepted on "
            "huggingface.co -- set HF_TOKEN to run this integration test"
        )

    from pyannote.audio import Pipeline

    real_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=token
    )

    audio_path = Path(__file__).parent / "data" / "dev00.wav"
    file = {"uri": "dev00", "audio": str(audio_path)}

    runner = Runner(
        real_pipeline,
        pipeline_config_id="community-1",
        segmentation_source=BaselineSegmentation(),
        cache_dir=tmp_path,
    )

    hypothesis = runner.run(file)
    assert isinstance(hypothesis, Annotation)
    assert len(list(hypothesis.itertracks())) > 0

    start = time.monotonic()
    runner.run(file)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0  # cache hit: no model invocation, should be near-instant
