"""Tests for the closeout-4 harness change: `--segmentation-param` and the
`min_duration_off` component of the FINAL cache key.

WHY THIS EXISTS. `segmentation.min_duration_off` only edits the final timeline
(speaker_diarization.py:718-736, after clustering). Before this change the
harness could not set it and neither cache key contained it, so a sweep would
have served the cached baseline RTTM at every value: a flat curve that reads
as a real negative result.

The pipeline stand-in below is a real `pyannote.pipeline.Pipeline` with a
real `ParamDict` segmentation block, so `instantiate({"segmentation": ...})`
rejects unknown names with the library's own error, not a mock's.

No GPU, no network, no audio.
"""

import hashlib
from unittest.mock import patch

import pytest
from pyannote.pipeline import Pipeline as PipelineBase
from pyannote.pipeline.parameter import ParamDict, Uniform

from harness.cache_id import intermediate_config_id, pipeline_config_id
import run_harness
from run_harness import build_pipeline, main

CHECKPOINT = "pyannote/speaker-diarization-community-1"

# The final key for the baseline configuration (VBx 0.6/0.07/0.8, batch 32,
# min_duration_off at its shipped value) AS COMPUTED BY THE CODE BEFORE THIS
# CHANGE. Hard-coded so that a change to what "no override" produces is a test
# failure rather than something the test silently follows.
BASELINE_FINAL_KEY = "241ecd865d157bd823106180f9691005b3f6245043a8fadb1c62480b9aedbced"


class _FakeVBx(PipelineBase):
    def __init__(self):
        super().__init__()
        self.threshold = Uniform(0.5, 0.8)
        self.Fa = Uniform(0.01, 0.5)
        self.Fb = Uniform(0.01, 15.0)


class _FakeSpeakerDiarization(PipelineBase):
    """Declares the same segmentation block a powerset checkpoint does
    (speaker_diarization.py:262-265) and is instantiated at the shipped values
    (community-1 config.yaml: min_duration_off 0.0)."""

    def __init__(self):
        super().__init__()
        self.klustering = "VBxClustering"
        self.embedding_batch_size = 32
        self.segmentation_batch_size = 32
        self._expects_num_speakers = False
        self.segmentation = ParamDict(min_duration_off=Uniform(0.0, 1.0))
        self.clustering = _FakeVBx()
        self.instantiate(
            {
                "segmentation": {"min_duration_off": 0.0},
                "clustering": {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8},
            }
        )

    def to(self, device):
        return self


def _fake_from_pretrained(checkpoint, token=None):
    return _FakeSpeakerDiarization()


def _build(**kwargs):
    with patch("run_harness.Pipeline") as mock_cls:
        mock_cls.from_pretrained.side_effect = _fake_from_pretrained
        return build_pipeline(CHECKPOINT, token=None, **kwargs)


def _seg(value):
    return {"min_duration_off": value}


def _expected_final_key(extra=()):
    """The final-key formula written out independently of harness.cache_id."""
    parts = [
        "final-v1",
        f"checkpoint={CHECKPOINT}",
        "segmentation_batch_size=32",
        "embedding_batch_size=32",
        "clustering=VBxClustering",
        "Fa=0.07",
        "Fb=0.8",
        "threshold=0.6",
        *extra,
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


# --- 1. different overridden values -> different final keys ----------------


def test_two_overridden_values_give_two_different_final_keys():
    a = _build(segmentation_params=_seg(0.3))
    b = _build(segmentation_params=_seg(0.5))
    assert a.segmentation.min_duration_off == 0.3  # reached the pipeline
    assert pipeline_config_id(CHECKPOINT, a) != pipeline_config_id(CHECKPOINT, b)


# --- 2. no override -> today's key -----------------------------------------


def test_no_override_gives_exactly_todays_final_key():
    pipeline = _build()
    key = pipeline_config_id(CHECKPOINT, pipeline)
    assert key == BASELINE_FINAL_KEY
    assert key == _expected_final_key()


def test_explicit_shipped_value_gives_todays_final_key():
    pipeline = _build(segmentation_params=_seg(0.0))
    assert pipeline_config_id(CHECKPOINT, pipeline) == BASELINE_FINAL_KEY


def test_no_override_assertion_can_fail_if_component_were_always_added():
    """Teeth for the two tests above: a key that always carried the component
    is not equal to the baseline key, so those assertions discriminate."""
    always = _expected_final_key(extra=["min_duration_off=0.0"])
    assert always != BASELINE_FINAL_KEY


# --- 3. unknown name fails loudly, before any scoring ----------------------


def _argv(tmp_path, *extra):
    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)
    return [
        "--data-root", str(data_root),
        "--per-file-csv", str(tmp_path / "per_file.csv"),
        "--summary", str(tmp_path / "summary.json"),
        "--runs-dir", str(tmp_path / "runs"),
        "--device", "cpu",
        *extra,
    ]


def test_unknown_segmentation_param_name_raises_before_any_scoring(tmp_path):
    with patch("run_harness.Pipeline") as mock_cls, patch(
        "run_harness.run_harness"
    ) as mock_run:
        mock_cls.from_pretrained.side_effect = _fake_from_pretrained
        with pytest.raises(ValueError, match="no_such_param"):
            main(_argv(tmp_path, "--segmentation-param", "no_such_param=1"))
    mock_run.assert_not_called()


def test_malformed_segmentation_param_raises_before_models_load(tmp_path):
    with patch("run_harness.Pipeline") as mock_cls, patch("run_harness.run_harness"):
        mock_cls.from_pretrained.side_effect = _fake_from_pretrained
        with pytest.raises(ValueError):
            main(_argv(tmp_path, "--segmentation-param", "min_duration_off"))
    mock_cls.from_pretrained.assert_not_called()


def test_segmentation_param_parsing_preserves_types():
    assert run_harness._parse_segmentation_params(["min_duration_off=0.5"]) == {
        "min_duration_off": 0.5
    }


def test_segmentation_param_reaches_pipeline_and_manifest_kwargs(tmp_path):
    with patch("run_harness.Pipeline") as mock_cls, patch(
        "run_harness.run_harness", return_value={"der": 0.0}
    ) as mock_run:
        mock_cls.from_pretrained.side_effect = _fake_from_pretrained
        main(_argv(tmp_path, "--segmentation-param", "min_duration_off=0.5"))
    kwargs = mock_run.call_args.kwargs
    assert kwargs["min_duration_off"] == 0.5
    assert kwargs["pipeline_config_id"] != BASELINE_FINAL_KEY


def test_no_flag_records_shipped_value_and_baseline_key(tmp_path):
    with patch("run_harness.Pipeline") as mock_cls, patch(
        "run_harness.run_harness", return_value={"der": 0.0}
    ) as mock_run:
        mock_cls.from_pretrained.side_effect = _fake_from_pretrained
        main(_argv(tmp_path))
    kwargs = mock_run.call_args.kwargs
    assert kwargs["min_duration_off"] == 0.0
    assert kwargs["pipeline_config_id"] == BASELINE_FINAL_KEY


# --- 4. intermediate key unaffected ----------------------------------------


def test_intermediate_key_identical_with_and_without_override():
    base = _build()
    over = _build(segmentation_params=_seg(0.5))
    assert intermediate_config_id(CHECKPOINT, base) == intermediate_config_id(
        CHECKPOINT, over
    )
    assert pipeline_config_id(CHECKPOINT, base) != pipeline_config_id(
        CHECKPOINT, over
    )
