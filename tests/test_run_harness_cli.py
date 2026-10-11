"""Tests for the gpu-support ticket: device resolution in run_harness.py's
CLI entry point.

`_resolve_device` doesn't exist yet at the time these tests are written --
the first run of this suite is expected to fail on import.

No real GPU is required -- torch.cuda.is_available is mocked so these tests
run identically on CPU-only and GPU-equipped machines.
"""

from unittest.mock import MagicMock, patch

import torch

from run_harness import _resolve_device, main


def test_device_flag_defaults_to_cuda_if_available():
    with patch("torch.cuda.is_available", return_value=True):
        assert _resolve_device(None) == torch.device("cuda")


def test_device_flag_defaults_to_cpu_if_unavailable():
    with patch("torch.cuda.is_available", return_value=False):
        assert _resolve_device(None) == torch.device("cpu")


def test_explicit_device_flag_overrides_autodetect_to_cpu():
    with patch("torch.cuda.is_available", return_value=True):
        assert _resolve_device("cpu") == torch.device("cpu")


def test_explicit_device_flag_overrides_autodetect_to_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        assert _resolve_device("cuda") == torch.device("cuda")


def test_explicit_device_flag_accepts_indexed_cuda_device():
    with patch("torch.cuda.is_available", return_value=True):
        assert _resolve_device("cuda:0") == torch.device("cuda:0")


def test_main_calls_pipeline_to_with_resolved_device(tmp_path):
    """Drives run_harness.py's actual CLI entry point (main()), not just
    _resolve_device in isolation -- confirms the resolved device is really
    threaded through to pipeline.to() before scoring starts."""
    fake_pipeline = MagicMock()
    fake_pipeline.to.return_value = fake_pipeline

    data_root = tmp_path / "data"
    data_root.mkdir()

    argv = [
        "--data-root", str(data_root),
        "--per-file-csv", str(tmp_path / "per_file.csv"),
        "--summary", str(tmp_path / "summary.json"),
        "--runs-dir", str(tmp_path / "runs"),
        "--device", "cpu",
    ]

    with patch("run_harness.Pipeline") as mock_pipeline_cls, \
         patch("run_harness.run_harness", return_value={"der": 0.0}) as mock_run_harness:
        mock_pipeline_cls.from_pretrained.return_value = fake_pipeline
        main(argv)

    fake_pipeline.to.assert_called_once_with(torch.device("cpu"))
    # the *moved* pipeline (fake_pipeline.to's return value) is what gets
    # scored, not the pre-move object
    assert mock_run_harness.call_args.args[1] is fake_pipeline.to.return_value
