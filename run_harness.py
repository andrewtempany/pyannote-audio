"""Orchestrator: adapter -> runner -> scorer -> reporter. See
TICKET-08-orchestrator-integration.md.

Constructs the three corpus-level metric accumulators (der, overlap_der,
jer) exactly once per run and threads them through every score() call, per
TICKET-05's resolved contract. Writes both output files only after every
file has been scored -- a failure partway through (e.g. OracleSegmentation's
still-stubbed NotImplementedError) never leaves a partial/misleading CSV or
summary behind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from harness.config import HarnessConfig
from harness.datasets import AMIDatasetAdapter
from harness.reporter import write_report
from harness.run_manifest import write_manifest
from harness.runner import Runner
from harness.scorer import score


def _resolve_device(requested: Optional[str]) -> torch.device:
    """--device flag -> torch.device. `requested=None` auto-detects: cuda if
    available, else cpu. An explicit value always wins over auto-detection."""
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _audio_path(config: HarnessConfig, uri: str) -> Path:
    """uri -> audio file path convention for this harness (a decision this
    ticket makes, deferred by TICKET-03's dataset adapter, which never
    touches audio at all): {data_root}/{condition}/audio/{uri}.wav."""
    return config.data_root / config.condition / "audio" / f"{uri}.wav"


def run_harness(
    config: HarnessConfig,
    pipeline: Any,
    pipeline_config_id: str,
    segmentation_source: Any,
    per_file_csv_path: Union[str, Path],
    summary_path: Union[str, Path],
    clustering_model: Optional[str] = None,
    runs_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    der = DiarizationErrorRate(collar=config.der_collar, skip_overlap=config.der_skip_overlap)
    overlap_der = DiarizationErrorRate(
        collar=config.der_collar, skip_overlap=config.der_skip_overlap
    )
    jer = JaccardErrorRate(collar=config.der_collar, skip_overlap=config.der_skip_overlap)

    runner = Runner(pipeline, pipeline_config_id, segmentation_source, config.cache_dir)
    adapter = AMIDatasetAdapter(config)

    rows = []
    for uri, reference, uem in adapter:
        file = {"uri": uri, "audio": str(_audio_path(config, uri))}
        hypothesis = runner.run(file)
        row = score(reference, hypothesis, uem, der, overlap_der, jer)
        row["uri"] = uri
        rows.append(row)

    summary = write_report(rows, der, overlap_der, jer, per_file_csv_path, summary_path)

    if runs_dir is not None:
        run_config = {
            "pipeline_config_id": pipeline_config_id,
            "segmentation_source_id": segmentation_source.id,
            "clustering_model": clustering_model,
            "extra_pipeline_steps": [],
            "split": config.split,
            "condition": config.condition,
            "der_collar": config.der_collar,
            "der_skip_overlap": config.der_skip_overlap,
        }
        write_manifest(run_config, summary, runs_dir)

    return summary


from pyannote.audio import Pipeline


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, float]:
    import argparse

    from harness.segmentation import BaselineSegmentation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--condition", default="IHM")
    parser.add_argument("--cache-dir", default=".harness_cache")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--per-file-csv", default="per_file.csv")
    parser.add_argument("--summary", default="summary.json")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--device", default=None,
        help="torch device (e.g. cpu, cuda, cuda:0). Default: cuda if available, else cpu.",
    )
    args = parser.parse_args(argv)

    config = HarnessConfig.load(
        data_root=args.data_root,
        split=args.split,
        condition=args.condition,
        cache_dir=args.cache_dir,
        dotenv_path=args.dotenv,
    )

    checkpoint = "pyannote/speaker-diarization-community-1"
    pipeline = Pipeline.from_pretrained(checkpoint, token=config.hf_token)
    pipeline = pipeline.to(_resolve_device(args.device))

    # No real clustering-model selection exists yet (see the run-manifest
    # ticket) -- hardcoded the same way `checkpoint` is above, until a
    # future ticket implements clustering-model selection.
    clustering_model = "pyannote-default"

    summary = run_harness(
        config,
        pipeline,
        pipeline_config_id=checkpoint,
        segmentation_source=BaselineSegmentation(),
        per_file_csv_path=args.per_file_csv,
        summary_path=args.summary,
        clustering_model=clustering_model,
        runs_dir=args.runs_dir,
    )
    print(summary)
    return summary


if __name__ == "__main__":
    main()
