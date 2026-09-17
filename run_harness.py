"""Orchestrator: adapter -> runner -> scorer -> reporter. See
TICKET-08-orchestrator-integration.md, extended by T1/T2 (oracle/ceiling
analysis batch).

Constructs the four corpus-level metric accumulators (der, overlap_der,
der_overlap_assigned, jer) exactly once per run and threads them through
every score() call, per the scorer's resolved contract. Writes both output
files only after every file has been scored -- a failure partway through
(e.g. OracleSegmentation's still-stubbed NotImplementedError) never leaves a
partial/misleading CSV or summary behind.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from harness.config import HarnessConfig
from harness.datasets import AMIDatasetAdapter
from harness.refinement import get_refinement_strategy, make_oracle_strategy
from harness.reporter import write_report
from harness.run_manifest import write_manifest
from harness.runner import Runner
from harness.scorer import score


def _current_git_commit() -> str:
    """Short git commit hash of the working tree HEAD, recorded on every
    manifest so a run's exact code state is reconstructable later."""
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent
    ).decode().strip()


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
    notes: str = "",
    run_condition: str = "baseline",
    refinement_strategy: str = "identity",
    oracle_scope: Optional[str] = None,
    corpus: str = "AMI",
    counts_toward_results: bool = False,
) -> Dict[str, float]:
    der = DiarizationErrorRate(collar=config.der_collar, skip_overlap=config.der_skip_overlap)
    overlap_der = DiarizationErrorRate(
        collar=config.der_collar, skip_overlap=config.der_skip_overlap
    )
    der_overlap_assigned = DiarizationErrorRate(
        collar=config.der_collar, skip_overlap=config.der_skip_overlap
    )
    jer = JaccardErrorRate(collar=config.der_collar, skip_overlap=config.der_skip_overlap)

    # "oracle" needs the current file's own reference Annotation, which the
    # fixed 5-arg refinement interface (embeddings, hard_clusters,
    # soft_clusters, centroids, segmentations) has no slot for -- so unlike
    # identity/nearest_centroid (set once, shared across every file),
    # oracle's pipeline.refinement is rebuilt per file inside the loop
    # below, right before that file is run. See T4's Implementation Notes.
    if refinement_strategy != "oracle":
        pipeline.refinement = get_refinement_strategy(refinement_strategy)

    refinement_id = (
        f"{refinement_strategy}:{oracle_scope or 'all_pairs'}"
        if refinement_strategy == "oracle"
        else refinement_strategy
    )
    runner = Runner(
        pipeline,
        pipeline_config_id,
        segmentation_source,
        config.cache_dir,
        refinement_id=refinement_id,
    )
    adapter = AMIDatasetAdapter(config)

    run_started_at = time.monotonic()
    rows = []
    for uri, reference, uem in adapter:
        if refinement_strategy == "oracle":
            pipeline.refinement = make_oracle_strategy(
                reference, oracle_scope=oracle_scope or "all_pairs"
            )
        file = {"uri": uri, "audio": str(_audio_path(config, uri))}
        hypothesis = runner.run(file)
        row = score(reference, hypothesis, uem, der, overlap_der, der_overlap_assigned, jer)
        row["uri"] = uri
        rows.append(row)

    duration_seconds = time.monotonic() - run_started_at

    summary = write_report(
        rows, der, overlap_der, der_overlap_assigned, jer, per_file_csv_path, summary_path
    )

    if runs_dir is not None:
        run_config = {
            "pipeline_config_id": pipeline_config_id,
            "segmentation_source_id": segmentation_source.id,
            "clustering_model": clustering_model,
            "extra_pipeline_steps": [],
            "split": config.split,
            "condition": run_condition,
            "der_collar": config.der_collar,
            "der_skip_overlap": config.der_skip_overlap,
            "notes": notes,
            "refinement_strategy": refinement_strategy,
            "oracle_scope": oracle_scope,
            "corpus": corpus,
            "mic_condition": config.condition,
            "git_commit": _current_git_commit(),
            "counts_toward_results": counts_toward_results,
        }
        write_manifest(run_config, summary, runs_dir, duration_seconds=duration_seconds)

    return summary


from pyannote.audio import Pipeline


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, float]:
    import argparse

    from pyannote.database.util import load_rttm

    from harness.segmentation import BaselineSegmentation, OracleSegmentation

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
        "--notes", default="",
        help="Free-text description of the experiment (e.g. 'oracle segmentation', "
        "'dbscan clustering', 'random forest overlap detector'), recorded in the run "
        "manifest and shown in runs/comparison.csv.",
    )
    parser.add_argument(
        "--device", default=None,
        help="torch device (e.g. cpu, cuda, cuda:0). Default: cuda if available, else cpu.",
    )
    parser.add_argument(
        "--run-condition", default="baseline",
        choices=["baseline", "oracle_segmentation", "oracle_assignment", "nearest_centroid"],
        help="Which oracle/ceiling-analysis condition this run belongs to (not to be "
        "confused with --condition, the AMI mic condition). Recorded in the manifest's "
        "'condition' field; T5's cross-condition deltas key on this.",
    )
    parser.add_argument(
        "--refinement-strategy", default="identity",
        choices=["identity", "oracle", "nearest_centroid"],
        help="Post-clustering refinement strategy (see harness/refinement.py).",
    )
    parser.add_argument(
        "--oracle-scope", default="all_pairs",
        choices=["all_pairs", "overlap_degraded"],
        help="Scope of pairs the 'oracle' refinement strategy refines (T4). "
        "'overlap_degraded' (primary): only pairs whose support is predominantly "
        "coincident with another speaker. 'all_pairs' (secondary): every pair, the "
        "full post-clustering ceiling. Ignored unless --refinement-strategy oracle.",
    )
    parser.add_argument(
        "--oracle-rttm", default=None,
        help="Path to a reference RTTM (e.g. an only_words RTTM) used to build ground-truth "
        "segmentation when --run-condition oracle_segmentation is selected (T3). Required "
        "in that case; ignored otherwise.",
    )
    parser.add_argument("--corpus", default="AMI")
    parser.add_argument(
        "--counts-toward-results", action="store_true",
        help="Mark this run as counting toward the oracle/ceiling-analysis results table "
        "(T5). Defaults to False for exploratory/debugging runs.",
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

    if args.run_condition == "oracle_segmentation":
        if not args.oracle_rttm:
            parser.error("--run-condition oracle_segmentation requires --oracle-rttm")
        segmentation_source = OracleSegmentation(
            reference_lookup=load_rttm(args.oracle_rttm)
        )
    else:
        segmentation_source = BaselineSegmentation()

    summary = run_harness(
        config,
        pipeline,
        pipeline_config_id=checkpoint,
        segmentation_source=segmentation_source,
        per_file_csv_path=args.per_file_csv,
        summary_path=args.summary,
        clustering_model=clustering_model,
        runs_dir=args.runs_dir,
        notes=args.notes,
        run_condition=args.run_condition,
        refinement_strategy=args.refinement_strategy,
        oracle_scope=args.oracle_scope,
        corpus=args.corpus,
        counts_toward_results=args.counts_toward_results,
    )
    print(summary)
    return summary


if __name__ == "__main__":
    main()
