"""Run manifest: records the exact config a harness run used, alongside the
existing per-file CSV / summary JSON from harness/reporter.py. See the
run-manifest ticket in Obsidian-Diarisation/Tickets/Open/run-manifest.md,
extended by T1 (oracle/ceiling analysis batch: condition/refinement_strategy
controlled vocabularies, corpus/mic_condition/git_commit required fields,
counts_toward_results defaulting to False).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union

REQUIRED_RUN_CONFIG_FIELDS = (
    "pipeline_config_id",
    "segmentation_source_id",
    "clustering_model",
    "extra_pipeline_steps",
    "split",
    "condition",
    "der_collar",
    "der_skip_overlap",
    "refinement_strategy",
    "corpus",
    "mic_condition",
    "git_commit",
)

# T1: condition identifies which of the four oracle/ceiling-analysis
# conditions a run belongs to; T5's cross-condition deltas key on this field
# alone, never on --notes, so it must be a closed vocabulary.
VALID_CONDITIONS = (
    "baseline",
    "oracle_segmentation",
    "oracle_assignment",
    "nearest_centroid",
)

# T1: refinement_strategy identifies which post-clustering refinement a run
# used. "oracle" is accepted here even though T4 (a separate ticket) hasn't
# implemented that strategy yet, so this vocabulary doesn't need revisiting
# when T4 lands.
VALID_REFINEMENT_STRATEGIES = ("identity", "oracle", "nearest_centroid")


class RunManifestError(ValueError):
    """Raised when a manifest can't be written: missing required run_config
    fields, an invalid controlled-vocabulary value, or a run_id collision
    that would silently overwrite an existing manifest."""


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_id(run_config: Dict[str, Any], timestamp: str) -> str:
    config_hash = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    return f"{timestamp}-{config_hash}"


def write_manifest(
    run_config: Dict[str, Any],
    summary: Dict[str, Any],
    runs_dir: Union[str, Path],
    duration_seconds: Any = None,
) -> Path:
    missing = [field for field in REQUIRED_RUN_CONFIG_FIELDS if field not in run_config]
    if missing:
        raise RunManifestError(
            f"run_config is missing required field(s): {', '.join(missing)}"
        )

    if run_config["condition"] not in VALID_CONDITIONS:
        raise RunManifestError(
            f"run_config['condition'] must be one of {VALID_CONDITIONS}, "
            f"got {run_config['condition']!r}"
        )

    if run_config["refinement_strategy"] not in VALID_REFINEMENT_STRATEGIES:
        raise RunManifestError(
            f"run_config['refinement_strategy'] must be one of "
            f"{VALID_REFINEMENT_STRATEGIES}, got {run_config['refinement_strategy']!r}"
        )

    run_config = {**run_config, "counts_toward_results": run_config.get("counts_toward_results", False)}

    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _utc_timestamp()
    run_id = _run_id(run_config, timestamp)
    manifest_path = runs_dir / f"{run_id}.json"

    if manifest_path.exists():
        raise RunManifestError(
            f"manifest {manifest_path} already exists for run_id {run_id!r} -- "
            "refusing to overwrite an existing manifest"
        )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config,
        "summary": summary,
        "duration_seconds": duration_seconds,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
