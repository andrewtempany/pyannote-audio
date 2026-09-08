"""Run manifest: records the exact config a harness run used, alongside the
existing per-file CSV / summary JSON from harness/reporter.py. See the
run-manifest ticket in Obsidian-Diarisation/Tickets/Open/run-manifest.md.
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
)


class RunManifestError(ValueError):
    """Raised when a manifest can't be written: missing required run_config
    fields, or a run_id collision that would silently overwrite an existing
    manifest."""


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
) -> Path:
    missing = [field for field in REQUIRED_RUN_CONFIG_FIELDS if field not in run_config]
    if missing:
        raise RunManifestError(
            f"run_config is missing required field(s): {', '.join(missing)}"
        )

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
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
