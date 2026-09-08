"""Aggregate a directory of run manifests (harness/run_manifest.py) into a
flat comparison table, one row per run. Generated on demand, not
hand-maintained. See the run-manifest ticket in
Obsidian-Diarisation/Tickets/Open/run-manifest.md.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Union


def _flatten_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    row = {"run_id": manifest["run_id"], "created_at": manifest["created_at"]}
    row.update(manifest["run_config"])
    row.update(manifest["summary"])
    return row


def aggregate_runs(runs_dir: Union[str, Path], output_csv_path: Union[str, Path]) -> Path:
    runs_dir = Path(runs_dir)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_paths = sorted(runs_dir.glob("*.json"))
    rows = [_flatten_manifest(json.loads(p.read_text())) for p in manifest_paths]

    if not rows:
        output_csv_path.write_text("")
        return output_csv_path

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return output_csv_path
