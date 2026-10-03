"""Driver and ledger for the clustering-benchmark sweep.

One sweep point = one scored run over the 16 AMI IHM test meetings with one
clustering configuration. This module runs a point through `run_harness.py`
(never an ad hoc scoring path), checks the ticket's hard-fail conditions on
the result, and keeps the ledger files under SWEEP_DIR up to date:

    points.jsonl              one JSON record per point (source of truth)
    SWEEP_INDEX.md            regenerated from points.jsonl after every point
    all_points_per_file.csv   long-format per-file rows, plus point_id/method/half

Half-level DER is sum(MD + FA + Conf) / sum(reference speech) over the
meetings in that half, with reference speech recovered per file as
(MD + FA + Conf) / der. It is NOT the mean of per-file DER. On all 16 meetings
it must reproduce the run's summary DER to 1e-9, and `half_metrics` checks
that on every point.

Usage:
    python -m harness.clustering_benchmark baseline
    python -m harness.clustering_benchmark run AGG-1a-01 WS1 agglomerative \\
        method=centroid min_cluster_size=12 threshold=0.3
    python -m harness.clustering_benchmark index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SWEEP_DIR = REPO / "runs" / "sweeps" / "clustering-benchmark"
RUNS_DIR = REPO / "runs"
DATA_ROOT = REPO.parent / "AMI-diarization-setup" / "harness-data"
PYTHON = REPO / "diarisation-env" / "Scripts" / "python.exe"

BASELINE_RUN_ID = "20260925T101811Z-db977741"
BASELINE_CSV = REPO / "per_file_baseline_batchkey_rebuild.csv"

# Fixed split (ticket §1): meetings ending in a/c tune, b/d report. Each
# recording series appears twice in each half.
TUNE_SUFFIXES = ("a", "c")

COMPONENTS = ("missed_detection", "false_alarm", "confusion")
REGIONS = ("region_t_and_d", "region_t_minus_d", "region_d_minus_t")
EXPECTED_TALLY = "intermediate cache: 16 full hit(s), 0 partial, 0 miss(es), 0 write(s)"
METHOD_CLASS = {
    "agglomerative": "AgglomerativeClustering",
    "hdbscan": "HDBSCANClustering",
    "spectral": "SpectralEigengapClustering",
    "vbx": "VBxClustering",
}


class HardFail(RuntimeError):
    """A ticket §7 hard-fail condition. The point is recorded with the failure
    and the caller must stop that workstream."""


def half_of(uri: str) -> str:
    return "tune" if uri.endswith(TUNE_SUFFIXES) else "report"


def half_metrics(per_file: pd.DataFrame) -> dict:
    """Corpus-level aggregates over the rows given (any subset of meetings)."""
    errors = per_file[list(COMPONENTS)].sum(axis=1)
    reference = errors / per_file["der"]
    md, fa, conf = (float(per_file[c].sum()) for c in COMPONENTS)
    total_error = md + fa + conf
    signed = per_file["count_hyp"] - per_file["count_ref"]
    return {
        "n": int(len(per_file)),
        "der": total_error / float(reference.sum()),
        "reference_s": float(reference.sum()),
        "md": md,
        "fa": fa,
        "conf": conf,
        "md_share": md / total_error,
        "fa_share": fa / total_error,
        "conf_share": conf / total_error,
        **{r: float(per_file[r].sum()) for r in REGIONS},
        "count_mae": float(per_file["count_error"].mean()),
        "exact": int((per_file["count_error"] == 0).sum()),
        "count_bias": float(signed.mean()),
        "over": int((signed > 0).sum()),
        "under": int((signed < 0).sum()),
    }


def point_metrics(per_file_csv: Path, summary: dict) -> dict:
    per_file = pd.read_csv(per_file_csv)
    if len(per_file) != 16:
        raise HardFail(f"{per_file_csv}: expected 16 meetings, got {len(per_file)}")
    halves = per_file["uri"].map(half_of)
    out = {
        "all": half_metrics(per_file),
        "tune": half_metrics(per_file[halves == "tune"]),
        "report": half_metrics(per_file[halves == "report"]),
        # Corpus-level ratios exist only for all 16 (ticket §1); halves of
        # these are per-file only and live in all_points_per_file.csv.
        "der_overlap_system": summary["der_overlap_system"],
        "der_overlap_assigned": summary["der_overlap_assigned"],
        "jer": summary["jer"],
    }
    drift = abs(out["all"]["der"] - summary["der"])
    if drift > 1e-9:
        raise HardFail(f"half-level DER does not reproduce summary DER (drift {drift:g})")
    return out


# -- ledger -----------------------------------------------------------------


def _ledger_path() -> Path:
    return SWEEP_DIR / "points.jsonl"


def load_points() -> list[dict]:
    path = _ledger_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_point(record: dict) -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with open(_ledger_path(), "a") as handle:
        handle.write(json.dumps(record) + "\n")


def _append_per_file(point_id: str, method: str, per_file_csv: Path) -> None:
    per_file = pd.read_csv(per_file_csv)
    per_file.insert(0, "half", per_file["uri"].map(half_of))
    per_file.insert(0, "method", method)
    per_file.insert(0, "point_id", point_id)
    target = SWEEP_DIR / "all_points_per_file.csv"
    if target.exists():
        existing = pd.read_csv(target)
        existing = existing[existing["point_id"] != point_id]
        per_file = pd.concat([existing, per_file], ignore_index=True)
    per_file.to_csv(target, index=False)


def _fmt(value, spec=".4f"):
    return "" if value is None else format(value, spec)


def write_index() -> Path:
    points = load_points()
    lines = ["# Clustering benchmark: sweep index", ""]
    lines.append(
        "Main columns are all 16 meetings. DER is a percentage; seconds are summed. "
        "Tune half = meetings ending a/c, report half = b/d. Regenerated from "
        "`points.jsonl` by `python -m harness.clustering_benchmark index`; do not hand-edit."
    )
    lines.append("")
    reference = SWEEP_DIR / "baseline_reference.json"
    if reference.exists():
        base = json.loads(reference.read_text())
        lines += [
            f"## VBx baseline reference (run `{base['run_id']}`, not re-run)", "",
            "| half | DER | MD s | FA s | Conf s | Conf % of error | count MAE | signed bias | exact | over / under |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for half in ("tune", "report", "all"):
            h = base["metrics"][half]
            lines.append(
                f"| {half} | {100*h['der']:.2f} | {h['md']:.1f} | {h['fa']:.1f} | {h['conf']:.1f} | "
                f"{100*h['conf_share']:.1f} | {h['count_mae']:.3f} | {h['count_bias']:+.3f} | "
                f"{h['exact']}/{h['n']} | {h['over']} / {h['under']} |"
            )
        m = base["metrics"]
        lines += ["", f"All 16: der_overlap_system {m['der_overlap_system']:.4f}, "
                      f"der_overlap_assigned {m['der_overlap_assigned']:.4f}, JER {m['jer']:.4f}.", "",
                  "## Sweep points", ""]
    header = (
        "| point_id | WS | method | params | run_id | DER tune | DER report | DER all | "
        "MD s | FA s | Conf s | Conf % of error | der_overlap_system | der_overlap_assigned | "
        "JER | overlap caught / missed / invented s | count MAE | signed count bias | "
        "exact/16 | outlier rate (HDBSCAN) | est. k range (spectral) | notes |"
    )
    lines += [header, "|" + "---|" * 22]
    for p in points:
        if "metrics" not in p:
            lines.append(
                f"| {p['point_id']} | {p['ws']} | {p['method']} | {p['params_text']} | "
                f"{p.get('run_id', '')} |" + " |" * 16 + f" FAILED: {p.get('failure', '')} |"
            )
            continue
        m = p["metrics"]
        a = m["all"]
        lines.append(
            "| " + " | ".join([
                p["point_id"], p["ws"], p["method"], p["params_text"], p.get("run_id", ""),
                _fmt(100 * m["tune"]["der"], ".2f"),
                _fmt(100 * m["report"]["der"], ".2f"),
                _fmt(100 * a["der"], ".2f"),
                _fmt(a["md"], ".1f"), _fmt(a["fa"], ".1f"), _fmt(a["conf"], ".1f"),
                _fmt(100 * a["conf_share"], ".1f"),
                _fmt(m["der_overlap_system"]), _fmt(m["der_overlap_assigned"]), _fmt(m["jer"]),
                f"{a['region_t_and_d']:.0f} / {a['region_t_minus_d']:.0f} / {a['region_d_minus_t']:.0f}",
                _fmt(a["count_mae"], ".3f"),
                _fmt(a["count_bias"], "+.3f"),
                str(a["exact"]),
                _fmt(p.get("outlier_rate"), ".3f"),
                p.get("k_range", ""),
                p.get("notes", ""),
            ]) + " |"
        )
    path = SWEEP_DIR / "SWEEP_INDEX.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# -- running a point ----------------------------------------------------------


def _parse_params(raw: list[str]) -> dict[str, str]:
    params = {}
    for item in raw:
        name, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"param must be name=value, got {item!r}")
        params[name] = value
    return params


def _described_params(description: str) -> tuple[str, dict[str, str]]:
    match = re.fullmatch(r"(\w+)\((.*)\)", description.strip())
    if not match:
        raise HardFail(f"cannot parse clustering_config {description!r}")
    pairs = dict(re.findall(r"(\w+)=('[^']*'|[^,]+)", match.group(2)))
    return match.group(1), {k: v.strip().strip("'") for k, v in pairs.items()}


def _same_value(requested: str, recorded: str) -> bool:
    try:
        return float(requested) == float(recorded)
    except ValueError:
        return requested == recorded


def _diagnostic_summary(method: str, diag_dir: Path) -> dict:
    path = diag_dir / "diagnostics.json"
    if not path.exists():
        return {}
    diagnostics = json.loads(path.read_text())
    out: dict = {"diagnostic_files": len(diagnostics)}
    if not diagnostics:
        return out
    if method == "hdbscan":
        outliers = sum(d["num_outliers"] for d in diagnostics.values())
        train = sum(d["num_train"] for d in diagnostics.values())
        out["outlier_rate"] = outliers / train
    if method == "spectral":
        ks = [d["k"] for d in diagnostics.values()]
        out["k_range"] = f"{min(ks)}-{max(ks)}"
        out["max_cluster_seconds"] = max(d["cluster_seconds"] for d in diagnostics.values())
    return out


def run_point(point_id: str, ws: str, method: str, params: dict[str, str],
              cache_dir: Path | None = None, note: str = "", force: bool = False) -> dict:
    if not force and any(p["point_id"] == point_id for p in load_points()):
        raise SystemExit(f"{point_id} already recorded; pass --force to rerun")

    out_dir = SWEEP_DIR / method / point_id
    out_dir.mkdir(parents=True, exist_ok=True)
    params_text = ", ".join(f"{k}={v}" for k, v in params.items())

    command = [
        str(PYTHON), "run_harness.py",
        "--data-root", str(DATA_ROOT), "--condition", "IHM", "--split", "test",
        "--cache-dir", str(cache_dir or REPO / ".harness_cache"),
        "--per-file-csv", str(out_dir / "per_file.csv"),
        "--summary", str(out_dir / "summary.json"),
        "--runs-dir", str(RUNS_DIR),
        "--notes", f"clustering-benchmark {ws} {point_id}: {params_text}" + (f" ({note})" if note else ""),
        "--run-condition", f"clustering_{method}",
        "--refinement-strategy", "identity",
        "--clustering-model", method,
        "--clustering-diagnostics-dir", str(out_dir),
    ]
    for name, value in params.items():
        command += ["--clustering-param", f"{name}={value}"]

    before = set(RUNS_DIR.glob("*.json"))
    started = time.monotonic()
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    (out_dir / "run.log").write_text(result.stdout + "\n--- stderr ---\n" + result.stderr,
                                     encoding="utf-8")

    record = {"point_id": point_id, "ws": ws, "method": method, "params": params,
              "params_text": params_text, "notes": note, "wall_seconds": round(elapsed, 1)}
    try:
        if result.returncode != 0:
            raise HardFail(f"run_harness exited {result.returncode}; see {out_dir / 'run.log'}")

        new_manifests = sorted(set(RUNS_DIR.glob("*.json")) - before)
        if len(new_manifests) != 1:
            raise HardFail(f"expected one new manifest, found {len(new_manifests)}")
        manifest = json.loads(new_manifests[0].read_text())
        record["run_id"] = manifest["run_id"]
        record["final_key"] = manifest["run_config"]["pipeline_config_id"]

        # §7.2: every point after setup must be fully served from the
        # intermediate tier. A miss or write means the GPU recomputed.
        tally = next((l for l in result.stdout.splitlines() if l.startswith("intermediate cache:")), "")
        record["cache_tally"] = tally
        if not tally.startswith(EXPECTED_TALLY):
            raise HardFail(f"intermediate cache tally was {tally!r}")

        # §7.3: the manifest must name the requested class and values.
        description = manifest["run_config"]["clustering_config"]
        record["clustering_config"] = description
        class_name, recorded = _described_params(description)
        if class_name != METHOD_CLASS[method]:
            raise HardFail(f"manifest names {class_name}, requested {METHOD_CLASS[method]}")
        for name, value in params.items():
            if name not in recorded or not _same_value(value, recorded[name]):
                raise HardFail(f"manifest has {name}={recorded.get(name)!r}, requested {value!r}")

        # §7.1: a different configuration must never share a final key.
        for other in load_points():
            if other.get("final_key") == record["final_key"] and other["params"] != params \
                    and other["point_id"] != point_id:
                raise HardFail(f"final cache key collides with {other['point_id']}")

        summary = json.loads((out_dir / "summary.json").read_text())
        record["metrics"] = point_metrics(out_dir / "per_file.csv", summary)
        record.update(_diagnostic_summary(method, out_dir))
        record["per_file_sha256"] = hashlib.sha256((out_dir / "per_file.csv").read_bytes()).hexdigest()

        # §7.4: three consecutive byte-identical per-file CSVs look exactly
        # like a stuck setting.
        previous = [p for p in load_points() if p["point_id"] != point_id][-2:]
        if len(previous) == 2 and all(p.get("per_file_sha256") == record["per_file_sha256"] for p in previous):
            raise HardFail("three consecutive points produced byte-identical per-file CSVs")
    except HardFail as failure:
        record["failure"] = str(failure)
        _store(record, out_dir, method, force)
        raise

    _store(record, out_dir, method, force)
    return record


def _store(record: dict, out_dir: Path, method: str, force: bool) -> None:
    if force:
        kept = [p for p in load_points() if p["point_id"] != record["point_id"]]
        _ledger_path().write_text("".join(json.dumps(p) + "\n" for p in kept))
    _append_point(record)
    if "metrics" in record:
        _append_per_file(record["point_id"], method, out_dir / "per_file.csv")
    write_index()


# -- baseline -----------------------------------------------------------------


def baseline() -> dict:
    manifest = json.loads((RUNS_DIR / f"{BASELINE_RUN_ID}.json").read_text())
    metrics = point_metrics(BASELINE_CSV, manifest["summary"])
    out = SWEEP_DIR / "baseline_reference.json"
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_id": BASELINE_RUN_ID, "metrics": metrics}, indent=1))
    _append_per_file("VBX-baseline", "vbx", BASELINE_CSV)
    return metrics


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("baseline")
    sub.add_parser("index")
    run = sub.add_parser("run")
    run.add_argument("point_id")
    run.add_argument("ws")
    run.add_argument("method", choices=sorted(METHOD_CLASS))
    run.add_argument("params", nargs="*")
    run.add_argument("--cache-dir", type=Path, default=None)
    run.add_argument("--note", default="")
    run.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "baseline":
        print(json.dumps(baseline(), indent=1))
    elif args.command == "index":
        print(write_index())
    else:
        try:
            record = run_point(args.point_id, args.ws, args.method, _parse_params(args.params),
                               cache_dir=args.cache_dir, note=args.note, force=args.force)
        except HardFail as failure:
            print(f"HARD FAIL {args.point_id}: {failure}")
            sys.exit(2)
        m = record["metrics"]
        print(f"{record['point_id']} {record['run_id']} DER all {100*m['all']['der']:.2f} "
              f"tune {100*m['tune']['der']:.2f} report {100*m['report']['der']:.2f} "
              f"MAE {m['all']['count_mae']:.3f} exact {m['all']['exact']}/16 "
              f"({record['wall_seconds']}s) | {record['cache_tally']}")


if __name__ == "__main__":
    main()
