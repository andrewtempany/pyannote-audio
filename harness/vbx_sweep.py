"""Resumable driver for the closeout-3 section 2 VBx hyperparameter sweep.

One point = one scored run over the 16 AMI IHM test meetings with one
(Fa, Fb, threshold) setting, run through `run_harness.py` (never an ad hoc
scoring path). Conventions mirror harness/clustering_benchmark.py: a JSONL
ledger (source of truth), long-format per-file rows with a tune/report half
column (tune = meetings ending a/c, report = b/d), and the same hard-fail
checks (intermediate cache must be 16/16 full hits, manifest clustering_config
must echo the request, no two different settings may share a final key).

Outputs under runs/sweeps/vbx-sweep/:
    points.jsonl              one record per completed point
    points.csv                flat summary of every completed point
    all_points_per_file.csv   per-file rows plus point_id/Fa/Fb/threshold/half
    <point_id>/               per-point per_file.csv, summary.json, run.log

Resumable: points already in points.jsonl (with metrics) are skipped.

Usage (from the repo root):
    diarisation-env/Scripts/python.exe -m harness.vbx_sweep --phase fb      # baseline + Fb axis
    diarisation-env/Scripts/python.exe -m harness.vbx_sweep --phase rest    # Fa axis, threshold axis, rest of grid
    diarisation-env/Scripts/python.exe -m harness.vbx_sweep --phase report  # rebuild points.csv only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from harness.clustering_benchmark import (
    EXPECTED_TALLY,
    HardFail,
    _described_params,
    _same_value,
    half_of,
    point_metrics,
)

REPO = Path(__file__).resolve().parent.parent
SWEEP_DIR = REPO / "runs" / "sweeps" / "vbx-sweep"
RUNS_DIR = REPO / "runs"
DATA_ROOT = REPO.parent / "AMI-diarization-setup" / "harness-data"
PYTHON = REPO / "diarisation-env" / "Scripts" / "python.exe"
POINT_TIMEOUT_S = 1800

# Shipped values, and the known baseline DER (run 0fa3b41d / vbx-identity).
SHIPPED = {"Fa": 0.07, "Fb": 0.8, "threshold": 0.6}
BASELINE_DER = 0.17048543579940634
# The shipped key is served from the final-RTTM cache (zero intermediate tally), whose RTTMs are
# millisecond-quantised: ~1e-5 DER vs the recomputed 0fa3b41d value (measured 8.0e-6; per-file max
# 2.1e-5, speaker counts identical). So the tolerance is 5e-5, and the drift is recorded.
BASELINE_TOL = 5e-5
BASELINE_FINAL_KEY = "241ecd865d157bd823106180f9691005b3f6245043a8fadb1c62480b9aedbced"

# Grid. Declared ranges (clustering.py:568-570): Fb U(0.01,15), Fa U(0.01,0.5),
# threshold U(0.5,0.8). Each axis is a regular grid with the point nearest the
# shipped value (in the axis's own spacing) replaced by the shipped value.
FB_VALUES = [0.01, 0.02258, 0.05097, 0.1151, 0.2598, 0.8, 1.324, 2.989, 6.748, 15.0]
FA_VALUES = [0.01, 0.0219, 0.07, 0.1046, 0.2288, 0.5]
THR_VALUES = [0.5, 0.56, 0.6, 0.68, 0.74, 0.8]


def point_id(fa: float, fb: float, thr: float) -> str:
    return f"VBX-Fa{fa:g}-Fb{fb:g}-T{thr:g}"


def phase_points(phase: str) -> list[tuple[float, float, float]]:
    fa0, fb0, thr0 = SHIPPED["Fa"], SHIPPED["Fb"], SHIPPED["threshold"]
    order: list[tuple[float, float, float]] = [(fa0, fb0, thr0)]
    if phase == "fb":
        order += [(fa0, fb, thr0) for fb in FB_VALUES]
    else:
        order += [(fa0, fb, thr0) for fb in FB_VALUES]
        order += [(fa, fb0, thr0) for fa in FA_VALUES]
        order += [(fa0, fb0, thr) for thr in THR_VALUES]
        order += [(fa, fb, thr) for fa in FA_VALUES for fb in FB_VALUES for thr in THR_VALUES]
    seen, out = set(), []
    for p in order:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# -- ledger -----------------------------------------------------------------


def _ledger() -> Path:
    return SWEEP_DIR / "points.jsonl"


def load_points() -> list[dict]:
    if not _ledger().exists():
        return []
    return [json.loads(l) for l in _ledger().read_text(encoding="utf-8").splitlines() if l.strip()]


def _append_per_file(rec: dict, csv: Path) -> None:
    df = pd.read_csv(csv)
    df.insert(0, "half", df["uri"].map(half_of))
    df.insert(0, "threshold", rec["params"]["threshold"])
    df.insert(0, "Fb", rec["params"]["Fb"])
    df.insert(0, "Fa", rec["params"]["Fa"])
    df.insert(0, "point_id", rec["point_id"])
    target = SWEEP_DIR / "all_points_per_file.csv"
    if target.exists():
        old = pd.read_csv(target)
        df = pd.concat([old[old["point_id"] != rec["point_id"]], df], ignore_index=True)
    df.to_csv(target, index=False)


def write_points_csv() -> Path:
    rows = []
    for p in load_points():
        if "metrics" not in p:
            continue
        m = p["metrics"]
        row = {"point_id": p["point_id"], **p["params"], "run_id": p["run_id"],
               "final_key": p["final_key"], "clustering_line": p["clustering_line"],
               "clustering_config": p["clustering_config"], "cache_tally": p["cache_tally"]}
        for half in ("all", "tune", "report"):
            h = m[half]
            pre = "" if half == "all" else f"{half}_"
            row[f"{pre}der"] = h["der"]
            if half == "all":
                row.update({"md_s": h["md"], "fa_s": h["fa"], "conf_s": h["conf"]})
            row[f"{pre}count_mae"] = h["count_mae"]
            row[f"{pre}count_bias"] = h["count_bias"]
            row[f"{pre}exact"] = h["exact"]
        row.update({"der_overlap_system": m["der_overlap_system"],
                    "der_overlap_assigned": m["der_overlap_assigned"], "jer": m["jer"],
                    "region_t_and_d": m["all"]["region_t_and_d"],
                    "region_t_minus_d": m["all"]["region_t_minus_d"],
                    "region_d_minus_t": m["all"]["region_d_minus_t"],
                    "mean_hyp_count": p["mean_hyp_count"], "mean_ref_count": p["mean_ref_count"],
                    "wall_seconds": p["wall_seconds"]})
        rows.append(row)
    path = SWEEP_DIR / "points.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# -- running a point --------------------------------------------------------


def run_point(fa: float, fb: float, thr: float) -> dict:
    pid = point_id(fa, fb, thr)
    params = {"Fa": fa, "Fb": fb, "threshold": thr}
    out_dir = SWEEP_DIR / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    text = f"Fa={fa:g}, Fb={fb:g}, threshold={thr:g}"
    command = [
        str(PYTHON), "run_harness.py",
        "--data-root", str(DATA_ROOT), "--condition", "IHM", "--split", "test",
        "--device", "cuda",
        "--per-file-csv", str(out_dir / "per_file.csv"),
        "--summary", str(out_dir / "summary.json"),
        "--notes", f"closeout-3 s2 VBx sweep {pid}: shipped pipeline with VBx clustering overridden to {text}; "
                   "identity refinement, IHM test, warm shared cache",
        "--refinement-strategy", "identity",
    ]
    for name, value in params.items():
        command += ["--clustering-param", f"{name}={value:g}"]

    before = set(RUNS_DIR.glob("*.json"))
    started = time.monotonic()
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    try:
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", env=env, timeout=POINT_TIMEOUT_S)
        stdout, stderr, rc = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout, stderr, rc = (exc.stdout or ""), (exc.stderr or ""), -999
    elapsed = time.monotonic() - started
    (out_dir / "run.log").write_text(str(stdout) + "\n--- stderr ---\n" + str(stderr), encoding="utf-8")

    rec = {"point_id": pid, "params": params, "params_text": text, "wall_seconds": round(elapsed, 1)}
    try:
        known_crash = next((m for m in ("OverflowError: Python integer", "ValueError: negative dimensions are not allowed")
                            if m in str(stderr) + str(stdout)), None)
        if rc != 0 and known_crash:
            # Finding, not a harness fault: VBx yields >127 clusters and src/.../clustering.py:138
            # writes them into an int8 array. Recorded as a crashed point; src/ is not edited.
            rec["clustering_line"] = next((l for l in str(stdout).splitlines() if l.startswith("clustering:")), "")
            rec["crash"] = [l for l in str(stderr).splitlines() if known_crash in l][-1].strip()
            with open(_ledger(), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            return rec
        if rc != 0:
            raise HardFail(f"run_harness exited {rc}; see {out_dir / 'run.log'}")
        new = sorted(set(RUNS_DIR.glob("*.json")) - before)
        if len(new) != 1:
            raise HardFail(f"expected one new manifest, found {len(new)}")
        manifest = json.loads(new[0].read_text(encoding="utf-8"))
        rec["run_id"] = manifest["run_id"]
        rec["final_key"] = manifest["run_config"]["pipeline_config_id"]
        rec["intermediate_key"] = manifest["run_config"].get("intermediate_config_id")
        rec["clustering_line"] = next((l for l in stdout.splitlines() if l.startswith("clustering:")), "")
        tally = next((l for l in stdout.splitlines() if l.startswith("intermediate cache:")), "")
        rec["cache_tally"] = tally
        zero = "intermediate cache: 0 full hit(s), 0 partial, 0 miss(es), 0 write(s)"
        if tally.startswith(zero) and rec["final_key"] == BASELINE_FINAL_KEY:
            rec["note"] = "served from final-RTTM cache (shipped key)"
        elif not tally.startswith(EXPECTED_TALLY):
            raise HardFail(f"INTERMEDIATE CACHE NOT 16/16 FULL HITS (halts GPU queue): {tally!r}")

        desc = manifest["run_config"]["clustering_config"]
        rec["clustering_config"] = desc
        cls, recorded = _described_params(desc)
        if cls != "VBxClustering":
            raise HardFail(f"manifest names {cls}, requested VBxClustering")
        for name, value in params.items():
            if name not in recorded or not _same_value(f"{value:g}", recorded[name]):
                raise HardFail(f"manifest has {name}={recorded.get(name)!r}, requested {value!r}")

        for other in load_points():
            if other.get("final_key") == rec["final_key"] and other["point_id"] != pid:
                raise HardFail(f"final cache key collides with {other['point_id']}")

        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        rec["metrics"] = point_metrics(out_dir / "per_file.csv", summary)
        pf = pd.read_csv(out_dir / "per_file.csv")
        rec["mean_hyp_count"] = float(pf["count_hyp"].mean())
        rec["mean_ref_count"] = float(pf["count_ref"].mean())
        rec["hyp_counts"] = dict(zip(pf["uri"], pf["count_hyp"].astype(int)))
        if (fa, fb, thr) == tuple(SHIPPED.values()):
            drift = abs(rec["metrics"]["all"]["der"] - BASELINE_DER)
            rec["baseline_der_drift"] = drift
            if drift > BASELINE_TOL:
                raise HardFail(f"shipped-baseline DER {rec['metrics']['all']['der']!r} != known "
                               f"{BASELINE_DER!r} (drift {drift:g})")
    except HardFail as failure:
        rec["failure"] = str(failure)
        with open(_ledger(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        raise

    with open(_ledger(), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    _append_per_file(rec, out_dir / "per_file.csv")
    write_points_csv()
    return rec


def fb_axis_noop_check() -> None:
    """Hard-fail if DER and speaker count are both identical at every Fb."""
    done = {p["params"]["Fb"]: p for p in load_points() if "metrics" in p
            and p["params"]["Fa"] == SHIPPED["Fa"] and p["params"]["threshold"] == SHIPPED["threshold"]}
    crashed = {p["params"]["Fb"] for p in load_points() if "crash" in p
               and p["params"]["Fa"] == SHIPPED["Fa"] and p["params"]["threshold"] == SHIPPED["threshold"]}
    if len(done) + len(crashed) < len(FB_VALUES):
        return
    ders = {round(p["metrics"]["all"]["der"], 12) for p in done.values()}
    cnts = {tuple(sorted(p["hyp_counts"].items())) for p in done.values()}
    if len(ders) == 1 and len(cnts) == 1:
        raise HardFail("Fb axis: DER and speaker counts identical at every Fb (suspected silent no-op)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["fb", "rest", "report"], required=True)
    args = ap.parse_args(argv)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    if args.phase == "report":
        print(write_points_csv())
        return
    todo = phase_points(args.phase)
    for i, (fa, fb, thr) in enumerate(todo, 1):
        pid = point_id(fa, fb, thr)
        if any(p["point_id"] == pid and ("metrics" in p or "crash" in p) for p in load_points()):
            continue
        try:
            rec = run_point(fa, fb, thr)
        except HardFail as failure:
            print(f"HARD FAIL {pid}: {failure}", flush=True)
            sys.exit(2)
        if "crash" in rec:
            print(f"[{i}/{len(todo)}] {pid} CRASHED {rec['crash']}", flush=True)
            continue
        m = rec["metrics"]
        print(f"[{i}/{len(todo)}] {pid} {rec['run_id']} DER {m['all']['der']:.8f} "
              f"tune {m['tune']['der']:.4f} report {m['report']['der']:.4f} "
              f"MAE {m['all']['count_mae']:.3f} bias {m['all']['count_bias']:+.3f} "
              f"hyp {rec['mean_hyp_count']:.3f} ({rec['wall_seconds']}s) | {rec['cache_tally'][:60]}",
              flush=True)
        if args.phase == "fb":
            try:
                fb_axis_noop_check()
            except HardFail as failure:
                print(f"HARD FAIL: {failure}", flush=True)
                sys.exit(2)
    print("PHASE DONE", args.phase, flush=True)


if __name__ == "__main__":
    main()
