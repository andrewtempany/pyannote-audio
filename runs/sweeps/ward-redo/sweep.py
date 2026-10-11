"""Closeout 3 section 3: ward linkage redo on its own threshold scale.

Mirrors the clustering benchmark's agglomerative points exactly (run_harness.py with
--run-condition clustering_agglomerative --clustering-model agglomerative, params
method=ward, min_cluster_size=12, threshold=T; identity refinement; IHM test; shared cache).
Run from repo root:
    diarisation-env/Scripts/python.exe runs/sweeps/ward-redo/sweep.py --phase gate
    diarisation-env/Scripts/python.exe runs/sweeps/ward-redo/sweep.py --phase grid
    diarisation-env/Scripts/python.exe runs/sweeps/ward-redo/sweep.py --phase report
Hard-fails exit 2 (nothing loosened): intermediate tally other than 16 full hits (0/0/0 allowed
only when the final key repeats an earlier recorded one), manifest clustering_config != request,
duplicate final keys, gate failure (threshold 10 not used / identical counts).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from harness.clustering_benchmark import EXPECTED_TALLY, HardFail, half_metrics, half_of, point_metrics  # noqa: E402
from harness.clustering_benchmark_report import _der  # noqa: E402

OUT = REPO / "runs" / "sweeps" / "ward-redo"
RUNS_DIR = REPO / "runs"
DATA_ROOT = REPO.parent / "AMI-diarization-setup" / "harness-data"
PYTHON = REPO / "diarisation-env" / "Scripts" / "python.exe"
LEDGER = OUT / "points.jsonl"
FRESH_BASELINE_CSV = REPO / "runs/sweeps/assignment-classifier/test_runs/vbx-identity/per_file.csv"
MCS = 12   # the benchmark's only ward min_cluster_size (AGG-1d-21..25)
GATE = [10.0, 1.4]
GRID = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0]
BOOT_DRAWS, BOOT_SEED = 2000, 0


def pid(t, gate=False):
    return f"WARD-{'G' if gate else 'T'}{t:g}"


def expected_cfg(t):
    return f"AgglomerativeClustering(method='ward', min_cluster_size={MCS}, threshold={t:g})"


def load():
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_point(point, t):
    out_dir = OUT / point
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), "run_harness.py", "--data-root", str(DATA_ROOT), "--condition", "IHM",
           "--split", "test", "--cache-dir", str(REPO / ".harness_cache"),
           "--per-file-csv", str(out_dir / "per_file.csv"), "--summary", str(out_dir / "summary.json"),
           "--runs-dir", str(RUNS_DIR),
           "--notes", f"closeout-3 ward redo {point}: agglomerative method=ward, min_cluster_size={MCS}, "
                      f"threshold={t:g} (euclidean ward scale, outside declared Uniform(0,2)); identity refinement; IHM test; warm cache",
           "--run-condition", "clustering_agglomerative", "--refinement-strategy", "identity",
           "--clustering-model", "agglomerative", "--clustering-diagnostics-dir", str(out_dir),
           "--clustering-param", "method=ward", "--clustering-param", f"min_cluster_size={MCS}",
           "--clustering-param", f"threshold={t:g}"]
    before = set(RUNS_DIR.glob("*.json"))
    t0 = time.monotonic()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=env, timeout=1800)
    wall = round(time.monotonic() - t0, 1)
    (out_dir / "run.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise HardFail(f"run_harness exited {r.returncode}; see {out_dir / 'run.log'}")
    new = sorted(set(RUNS_DIR.glob("*.json")) - before)
    if len(new) != 1:
        raise HardFail(f"expected one new manifest, found {len(new)}")
    man = json.loads(new[0].read_text(encoding="utf-8"))
    printed = next((l for l in r.stdout.splitlines() if l.startswith("clustering:")), "")
    rec = {"point_id": point, "requested": t, "run_id": man["run_id"], "manifest": new[0].name,
           "wall_seconds": wall, "printed_clustering_line": printed,
           "final_key": man["run_config"]["pipeline_config_id"],
           "clustering_config": man["run_config"]["clustering_config"]}
    tally = next((l for l in r.stdout.splitlines() if l.startswith("intermediate cache:")), "")
    rec["cache_tally"] = tally
    if expected_cfg(t) != rec["clustering_config"] or expected_cfg(t) not in printed:
        raise HardFail(f"{point}: clustering_config {rec['clustering_config']!r} / printed {printed!r} != {expected_cfg(t)!r}")
    if tally.startswith("intermediate cache: 0 full hit(s), 0 partial, 0 miss(es)"):
        rec["served_from_final_cache"] = True   # only legitimate for a repeat (t=1.4 == AGG-1d-25)
    elif not tally.startswith(EXPECTED_TALLY):
        raise HardFail(f"INTERMEDIATE CACHE NOT 16/16 FULL HITS (halts GPU queue): {tally!r}")
    for o in load():
        if o["final_key"] == rec["final_key"] and o["requested"] != t:
            raise HardFail(f"{point}: final key collides with {o['point_id']}")
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    rec["metrics"] = point_metrics(out_dir / "per_file.csv", summary)
    pf = pd.read_csv(out_dir / "per_file.csv")
    rec["mean_hyp_count"] = float(pf["count_hyp"].mean())
    rec["counts_hyp"] = dict(zip(pf["uri"], map(int, pf["count_hyp"])))
    rec["counts_ref"] = dict(zip(pf["uri"], map(int, pf["count_ref"])))
    return rec


def append(rec):
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def do(point, t):
    if any(p["point_id"] == point for p in load()):
        return
    try:
        rec = run_point(point, t)
    except HardFail as f:
        print(f"HARD FAIL {point}: {f}", flush=True)
        sys.exit(2)
    append(rec)
    m = rec["metrics"]["all"]
    print(f"{point} {rec['run_id']} DER {m['der']:.8f} meanhyp {rec['mean_hyp_count']:.3f} "
          f"({rec['wall_seconds']}s)\n  {rec['printed_clustering_line']}\n  {rec['cache_tally'][:70]}", flush=True)


def phase_gate():
    for t in GATE:
        do(pid(t, gate=True), t)
    a = {p["point_id"]: p for p in load()}
    hi, lo = a[pid(10.0, True)], a[pid(1.4, True)]
    print("uri  ref  hyp@10  hyp@1.4")
    for u in hi["counts_hyp"]:
        print(f"{u} {hi['counts_ref'][u]} {hi['counts_hyp'][u]} {lo['counts_hyp'][u]}")
    same = all(hi["counts_hyp"][u] == lo["counts_hyp"][u] for u in hi["counts_hyp"])
    print("manifest clustering_config @10:", hi["clustering_config"])
    print("printed line @10:", hi["printed_clustering_line"])
    print("final keys differ:", hi["final_key"] != lo["final_key"], "| counts identical everywhere:", same)
    if same or hi["final_key"] == lo["final_key"]:
        print("GATE FAILED")
        sys.exit(2)
    print("GATE PASSED")


def phase_grid():
    for t in GRID:
        do(pid(t), t)
    print("PHASE DONE grid")


def pf_of(point):
    df = pd.read_csv(OUT / point / "per_file.csv")
    df["half"] = df["uri"].map(half_of)
    return df


def boot_delta(point_df, base_df, half="report", draws=BOOT_DRAWS, seed=BOOT_SEED):
    """Same algorithm as harness/clustering_benchmark_report.py:57 paired_bootstrap_delta, with
    tables passed in (validated against it in validate_bootstrap)."""
    p = point_df[point_df["half"] == half].set_index("uri")
    b = base_df[base_df["half"] == half].set_index("uri")
    uris = sorted(p.index)
    assert uris == sorted(b.index) and len(uris) == 8
    rng = np.random.default_rng(seed)
    d = np.empty(draws)
    for i in range(draws):
        s = rng.choice(uris, size=len(uris), replace=True)
        d[i] = _der(p.loc[s]) - _der(b.loc[s])
    return {"delta": _der(p.loc[uris]) - _der(b.loc[uris]),
            "low": float(np.percentile(d, 5)), "high": float(np.percentile(d, 95))}


def validate_bootstrap():
    from harness.clustering_benchmark_report import paired_bootstrap_delta, per_file
    t = per_file()
    ref = paired_bootstrap_delta("HDB-2b-04")
    mine = boot_delta(t[t["point_id"] == "HDB-2b-04"].copy(), t[t["point_id"] == "VBX-baseline"].copy())
    assert all(abs(ref[k] - mine[k]) < 1e-12 for k in ref), (ref, mine)
    return ref, mine


def phase_report():
    from harness.assignment_runs import bootstrap, per_file as ar_pf
    pts = [p for p in load() if p["point_id"].startswith("WARD-T")]
    assert len(pts) == len(GRID), len(pts)
    pts.sort(key=lambda p: p["requested"])
    rows = []
    for p in pts:
        m = p["metrics"]
        rows.append({"threshold": p["requested"], "run_id": p["run_id"], "final_key": p["final_key"],
                     "clustering_config": p["clustering_config"], "cache_tally": p["cache_tally"],
                     "der": m["all"]["der"], "tune_der": m["tune"]["der"], "report_der": m["report"]["der"],
                     "md_s": m["all"]["md"], "fa_s": m["all"]["fa"], "conf_s": m["all"]["conf"],
                     "der_overlap_system": m["der_overlap_system"], "der_overlap_assigned": m["der_overlap_assigned"],
                     "jer": m["jer"], "region_t_and_d": m["all"]["region_t_and_d"],
                     "region_t_minus_d": m["all"]["region_t_minus_d"], "region_d_minus_t": m["all"]["region_d_minus_t"],
                     "count_mae": m["all"]["count_mae"], "count_bias": m["all"]["count_bias"],
                     "exact": m["all"]["exact"], "mean_hyp_count": p["mean_hyp_count"]})
    tab = pd.DataFrame(rows)
    checks = {"unique_final_keys": bool(tab["final_key"].nunique() == len(tab)),
              "manifest_config_matches_request": all(r["clustering_config"] == expected_cfg(r["threshold"]) for r in rows)}
    ref, mine = validate_bootstrap()
    fresh = pd.read_csv(FRESH_BASELINE_CSV)
    fresh["half"] = fresh["uri"].map(half_of)
    fm = {h: half_metrics(fresh if h == "all" else fresh[fresh["half"] == h]) for h in ("all", "tune", "report")}
    best = min(pts, key=lambda p: p["metrics"]["tune"]["der"])   # TUNE half only
    sp = pf_of(best["point_id"])
    res = {"grid": GRID, "min_cluster_size": MCS, "selection_half": "tune (meetings ending a/c)",
           "selected_threshold": best["requested"], "selected_point": best["point_id"],
           "selected_run_id": best["run_id"], "tune_best_at_grid_edge": best["requested"] in (GRID[0], GRID[-1]),
           "tune_der_by_threshold": {p["requested"]: p["metrics"]["tune"]["der"] for p in pts},
           "checks": checks, "bootstrap_validation": {"reference": ref, "reimplementation": mine},
           "fresh_baseline_der": {h: fm[h]["der"] for h in fm},
           "selected_report_der": best["metrics"]["report"]["der"],
           "delta_report_vs_fresh_baseline": boot_delta(sp, fresh),
           "bootstrap_function_report_half": "reimplementation of harness/clustering_benchmark_report.py:57 paired_bootstrap_delta (tables passed in), validated equal on HDB-2b-04",
           "selected_der_16": best["metrics"]["all"]["der"],
           "delta_16_vs_fresh_baseline": best["metrics"]["all"]["der"] - fm["all"]["der"]}
    a = ar_pf(OUT / best["point_id"])
    b = ar_pf(FRESH_BASELINE_CSV.parent)
    assert list(a.index) == list(b.index)
    lo, hi = bootstrap((b["error"].values, b["total"].values), (a["error"].values, a["total"].values),
                       lambda e, t: e.sum() / t.sum())
    res["delta_16_interval_assignment_runs_bootstrap"] = {
        "low": lo, "high": hi,
        "function": "harness/assignment_runs.py:150 bootstrap(), 2000 draws seed 0, 90% percentile"}
    tab["delta_all_vs_fresh"] = tab["der"] - fm["all"]["der"]
    tab["delta_report_vs_fresh"] = tab["report_der"] - fm["report"]["der"]
    tab.to_csv(OUT / "points.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(tab.drop(columns=["final_key", "clustering_config", "cache_tally", "run_id"]).round(4).to_string())
    print(tab[["threshold", "final_key", "clustering_config", "cache_tally"]].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gate", "grid", "report"], required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"gate": phase_gate, "grid": phase_grid, "report": phase_report}[a.phase]()
