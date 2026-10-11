"""Closeout 4 (sections 2-4): min_duration_off sweep on VBx, plus one HDBSCAN check.

Every point goes through run_harness.py with an explicit
`--segmentation-param min_duration_off=<v>` (including 0.0). Resumable: points already in
points.jsonl are skipped. Run from the repo root:

    diarisation-env/Scripts/python.exe runs/sweeps/min-duration-off/sweep.py --phase vbx
    diarisation-env/Scripts/python.exe runs/sweeps/min-duration-off/sweep.py --phase hdbscan
    diarisation-env/Scripts/python.exe runs/sweeps/min-duration-off/sweep.py --phase report

Hard-fails (exit 2, nothing loosened): intermediate-cache tally other than 16 full hits (the
0/0/0 tally is allowed only when the final key was already in the final-RTTM cache, i.e. the
min_duration_off=0.0 points); manifest min_duration_off != requested; two points sharing a final
key; check 1 (0.0 explicit) not equal to the baseline final key and DER; check 2 (0.5 s changes
no meeting's missed detection); MD identical at every VBx value.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from harness.clustering_benchmark import (  # noqa: E402
    EXPECTED_TALLY, HardFail, half_metrics, half_of, point_metrics,
)
from harness.clustering_benchmark_report import _der  # noqa: E402

OUT = REPO / "runs" / "sweeps" / "min-duration-off"
RUNS_DIR = REPO / "runs"
DATA_ROOT = REPO.parent / "AMI-diarization-setup" / "harness-data"
PYTHON = REPO / "diarisation-env" / "Scripts" / "python.exe"
LEDGER = OUT / "points.jsonl"

# Order matters: 0.0 (check 1) then 0.5 (check 2), then the rest.
VALUES = [0.0, 0.5, 0.05, 0.1, 0.2, 0.3, 0.75, 1.0]
BASELINE_FINAL_KEY = "241ecd865d157bd823106180f9691005b3f6245043a8fadb1c62480b9aedbced"
CACHED_BASELINE_DER = 0.17049342382952828   # shipped config served from the final-RTTM cache
FRESH_BASELINE_CSV = REPO / "runs/sweeps/assignment-classifier/test_runs/vbx-identity/per_file.csv"
HDB_MANIFEST = RUNS_DIR / "20261003T012018Z-b15caf86.json"
HDB_BENCH_DER = 0.16855098234923785
HDB_PARAMS = {"min_cluster_size": "40", "min_samples": "0", "cluster_selection_method": "leaf"}
BOOT_DRAWS, BOOT_SEED = 2000, 0


def pid_vbx(v):
    return f"VBX-MDO{v:g}"


def pid_hdb(v):
    return f"HDB-MDO{v:g}"


def load():
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def find_key(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def run_point(pid, method, v, extra_args):
    out_dir = OUT / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), "run_harness.py", "--data-root", str(DATA_ROOT), "--condition", "IHM",
           "--split", "test", "--device", "cuda",
           "--per-file-csv", str(out_dir / "per_file.csv"), "--summary", str(out_dir / "summary.json"),
           "--refinement-strategy", "identity",
           "--notes", f"closeout-4 {pid}: shipped pipeline, {method} clustering, segmentation "
                      f"min_duration_off={v:g} s (gap filling after clustering); identity refinement, "
                      "IHM test, warm shared cache",
           "--segmentation-param", f"min_duration_off={v:g}"] + extra_args
    before = set(RUNS_DIR.glob("*.json"))
    t0 = time.monotonic()
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=1800)
    wall = round(time.monotonic() - t0, 1)
    (out_dir / "run.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise HardFail(f"run_harness exited {r.returncode}; see {out_dir / 'run.log'}")
    new = sorted(set(RUNS_DIR.glob("*.json")) - before)
    if len(new) != 1:
        raise HardFail(f"expected one new manifest, found {len(new)}")
    man = json.loads(new[0].read_text(encoding="utf-8"))
    rec = {"point_id": pid, "method": method, "requested": v, "run_id": man["run_id"],
           "manifest": new[0].name, "wall_seconds": wall, "command": " ".join(cmd[1:]),
           "final_key": man["run_config"]["pipeline_config_id"],
           "intermediate_key": man["run_config"].get("intermediate_config_id"),
           "clustering_config": man["run_config"]["clustering_config"],
           "manifest_min_duration_off": find_key(man, "min_duration_off")}
    tally = next((l for l in r.stdout.splitlines() if l.startswith("intermediate cache:")), "")
    rec["cache_tally"] = tally
    zero = "intermediate cache: 0 full hit(s), 0 partial, 0 miss(es)"
    if tally.startswith(zero):
        if v != 0.0:
            raise HardFail(f"{pid}: 0/0/0 tally at nonzero value (unexpected final-cache hit): {tally!r}")
        rec["served_from_final_cache"] = True
    elif not tally.startswith(EXPECTED_TALLY):
        raise HardFail(f"INTERMEDIATE CACHE NOT 16/16 FULL HITS (halts GPU queue): {tally!r}")
    if rec["manifest_min_duration_off"] is None or abs(rec["manifest_min_duration_off"] - v) > 1e-12:
        raise HardFail(f"{pid}: manifest min_duration_off={rec['manifest_min_duration_off']!r}, requested {v}")
    for o in load():
        if o["final_key"] == rec["final_key"]:
            raise HardFail(f"{pid}: final key collides with {o['point_id']}")
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    rec["metrics"] = point_metrics(out_dir / "per_file.csv", summary)
    pf = pd.read_csv(out_dir / "per_file.csv")
    rec["mean_hyp_count"] = float(pf["count_hyp"].mean())
    rec["summary_der"] = summary["der"]
    return rec


def append(rec):
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def per_meeting_md(pid):
    return pd.read_csv(OUT / pid / "per_file.csv").set_index("uri")["missed_detection"]


def after_point_checks(rec):
    if rec["method"] != "vbx":
        return
    v = rec["requested"]
    if v == 0.0:
        ok_key = rec["final_key"] == BASELINE_FINAL_KEY
        ok_der = rec["summary_der"] == CACHED_BASELINE_DER  # harness-reported DER, exact equality
        print(f"CHECK 1: final key == baseline key: {ok_key} ({rec['final_key']}); "
              f"summary DER {rec['summary_der']!r} == {CACHED_BASELINE_DER!r}: {ok_der} "
              f"(recomputed from per-file CSV: {rec['metrics']['all']['der']!r}); "
              f"tally: {rec['cache_tally'][:70]}", flush=True)
        if not (ok_key and ok_der):
            raise HardFail("check 1 failed (explicit min_duration_off=0.0 is not today's baseline)")
    if v == 0.5:
        base = per_meeting_md(pid_vbx(0.0))
        now = per_meeting_md(pid_vbx(0.5))
        diff = (now - base)
        print("CHECK 2: per-meeting MD at 0.5 minus MD at 0.0 (s):", flush=True)
        print(diff.round(3).to_string(), flush=True)
        if (diff.abs() < 1e-6).all():
            raise HardFail("check 2 failed: no meeting's missed detection changed at 0.5 s (override not reaching pipeline)")


def phase_vbx():
    for v in VALUES:
        pid = pid_vbx(v)
        if any(p["point_id"] == pid for p in load()):
            continue
        try:
            rec = run_point(pid, "vbx", v, [])
            after_point_checks(rec)
        except HardFail as f:
            print(f"HARD FAIL {pid}: {f}", flush=True)
            sys.exit(2)
        append(rec)
        m = rec["metrics"]["all"]
        print(f"{pid} {rec['run_id']} DER {m['der']:.8f} MD {m['md']:.3f} FA {m['fa']:.3f} "
              f"Conf {m['conf']:.3f} ({rec['wall_seconds']}s) | {rec['cache_tally'][:62]}", flush=True)
    mds = {round(p["metrics"]["all"]["md"], 6) for p in load() if p["method"] == "vbx"}
    if len(mds) == 1:
        print("HARD FAIL: MD identical at every value (suspected silent no-op)")
        sys.exit(2)
    print("PHASE DONE vbx")


# -- analysis ---------------------------------------------------------------

def pf_of(pid):
    df = pd.read_csv(OUT / pid / "per_file.csv")
    df["half"] = df["uri"].map(half_of)
    return df


def boot_delta(point_df, base_df, half="report", draws=BOOT_DRAWS, seed=BOOT_SEED):
    """Same algorithm as harness/clustering_benchmark_report.py:57 paired_bootstrap_delta
    (rng.choice over sorted report-half uris, same _der, 5th/95th percentiles), but taking the
    two per-file tables as arguments instead of reading that module's fixed benchmark table."""
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
    """boot_delta must reproduce paired_bootstrap_delta on an input that function can read."""
    from harness.clustering_benchmark_report import paired_bootstrap_delta, per_file
    t = per_file()
    a = t[t["point_id"] == "HDB-2b-04"].copy()
    b = t[t["point_id"] == "VBX-baseline"].copy()
    ref = paired_bootstrap_delta("HDB-2b-04")
    mine = boot_delta(a, b)
    assert all(abs(ref[k] - mine[k]) < 1e-12 for k in ref), (ref, mine)
    return ref, mine


def phase_hdbscan():
    pts = [p for p in load() if p["method"] == "vbx"]
    if len(pts) != len(VALUES):
        print("VBx sweep incomplete"); sys.exit(2)
    sel = selection(pts)
    man = json.loads(HDB_MANIFEST.read_text(encoding="utf-8"))
    cfg = man["run_config"]["clustering_config"]
    print("b15caf86 clustering_config:", cfg, flush=True)
    assert cfg == "HDBSCANClustering(cluster_selection_method='leaf', min_cluster_size=40, min_samples=0)"
    args = ["--run-condition", "clustering_hdbscan", "--clustering-model", "hdbscan"]
    for k, val in HDB_PARAMS.items():
        args += ["--clustering-param", f"{k}={val}"]
    v_sel = sel["requested"]
    vals = [0.0, v_sel] if v_sel != 0.0 else [0.0, sel["tune_best_nonzero"]]
    for v in vals:
        pid = pid_hdb(v)
        if any(p["point_id"] == pid for p in load()):
            continue
        try:
            rec = run_point(pid, "hdbscan", v, args)
            if rec["clustering_config"] != cfg:
                raise HardFail(f"{pid}: clustering_config {rec['clustering_config']!r} != b15caf86's")
            if v == 0.0:
                der = rec["metrics"]["all"]["der"]
                print(f"HDB 0.0 reproduces b15caf86? final key {rec['final_key']} vs "
                      f"{man['run_config']['pipeline_config_id']}; DER {der!r} vs benchmark "
                      f"{HDB_BENCH_DER!r} vs manifest {man['summary']['der']!r}; tally {rec['cache_tally'][:70]}",
                      flush=True)
                if rec["final_key"] != man["run_config"]["pipeline_config_id"]:
                    raise HardFail("HDBSCAN 0.0 final key differs from b15caf86's")
                if abs(der - HDB_BENCH_DER) > 1e-12 and not rec.get("served_from_final_cache"):
                    raise HardFail("HDBSCAN 0.0 DER differs from b15caf86 and was not served from final cache")
        except HardFail as f:
            print(f"HARD FAIL {pid}: {f}", flush=True)
            sys.exit(2)
        append(rec)
        m = rec["metrics"]["all"]
        print(f"{pid} {rec['run_id']} DER {m['der']:.8f} MD {m['md']:.3f} FA {m['fa']:.3f} "
              f"({rec['wall_seconds']}s) | {rec['cache_tally'][:62]}", flush=True)
    print("PHASE DONE hdbscan")


def selection(pts):
    """Lowest TUNE-half DER over the VBx sweep values (report half is never consulted)."""
    best = min(pts, key=lambda p: p["metrics"]["tune"]["der"])
    out = dict(best)
    nz = [p for p in pts if p["requested"] != 0.0]
    out["tune_best_nonzero"] = min(nz, key=lambda p: p["metrics"]["tune"]["der"])["requested"]
    return out


def phase_report():
    pts = [p for p in load() if p["method"] == "vbx"]
    rows = []
    for p in sorted(pts, key=lambda p: p["requested"]):
        m = p["metrics"]
        rows.append({"point_id": p["point_id"], "min_duration_off": p["requested"], "run_id": p["run_id"],
                     "final_key": p["final_key"], "intermediate_key": p["intermediate_key"],
                     "manifest_min_duration_off": p["manifest_min_duration_off"],
                     "cache_tally": p["cache_tally"],
                     "der": m["all"]["der"], "md_s": m["all"]["md"], "fa_s": m["all"]["fa"],
                     "conf_s": m["all"]["conf"], "der_overlap_system": m["der_overlap_system"],
                     "der_overlap_assigned": m["der_overlap_assigned"], "jer": m["jer"],
                     "region_t_and_d": m["all"]["region_t_and_d"], "region_t_minus_d": m["all"]["region_t_minus_d"],
                     "region_d_minus_t": m["all"]["region_d_minus_t"],
                     "count_mae": m["all"]["count_mae"], "count_bias": m["all"]["count_bias"],
                     "exact": m["all"]["exact"], "tune_der": m["tune"]["der"], "report_der": m["report"]["der"]})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "points.csv", index=False)

    checks = {"unique_final_keys": tab["final_key"].nunique() == len(tab),
              "single_intermediate_key": tab["intermediate_key"].nunique() == 1,
              "manifest_equals_requested": bool(((tab["manifest_min_duration_off"] - tab["min_duration_off"]).abs() < 1e-12).all()),
              "md_not_constant": tab["md_s"].round(6).nunique() > 1}
    print("CHECKS", checks)

    ref, mine = validate_bootstrap()
    print("bootstrap validation vs paired_bootstrap_delta(HDB-2b-04):", ref, mine)

    sel = selection(pts)
    fresh = pd.read_csv(FRESH_BASELINE_CSV)
    fresh["half"] = fresh["uri"].map(half_of)
    fresh_m = {h: half_metrics(fresh if h == "all" else fresh[fresh["half"] == h]) for h in ("all", "tune", "report")}
    cached0 = pf_of(pid_vbx(0.0))
    selpf = pf_of(sel["point_id"])
    res = {"selected_value": sel["requested"], "selected_point": sel["point_id"],
           "selected_tune_der": sel["metrics"]["tune"]["der"],
           "tune_der_by_value": {p["requested"]: p["metrics"]["tune"]["der"] for p in pts},
           "fresh_baseline_der": {h: fresh_m[h]["der"] for h in fresh_m},
           "selection_is_shipped": sel["requested"] == 0.0,
           "bootstrap_function": "reimplementation of harness/clustering_benchmark_report.py:57 paired_bootstrap_delta "
                                 "(reads a fixed table, so not callable on these CSVs); _der imported from :50; validated equal on HDB-2b-04",
           "checks": checks}
    res["delta_report_vs_fresh"] = boot_delta(selpf, fresh)
    res["delta_report_vs_cached0"] = boot_delta(selpf, cached0)
    # per-value deltas, for the table
    tab["delta_all_vs_fresh"] = [p["metrics"]["all"]["der"] - fresh_m["all"]["der"] for p in sorted(pts, key=lambda p: p["requested"])]
    tab["delta_report_vs_fresh"] = [p["metrics"]["report"]["der"] - fresh_m["report"]["der"] for p in sorted(pts, key=lambda p: p["requested"])]
    tab.to_csv(OUT / "points.csv", index=False)

    hd = {p["requested"]: p for p in load() if p["method"] == "hdbscan"}
    if hd:
        res["hdbscan"] = {}
        base = pf_of(pid_hdb(0.0)) if 0.0 in hd else None
        for v, p in sorted(hd.items()):
            e = {"run_id": p["run_id"], "final_key": p["final_key"], "tally": p["cache_tally"],
                 "der": p["metrics"]["all"]["der"], "report_der": p["metrics"]["report"]["der"],
                 "md_s": p["metrics"]["all"]["md"], "fa_s": p["metrics"]["all"]["fa"], "conf_s": p["metrics"]["all"]["conf"]}
            if base is not None and v != 0.0:
                b0 = p["metrics"]["all"]["der"] - hd[0.0]["metrics"]["all"]["der"]
                e["delta_all_vs_hdb0"] = b0
                e["delta_report_vs_hdb0"] = boot_delta(pf_of(pid_hdb(v)), base)
                # like-for-like fresh baseline: benchmark point HDB-2b-04 (run b15caf86, computed, not cache-served)
                t = pd.read_csv(REPO / "runs/sweeps/clustering-benchmark/all_points_per_file.csv")
                t = t[t["point_id"] == "HDB-2b-04"].copy()
                t["half"] = t["uri"].map(half_of)
                e["delta_all_vs_b15caf86"] = p["metrics"]["all"]["der"] - half_metrics(t)["der"]
                e["delta_report_vs_b15caf86"] = boot_delta(pf_of(pid_hdb(v)), t)
            res["hdbscan"][str(v)] = e
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res, indent=2, default=float))
    figure(tab, fresh_m["all"])
    print(tab.drop(columns=["final_key", "intermediate_key", "cache_tally", "run_id"]).to_string())


def figure(tab, fresh_all):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=140)
    spec = [("md_s", "Missed detection (s)", "#2a78d6", fresh_all["md"]),
            ("fa_s", "False alarm (s)", "#eb6834", fresh_all["fa"]),
            ("der", "DER (fraction of reference)", "#1baf7a", fresh_all["der"])]
    for ax, (col, label, color, base) in zip(axes, spec):
        ax.plot(tab["min_duration_off"], tab[col], "o-", color=color, lw=2)
        ax.axhline(base, color="#52514e", lw=1, ls="--", label="shipped (fresh baseline)")
        ax.set_xlabel("min_duration_off (s)")
        ax.set_title(label, fontsize=10, loc="left")
        ax.grid(color="#e4e3df")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("VBx baseline, AMI IHM test (16 meetings): effect of min_duration_off", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "md_fa_der_vs_min_duration_off.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["vbx", "hdbscan", "report"], required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"vbx": phase_vbx, "hdbscan": phase_hdbscan, "report": phase_report}[a.phase]()
