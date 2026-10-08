"""Closeout 2 sections 2-4 and 6: SDM (single distant microphone) runs with IHM-selected settings.

Run from the repo root (one phase at a time, strictly one GPU job):
    diarisation-env/Scripts/python.exe runs/sweeps/sdm/sweep.py --phase gate     # cold VBx baseline
    diarisation-env/Scripts/python.exe runs/sweeps/sdm/sweep.py --phase runs     # the other six (warm)
    diarisation-env/Scripts/python.exe runs/sweeps/sdm/sweep.py --phase report   # tables, intervals, predictions

Every run goes through run_harness.py with
    --data-root C:/Users/Andrew/Code/AMI-diarization-setup/harness-data --condition SDM --split test
    --cache-dir .harness_cache_sdm
(run_harness.py feeds config.cache_dir to BOTH IntermediateCache and Runner, so one flag isolates
both cache tiers; the keys contain the uri but not the mic condition, so sharing .harness_cache with
IHM would return IHM's data.) The shared .harness_cache is fingerprinted before and after every run;
any change is a hard-fail.

Settings are copied from the IHM manifests / ledgers (not retyped): see IHM_SOURCES below.
Hard-fails exit 2 and nothing is loosened: cold run with any intermediate hit; warm run without
exactly "16 full hit(s), 0 partial, 0 miss(es), 0 write(s)"; manifest clustering_config /
refinement_strategy / mic_condition / split different from the request; shared cache touched.
The gate phase exits 3 (after printing and recording everything) when |DER - 19.9| > 1.0.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from harness.assignment_runs import bootstrap, per_file as ar_per_file  # noqa: E402
from harness.clustering_benchmark import EXPECTED_TALLY, HardFail, half_of, point_metrics  # noqa: E402

OUT = REPO / "runs" / "sweeps" / "sdm"
RUNS_DIR = REPO / "runs"
DATA_ROOT = "C:/Users/Andrew/Code/AMI-diarization-setup/harness-data"
CACHE_DIR = ".harness_cache_sdm"
SHARED_CACHE = REPO / ".harness_cache"
PYTHON = REPO / "diarisation-env" / "Scripts" / "python.exe"
LEDGER = OUT / "points.jsonl"
PUBLISHED = 19.9
BOOT_DRAWS, BOOT_SEED = 2000, 0
BM = REPO / "runs/sweeps/clustering-benchmark"
AC = REPO / "runs/sweeps/assignment-classifier/test_runs"

# method -> (IHM manifest, IHM per-file csv dir)
IHM_MANIFEST = {
    "agglomerative": "20261003T003611Z-da03276c.json",
    "hdbscan": "20261003T012018Z-b15caf86.json",
    "spectral": "20261003T012326Z-fda86985.json",
    "nc_target": "20261003T050432Z-703683ce.json",
    "knn_target_k25": "20261003T055818Z-e2ef8a62.json",
}
IHM_DIR = {
    "vbx": AC / "vbx-identity",
    "agglomerative": BM / "agglomerative" / "AGG-1a-07",
    "hdbscan": BM / "hdbscan" / "HDB-2b-04",
    "spectral": BM / "spectral" / "SPC-3a-03",
    "nc_target": AC / "vbx-nc_target",
    "knn_target_k25": AC / "vbx-knn_target_k25",
    "ward": REPO / "runs/sweeps/ward-redo/WARD-T8",
}
BENCH_POINT = {"agglomerative": "AGG-1a-07", "hdbscan": "HDB-2b-04", "spectral": "SPC-3a-03"}
ORDER = ["vbx", "agglomerative", "hdbscan", "spectral", "nc_target", "knn_target_k25", "ward"]
LABEL = {"vbx": "VBx baseline", "agglomerative": "Agglomerative (centroid, 0.75, mcs 12)",
         "hdbscan": "HDBSCAN (leaf, mcs 40)", "spectral": "Spectral eigengap (pruning 0.20)",
         "nc_target": "Nearest centroid (nc_target)", "knn_target_k25": "kNN k=25 (knn_target_k25)",
         "ward": "Ward (threshold 8, mcs 12) [added after ticket]"}
WARD_PARAMS = {"method": "ward", "min_cluster_size": "12", "threshold": "8"}
WARD_CFG = "AgglomerativeClustering(method='ward', min_cluster_size=12, threshold=8)"


def request_for(method: str) -> dict:
    """The request (what we ask run_harness to do), copied from IHM records."""
    if method == "vbx":
        return {"run_condition": "baseline", "refinement": "identity", "clustering_model": None, "params": {},
                "cfg": "VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6)"}
    if method == "ward":
        return {"run_condition": "clustering_agglomerative", "refinement": "identity",
                "clustering_model": "agglomerative", "params": WARD_PARAMS, "cfg": WARD_CFG}
    man = json.loads((RUNS_DIR / IHM_MANIFEST[method]).read_text(encoding="utf-8"))["run_config"]
    if method in BENCH_POINT:
        pts = [json.loads(l) for l in (BM / "points.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        p = next(p for p in pts if p["point_id"] == BENCH_POINT[method])
        assert p["clustering_config"] == man["clustering_config"], "IHM ledger vs manifest disagree"
        return {"run_condition": man["condition"], "refinement": man["refinement_strategy"],
                "clustering_model": method, "params": p["params"], "cfg": man["clustering_config"]}
    return {"run_condition": man["condition"], "refinement": man["refinement_strategy"],
            "clustering_model": None, "params": {}, "cfg": man["clustering_config"]}


def fingerprint(root: Path):
    n, size, newest = 0, 0, 0.0
    for dp, _, fs in os.walk(root):
        for f in fs:
            st = os.stat(os.path.join(dp, f))
            n += 1
            size += st.st_size
            newest = max(newest, st.st_mtime)
    return (n, size, newest)


def load():
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_point(method: str, cold: bool) -> dict:
    req = request_for(method)
    out_dir = OUT / method
    out_dir.mkdir(parents=True, exist_ok=True)
    pstr = ", ".join(f"{k}={v}" for k, v in req["params"].items())
    notes = (f"closeout-2 SDM {method}: IHM-selected settings carried across, not re-tuned"
             f"{' (' + req['cfg'] + ')' if method != 'vbx' else ' (shipped defaults, cold baseline gate)'}"
             f"; refinement {req['refinement']}; AMI SDM (Array1-01) test; cache {CACHE_DIR}")
    cmd = [str(PYTHON), "run_harness.py", "--data-root", DATA_ROOT, "--condition", "SDM", "--split", "test",
           "--cache-dir", CACHE_DIR, "--per-file-csv", str(out_dir / "per_file.csv"),
           "--summary", str(out_dir / "summary.json"), "--runs-dir", str(RUNS_DIR), "--notes", notes,
           "--run-condition", req["run_condition"], "--refinement-strategy", req["refinement"],
           "--clustering-diagnostics-dir", str(out_dir)]
    if req["clustering_model"]:
        cmd += ["--clustering-model", req["clustering_model"]]
    for k, v in req["params"].items():
        cmd += ["--clustering-param", f"{k}={v}"]
    before = set(RUNS_DIR.glob("*.json"))
    shared_before = fingerprint(SHARED_CACHE)
    t0 = time.monotonic()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=env, timeout=7200)
    wall = round(time.monotonic() - t0, 1)
    (out_dir / "run.log").write_text(" ".join(cmd) + "\n\n" + r.stdout + "\n--- stderr ---\n" + r.stderr,
                                     encoding="utf-8")
    if r.returncode != 0:
        raise HardFail(f"run_harness exited {r.returncode}; see {out_dir / 'run.log'}")
    if fingerprint(SHARED_CACHE) != shared_before:
        raise HardFail("SHARED .harness_cache CHANGED DURING AN SDM RUN (hard-fail 3)")
    new = sorted(set(RUNS_DIR.glob("*.json")) - before)
    if len(new) != 1:
        raise HardFail(f"expected one new manifest, found {len(new)}")
    man = json.loads(new[0].read_text(encoding="utf-8"))
    rc = man["run_config"]
    printed = next((l for l in r.stdout.splitlines() if l.startswith("clustering:")), "")
    tally = next((l for l in r.stdout.splitlines() if l.startswith("intermediate cache:")), "")
    checks = {
        "clustering_config_matches_request": rc["clustering_config"] == req["cfg"] and req["cfg"] in printed,
        "refinement_matches_request": rc["refinement_strategy"] == req["refinement"],
        "mic_condition_SDM": rc.get("mic_condition") == "SDM",
        "split_test": rc["split"] == "test",
        "run_condition_matches": rc["condition"] == req["run_condition"],
        "collar0_overlap_scored": rc["der_collar"] == 0.0 and rc["der_skip_overlap"] is False,
        "cache_dir_flag_in_logged_command": ["--cache-dir", CACHE_DIR] == cmd[cmd.index("--cache-dir"):cmd.index("--cache-dir") + 2],
        "shared_cache_untouched": True,
    }
    m = re.search(r"intermediate cache: (\d+) full hit\(s\), (\d+) partial, (\d+) miss\(es\), (\d+) write\(s\)", tally)
    if not m:
        raise HardFail(f"no cache tally printed: {tally!r}")
    hits, partial, miss, writes = map(int, m.groups())
    if cold:
        if hits != 0 or miss != 16:
            raise HardFail(f"COLD RUN NOT COLD: tally {tally!r} (need 0 hits / 16 misses)")
    elif not tally.startswith(EXPECTED_TALLY):
        raise HardFail(f"WARM RUN NOT 16/16 FULL HITS: {tally!r}")
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise HardFail(f"{method}: failed programmatic checks {bad}: config {rc['clustering_config']!r}")
    for o in load():
        if o["final_key"] == rc["pipeline_config_id"] and o["refinement"] == req["refinement"]:
            raise HardFail(f"{method}: final key + refinement collides with {o['method']}")
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    pf = pd.read_csv(out_dir / "per_file.csv")
    rec = {"method": method, "run_id": man["run_id"], "manifest": new[0].name, "wall_seconds": wall,
           "final_key": rc["pipeline_config_id"], "intermediate_key": rc["intermediate_config_id"],
           "refinement": rc["refinement_strategy"], "clustering_config": rc["clustering_config"],
           "mic_condition": rc.get("mic_condition"), "cache_tally": tally, "cold": cold, "checks": checks,
           "command": cmd, "mean_hyp_count": float(pf["count_hyp"].mean()),
           "metrics": point_metrics(out_dir / "per_file.csv", summary)}
    return rec


def do(method: str, cold: bool):
    if any(p["method"] == method for p in load()):
        return
    try:
        rec = run_point(method, cold)
    except HardFail as f:
        print(f"HARD FAIL {method}: {f}", flush=True)
        sys.exit(2)
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    m = rec["metrics"]["all"]
    print(f"{method} {rec['run_id']} DER {m['der']:.8f} ({rec['wall_seconds']}s)\n  {rec['clustering_config']}"
          f"\n  {rec['cache_tally']}\n  checks {rec['checks']}", flush=True)


def phase_gate():
    do("vbx", cold=True)
    rec = next(p for p in load() if p["method"] == "vbx")
    der = rec["metrics"]["all"]["der"] * 100
    gap = der - PUBLISHED
    print(f"\nGATE: SDM VBx DER {der:.4f} vs published {PUBLISHED}: gap {gap:+.4f}; tally {rec['cache_tally']}")
    print("CAVEAT: 19.9 is from repo README.md:84-101 (legacy 3.1 | community-1 | precision-2 = 22.7 | 19.9 | 15.6); "
          "collar/overlap settings are not stated there ('no collar, overlap not skipped' is inferred from IHM 17.0 ~ 17.05).")
    ihm = pd.read_csv(IHM_DIR["vbx"] / "per_file.csv").set_index("uri")
    sdm = pd.read_csv(OUT / "vbx" / "per_file.csv").set_index("uri")
    ref = (sdm["missed_detection"] + sdm["false_alarm"] + sdm["confusion"]) / sdm["der"]
    tab = pd.DataFrame({"ref_s": ref, "SDM_DER%": sdm["der"] * 100, "IHM_DER%": ihm["der"] * 100,
                        "MD%": sdm["missed_detection"] / ref * 100, "FA%": sdm["false_alarm"] / ref * 100,
                        "Conf%": sdm["confusion"] / ref * 100, "hyp": sdm["count_hyp"], "ref": sdm["count_ref"]})
    tab["delta"] = tab["SDM_DER%"] - tab["IHM_DER%"]
    pd.set_option("display.width", 200)
    print(tab.round(2).to_string())
    m = rec["metrics"]["all"]
    print(f"corpus MD {m['md_share']*100:.2f}% share {m['md']:.1f}s FA {m['fa']:.1f}s Conf {m['conf']:.1f}s")
    if abs(gap) > 1.0:
        print("GATE STOP: |gap| > 1.0")
        sys.exit(3)
    print("GATE FLAG: gap between 0.5 and 1.0 -- proceed and flag" if abs(gap) > 0.5 else "GATE PASSED (<= 0.5)")


def phase_runs():
    g = [p for p in load() if p["method"] == "vbx"]
    if not g:
        print("no gate result; run --phase gate first")
        sys.exit(2)
    for m in ORDER[1:]:
        do(m, cold=False)
    print("PHASE DONE runs")


def abs_interval(df):
    zeros = (np.zeros(len(df)), np.ones(len(df)))
    return bootstrap(zeros, (df["error"].values, df["total"].values), lambda e, t: e.sum() / t.sum())


def phase_report():
    pts = {p["method"]: p for p in load()}
    assert set(pts) == set(ORDER), set(ORDER) - set(pts)
    der_fn = lambda e, t: e.sum() / t.sum()
    sdm = {m: ar_per_file(OUT / m) for m in ORDER}
    ihm = {m: ar_per_file(IHM_DIR[m]) for m in ORDER}
    base_s, base_i = sdm["vbx"], ihm["vbx"]
    rows = []
    for m in ORDER:
        s, i = sdm[m], ihm[m]
        assert list(s.index) == list(i.index) == list(base_s.index)
        assert np.allclose(s["total"].values, i["total"].values, rtol=1e-3), f"{m}: reference duration differs IHM vs SDM"
        met = pts[m]["metrics"]
        d_s, d_i = der_fn(s["error"].values, s["total"].values), der_fn(i["error"].values, i["total"].values)
        assert abs(d_s - met["all"]["der"]) < 1e-9
        lo, hi = abs_interval(s)
        dl, dh = bootstrap((base_s["error"].values, base_s["total"].values), (s["error"].values, s["total"].values), der_fn)
        gl, gh = bootstrap((i["error"].values, i["total"].values), (s["error"].values, s["total"].values), der_fn)
        il, ih = abs_interval(i)
        dv = (s["der"] - i["der"]) * 100
        sdm_exact = met["all"]["exact"]
        ihm_mae, ihm_exact = float(i["count_error"].mean()), int((i["count_error"] == 0).sum())
        rows.append({
            "method": m, "label": LABEL[m], "run_id": pts[m]["run_id"], "manifest": pts[m]["manifest"],
            "final_key": pts[m]["final_key"], "clustering_config": pts[m]["clustering_config"],
            "refinement": pts[m]["refinement"], "cache_tally": pts[m]["cache_tally"],
            "sdm_der": d_s * 100, "sdm_ci_lo": lo * 100, "sdm_ci_hi": hi * 100,
            "sdm_tune_der": met["tune"]["der"] * 100, "sdm_report_der": met["report"]["der"] * 100,
            "md_s": met["all"]["md"], "fa_s": met["all"]["fa"], "conf_s": met["all"]["conf"],
            "md_pct": met["all"]["md"] / (met["all"]["reference_s"]) * 100,
            "fa_pct": met["all"]["fa"] / (met["all"]["reference_s"]) * 100,
            "conf_pct": met["all"]["conf"] / (met["all"]["reference_s"]) * 100,
            "der_overlap_system": met["der_overlap_system"] * 100, "der_overlap_assigned": met["der_overlap_assigned"] * 100,
            "jer": met["jer"] * 100, "region_t_and_d": met["all"]["region_t_and_d"],
            "region_t_minus_d": met["all"]["region_t_minus_d"], "region_d_minus_t": met["all"]["region_d_minus_t"],
            "count_mae": met["all"]["count_mae"], "count_bias": met["all"]["count_bias"], "exact": sdm_exact,
            "delta_vs_sdm_vbx": (d_s - der_fn(base_s["error"].values, base_s["total"].values)) * 100,
            "delta_lo": dl * 100, "delta_hi": dh * 100,
            "ihm_der": d_i * 100, "ihm_ci_lo": il * 100, "ihm_ci_hi": ih * 100,
            "ihm_delta_vs_ihm_vbx": (d_i - der_fn(base_i["error"].values, base_i["total"].values)) * 100,
            "degradation": (d_s - d_i) * 100, "degr_lo": gl * 100, "degr_hi": gh * 100,
            "degr_meeting_min": float(dv.min()), "degr_meeting_max": float(dv.max()),
            "degr_meeting_min_uri": dv.idxmin(), "degr_meeting_max_uri": dv.idxmax(),
            "ihm_count_mae": ihm_mae, "ihm_exact": ihm_exact,
            "sdm_mean_hyp": pts[m]["mean_hyp_count"],
        })
    t = pd.DataFrame(rows)
    # relative measures (prediction 5) and transfer (prediction 4)
    t["rel_der"] = t["sdm_der"] / t["ihm_der"] - 1
    t["rel_count_mae"] = t["count_mae"] / t["ihm_count_mae"] - 1
    t["transfer_excess"] = t["delta_vs_sdm_vbx"] - t["ihm_delta_vs_ihm_vbx"]
    t.to_csv(OUT / "points.csv", index=False)
    # per-meeting CSV
    long = []
    for m in ORDER:
        s, i = sdm[m], ihm[m]
        for u in s.index:
            long.append({"method": m, "uri": u, "half": half_of(u), "sdm_der": s.loc[u, "der"], "ihm_der": i.loc[u, "der"],
                         "sdm_hyp": s.loc[u, "count_hyp"], "ihm_hyp": i.loc[u, "count_hyp"], "ref": s.loc[u, "count_ref"]})
    pd.DataFrame(long).to_csv(OUT / "per_meeting.csv", index=False)

    # predictions
    p = {}
    base = t[t.method == "vbx"].iloc[0]
    p["1_baseline_within_1pt_of_19.9"] = {"sdm_vbx_der": base.sdm_der, "gap": base.sdm_der - PUBLISHED,
                                         "held": bool(abs(base.sdm_der - PUBLISHED) <= 1.0)}
    p["2_all_methods_worse_by_2_to_4"] = {"degradations": dict(zip(t.method, t.degradation.round(3))),
                                          "held": bool(((t.degradation >= 2.0) & (t.degradation <= 4.0)).all())}
    r_i, r_s = t.ihm_der.rank(), t.sdm_der.rank()
    rho = float(np.corrcoef(r_i, r_s)[0, 1])
    p["3_ranking_preserved"] = {"ihm_rank": dict(zip(t.method, r_i)), "sdm_rank": dict(zip(t.method, r_s)),
                                "spearman": rho, "held": bool(rho >= 0.7)}
    nb = t[t.method != "vbx"]
    p["4_transfer_imperfect"] = {"excess_vs_ihm_delta": dict(zip(nb.method, nb.transfer_excess.round(3))),
                                 "max_excess": float(nb.transfer_excess.max()),
                                 "held_at_0.5": bool((nb.transfer_excess > 0.5).any()),
                                 "held_at_0.3": bool((nb.transfer_excess > 0.3).any())}
    cnt_more = (t.rel_count_mae > t.rel_der)
    p["5_counting_degrades_more"] = {"rel_der": dict(zip(t.method, t.rel_der.round(3))),
                                     "rel_count_mae": dict(zip(t.method, t.rel_count_mae.round(3))),
                                     "count_exceeds_der": dict(zip(t.method, map(bool, cnt_more))),
                                     "exact_ihm_to_sdm": {m: (int(a), int(b)) for m, a, b in zip(t.method, t.ihm_exact, t.exact)},
                                     "held": bool(cnt_more[t.method == "vbx"].iloc[0] and cnt_more.sum() >= 4)}
    (OUT / "results.json").write_text(json.dumps({"predictions": p, "rows": rows}, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 260)
    cols = ["method", "sdm_der", "sdm_ci_lo", "sdm_ci_hi", "delta_vs_sdm_vbx", "delta_lo", "delta_hi", "ihm_der",
            "degradation", "degr_lo", "degr_hi", "sdm_tune_der", "sdm_report_der"]
    print(t[cols].round(3).to_string())
    print(t[["method", "md_pct", "fa_pct", "conf_pct", "der_overlap_system", "der_overlap_assigned", "jer",
             "region_t_and_d", "region_t_minus_d", "region_d_minus_t", "count_mae", "count_bias", "exact",
             "sdm_mean_hyp", "ihm_count_mae", "ihm_exact"]].round(3).to_string())
    print(t[["method", "run_id", "refinement", "cache_tally", "clustering_config"]].to_string())
    print(json.dumps(p, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gate", "runs", "report"], required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"gate": phase_gate, "runs": phase_runs, "report": phase_report}[a.phase]()
