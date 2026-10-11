"""Closeout 1 section 2: 90% paired bootstrap (16 test meetings, 2000 draws, seed 0).

Reuses harness.assignment_runs.bootstrap (and per_file(), _der()) unchanged, so the
resampling stream and DER aggregation (sum(error)/sum(error/der) = total error /
total reference speech) are identical to the classifier work's ledger.

Run from repo root:  diarisation-env/Scripts/python.exe <this file>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\Andrew\Code\pyannote-audio")
sys.path.insert(0, str(REPO))
from harness import assignment_runs as ar  # noqa: E402

CB = REPO / "runs/sweeps/clustering-benchmark/all_points_per_file.csv"
AC = REPO / "runs/sweeps/assignment-classifier/test_runs"
COLS = ["missed_detection", "false_alarm", "confusion"]


def prep(df):
    df = df.set_index("uri").sort_index()
    df["error"] = df[COLS].sum(axis=1)
    df["total"] = df["error"] / df["der"]
    return df


cb = pd.read_csv(CB)


def from_cb(pid):
    return prep(cb[cb.point_id == pid].drop(columns=["point_id", "method", "half"]).copy())


def from_csv(p):
    return prep(pd.read_csv(p))


def manifest(run_id):
    return json.load(open(next((REPO / "runs").glob(f"*{run_id}.json"))))["summary"]


SOURCES = {  # label: (run id, csv description, loader)
    "VBx baseline": ("0fa3b41d", "runs/sweeps/assignment-classifier/test_runs/vbx-identity/per_file.csv",
                     lambda: from_csv(AC / "vbx-identity/per_file.csv")),
    "Agglomerative AGG-1a-07": ("da03276c", "runs/sweeps/clustering-benchmark/all_points_per_file.csv [point_id=AGG-1a-07]",
                                lambda: from_cb("AGG-1a-07")),
    "HDBSCAN HDB-2b-04": ("b15caf86", "runs/sweeps/clustering-benchmark/all_points_per_file.csv [point_id=HDB-2b-04]",
                          lambda: from_cb("HDB-2b-04")),
    "Spectral SPC-3a-03": ("fda86985", "runs/sweeps/clustering-benchmark/all_points_per_file.csv [point_id=SPC-3a-03]",
                           lambda: from_cb("SPC-3a-03")),
    "NC nc_target": ("703683ce", "runs/sweeps/assignment-classifier/test_runs/vbx-nc_target/per_file.csv",
                     lambda: from_csv(AC / "vbx-nc_target/per_file.csv")),
    "kNN k=25 (VBx)": ("e2ef8a62", "runs/sweeps/assignment-classifier/test_runs/vbx-knn_target_k25/per_file.csv",
                       lambda: from_csv(AC / "vbx-knn_target_k25/per_file.csv")),
    "NC all-voices (Sept)": ("32241285", "per_file_nearest_centroid_t5.csv (repo root; matched to manifest by exact MD/FA/Conf sums)",
                             lambda: from_csv(REPO / "per_file_nearest_centroid_t5.csv")),
    "Oracle all_pairs": ("1db2ba34", "per_file_oracle_all_pairs.csv (repo root)",
                         lambda: from_csv(REPO / "per_file_oracle_all_pairs.csv")),
    "Oracle overlap_degraded": ("c93fb9f9", "per_file_oracle_overlap_degraded.csv (repo root)",
                                lambda: from_csv(REPO / "per_file_oracle_overlap_degraded.csv")),
    "HDBSCAN identity (assignment-classifier)": ("b96eddd3", "runs/sweeps/assignment-classifier/test_runs/hdbscan-identity/per_file.csv",
                                                 lambda: from_csv(AC / "hdbscan-identity/per_file.csv")),
}

ZERO = None


def abs_ci(d):
    z = (np.zeros(len(d)), np.ones(len(d)))
    return ar.bootstrap(z, (d["error"].values, d["total"].values), ar._der)


def delta_ci(base, d):
    return ar.bootstrap((base["error"].values, base["total"].values), (d["error"].values, d["total"].values), ar._der)


frames = {}
base = None
rows = []
for label, (rid, csv, load) in SOURCES.items():
    d = load()
    frames[label] = d
    assert len(d) == 16, (label, len(d))
    assert list(d.index) == list(frames.get("VBx baseline", d).index), label
    der = ar._der(d["error"].values, d["total"].values)
    # check against manifest (same-run provenance)
    s = manifest(rid)
    man_der = s["der"]
    sums = [d[c].sum() - s[c] for c in COLS if c in s]
    rows.append((label, rid, csv, der, man_der, max([abs(x) for x in sums], default=float("nan"))))

base = frames["VBx baseline"]
print("label | run | DER(csv) | DER(manifest) | max|sum diff| s | abs CI | delta vs VBx baseline | delta CI")
out = []
for (label, rid, csv, der, man_der, sd) in rows:
    d = frames[label]
    a_lo, a_hi = abs_ci(d)
    dd = der - ar._der(base["error"].values, base["total"].values)
    d_lo, d_hi = delta_ci(base, d)
    print(f"{label} | {rid} | {100*der:.4f} | {100*man_der:.4f} | {sd:.6f} | [{100*a_lo:.2f}, {100*a_hi:.2f}] | "
          f"{100*dd:+.3f} | [{100*d_lo:+.2f}, {100*d_hi:+.2f}]")
    out.append(dict(label=label, run_id=rid, csv=csv, der=100 * der, man_der=100 * man_der, sum_diff=sd,
                    abs_lo=100 * a_lo, abs_hi=100 * a_hi, delta=100 * dd, d_lo=100 * d_lo, d_hi=100 * d_hi))
pd.DataFrame(out).to_csv(Path(__file__).with_name("closeout1_bootstrap_out.csv"), index=False)

# cross-check: VBx baseline in the benchmark csv vs assignment-classifier identity csv
bb = from_cb("VBX-baseline")
print("VBX-baseline (benchmark csv) DER:", 100 * ar._der(bb["error"].values, bb["total"].values),
      "max abs per-file der diff vs ac identity:", float((bb["der"] - base["der"]).abs().max()))

# reproduction of existing ledger intervals as a sanity check on the reused function
for lab, led in [("NC nc_target", (-0.26, -0.06)), ("kNN k=25 (VBx)", (-0.35, -0.06))]:
    lo, hi = delta_ci(base, frames[lab])
    print("ledger check", lab, round(100 * lo, 2), round(100 * hi, 2), "ledger:", led)
# the 16.79 variant vs nc_target (paired)
lo, hi = ar.bootstrap((frames["NC nc_target"]["error"].values, frames["NC nc_target"]["total"].values),
                      (frames["NC all-voices (Sept)"]["error"].values, frames["NC all-voices (Sept)"]["total"].values), ar._der)
print("16.79 minus nc_target paired:", 100 * (rows[6][3] - rows[4][3]), [100 * lo, 100 * hi])
