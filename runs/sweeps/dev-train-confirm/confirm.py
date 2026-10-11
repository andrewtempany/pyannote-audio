"""Closeout 3 s4: confirm knn_target_k25 and nc_target on AMI IHM dev (18) and train (136).

Rerun:  python runs/sweeps/dev-train-confirm/confirm.py run      (GPU; one harness call at a time)
        python runs/sweeps/dev-train-confirm/confirm.py report   (CPU; bootstrap + table from the CSVs)
Baselines are the existing cold-run per-file CSVs (assignment-classifier ws1/*_vbx_identity).
"""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from harness import assignment_common as ac
from harness.assignment_runs import bootstrap, per_file, _der

OUT = Path(__file__).resolve().parent
AC = REPO / "runs" / "sweeps" / "assignment-classifier"
BASE = {"dev": AC / "ws1" / "dev_vbx_identity", "train": AC / "ws1" / "train_vbx_identity"}
EXPECT = {"dev": 18, "train": 136}
STRATS = ["knn_target_k25", "nc_target"]


def check_bootstrap():
    """Reproduce the published test nc_target interval before trusting the function on other splits."""
    b = per_file(AC / "ws0" / "vbx_identity"); t = per_file(AC / "test_runs" / "vbx-nc_target")
    lo, hi = bootstrap((b["error"].values, b["total"].values), (t["error"].values, t["total"].values), _der)
    assert (round(100 * lo, 2), round(100 * hi, 2)) == (-0.26, -0.06), (lo, hi)
    print("bootstrap check OK on test nc_target:", round(100 * lo, 2), round(100 * hi, 2))


def run_all():
    for split in ("dev", "train"):
        for s in STRATS:
            d = OUT / f"{split}_{s}"
            if (d / "summary.json").exists():
                print("skip (done):", d); continue
            d.mkdir(exist_ok=True)
            notes = (f"closeout-3 s4 confirmation: VBx (community-1 default) + {s}, AMI IHM {split.upper()} "
                     f"({EXPECT[split]} meetings), warm shared cache; compare to {split} VBx identity baseline")
            cmd = [str(ac.PYTHON if hasattr(ac, 'PYTHON') else REPO / "diarisation-env/Scripts/python.exe"),
                   "run_harness.py", "--data-root", str(ac.DATA_ROOT), "--condition", "IHM", "--split", split,
                   "--device", "cuda", "--per-file-csv", str(d / "per_file.csv"), "--summary", str(d / "summary.json"),
                   "--run-condition", "baseline", "--refinement-strategy", s,
                   "--clustering-model", "pyannote-default", "--notes", notes]
            (d / "command.txt").write_text(" ".join(cmd) + "\n")
            p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            (d / "run.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
            m = re.search(r"intermediate cache: (\d+) full hit\(s\), (\d+) partial, (\d+) miss", p.stdout)
            print(split, s, "rc", p.returncode, "tally", m.groups() if m else None)
            if p.returncode != 0 or not m or m.group(2) != "0" or m.group(3) != "0" or int(m.group(1)) != EXPECT[split]:
                print("STOP: bad exit or intermediate miss/partial; see", d / "run.log"); sys.exit(2)


def report():
    check_bootstrap()
    rows = []
    for split in ("dev", "train"):
        b = per_file(BASE[split]); assert len(b) == EXPECT[split]
        bd = _der(b["error"].values, b["total"].values)
        rows.append(dict(split=split, strategy="vbx_identity", der=bd, delta=0, lo=0, hi=0, n=len(b)))
        for s in STRATS:
            t = per_file(OUT / f"{split}_{s}"); assert list(t.index) == list(b.index)
            td = _der(t["error"].values, t["total"].values)
            lo, hi = bootstrap((b["error"].values, b["total"].values), (t["error"].values, t["total"].values), _der)
            rows.append(dict(split=split, strategy=s, der=td, delta=td - bd, lo=lo, hi=hi, n=len(t)))
    df = pd.DataFrame(rows)
    for c in ("der", "delta", "lo", "hi"): df[c] *= 100
    df.to_csv(OUT / "results.csv", index=False, float_format="%.4f")
    print(df.to_string())


if __name__ == "__main__":
    {"run": run_all, "report": report}[sys.argv[1]]()
