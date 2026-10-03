"""Test-run driver and ledger for the assignment-classifier ticket.

    python -m harness.assignment_runs run <run_point> <method> <strategy> [--note ...]
    python -m harness.assignment_runs index        # rebuild RUN_INDEX.md from runs.jsonl

Each point is one scored 16-meeting test run through `run_harness.py`
(manifest in runs/), plus pair accuracy computed with the evaluator's test
labels. Hard-fails enforced on every point (ticket §7):

  1. MD, FA and the region census must equal the method's `identity` run exactly.
  3. No two points with different strategies may share a refinement_id
     (the only part of the final cache key that differs between points).
  7. Intermediate cache tally must be 16 full hits, 0 partial, 0 misses, 0 writes.

Hard-fail 2 (non-target pair changed) is enforced inside every strategy
(`assignment_strategies.checked_decision`). The `identity` reference rows are
the WS0 runs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from harness import assignment_common as ac

LEDGER = ac.SWEEP_DIR / "runs.jsonl"
RUN_INDEX = ac.SWEEP_DIR / "RUN_INDEX.md"
POINTS_DIR = ac.SWEEP_DIR / "test_runs"
PYTHON = ac.REPO / "diarisation-env" / "Scripts" / "python.exe"
IDENTITY = {"vbx": ac.SWEEP_DIR / "ws0" / "vbx_identity", "hdbscan": ac.SWEEP_DIR / "ws0" / "hdb_identity"}
FIXED = ["missed_detection", "false_alarm", "region_t_and_d", "region_t_minus_d", "region_d_minus_t"]
RUN_CONDITION = {"vbx": "baseline", "hdbscan": "clustering_hdbscan"}
BOOT_DRAWS, BOOT_SEED = 2000, 0


class HardFail(RuntimeError):
    pass


def harness_command(method: str, strategy: str, out_dir: Path, notes: str) -> list:
    spec = ac.METHODS[method]
    cmd = [str(PYTHON), "run_harness.py", "--data-root", str(ac.DATA_ROOT), "--condition", "IHM",
           "--split", "test", "--per-file-csv", str(out_dir / "per_file.csv"),
           "--summary", str(out_dir / "summary.json"), "--run-condition", RUN_CONDITION[method],
           "--refinement-strategy", strategy, "--clustering-model", spec["clustering_model"],
           "--notes", notes]
    for name, value in spec["clustering_params"].items():
        cmd += ["--clustering-param", f"{name}={value}"]
    if method == "hdbscan":
        cmd += ["--clustering-diagnostics-dir", str(out_dir)]
    return cmd


def offline_pair_accuracy(method: str, strategy: str) -> pd.DataFrame:
    """Per-meeting correct / labelled target pairs for `strategy` on test."""
    from harness.assignment_eval import _labels_path
    from harness.assignment_strategies import checked_decision, decider

    labels = pd.read_csv(_labels_path(method))
    decide = (lambda m: m.hard_clusters.copy()) if strategy == "identity" else decider(strategy)
    pipeline = ac.load_pipeline(method)
    cache = ac.intermediate_cache(pipeline)
    rows = []
    for uri in ac.meeting_uris("test"):
        seg, emb = ac.load_intermediates(cache, uri)
        m = ac.cluster_meeting(pipeline, seg, emb, uri)
        out = checked_decision(m, decide)
        lab = labels[labels["uri"] == uri]
        dec = out[lab["chunk_idx"].to_numpy(), lab["speaker_idx"].to_numpy()]
        ok = lab["label"].to_numpy() >= 0
        correct = (dec == lab["label"].to_numpy()) & ok
        for b in ac.OVERLAP_BINS:
            sel = ok & (lab["overlap_bin"].to_numpy() == b)
            rows.append({"uri": uri, "bin": b, "correct": int(correct[sel].sum()), "labelled": int(sel.sum())})
    return pd.DataFrame(rows)


def run_point(run_point: str, method: str, strategy: str, note: str = "", model_id: str = "") -> dict:
    from harness.refinement import refinement_id_for

    out_dir = POINTS_DIR / run_point
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; a test point is run once")
    out_dir.mkdir(parents=True)
    rid = refinement_id_for(strategy)
    ledger = [json.loads(l) for l in LEDGER.read_text().splitlines()] if LEDGER.exists() else []
    for prev in ledger:
        if prev["refinement_id"] == rid and prev["method"] == method:
            raise HardFail(f"refinement_id {rid} already used by {prev['run_point']} (hard-fail 3)")

    notes = f"assignment-classifier {run_point}: {method} + {strategy}" + (f" ({note})" if note else "")
    started = time.time()
    before = set(Path(ac.REPO / "runs").glob("*.json"))
    proc = subprocess.run(harness_command(method, strategy, out_dir, notes), cwd=ac.REPO,
                          capture_output=True, text=True)
    (out_dir / "run.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"run_harness failed ({proc.returncode}); see {out_dir / 'run.log'}")
    manifests = [m for m in sorted(set(Path(ac.REPO / "runs").glob("*.json")) - before)
                 if json.loads(m.read_text())["run_config"].get("notes") == notes]
    if len(manifests) != 1:
        raise RuntimeError(f"expected one new manifest with notes {notes!r}, found {manifests}")
    return finalize(run_point, method, strategy, note, model_id, rid, manifests[0].stem,
                    proc.stdout, round(time.time() - started))


def finalize(run_point, method, strategy, note, model_id, rid, run_id, stdout, seconds) -> dict:
    """Checks and ledger record for a completed harness run (also used to record a run
    whose harness step finished but whose bookkeeping was interrupted)."""
    out_dir = POINTS_DIR / run_point
    tally = re.search(r"intermediate cache: (.*?) \(", stdout)
    if not tally or tally.group(1) != "16 full hit(s), 0 partial, 0 miss(es), 0 write(s)":
        raise HardFail(f"cache tally {tally.group(1) if tally else '?'} (hard-fail 7)")

    summary = json.loads((out_dir / "summary.json").read_text())
    ident = json.loads((IDENTITY[method] / "summary.json").read_text())
    moved = [k for k in FIXED if summary[k] != ident[k]]
    if moved:
        raise HardFail(f"{moved} differ from identity (hard-fail 1)")

    acc = offline_pair_accuracy(method, strategy)
    acc.to_csv(out_dir / "pair_accuracy.csv", index=False)
    record = {"run_point": run_point, "method": method, "strategy": strategy, "model_id": model_id,
              "refinement_id": rid, "run_id": run_id, "seconds": seconds,
              "summary": summary, "note": note}
    with LEDGER.open("a") as f:
        f.write(json.dumps(record) + "\n")
    write_index()
    return record


# -- reporting -------------------------------------------------------------------

def per_file(point_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(point_dir / "per_file.csv").set_index("uri").sort_index()
    df["error"] = df["missed_detection"] + df["false_alarm"] + df["confusion"]
    df["total"] = df["error"] / df["der"]
    return df


def bootstrap(values_a, values_b, stat, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """90% paired percentile interval of stat(b) - stat(a) over meetings."""
    rng = np.random.default_rng(seed)
    n = len(values_a[0])
    diffs = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        diffs.append(stat(*(v[idx] for v in values_b)) - stat(*(v[idx] for v in values_a)))
    return float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))


def _der(err, tot):
    return err.sum() / tot.sum()


def _acc(c, l):
    return c.sum() / max(l.sum(), 1)


def point_stats(record: dict) -> dict:
    method = record["method"]
    pdir = POINTS_DIR / record["run_point"]
    base = per_file(IDENTITY[method])
    this = per_file(pdir)
    acc = pd.read_csv(pdir / "pair_accuracy.csv")
    tot = acc.groupby("uri")[["correct", "labelled"]].sum().sort_index()
    out = {"acc": _acc(tot["correct"].values, tot["labelled"].values)}
    for b in ac.OVERLAP_BINS:
        s = acc[acc["bin"] == b]
        out[f"acc_{b}"] = s["correct"].sum() / max(s["labelled"].sum(), 1)
    # interval of the level itself: stat(b) - stat(a) with a contributing 0
    zeros = (np.zeros(len(tot)), np.ones(len(tot)))
    out["acc_ci"] = bootstrap(zeros, (tot["correct"].values, tot["labelled"].values), _acc)
    out["d_der"] = _der(this["error"].values, this["total"].values) - _der(base["error"].values, base["total"].values)
    out["d_der_ci"] = bootstrap((base["error"].values, base["total"].values),
                                (this["error"].values, this["total"].values), _der)
    ident_acc_path = POINTS_DIR / f"{method}-identity" / "pair_accuracy.csv"
    if ident_acc_path.exists():
        ia = pd.read_csv(ident_acc_path).groupby("uri")[["correct", "labelled"]].sum().sort_index()
        out["d_acc"] = out["acc"] - _acc(ia["correct"].values, ia["labelled"].values)
        out["d_acc_ci"] = bootstrap((ia["correct"].values, ia["labelled"].values),
                                    (tot["correct"].values, tot["labelled"].values), _acc)
    errs = this["count_hyp"] - this["count_ref"]
    out["count_mae"], out["count_bias"] = float(np.abs(errs).mean()), float(errs.mean())
    return out


def write_index():
    records = [json.loads(l) for l in LEDGER.read_text().splitlines()] if LEDGER.exists() else []
    lines = RUN_INDEX.read_text(encoding="utf-8").splitlines()[:6]
    for r in records:
        s, st = r["summary"], point_stats(r)
        sha = r["refinement_id"].rsplit(":", 1)[-1][:8] if r["model_id"] else "–"
        bins = " / ".join(f"{100 * st[f'acc_{b}']:.1f}" for b in ac.OVERLAP_BINS)
        dacc = (f"; Δacc vs identity {100 * st['d_acc']:+.1f} [{100 * st['d_acc_ci'][0]:+.1f}, {100 * st['d_acc_ci'][1]:+.1f}]"
                if "d_acc" in st else "")
        lines.append(
            f"| {r['run_point']} | {r['method']} | {r['strategy'].split(':')[0]} | {r['model_id'] or '–'} | {sha} "
            f"| {r['run_id']} | {100 * st['acc']:.1f} [{100 * st['acc_ci'][0]:.1f}, {100 * st['acc_ci'][1]:.1f}] | {bins} "
            f"| {100 * s['der']:.2f} | {s['missed_detection']:.1f} | {s['false_alarm']:.1f} | {s['confusion']:.1f} "
            f"| {s['der_overlap_assigned']:.4f} | {s['jer']:.4f} | {st['count_mae']:.3f} / {st['count_bias']:+.3f} "
            f"| {100 * st['d_der']:+.2f} [{100 * st['d_der_ci'][0]:+.2f}, {100 * st['d_der_ci'][1]:+.2f}] "
            f"| {r['note']}{dacc} |")
    RUN_INDEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("run_point"); r.add_argument("method", choices=list(ac.METHODS)); r.add_argument("strategy")
    r.add_argument("--note", default=""); r.add_argument("--model-id", default="")
    sub.add_parser("index")
    sub.add_parser("identity-acc")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        rec = run_point(args.run_point, args.method, args.strategy, args.note, args.model_id)
        print(json.dumps({k: rec[k] for k in ("run_point", "run_id", "refinement_id")}), rec["summary"]["der"])
    elif args.cmd == "identity-acc":
        # register the WS0 identity runs as the reference rows of the ledger
        import shutil

        run_ids = {"vbx": "20261003T044356Z-0fa3b41d", "hdbscan": "20261003T044452Z-b96eddd3"}
        for method in ac.METHODS:
            d = POINTS_DIR / f"{method}-identity"
            d.mkdir(parents=True, exist_ok=True)
            for name in ("per_file.csv", "summary.json"):
                shutil.copy(IDENTITY[method] / name, d / name)
            offline_pair_accuracy(method, "identity").to_csv(d / "pair_accuracy.csv", index=False)
            record = {"run_point": f"{method}-identity", "method": method, "strategy": "identity",
                      "model_id": "", "refinement_id": "identity", "run_id": run_ids[method], "seconds": 0,
                      "summary": json.loads((IDENTITY[method] / "summary.json").read_text()),
                      "note": "WS0 reference run (floor)"}
            with LEDGER.open("a") as f:
                f.write(json.dumps(record) + "\n")
        write_index()
    else:
        write_index()


if __name__ == "__main__":
    main(sys.argv[1:])
