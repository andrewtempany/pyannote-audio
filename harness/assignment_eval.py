"""Pair-accuracy evaluator for the assignment-classifier ticket.

Pair accuracy = correct target pairs / labelled target pairs. It is reported
overall and per overlap bin, and kept per meeting so WS5 can bootstrap it.

Train/dev labels come from the dataset rows (built with purpose "training").
Test labels are computed HERE, in a separate file
(`data/test_<method>_labels.csv.gz`), with purpose "pair-eval". This module
and the scorer are the only readers of test references (ticket hard-fail 4).
Nothing in feature building, training or selection imports this file's test
path.

    python -m harness.assignment_eval test-labels --method vbx
    python -m harness.assignment_eval floors          # writes dataset_stats.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from harness import assignment_common as ac
from harness.assignment_dataset import DATA_DIR, load

PAIR = ["uri", "chunk_idx", "speaker_idx"]


def _labels_path(method: str) -> Path:
    return DATA_DIR / f"test_{method}_labels.csv.gz"


def build_test_labels(method: str) -> Path:
    from harness.assignment_labels import load_reference, pair_labels

    pipeline = ac.load_pipeline(method)
    cache = ac.intermediate_cache(pipeline)
    frames = []
    for uri in ac.meeting_uris("test"):
        seg, emb = ac.load_intermediates(cache, uri)
        m = ac.cluster_meeting(pipeline, seg, emb, uri)
        labels = pair_labels(m, load_reference("test", uri, "pair-eval"))
        c, s = np.where(m.target)
        frames.append(pd.DataFrame({
            "uri": uri, "chunk_idx": c, "speaker_idx": s, "label": labels[c, s],
            "overlap_bin": ac.overlap_bin(ac.overlap_fraction(seg)[c, s]),
        }))
    out = _labels_path(method)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False, compression="gzip")
    return out


def pair_table(split: str, method: str) -> pd.DataFrame:
    """One row per target pair: ids, current_label, nc_label, label, overlap_bin."""
    df = load(split, method)
    pairs = df.drop_duplicates(PAIR)[PAIR + ["current_label", "nc_label", "overlap_bin",
                                            "is_held_back", "is_outlier", "label"]]
    if split == "test":
        labels = pd.read_csv(_labels_path(method))
        pairs = pairs.drop(columns=["label"]).merge(labels[PAIR + ["label"]], on=PAIR, how="left",
                                                    validate="one_to_one")
        if pairs["label"].isna().any():
            raise RuntimeError("test pairs without an evaluator label row")
    return pairs.reset_index(drop=True)


def pair_accuracy(pairs: pd.DataFrame, decision: pd.Series) -> dict:
    """Accuracy of `decision` (aligned with `pairs`) over labelled pairs."""
    labelled = pairs["label"] >= 0
    correct = (decision.values == pairs["label"].values) & labelled.values
    out = {"n_target": int(len(pairs)), "n_labelled": int(labelled.sum()),
           "acc": float(correct.sum() / max(labelled.sum(), 1))}
    for b in ac.OVERLAP_BINS:
        sel = labelled.values & (pairs["overlap_bin"].values == b)
        out[f"acc_{b}"] = float(correct[sel].sum() / sel.sum()) if sel.sum() else float("nan")
        out[f"n_{b}"] = int(sel.sum())
    per_meeting = pd.DataFrame({"uri": pairs["uri"], "correct": correct, "labelled": labelled})
    out["per_meeting"] = per_meeting.groupby("uri")[["correct", "labelled"]].sum()
    return out


def floors(splits=("train", "dev", "test"), methods=("vbx", "hdbscan")) -> Path:
    lines = ["# Assignment classifier — dataset stats and floors", "",
             "Pair accuracy over labelled target pairs. `identity` = the clustering's own label; "
             "`nc` = plain nearest centroid (argmax cosine). Overlap bins by the pair's overlap fraction.", "",
             "| split | method | meetings | target pairs | held-back | outliers | labelled % | candidates (mean) "
             "| identity acc | nc acc | identity by bin (0 / ≤.25 / ≤.5 / >.5) | nc by bin | n by bin |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for split in splits:
        for method in methods:
            path = DATA_DIR / f"{split}_{method}.csv.gz"
            if not path.exists() or (split == "test" and not _labels_path(method).exists()):
                lines.append(f"| {split} | {method} | – (not built) |" + " |" * 10)
                continue
            pairs = pair_table(split, method)
            cand = load(split, method).groupby("uri")["num_candidates"].first()
            ident = pair_accuracy(pairs, pairs["current_label"])
            nc = pair_accuracy(pairs, pairs["nc_label"])
            fmt = lambda r: " / ".join(f"{100 * r[f'acc_{b}']:.1f}" for b in ac.OVERLAP_BINS)
            lines.append(
                f"| {split} | {method} | {pairs['uri'].nunique()} | {len(pairs)} | {int(pairs['is_held_back'].sum())} "
                f"| {int(((pairs['is_outlier'] == 1) & (pairs['is_held_back'] == 0)).sum())} "
                f"| {100 * ident['n_labelled'] / len(pairs):.1f} | {cand.mean():.2f} "
                f"| {100 * ident['acc']:.1f} | {100 * nc['acc']:.1f} | {fmt(ident)} | {fmt(nc)} "
                f"| {' / '.join(str(ident[f'n_{b}']) for b in ac.OVERLAP_BINS)} |")
    out = ac.SWEEP_DIR / "dataset_stats.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("test-labels")
    t.add_argument("--method", required=True, choices=list(ac.METHODS))
    sub.add_parser("floors")
    args = parser.parse_args(argv)
    if args.cmd == "test-labels":
        print(build_test_labels(args.method))
    else:
        floors()


if __name__ == "__main__":
    main(sys.argv[1:])
