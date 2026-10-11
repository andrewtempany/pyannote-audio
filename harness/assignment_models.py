"""Learned assignment classifiers (assignment-classifier ticket, WS4/WS5).

Each model scores one (target pair, candidate) row: P(candidate is correct).
At inference a pair takes the argmax over its candidates
(`classifier:<model_id>`), or the per-chunk Hungarian over those scores
(`classifier_hungarian:<model_id>`, the same constraint as `constrained_target`).

    python -m harness.assignment_models grid --method vbx       # dev_selection.csv rows
    python -m harness.assignment_models freeze                   # best per type x method + pooled
    python -m harness.assignment_models train-one ...            # WS5 dev-only / ablations

Fitting reads train rows only; selection reads dev rows only. Nothing here
reads test data (ticket hard-fail 5). Seeds are 0 everywhere.

Frozen models live in runs/sweeps/assignment-classifier/models/<model_id>/:
model.joblib, config.json (features, settings, training meetings, seed) and
sha256.txt (of model.joblib). The sha256 is part of the strategy's
refinement_id, so a retrained model can never share a final cache key (A8).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harness import assignment_common as ac
from harness.assignment_dataset import FEATURES, load

MODELS_DIR = ac.SWEEP_DIR / "models"
SELECTION_CSV = ac.SWEEP_DIR / "dev_selection.csv"
PAIR = ["uri", "chunk_idx", "speaker_idx"]
SEED = 0

GRIDS = {
    "lr": [{"C": c, "class_weight": w} for c in (0.01, 0.1, 1, 10) for w in (None, "balanced")],
    "hgb": [
        {"learning_rate": lr, "max_depth": d, "max_iter": it, "l2_regularization": l2}
        for lr, d, it, l2 in itertools.product((0.05, 0.1), (3, 6, None), (200, 500), (0, 1))
    ],
    "mlp": [{"hidden_layer_sizes": h, "alpha": a} for h in ((64,), (128, 64)) for a in (1e-4, 1e-3)],
}
# simplicity order for tie-breaks (ticket: ties within 0.2 points go to the simpler model)
TYPE_ORDER = {"lr": 0, "hgb": 1, "mlp": 2}


def make_model(kind: str, params: dict):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED, **params))
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=SEED, **params)
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            early_stopping=True, validation_fraction=0.1, random_state=SEED, max_iter=200, **params))
    raise KeyError(kind)


def complexity(kind: str, params: dict) -> tuple:
    """Smaller = simpler, for tie-breaks within a type."""
    if kind == "lr":
        return (TYPE_ORDER[kind], params["C"])
    if kind == "hgb":
        depth = params["max_depth"] or 99
        return (TYPE_ORDER[kind], depth, params["max_iter"], -params["l2_regularization"], params["learning_rate"])
    return (TYPE_ORDER[kind], sum(params["hidden_layer_sizes"]), -params["alpha"])


def matrix(df: pd.DataFrame, features=FEATURES) -> np.ndarray:
    return df[list(features)].astype(float).fillna(-1.0).to_numpy()


def training_rows(split: str, methods) -> pd.DataFrame:
    df = pd.concat([load(split, m) for m in methods], ignore_index=True)
    return df[df["is_label"].notna()].reset_index(drop=True)


# -- decisions from row scores ---------------------------------------------------

def argmax_decisions(rows: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """One row per pair: the candidate with the highest score."""
    tmp = rows[PAIR + ["method", "candidate"]].copy()
    tmp["score"] = scores
    idx = tmp.groupby(["method"] + PAIR, sort=False)["score"].idxmax()
    best = tmp.loc[idx, ["method"] + PAIR + ["candidate"]].rename(columns={"candidate": "decision"})
    return best.reset_index(drop=True)


def dev_pair_accuracy(rows: pd.DataFrame, scores: np.ndarray) -> float:
    """Accuracy over labelled dev pairs (rows carry `label`)."""
    dec = argmax_decisions(rows, scores)
    lab = rows.drop_duplicates(["method"] + PAIR)[["method"] + PAIR + ["label"]]
    merged = dec.merge(lab, on=["method"] + PAIR)
    merged = merged[merged["label"] >= 0]
    return float((merged["decision"] == merged["label"]).mean())


# -- grid search -----------------------------------------------------------------

def _fit_score(kind, params, X, y, dev_rows, X_dev, features):
    started = time.time()
    model = make_model(kind, params).fit(X, y)
    acc = dev_pair_accuracy(dev_rows, model.predict_proba(X_dev)[:, 1])
    return {"type": kind, "params": json.dumps(params), "dev_pair_acc": acc,
            "fit_seconds": round(time.time() - started, 1), "features": ",".join(features)}


def grid(train_methods, dev_methods, tag: str, kinds=("lr", "hgb", "mlp"), n_jobs: int = 4,
         features=FEATURES) -> pd.DataFrame:
    from joblib import Parallel, delayed

    train = training_rows("train", train_methods)
    dev = training_rows("dev", dev_methods)
    X, y = matrix(train, features), train["is_label"].astype(int).to_numpy()
    X_dev = matrix(dev, features)
    jobs = [(k, p) for k in kinds for p in GRIDS[k]]
    print(f"[{tag}] {len(jobs)} fits on {len(train)} train rows, dev {len(dev)} rows", flush=True)
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fit_score)(k, p, X, y, dev, X_dev, features) for k, p in jobs)
    out = pd.DataFrame(results)
    out.insert(0, "tag", tag)
    out.insert(1, "train_methods", "+".join(train_methods))
    if SELECTION_CSV.exists():
        prev = pd.read_csv(SELECTION_CSV)
        out = pd.concat([prev[prev["tag"] != tag], out], ignore_index=True)
    out.to_csv(SELECTION_CSV, index=False)
    return out[out["tag"] == tag]


def select(results: pd.DataFrame, kind: str) -> dict:
    """Best setting of one type: highest dev acc; ties within 0.2 points -> simplest."""
    sub = results[results["type"] == kind].copy()
    best = sub["dev_pair_acc"].max()
    near = sub[sub["dev_pair_acc"] >= best - 0.002].copy()
    near["cx"] = [complexity(kind, json.loads(p)) for p in near["params"]]
    row = near.sort_values("cx").iloc[0]
    return {"type": kind, "params": json.loads(row["params"]), "dev_pair_acc": float(row["dev_pair_acc"])}


# -- freezing and loading --------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(model_id: str, kind: str, params: dict, train_split: str, methods, features=FEATURES,
           dev_pair_acc=None, note: str = "") -> Path:
    out = MODELS_DIR / model_id
    if out.exists():
        raise FileExistsError(f"{out} exists; frozen models are never overwritten (hard-fail 5)")
    rows = training_rows(train_split, methods)
    model = make_model(kind, params).fit(matrix(rows, features), rows["is_label"].astype(int).to_numpy())
    out.mkdir(parents=True)
    joblib.dump(model, out / "model.joblib")
    sha = _sha256(out / "model.joblib")
    (out / "sha256.txt").write_text(sha + "\n")
    (out / "config.json").write_text(json.dumps({
        "model_id": model_id, "type": kind, "params": params, "features": list(features),
        "train_split": train_split, "train_methods": list(methods),
        "train_meetings": sorted(rows["uri"].unique().tolist()), "train_rows": int(len(rows)),
        "seed": SEED, "dev_pair_acc": dev_pair_acc, "imputation": "NaN -> -1 (missing flags are features)",
        "note": note,
    }, indent=2))
    print(f"froze {model_id}: {kind} {params} sha256 {sha[:8]}")
    return out


def model_sha256(model_id: str) -> str:
    folder = MODELS_DIR / model_id
    recorded = (folder / "sha256.txt").read_text().strip()
    actual = _sha256(folder / "model.joblib")
    if recorded != actual:
        raise RuntimeError(f"{model_id}: model.joblib sha256 {actual[:8]} != recorded {recorded[:8]}")
    return recorded


def load_model(model_id: str):
    folder = MODELS_DIR / model_id
    model_sha256(model_id)
    return joblib.load(folder / "model.joblib"), json.loads((folder / "config.json").read_text())


def model_scores(model, config, m: ac.ClusteredMeeting) -> np.ndarray:
    """(chunks, speakers, K) P(correct) at target pairs, -inf elsewhere."""
    from harness.assignment_dataset import meeting_rows

    rows = meeting_rows(m, None, "inference", "inference")
    scores = np.full(m.hard_clusters.shape + (m.centroids.shape[0],), -np.inf)
    if len(rows):
        p = model.predict_proba(matrix(rows, config["features"]))[:, 1]
        scores[rows["chunk_idx"].to_numpy(), rows["speaker_idx"].to_numpy(), rows["candidate"].to_numpy()] = p
    return scores


def model_decider(model_id: str, hungarian: bool = False):
    from harness.assignment_strategies import _apply_on_targets, hungarian_targets

    model, config = load_model(model_id)

    def decide(m: ac.ClusteredMeeting) -> np.ndarray:
        scores = model_scores(model, config, m)
        if hungarian:
            return hungarian_targets(m, scores)
        return _apply_on_targets(m, np.argmax(scores, axis=2))

    return decide


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--method", required=True, choices=["vbx", "hdbscan", "pooled"])
    g.add_argument("--kinds", default="lr,hgb,mlp")
    g.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if args.cmd == "grid":
        methods = ["vbx", "hdbscan"] if args.method == "pooled" else [args.method]
        res = grid(methods, methods, args.method, kinds=tuple(args.kinds.split(",")), n_jobs=args.n_jobs)
        for kind in args.kinds.split(","):
            print(select(res, kind))


if __name__ == "__main__":
    main(sys.argv[1:])
