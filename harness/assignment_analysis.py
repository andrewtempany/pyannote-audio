"""WS5 analyses on DEV only (assignment-classifier ticket).

    python -m harness.assignment_analysis importance   # permutation importance by feature group
    python -m harness.assignment_analysis ablation     # retrain without temporal / similarity, eval on dev

Both use the best model per method (`<method>-hgb`, see the ticket's WS5 notes)
and dev pair accuracy (argmax decision). Writes
runs/sweeps/assignment-classifier/feature_importance.csv and ablation.csv.
Seeds 0; permutation repeats 5. Nothing here reads test data.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from harness import assignment_common as ac
from harness import assignment_models as am
from harness.assignment_dataset import FEATURE_GROUPS, FEATURES

METHODS = ("vbx", "hdbscan")
REPEATS = 5


def importance() -> pd.DataFrame:
    rows = []
    for method in METHODS:
        model, config = am.load_model(f"{method}-hgb")
        dev = am.training_rows("dev", [method])
        X = am.matrix(dev, config["features"])
        base = am.dev_pair_accuracy(dev, model.predict_proba(X)[:, 1])
        rng = np.random.default_rng(0)
        for group, cols in FEATURE_GROUPS.items():
            idx = [config["features"].index(c) for c in cols]
            drops = []
            for _ in range(REPEATS):
                Xp = X.copy()
                perm = rng.permutation(len(Xp))
                Xp[:, idx] = X[perm][:, idx]  # permute the group's columns jointly
                drops.append(base - am.dev_pair_accuracy(dev, model.predict_proba(Xp)[:, 1]))
            rows.append({"method": method, "group": group, "base_dev_acc": base,
                         "acc_drop_mean": float(np.mean(drops)), "acc_drop_sd": float(np.std(drops))})
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ac.SWEEP_DIR / "feature_importance.csv", index=False)
    return out


def ablation() -> pd.DataFrame:
    rows = []
    for method in METHODS:
        config = am.load_model(f"{method}-hgb")[1]
        train = am.training_rows("train", [method])
        dev = am.training_rows("dev", [method])
        for dropped in (None, "temporal", "similarity"):
            feats = [f for f in FEATURES if dropped is None or f not in FEATURE_GROUPS[dropped]]
            model = am.make_model("hgb", config["params"]).fit(
                am.matrix(train, feats), train["is_label"].astype(int).to_numpy())
            acc = am.dev_pair_accuracy(dev, model.predict_proba(am.matrix(dev, feats))[:, 1])
            rows.append({"method": method, "without": dropped or "(none)", "n_features": len(feats), "dev_pair_acc": acc})
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ac.SWEEP_DIR / "ablation.csv", index=False)
    return out


if __name__ == "__main__":
    {"importance": importance, "ablation": ablation}[sys.argv[1]]()
