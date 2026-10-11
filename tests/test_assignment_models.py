"""Assignment-classifier WS4: model freezing, hashing (A8) and the classifier strategy.

Uses a temporary models folder and small fits on dev rows (a code check, not
a selection). Skipped when the dev dataset or test intermediates are absent.
"""

import numpy as np
import pytest

from harness import assignment_common as ac
from harness import assignment_dataset as ad
from harness import assignment_models as am
from harness import assignment_strategies as st


@pytest.fixture()
def tmp_models(tmp_path, monkeypatch):
    if not (ad.DATA_DIR / "dev_hdbscan.csv.gz").exists():
        pytest.skip("dev dataset not built")
    dev = am.training_rows("dev", ["hdbscan"])
    small = dev[dev["uri"].isin(sorted(dev["uri"].unique())[:3])]
    monkeypatch.setattr(am, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(am, "training_rows", lambda split, methods: small)
    return tmp_path


def test_freeze_writes_hash_and_never_overwrites(tmp_models):
    folder = am.freeze("m1", "lr", {"C": 1, "class_weight": None}, "dev", ["hdbscan"])
    assert {p.name for p in folder.iterdir()} == {"model.joblib", "config.json", "sha256.txt"}
    assert am.model_sha256("m1") == (folder / "sha256.txt").read_text().strip()
    with pytest.raises(FileExistsError):
        am.freeze("m1", "lr", {"C": 1, "class_weight": None}, "dev", ["hdbscan"])


def test_a8_different_models_give_different_refinement_ids(tmp_models):
    am.freeze("m1", "lr", {"C": 1, "class_weight": None}, "dev", ["hdbscan"])
    am.freeze("m2", "lr", {"C": 0.01, "class_weight": None}, "dev", ["hdbscan"])
    ids = {st.refinement_id(f"{p}m{i}") for p in st.MODEL_PREFIXES for i in (1, 2)}
    assert len(ids) == 4
    assert am.model_sha256("m1") in st.refinement_id("classifier:m1")


def test_tampered_model_file_is_refused(tmp_models):
    folder = am.freeze("m1", "lr", {"C": 1, "class_weight": None}, "dev", ["hdbscan"])
    with open(folder / "model.joblib", "ab") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError):
        am.model_sha256("m1")


def test_classifier_strategy_changes_targets_only(tmp_models):
    am.freeze("m1", "hgb", {"learning_rate": 0.1, "max_depth": 3, "max_iter": 50,
                            "l2_regularization": 0}, "dev", ["hdbscan"])
    pipeline = ac.load_pipeline("hdbscan")
    cache = ac.intermediate_cache(pipeline)
    if not cache.path("IS1009a").exists():
        pytest.skip("intermediates not cached")
    seg, emb = ac.load_intermediates(cache, "IS1009a")
    m = ac.cluster_meeting(pipeline, seg, emb, "IS1009a")
    for name in ("classifier:m1", "classifier_hungarian:m1"):
        out = st.checked_decision(m, st.decider(name))  # raises on a non-target change
        assert out.shape == m.hard_clusters.shape
        assert out[m.target].min() >= 0 and out[m.target].max() < m.centroids.shape[0]
