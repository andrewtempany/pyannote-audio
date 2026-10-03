"""Assignment-classifier WS2: pair dataset builder.

Synthetic checks for the temporal matcher, the test-reference guard (ticket
hard-fail 4), and real-data shape checks on one cached test meeting.
"""

import numpy as np
import pytest
from pyannote.core import SlidingWindow, SlidingWindowFeature

from harness import assignment_common as ac
from harness import assignment_dataset as ad
from harness import assignment_labels as al


def _seg(data, step=1.0, duration=10.0):
    return SlidingWindowFeature(data.astype(np.float32), SlidingWindow(start=0.0, duration=duration, step=step))


def test_neighbour_matches_follows_a_voice_across_chunks():
    # 10 frames per 10 s chunk (1 s frames), step 1 s: speaker 0 talks 2-8 s
    # absolute; in chunk c that is frames 2-c..8-c. Local slots swap in chunk 1.
    chunks, frames = 3, 10
    data = np.zeros((chunks, frames, 2))
    for c in range(chunks):
        slot = 1 if c == 1 else 0
        for t in range(2, 9):
            f = t - c
            if 0 <= f < frames:
                data[c, f, slot] = 1
    m = ad.neighbour_matches(_seg(data))
    assert m[1][0, 0] == 1      # chunk 0 slot 0 -> chunk 1 slot 1
    assert m[-1][1, 1] == 0     # chunk 1 slot 1 -> chunk 0 slot 0
    assert m[2][0, 0] == 0      # chunk 0 slot 0 -> chunk 2 slot 0
    assert m[1][0, 1] == -1     # silent slot has no match
    assert m[-1][0, 0] == -1    # no chunk before chunk 0


def test_load_reference_refuses_test_split_for_training():
    with pytest.raises(al.TestReferenceAccess):
        al.load_reference("test", "IS1009a", "training")
    with pytest.raises(ValueError):
        al.load_reference("dev", "IS1008a", "features")


@pytest.fixture(scope="module")
def hdbscan_pipeline():
    try:
        pipeline = ac.load_pipeline("hdbscan")
    except Exception as exc:
        pytest.skip(f"pipeline unavailable: {exc}")
    if not ac.intermediate_cache(pipeline).path("IS1009a").exists():
        pytest.skip("test-split intermediates not cached")
    return pipeline


def test_building_test_rows_never_reads_a_reference(hdbscan_pipeline, monkeypatch):
    requested = []

    def spy(split):
        requested.append(split)
        raise AssertionError(f"reference file for split {split!r} was read")

    monkeypatch.setattr(al, "_split_references", spy)
    ad._WORKER.update(pipeline=hdbscan_pipeline, cache=ac.intermediate_cache(hdbscan_pipeline),
                      method="hdbscan", split="test")
    df = ad._one("IS1009a")
    assert requested == []
    assert df["is_label"].isna().all() and (df["label"] == -1).all()


def test_rows_on_real_meeting(hdbscan_pipeline):
    pipeline = hdbscan_pipeline
    seg, emb = ac.load_intermediates(ac.intermediate_cache(pipeline), "IS1009a")
    m = ac.cluster_meeting(pipeline, seg, emb, "IS1009a")
    df = ad.meeting_rows(m, None, "test", "hdbscan")
    k = m.centroids.shape[0]
    per_pair = df.groupby(["chunk_idx", "speaker_idx"]).size()
    assert len(per_pair) == m.target.sum() and (per_pair == k).all()
    # HDBSCAN assigns by plain nearest centroid, so current == nc for every pair
    assert (df["current_label"] == df["nc_label"]).all()
    # rank 0 is the nearest centroid
    assert (df.loc[df["rank"] == 0, "candidate"].values == df.loc[df["rank"] == 0, "nc_label"].values).all()
    # margin is positive exactly for the nearest candidate (ties aside)
    assert (df.loc[df["rank"] == 0, "margin"] >= 0).all()
    assert (df.loc[df["rank"] > 0, "margin"] <= 0).all()
    for col in ["cand_taken", "cand_size", "clean_ratio", "active_ratio"]:
        assert df[col].between(0, 1).all(), col
    assert df[["ctx_agree_k1", "ctx_agree_k5", "cand_activity_30s"]].stack().dropna().between(0, 1).all()
    # target pairs are held back or outliers, never both kept and inlier
    assert ((df["is_held_back"] == 1) | (df["is_outlier"] == 1)).all()
