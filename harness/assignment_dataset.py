"""Pair dataset for the assignment-classifier ticket (WS2).

One row per (target pair, candidate cluster), for every meeting of a split
and one clustering method, rebuilt on CPU from the pre-clustering cache:

    python -m harness.assignment_dataset build --split dev --method vbx [--workers 6]

writes `runs/sweeps/assignment-classifier/data/<split>_<method>.csv.gz`
(gzip CSV rather than parquet: pyarrow is not installed in diarisation-env).

Labels (`is_label`) come from `harness/assignment_labels.py` for train and dev
only. Test rows always carry `is_label = NaN`: test references are read
solely by the pair evaluator (`harness/assignment_eval.py`), never here.

Feature groups (see FEATURE_GROUPS) follow the ticket's WS2 list exactly.
`knn_sim_k5` / `knn_sim_k25` are inputs to the `knn_target` strategy's k
selection only and are not classifier features.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from harness import assignment_common as ac

DATA_DIR = ac.SWEEP_DIR / "data"

FEATURE_GROUPS = {
    "similarity": ["cos_centroid", "rank", "margin", "soft_score", "soft_missing", "knn_sim"],
    "quality": ["overlap_fraction", "clean_ratio", "active_ratio", "emb_norm"],
    "temporal": ["ctx_agree_k1", "ctx_agree_k5", "ctx_missing_k1", "ctx_missing_k5", "cand_activity_30s"],
    "constraint_size": ["cand_taken", "cand_size"],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

KNN_K = 10
CONTEXT_OFFSETS = 5
MIN_MATCH_SECONDS = 0.5
ACTIVITY_WINDOW_SECONDS = 30.0


def _unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, 1e-8, None)


def _knn_sims(target_vecs, member_vecs, ks=(5, 10, 25)) -> dict:
    """Mean cosine to the k most similar members, per target; NaN if no members."""
    out = {k: np.full(len(target_vecs), np.nan) for k in ks}
    if len(member_vecs) == 0 or len(target_vecs) == 0:
        return out
    sims = target_vecs @ member_vecs.T
    sims.sort(axis=1)
    for k in ks:
        kk = min(k, sims.shape[1])
        out[k] = sims[:, -kk:].mean(axis=1)
    return out


def neighbour_matches(segmentations) -> dict:
    """For each chunk offset d in ±1..±5: (chunks, local_speakers) index of the
    local speaker in chunk c+d whose activity overlaps this pair's most within
    the shared time span, or -1 if none reaches MIN_MATCH_SECONDS.

    Frames are aligned by time: chunk c+d starts d*step seconds later, so frame
    f of chunk c lines up with frame f - round(d*step/frame_duration).
    """
    data = segmentations.data.astype(np.float32)
    num_chunks, num_frames, num_speakers = data.shape
    window = segmentations.sliding_window
    frame_duration = window.duration / num_frames
    matches = {}
    for d in [o for o in range(-CONTEXT_OFFSETS, CONTEXT_OFFSETS + 1) if o != 0]:
        shift = int(round(abs(d) * window.step / frame_duration))
        best = -np.ones((num_chunks, num_speakers), dtype=int)
        if shift >= num_frames or abs(d) >= num_chunks:
            matches[d] = best
            continue
        if d > 0:
            a = data[: num_chunks - d, shift:, :]          # chunk c, later frames
            b = data[d:, : num_frames - shift, :]          # chunk c+d, earlier frames
            rows = np.arange(0, num_chunks - d)
        else:
            a = data[-d:, : num_frames - shift, :]         # chunk c, earlier frames
            b = data[: num_chunks + d, shift:, :]          # chunk c+d (d<0), later frames
            rows = np.arange(-d, num_chunks)
        overlap = np.einsum("cfs,cfo->cso", a, b) * frame_duration
        arg = overlap.argmax(axis=2)
        val = overlap.max(axis=2)
        best[rows] = np.where(val >= MIN_MATCH_SECONDS, arg, -1)
        matches[d] = best
    return matches


def meeting_rows(m: ac.ClusteredMeeting, labels: Optional[np.ndarray],
                 split: str, method: str) -> pd.DataFrame:
    seg = m.segmentations
    data = seg.data
    num_chunks, num_frames, num_speakers = data.shape
    num_clusters = m.centroids.shape[0]
    current = m.hard_clusters
    target = m.target
    nontarget = m.active & ~target

    emb_norm = np.linalg.norm(m.embeddings, axis=-1)
    unit = _unit(m.embeddings)
    cos = unit @ _unit(m.centroids).T  # (c, s, K)

    single = np.sum(data, axis=2, keepdims=True) == 1
    clean_ratio = np.sum(data * single, axis=1) / num_frames
    active_ratio = np.sum(data, axis=1) / num_frames
    ovf = ac.overlap_fraction(seg)

    tc, ts = np.where(target)
    n = len(tc)
    if n == 0:
        return pd.DataFrame()

    # kNN against non-target members of each candidate
    knn = {k: np.full((n, num_clusters), np.nan) for k in (5, 10, 25)}
    tvec = unit[tc, ts]
    for k in range(num_clusters):
        mc, ms = np.where(nontarget & (current == k))
        sims = _knn_sims(tvec, unit[mc, ms])
        for kk in knn:
            knn[kk][:, k] = sims[kk]

    # temporal context: matched neighbours' current labels (non-target only)
    matches = neighbour_matches(seg)
    agree = {1: np.zeros((n, num_clusters)), 5: np.zeros((n, num_clusters))}
    seen = {1: np.zeros(n), 5: np.zeros(n)}
    for d, best in matches.items():
        o = best[tc, ts]
        nc = tc + d
        ok = (o >= 0) & (nc >= 0) & (nc < num_chunks)
        ok[ok] &= nontarget[nc[ok], o[ok]]
        lab = np.full(n, -1)
        lab[ok] = current[nc[ok], o[ok]]
        for horizon in (1, 5):
            if abs(d) <= horizon:
                seen[horizon] += ok
                agree[horizon][np.arange(n)[ok], lab[ok]] += 1
    ctx = {}
    for horizon in (1, 5):
        with np.errstate(invalid="ignore", divide="ignore"):
            ctx[horizon] = agree[horizon] / seen[horizon][:, None]

    # candidate activity within ±30 s and candidate size, from non-target pairs
    per_chunk = np.zeros((num_chunks, num_clusters))
    for k in range(num_clusters):
        per_chunk[:, k] = np.sum(nontarget & (current == k), axis=1)
    half = int(round(ACTIVITY_WINDOW_SECONDS / seg.sliding_window.step))
    csum = np.vstack([np.zeros((1, num_clusters)), np.cumsum(per_chunk, axis=0)])
    lo = np.clip(tc - half, 0, num_chunks)
    hi = np.clip(tc + half + 1, 0, num_chunks)
    window_counts = csum[hi] - csum[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        activity = window_counts / window_counts.sum(axis=1, keepdims=True)
    size = per_chunk.sum(axis=0) / max(per_chunk.sum(), 1)

    # cand_taken: another non-target active local speaker in the chunk has k
    taken = np.zeros((n, num_clusters))
    for other in range(num_speakers):
        mask = (other != ts) & nontarget[tc, other]
        taken[np.arange(n)[mask], current[tc[mask], other]] = 1

    soft = m.soft_clusters[tc, ts] if m.soft_clusters is not None else np.full((n, num_clusters), np.nan)
    tcos = cos[tc, ts]
    order = np.argsort(-tcos, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(n)[:, None], order] = np.arange(num_clusters)[None, :]
    best_other = np.empty((n, num_clusters))
    for k in range(num_clusters):
        others = np.delete(tcos, k, axis=1)
        best_other[:, k] = others.max(axis=1) if others.shape[1] else np.nan

    lab = labels[tc, ts] if labels is not None else None
    rows = {
        "split": split, "method": method, "uri": m.uri,
        "chunk_idx": np.repeat(tc, num_clusters),
        "speaker_idx": np.repeat(ts, num_clusters),
        "candidate": np.tile(np.arange(num_clusters), n),
        "num_candidates": num_clusters,
        "is_outlier": np.repeat(m.outlier[tc, ts], num_clusters).astype(int),
        "is_held_back": np.repeat(m.held_back[tc, ts], num_clusters).astype(int),
        "current_label": np.repeat(current[tc, ts], num_clusters),
        "nc_label": np.repeat(tcos.argmax(axis=1), num_clusters),
        "overlap_bin": np.repeat(ac.overlap_bin(ovf[tc, ts]), num_clusters),
        "cos_centroid": tcos.ravel(),
        "rank": rank.ravel(),
        "margin": (tcos - best_other).ravel(),
        "soft_score": soft.ravel(),
        "knn_sim": knn[10].ravel(),
        "knn_sim_k5": knn[5].ravel(),
        "knn_sim_k25": knn[25].ravel(),
        "overlap_fraction": np.repeat(ovf[tc, ts], num_clusters),
        "clean_ratio": np.repeat(clean_ratio[tc, ts], num_clusters),
        "active_ratio": np.repeat(active_ratio[tc, ts], num_clusters),
        "emb_norm": np.repeat(emb_norm[tc, ts], num_clusters),
        "ctx_agree_k1": ctx[1].ravel(),
        "ctx_agree_k5": ctx[5].ravel(),
        "cand_activity_30s": activity.ravel(),
        "cand_taken": taken.ravel(),
        "cand_size": np.tile(size, n),
    }
    df = pd.DataFrame(rows)
    df["soft_missing"] = df["soft_score"].isna().astype(int)
    df["ctx_missing_k1"] = df["ctx_agree_k1"].isna().astype(int)
    df["ctx_missing_k5"] = df["ctx_agree_k5"].isna().astype(int)
    if lab is None:
        df["label"] = -1
        df["is_label"] = np.nan
    else:
        df["label"] = np.repeat(lab, num_clusters)
        df["is_label"] = np.where(df["label"] >= 0, (df["candidate"] == df["label"]).astype(float), np.nan)
    return df


# -- split driver (multiprocessing; each worker loads the pipeline once) --------

_WORKER = {}


def _init(method: str, split: str):
    _WORKER["pipeline"] = ac.load_pipeline(method)
    _WORKER["cache"] = ac.intermediate_cache(_WORKER["pipeline"])
    _WORKER["method"], _WORKER["split"] = method, split


def _one(uri: str) -> pd.DataFrame:
    from harness.assignment_labels import load_reference, pair_labels

    pipeline, cache = _WORKER["pipeline"], _WORKER["cache"]
    split, method = _WORKER["split"], _WORKER["method"]
    seg, emb = ac.load_intermediates(cache, uri)
    m = ac.cluster_meeting(pipeline, seg, emb, uri)
    labels = None
    if split != "test":
        labels = pair_labels(m, load_reference(split, uri, "training"))
    return meeting_rows(m, labels, split, method)


def build_split(split: str, method: str, workers: int = 6, uris=None) -> Path:
    from multiprocessing import get_context

    uris = uris or ac.meeting_uris(split)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if workers <= 1:
        _init(method, split)
        frames = [_one(u) for u in uris]
    else:
        with get_context("spawn").Pool(workers, initializer=_init, initargs=(method, split)) as pool:
            frames = pool.map(_one, uris, chunksize=1)
    df = pd.concat(frames, ignore_index=True)
    out = DATA_DIR / f"{split}_{method}.csv.gz"
    df.to_csv(out, index=False, compression="gzip")
    print(f"{split}/{method}: {len(uris)} meetings, {df.groupby(['uri','chunk_idx','speaker_idx']).ngroups} "
          f"target pairs, {len(df)} rows -> {out} ({time.time() - started:.0f} s)")
    return out


def load(split: str, method: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{split}_{method}.csv.gz")


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split", required=True, choices=["train", "dev", "test"])
    b.add_argument("--method", required=True, choices=list(ac.METHODS))
    b.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    build_split(args.split, args.method, args.workers)


if __name__ == "__main__":
    main(sys.argv[1:])
