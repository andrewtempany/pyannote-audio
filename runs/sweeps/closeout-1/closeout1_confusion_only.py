"""Closeout 1 section 3 addendum: MD/FA are identical across methods, so correlate the confusion-only rate."""
import itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr
cb = pd.read_csv("runs/sweeps/clustering-benchmark/all_points_per_file.csv")
PTS = {"VBx": "VBX-baseline", "Agglo": "AGG-1a-07", "HDBSCAN": "HDB-2b-04", "Spectral": "SPC-3a-03"}
d = {m: cb[cb.point_id == p].set_index("uri").sort_index() for m, p in PTS.items()}
for c in ["missed_detection", "false_alarm"]:
    print(c, "max abs diff across methods per file:", max(float((d[m][c] - d["VBx"][c]).abs().max()) for m in d))
ref = (d["VBx"][["missed_detection", "false_alarm", "confusion"]].sum(axis=1) / d["VBx"]["der"])
print("MD+FA share of VBx total error:", round(float((d["VBx"].missed_detection + d["VBx"].false_alarm).sum() / d["VBx"][["missed_detection", "false_alarm", "confusion"]].sum().sum()), 3))
conf = pd.DataFrame({m: d[m]["confusion"] / ref * 100 for m in d})  # confusion as % of reference speech
print("CONFUSION RATE (% of ref speech)"); print(conf.round(2).to_string())
sp = pd.DataFrame({a: {b: spearmanr(conf[a], conf[b])[0] for b in conf} for a in conf})
print("Spearman on confusion rate"); print(sp.round(2).to_string())
print("mean off-diag:", round(np.mean([sp.loc[a, b] for a, b in itertools.combinations(conf.columns, 2)]), 3))
md_fa = (d["VBx"].missed_detection + d["VBx"].false_alarm) / ref * 100
print("Spearman of VBx total DER vs MD+FA rate:", round(spearmanr(d["VBx"]["der"], md_fa)[0], 2))
print("Spearman of VBx confusion rate vs MD+FA rate:", round(spearmanr(conf["VBx"], md_fa)[0], 2))
dl = conf.sub(conf["VBx"], axis=0).drop(columns="VBx")
print("Delta confusion vs VBx (pts)"); print(dl.round(2).to_string())
print("Spearman among deltas"); print(dl.corr(method="spearman").round(2).to_string())
P = pd.read_csv("runs/sweeps/closeout-1/closeout1_properties.csv", index_col=0)
for c in ["speakers", "overlap_frac", "duration_s", "turns"]:
    print("VBx confusion rate vs", c, round(spearmanr(conf["VBx"], P[c])[0], 2), " MD+FA rate vs", c, round(spearmanr(md_fa, P[c])[0], 2))
