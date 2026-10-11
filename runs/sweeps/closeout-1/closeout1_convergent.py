"""Closeout 1 section 3: convergent failure analysis. Read-only. Run from repo root with diarisation-env python."""
import itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from pyannote.database.util import load_rttm, load_uem

REPO = Path(r"C:\Users\Andrew\Code\pyannote-audio")
IHM = Path.home() / "Code/AMI-diarization-setup/harness-data/IHM"
cb = pd.read_csv(REPO / "runs/sweeps/clustering-benchmark/all_points_per_file.csv")
PTS = {"VBx": "VBX-baseline", "Agglo": "AGG-1a-07", "HDBSCAN": "HDB-2b-04", "Spectral": "SPC-3a-03"}
tab = pd.DataFrame({m: cb[cb.point_id == p].set_index("uri")["der"] for m, p in PTS.items()}).sort_index() * 100
assert len(tab) == 16
pd.set_option("display.width", 200)
print("PER-MEETING DER (%)"); print(tab.round(2).to_string())
print("column DER means (unweighted):", tab.mean().round(2).to_dict())

print("\nSPEARMAN pairwise (DER values)")
sp = pd.DataFrame(index=tab.columns, columns=tab.columns, dtype=float)
for a, b in itertools.product(tab.columns, tab.columns):
    sp.loc[a, b] = spearmanr(tab[a], tab[b])[0]
print(sp.round(2).to_string())
print("mean off-diagonal rho:", round(np.mean([sp.loc[a, b] for a, b in itertools.combinations(tab.columns, 2)]), 3))
print("PEARSON pairwise"); print(tab.corr().round(2).to_string())

ranks = tab.rank(ascending=False)  # 1 = worst
print("\nRANK (1 = worst DER)"); print(ranks.astype(int).to_string())
worst4 = {m: set(tab[m].nlargest(4).index) for m in tab}
print("\nworst quartile (top 4) per method:")
for m, s in worst4.items(): print(" ", m, sorted(s, key=lambda u: -tab.loc[u, m]))
print("in worst quartile of ALL four:", sorted(set.intersection(*worst4.values())))
print("in worst quartile of >=3:", sorted(u for u in tab.index if sum(u in s for s in worst4.values()) >= 3))
print("overlap size pairwise (of 4):")
for a, b in itertools.combinations(tab.columns, 2): print("  ", a, b, len(worst4[a] & worst4[b]))
best4 = {m: set(tab[m].nsmallest(4).index) for m in tab}
print("best quartile for ALL four:", sorted(set.intersection(*best4.values())))

# disagreement: rank range and per-meeting std of DER and of rank
dis = pd.DataFrame({"rank_range": ranks.max(axis=1) - ranks.min(axis=1), "rank_std": ranks.std(axis=1, ddof=0),
                    "der_range_pts": tab.max(axis=1) - tab.min(axis=1)})
print("\nDISAGREEMENT (sorted by rank_range, then der_range)"); print(dis.sort_values(["rank_range", "der_range_pts"], ascending=False).round(2).head(6).to_string())
print("DER range sorted"); print(dis.sort_values("der_range_pts", ascending=False).round(2).head(4).to_string())

# IS1009b
print("\nIS1009b:"); print(pd.concat([tab.loc["IS1009b"].rename("DER"), ranks.loc["IS1009b"].rename("rank_worst1")], axis=1).round(2).to_string())
print("IS1009b DER minus meeting-median per method:", (tab.loc["IS1009b"] - tab.median()).round(2).to_dict())
print("IS1009b delta vs VBx per method:", (tab.loc["IS1009b"] - tab.loc["IS1009b", "VBx"]).round(2).to_dict())
print("IS1009b DER-delta rank among 16 (most negative = biggest gain):")
dl = tab.sub(tab["VBx"], axis=0)
for m in ["Agglo", "HDBSCAN", "Spectral"]: print("  ", m, "delta", round(dl.loc["IS1009b", m], 2), "rank of gain", int(dl[m].rank().loc["IS1009b"]), "of 16")
# excluding IS1009b
t2 = tab.drop("IS1009b"); print("Spearman excluding IS1009b (n=15):")
for a, b in itertools.combinations(tab.columns, 2): print("  ", a, b, round(spearmanr(t2[a], t2[b])[0], 2))

# properties from reference
ref = load_rttm(str(IHM / "test.rttm")); uem = load_uem(str(IHM / "test.uem"))
rows = []
for uri in tab.index:
    a = ref[uri]; tl = uem[uri]
    ext = tl.extent()
    a = a.crop(tl)
    speech = a.get_timeline().support()
    ov = a.get_overlap()
    rows.append(dict(uri=uri, speakers=len(a.labels()), duration_s=ext.duration, speech_s=speech.duration(),
                     overlap_s=ov.duration(), overlap_frac=ov.duration() / speech.duration(),
                     turns=len(list(a.itertracks()))))
P = pd.DataFrame(rows).set_index("uri")
print("\nPROPERTIES"); print(P.round(3).to_string())
base = tab["VBx"]
print("\nbaseline per-meeting DER vs property: Spearman rho, p")
for c in ["speakers", "overlap_frac", "duration_s", "turns", "speech_s"]:
    r, p = spearmanr(base, P[c]); print(f"  {c}: rho={r:.2f} p={p:.2f}")
print("same for each method:")
for m in tab.columns:
    print(" ", m, {c: round(spearmanr(tab[m], P[c])[0], 2) for c in ["speakers", "overlap_frac", "duration_s", "turns"]})
print("DER by speaker count:", base.groupby(P["speakers"]).agg(["count", "mean"]).round(2).to_dict())
print("(DER as in the CSV `der` column x100 = per-file DER, error / reference speech)")
tab.round(4).to_csv(Path(__file__).with_name("closeout1_per_meeting_der.csv")); P.to_csv(Path(__file__).with_name("closeout1_properties.csv"))
