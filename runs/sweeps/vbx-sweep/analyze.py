"""Falsifiability checks and summary tables for the VBx sweep. Run from the repo root."""
import json, re, collections
from pathlib import Path
import pandas as pd
here = Path("runs/sweeps/vbx-sweep")
pts = [json.loads(l) for l in open(here / "points.jsonl", encoding="utf-8") if l.strip()]
done = [p for p in pts if "metrics" in p]
crash = [p for p in pts if "crash" in p]
failed = [p for p in pts if "failure" in p]
print("ledger records", len(pts), "| scored points", len(done), "| crashed points", len(crash), "| failed attempts", len(failed))
ids = {p["point_id"] for p in done}
assert len(ids) == len(done), "duplicate scored point ids"
keys = [p["final_key"] for p in done]
print("scored points", len(done), "unique final cache keys", len(set(keys)))
print("unique intermediate keys", len({p["intermediate_key"] for p in done}), "(expect 1: clustering is not in the intermediate key)")
print("tallies all 16/16:", sum(p["cache_tally"].startswith("intermediate cache: 16 full hit(s), 0 partial, 0 miss(es), 0 write(s)") for p in done), "of", len(done),
      "| zero-tally (final-RTTM cache):", [p["point_id"] for p in done if p["cache_tally"].startswith("intermediate cache: 0 full")])
bad = 0
for p in done:
    m = json.loads(Path("runs", p["run_id"] + ".json").read_text(encoding="utf-8"))
    cfg = m["run_config"]["clustering_config"]
    got = dict(re.findall(r"(\w+)=([\d.eE+-]+)", cfg))
    ok = cfg.startswith("VBxClustering(") and all(float(got[k]) == v for k, v in p["params"].items()) \
        and m["run_config"]["pipeline_config_id"] == p["final_key"]
    bad += not ok
    if not ok: print("MISMATCH", p["point_id"], cfg)
print("manifest clustering_config + final key match requested:", len(done) - bad, "of", len(done), "mismatches", bad)
print("crash types:", collections.Counter(p["crash"].split(":")[0] for p in crash))
print("crashed (Fa,Fb,thr):", sorted((p["params"]["Fa"], p["params"]["Fb"], p["params"]["threshold"]) for p in crash))
d = pd.read_csv(here / "points.csv")
base = d[(d.Fa == .07) & (d.Fb == .8) & (d.threshold == .6)].iloc[0]
print("baseline DER", repr(base.der), "drift", abs(base.der - 0.17048543579940634))
d["d_all"] = 100 * (d.der - base.der); d["d_tune"] = 100 * (d.tune_der - base.tune_der); d["d_report"] = 100 * (d.report_der - base.report_der)
print("largest DER gain vs shipped (points, all-16):", round(d.d_all.min(), 3), "pts at", d.loc[d.d_all.idxmin(), "point_id"], "| gain >1 pt:", int((d.d_all < -1).sum()))
cols = ["der", "tune_der", "report_der", "md_s", "fa_s", "conf_s", "mean_hyp_count", "count_mae", "count_bias", "exact", "jer", "der_overlap_system", "der_overlap_assigned"]
def show(df, key):
    t = df.sort_values(key)[[key, "run_id"] + cols].copy()
    for c in ("der", "tune_der", "report_der"): t[c] = (100 * t[c]).round(2)
    print(t.round(3).to_string(index=False))
print("\nFb axis"); show(d[(d.Fa == .07) & (d.threshold == .6)], "Fb")
print("\nFa axis (Fb=0.8, thr=0.6)"); show(d[(d.Fb == .8) & (d.threshold == .6)], "Fa")
print("\nthreshold axis (Fa=0.07, Fb=0.8)"); show(d[(d.Fa == .07) & (d.Fb == .8)], "threshold")
print("\nGrid summary over", len(d), "scored points")
print("DER all-16 (pct): min %.2f median %.2f max %.2f" % tuple(100 * d.der.quantile(q) for q in (0, .5, 1)))
print("points within +-0.3 DER pts of shipped:", int((d.d_all.abs() <= .3).sum()), "| better than shipped by >0.3:", int((d.d_all < -.3).sum()), "| worse by >0.3:", int((d.d_all > .3).sum()))
for half, col in (("all-16", "der"), ("tune half", "tune_der"), ("report half", "report_der")):
    r = d.loc[d[col].idxmin()]; print(f"lowest {half} DER among scored points: {100*r[col]:.2f} at Fa={r.Fa:g} Fb={r.Fb:g} thr={r.threshold:g} (this half only; other halves for the same point: all {100*r.der:.2f} tune {100*r.tune_der:.2f} report {100*r.report_der:.2f})")
print("count_bias sign: over-count", int((d.count_bias > 0).sum()), "under-count", int((d.count_bias < 0).sum()), "zero", int((d.count_bias == 0).sum()))
print("points with exact count on >=14/16:", int((d.exact >= 14).sum()))
print("\nFb x Fa slices of all-16 DER pct at thr=0.6 (blank = crashed)")
print((100 * d[d.threshold == .6].pivot(index="Fa", columns="Fb", values="der")).round(2).to_string())
print("\nFb x Fa mean hyp count at thr=0.6")
print(d[d.threshold == .6].pivot(index="Fa", columns="Fb", values="mean_hyp_count").round(2).to_string())
print("\nDER-by-Fb, median over the Fa,thr grid (scored points only), with n:")
g = d.groupby("Fb").agg(n=("der", "size"), der_med=("der", lambda s: 100 * s.median()), hyp_med=("mean_hyp_count", "median")); print(g.round(2).to_string())
