"""Report figures and headline-number check for the final report.

Run from the repo root:  python3 report/make_figures.py
Writes report/figures/fig{1,2,3}_*.{pdf,png} and prints every number the
figures use, recomputed from the per-file CSVs (nothing is typed in by hand).

Conventions match Closeout Results.md: DER = total error / total reference
speech over the 16 AMI test meetings; 90% paired percentile bootstrap,
2000 draws, seed 0, meetings sorted by uri, rng.integers(0, n, n).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "sweeps"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- style (IEEE single column = 3.5 in) ----------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})
W = 3.5


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ---------- data helpers ----------
def per_file(path, **filt):
    df = pd.read_csv(path)
    for k, v in filt.items():
        df = df[df[k] == v]
    df = df.sort_values("uri").reset_index(drop=True)
    err = df.missed_detection + df.false_alarm + df.confusion
    return pd.DataFrame({"uri": df.uri, "err": err, "tot": err / df.der})


def der(d, idx=None):
    if idx is not None:
        d = d.iloc[idx]
    return 100 * d.err.sum() / d.tot.sum()


def paired(a, b, draws=2000, seed=0):
    """Delta = DER(a) - DER(b), paired over meetings."""
    assert list(a.uri) == list(b.uri)
    rng = np.random.default_rng(seed)
    n = len(a)
    ds = []
    for _ in range(draws):
        i = rng.integers(0, n, n)
        ds.append(der(a, i) - der(b, i))
    return der(a) - der(b), *np.percentile(ds, [5, 95])


TR = RUNS / "assignment-classifier" / "test_runs"
CB = RUNS / "clustering-benchmark" / "all_points_per_file.csv"
ihm = {
    "VBx": per_file(TR / "vbx-identity" / "per_file.csv"),
    "Agglomerative": per_file(CB, point_id="AGG-1a-07"),
    "HDBSCAN": per_file(CB, point_id="HDB-2b-04"),
    "Spectral": per_file(CB, point_id="SPC-3a-03"),
    "Ward": per_file(RUNS / "ward-redo" / "WARD-T8" / "per_file.csv"),
    "Nearest centroid": per_file(TR / "vbx-nc_target" / "per_file.csv"),
    "kNN (k=25)": per_file(TR / "vbx-knn_target_k25" / "per_file.csv"),
    "Trees (argmax)": per_file(TR / "vbx-classifier-hgb" / "per_file.csv"),
    "MLP (argmax)": per_file(TR / "vbx-classifier-mlp" / "per_file.csv"),
    "Oracle, overlap-degraded": per_file(ROOT / "per_file_oracle_overlap_degraded.csv"),
    "Oracle, all pairs": per_file(ROOT / "per_file_oracle_all_pairs.csv"),
}
base = ihm["VBx"]

# SDM: same reference and UEM as IHM, so per-meeting totals are IHM's
sdm_raw = pd.read_csv(RUNS / "sdm" / "per_meeting.csv")
SDM_KEY = {"VBx": "vbx", "Agglomerative": "agglomerative", "HDBSCAN": "hdbscan",
           "Spectral": "spectral", "Ward": "ward", "Nearest centroid": "nc_target",
           "kNN (k=25)": "knn_target_k25"}
sdm = {}
for name, key in SDM_KEY.items():
    d = sdm_raw[sdm_raw.method == key].sort_values("uri").reset_index(drop=True)
    sdm[name] = pd.DataFrame({"uri": d.uri, "tot": base.tot.values,
                              "err": d.sdm_der.values * base.tot.values})

print("== DER and paired delta vs VBx (90% CI) ==")
rows = []
for name, d in ihm.items():
    r = {"method": name, "ihm_der": der(d)}
    if name != "VBx":
        r["ihm_d"], r["ihm_lo"], r["ihm_hi"] = paired(d, base)
    if name in sdm:
        r["sdm_der"] = der(sdm[name])
        if name != "VBx":
            r["sdm_d"], r["sdm_lo"], r["sdm_hi"] = paired(sdm[name], sdm["VBx"])
    rows.append(r)
res = pd.DataFrame(rows)
with pd.option_context("display.float_format", "{:+.2f}".format, "display.width", 140):
    print(res.to_string(index=False))
res.to_csv(OUT / "method_deltas.csv", index=False)

# ---------- Fig 1: error budget ----------
ref = base.tot.sum()
ihm_b = pd.read_csv(TR / "vbx-identity" / "per_file.csv")
ihm_o = pd.read_csv(ROOT / "per_file_oracle_v2.csv")  # IHM oracle segmentation, 3.96%
sdm_b_path = RUNS / "sdm" / "vbx" / "per_file.csv"
sdm_o = pd.read_csv(RUNS / "sdm-oracle" / "per_file.csv")


def comps(df):
    return [df.missed_detection.sum(), df.false_alarm.sum(), df.confusion.sum()]


budget = {
    "IHM, system segmentation": comps(ihm_b),
    "IHM, oracle segmentation": comps(ihm_o),
    "SDM, system segmentation": comps(pd.read_csv(sdm_b_path)),
    "SDM, oracle segmentation": comps(sdm_o),
}
print(f"\n== Error budget (% of reference speech; ref = {ref:.1f} s) ==")
for k, v in budget.items():
    p = [100 * x / ref for x in v]
    print(f"{k:28s} MD {p[0]:5.2f}  FA {p[1]:5.2f}  Conf {p[2]:5.2f}  DER {sum(p):6.2f}")

fig, ax = plt.subplots(figsize=(W, 1.65))
labels = list(budget)[::-1]
for y, k in enumerate(labels):
    p = [100 * x / ref for x in budget[k]]
    left = 0
    for val, col, nm in zip(p, (BLUE, ORANGE, AQUA), ("Missed", "False alarm", "Confusion")):
        ax.barh(y, val, left=left, color=col, height=0.62, edgecolor="white", linewidth=1)
        if val > 1.6:
            ax.text(left + val / 2, y, f"{val:.1f}", ha="center", va="center",
                    color="white", fontsize=6.5, fontweight="bold")
        left += val
    ax.text(left + 0.3, y, f"{left:.2f}%", va="center", color=INK, fontsize=7)
ax.set_yticks(range(len(labels)), labels)
ax.set_xlabel("Error, % of reference speech (DER)")
ax.set_xlim(0, 22.5)
ax.xaxis.grid(True, color=GRID, linewidth=0.5)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=l) for c, l in
                   ((BLUE, "Missed speech"), (ORANGE, "False alarm"), (AQUA, "Confusion"))],
          ncol=3, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False,
          handlelength=1, columnspacing=1)
save(fig, "fig1_error_budget")

# ---------- Fig 2: forest plot of deltas ----------
groups = [
    ("Clustering method", ["Agglomerative", "HDBSCAN", "Spectral", "Ward"]),
    ("Assignment on VBx", ["Nearest centroid", "kNN (k=25)",
                                                 "Trees (argmax)", "MLP (argmax)"]),
    ("Oracle ceilings", ["Oracle, overlap-degraded", "Oracle, all pairs"]),
]
order, heads = [], []
for g, names in groups:
    heads.append((len(order), g))
    order += names
r = res.set_index("method")
fig, ax = plt.subplots(figsize=(W, 3.0))
ax.axvspan(-0.3, 0.3, color="#eeeeec", zorder=0)
ax.axvline(0, color=INK2, linewidth=0.7, zorder=1)
ypos = {}
y = 0
for gi, (g, names) in enumerate(groups):
    if gi:
        y += 0.6
    ax.text(-0.02, y - 0.6, g, fontsize=7, fontweight="bold", color=INK, ha="right",
            transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData))
    for n in names:
        ypos[n] = y
        y += 1
    y += 0.4
for n, yy in ypos.items():
    row = r.loc[n]
    oracle = n.startswith("Oracle")
    ax.errorbar(row.ihm_d, yy, xerr=[[row.ihm_d - row.ihm_lo], [row.ihm_hi - row.ihm_d]],
                fmt="o", color=INK if oracle else BLUE, ms=4, capsize=0, elinewidth=1.2, zorder=3)
    if not np.isnan(row.get("sdm_d", np.nan)):
        ax.errorbar(row.sdm_d, yy + 0.32, xerr=[[row.sdm_d - row.sdm_lo], [row.sdm_hi - row.sdm_d]],
                    fmt="o", mfc="white", color=ORANGE, ms=4, capsize=0, elinewidth=1.0, zorder=3)
ax.set_yticks([ypos[n] + 0.1 for n in order], order)
ax.invert_yaxis()
ax.set_xlim(-2.2, 4.0)
ax.set_xlabel("ΔDER vs VBx default (points), 90% paired CI")
ax.tick_params(axis="y", length=0)
ax.text(0, -1.25, "±0.3 resolution", ha="center", fontsize=6.5, color=INK2)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", ms=4, label="AMI IHM"),
                   Line2D([], [], marker="o", color=ORANGE, mfc="white", ls="", ms=4, label="AMI SDM")],
          loc="lower right", frameon=False)
ax.xaxis.grid(True, color=GRID, linewidth=0.5)
ax.set_axisbelow(True)
save(fig, "fig2_method_deltas")

# ---------- Fig 3: VBx Fb axis ----------
pts = pd.read_csv(RUNS / "vbx-sweep" / "points.csv")
fb = pts[(pts.Fa == 0.07) & (pts.threshold == 0.6)].sort_values("Fb")
print("\n== VBx Fb axis (Fa 0.07, thr 0.6) ==")
print(fb[["Fb", "der", "mean_hyp_count", "count_mae"]].to_string(index=False))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 2.6), sharex=True,
                             gridspec_kw={"hspace": 0.12, "height_ratios": [1, 1]})
a1.plot(fb.Fb, 100 * fb.der, color=BLUE, lw=2, marker="o", ms=4)
a1.set_ylabel("DER (%)")
a1.set_ylim(15, 53)
a2.plot(fb.Fb, fb.mean_hyp_count, color=BLUE, lw=2, marker="o", ms=4)
a2.axhline(fb.mean_ref_count.iloc[0], color=INK2, lw=0.8, ls="--")
a2.text(0.03, fb.mean_ref_count.iloc[0] * 1.12, f"true mean {fb.mean_ref_count.iloc[0]:.2f}",
        fontsize=6.5, color=INK2)
a2.set_yscale("log")
a2.set_yticks([3, 5, 10, 20, 50], ["3", "5", "10", "20", "50"])
a2.minorticks_off()
a2.set_ylabel("Mean predicted\nspeakers")
a2.set_xscale("log")
a2.set_xlabel("Fb (log scale)")
for a in (a1, a2):
    a.axvline(0.8, color=ORANGE, lw=1, zorder=0)
    a.yaxis.grid(True, color=GRID, linewidth=0.5)
    a.set_axisbelow(True)
a1.text(0.9, 46, "shipped Fb = 0.8", color=INK2, fontsize=6.5)
a2.set_xlim(0.008, 20)
save(fig, "fig3_vbx_fb_axis")
print("\nfigures written to", OUT)
