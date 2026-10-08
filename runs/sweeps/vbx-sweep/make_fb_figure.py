"""Figure for the Fb axis (Fa=0.07, threshold=0.6): DER and speaker-count metrics vs Fb."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

here = Path(__file__).parent
d = pd.read_csv(here / "points.csv")
d = d[(d.Fa == 0.07) & (d.threshold == 0.6)].sort_values("Fb")
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for col, lab, st in [("der", "all 16 meetings", "-o"), ("tune_der", "tune half (a/c)", "--s"), ("report_der", "report half (b/d)", ":^")]:
    a.plot(d.Fb, 100 * d[col], st, label=lab, ms=5)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("Fb (log scale)"); a.set_ylabel("DER, % (log scale)")
a.set_title("DER vs Fb (Fa=0.07, threshold=0.6)")
b.plot(d.Fb, d.mean_hyp_count, "-o", label="mean hypothesis speakers / meeting", ms=5)
b.plot(d.Fb, d.count_mae, "--s", label="count MAE", ms=5)
b.axhline(d.mean_ref_count.iloc[0], color="grey", ls=":", label="mean reference speakers (3.94)")
b.set_xscale("log"); b.set_yscale("log"); b.set_xlabel("Fb (log scale)"); b.set_ylabel("speakers (log scale)")
b.set_title("Speaker count vs Fb")
for ax in (a, b):
    ax.axvline(0.8, color="k", lw=0.6, alpha=0.5); ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
fig.text(0.01, -0.03, "Vertical line = shipped Fb 0.8. Fb=0.01 crashed (>127 clusters, int8 overflow) so it has no point.", fontsize=8)
fig.savefig(here / "fb_axis.png", dpi=150, bbox_inches="tight")
