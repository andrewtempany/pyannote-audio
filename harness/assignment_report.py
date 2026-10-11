"""WS6 report for the assignment-classifier ticket.

    python -m harness.assignment_report      # figures/ + REPORT.md under runs/sweeps/assignment-classifier/

Reads only the ledger (runs.jsonl), per-point folders, dataset_stats.md and
the WS5 csvs. Colours: the dataviz reference palette's categorical slots in
fixed order (validator could not be run: no node on this machine), with marker
shape and direct labels as secondary encoding.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from harness import assignment_common as ac
from harness.assignment_runs import IDENTITY, LEDGER, POINTS_DIR, per_file, point_stats

FIG_DIR = ac.SWEEP_DIR / "figures"
SURFACE, TEXT_1, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
METHOD_NAME = {"vbx": "VBx", "hdbscan": "HDBSCAN leaf/40"}
BIN_LABELS = ["0", "0–0.25", "0.25–0.5", "0.5–1"]

# strategies shown in the by-bin chart, fixed order = fixed colour
BIN_SERIES = [
    ("identity", "clustering as-is"),
    ("nc_target", "nearest centroid"),
    ("knn_target_k25", "nearest neighbours (k=25)"),
    ("classifier:{m}-hgb", "trees, argmax"),
    ("classifier_hungarian:{m}-hgb", "trees + one-per-chunk"),
]
FAMILY = {"non-learned": 0, "learned, argmax": 1, "learned + one-per-chunk": 2}


def records():
    return [json.loads(l) for l in LEDGER.read_text().splitlines()]


def family(strategy: str) -> str:
    if strategy.startswith("classifier_hungarian:"):
        return "learned + one-per-chunk"
    if strategy.startswith("classifier:"):
        return "learned, argmax"
    return "non-learned"


def short(strategy: str) -> str:
    if ":" in strategy:
        prefix, model = strategy.split(":", 1)
        name = model.replace("vbx-", "").replace("hdbscan-", "").replace("pooled-hgb", "pooled")
        return name + (" +H" if prefix == "classifier_hungarian" else "")
    return {"identity": "as-is", "nc_target": "NC", "constrained_target": "NC +H",
            "knn_target_k25": "kNN25"}.get(strategy, strategy)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    return plt


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def chart_by_bin(stats):
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, method in zip(axes, ac.METHODS):
        _style(ax)
        for i, (pattern, label) in enumerate(BIN_SERIES):
            strategy = pattern.format(m=method)
            st = stats.get((method, strategy))
            if st is None:
                continue
            ys = [100 * st[f"acc_{b}"] for b in ac.OVERLAP_BINS]
            ax.plot(range(4), ys, color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.set_xticks(range(4), BIN_LABELS)
        ax.set_xlabel("overlap fraction of the voice", color=TEXT_2, fontsize=8)
        ax.set_title(METHOD_NAME[method] + (" (as-is = nearest centroid; both under the green line)"
                     if method == "hdbscan" else ""), color=TEXT_1, fontsize=10, loc="left")
    axes[0].set_ylabel("pair accuracy, test (%)", color=TEXT_2, fontsize=8)
    axes[0].set_ylim(20, 95)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=TEXT_1, loc="lower left")
    fig.suptitle("Hard-to-place voices: share given the correct speaker, by overlap", color=TEXT_1,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pair_accuracy_by_overlap.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_acc_vs_der(stats):
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, method in zip(axes, ac.METHODS):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.axhline(0, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
        seen, labels = set(), {}
        for (m, strategy), st in stats.items():
            if m != method or "devonly" in strategy:
                continue
            fam = family(strategy)
            i = FAMILY[fam]
            x, y = 100 * st["acc"], 100 * st["d_der"]
            ax.plot(x, y, linestyle="none", marker=MARKERS[i], markersize=8,
                    color=SERIES[i], markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=fam if fam not in seen else None)
            seen.add(fam)
            for key, members in labels.items():  # greedy clustering of nearby points
                if abs(key[0] - x) < 1.2 and abs(key[1] - y) < 0.025:
                    members.append((x, y, short(strategy)))
                    break
            else:
                labels[(x, y)] = [(x, y, short(strategy))]
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + 2.5)
        for group in labels.values():  # one label per cluster of near-coincident points
            x, y = group[0][0], group[0][1]
            ax.annotate(" = ".join(sorted({g[2] for g in group})), (x, y), xytext=(6, 2),
                        textcoords="offset points", fontsize=7, color=TEXT_2)
        ax.set_xlabel("pair accuracy, test (%)", color=TEXT_2, fontsize=8)
        ax.set_title(METHOD_NAME[method], color=TEXT_1, fontsize=10, loc="left")
    axes[0].set_ylabel("ΔDER vs clustering as-is (points; below 0 = better)", color=TEXT_2, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=TEXT_1, loc="upper left")
    fig.suptitle("Matching the label better does not lower DER", color=TEXT_1, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pair_accuracy_vs_delta_der.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_importance():
    plt = _pyplot()
    imp = pd.read_csv(ac.SWEEP_DIR / "feature_importance.csv")
    groups = ["similarity", "temporal", "constraint_size", "quality"]
    labels = ["similarity", "temporal context", "constraint / size", "voice quality"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    h = 0.36
    for i, method in enumerate(ac.METHODS):
        sub = imp[imp["method"] == method].set_index("group").loc[groups]
        y = np.arange(len(groups)) + (i - 0.5) * (h + 0.04)
        vals = 100 * sub["acc_drop_mean"].to_numpy()
        ax.barh(y, vals, height=h, color=SERIES[i], label=METHOD_NAME[method], edgecolor=SURFACE, linewidth=2)
        for yy, v in zip(y, vals):
            ax.text(v + 0.5, yy, f"{v:.1f}", va="center", fontsize=7.5, color=TEXT_2)
    ax.set_yticks(range(len(groups)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("drop in dev pair accuracy when the group is shuffled (points)", color=TEXT_2, fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_1, loc="lower right")
    ax.set_title("Which clues the tree model relies on", color=TEXT_1, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_group_importance.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def movers(record, n=3):
    """Per-meeting DER change vs the method's identity run (points)."""
    base = per_file(IDENTITY[record["method"]])
    this = per_file(POINTS_DIR / record["run_point"])
    delta = 100 * (this["der"] - base["der"])
    return delta.nsmallest(n), delta.nlargest(n)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    recs = records()
    stats = {(r["method"], r["strategy"]): point_stats(r) for r in recs}
    chart_by_bin(stats)
    chart_acc_vs_der(stats)
    chart_importance()

    lines = ["# Assignment classifier — report", "",
             "Test split, 16 AMI IHM meetings. Pair accuracy over labelled target pairs; intervals are 90% paired "
             "bootstrap over meetings (2000 draws, seed 0). ΔDER is against the same clustering's own output "
             "(`identity`). Share of ceiling = −ΔDER / 0.97 (all pairs) and / 0.47 (overlap-degraded); VBx only.", "",
             "| clustering | strategy | pair acc [CI] | acc by overlap bin (0 / ≤.25 / ≤.5 / >.5) | DER | MD | FA | Conf "
             "| der_overlap_assigned | JER | count MAE / bias | ΔDER [CI] | ceiling share (0.97 / 0.47) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        s, st = r["summary"], stats[(r["method"], r["strategy"])]
        share = (f"{-1e4 * st['d_der'] / 0.97:.0f}% / {-1e4 * st['d_der'] / 0.47:.0f}%"
                 if r["method"] == "vbx" else "n/a (no oracle run)")
        lines.append(
            f"| {r['method']} | {r['strategy']} | {100 * st['acc']:.1f} [{100 * st['acc_ci'][0]:.1f}, {100 * st['acc_ci'][1]:.1f}] "
            f"| {' / '.join(f'{100 * st[f'acc_{b}']:.1f}' for b in ac.OVERLAP_BINS)} | {100 * s['der']:.2f} "
            f"| {s['missed_detection']:.1f} | {s['false_alarm']:.1f} | {s['confusion']:.1f} | {s['der_overlap_assigned']:.4f} "
            f"| {s['jer']:.4f} | {st['count_mae']:.3f} / {st['count_bias']:+.3f} "
            f"| {100 * st['d_der']:+.2f} [{100 * st['d_der_ci'][0]:+.2f}, {100 * st['d_der_ci'][1]:+.2f}] | {share} |")
    lines += ["", "## Meetings that moved most (best learned point per method, by ΔDER)", ""]
    for method in ac.METHODS:
        learned = [r for r in recs if r["method"] == method and r["strategy"].startswith("classifier")
                   and "devonly" not in r["strategy"] and "pooled" not in r["strategy"]]
        best = min(learned, key=lambda r: stats[(method, r["strategy"])]["d_der"])
        gain, loss = movers(best)
        lines.append(f"- **{method}, {best['strategy']}:** gained most "
                     + ", ".join(f"{u} {v:+.2f}" for u, v in gain.items())
                     + "; lost most " + ", ".join(f"{u} {v:+.2f}" for u, v in loss.items()))
    (ac.SWEEP_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
