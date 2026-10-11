"""Analysis for the clustering-benchmark sweep (ticket §5 WS4).

Reads the ledger written by `harness.clustering_benchmark` and never re-scores
anything: every number comes from the per-file CSVs the harness wrote.

Selection rule (ticket §1): the best setting is the one with the lowest
TUNE-half DER; ties within 0.1 DER points go to the lower tune-half count MAE.
Report-half numbers are never consulted when selecting (§7.7).

Usage:
    python -m harness.clustering_benchmark_report best AGG-1a AGG-1b
    python -m harness.clustering_benchmark_report report
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from harness.clustering_benchmark import COMPONENTS, SWEEP_DIR, half_metrics, load_points

TIE_DER = 0.001  # 0.1 DER points, as a fraction


def scored_points(prefixes: tuple[str, ...] = ()) -> list[dict]:
    points = [p for p in load_points() if "metrics" in p]
    if prefixes:
        points = [p for p in points if p["point_id"].startswith(prefixes)]
    return points


def best_on_tune(points: list[dict]) -> dict:
    """Lowest tune-half DER; within 0.1 points of it, lowest tune-half MAE."""
    if not points:
        raise ValueError("no scored points to select from")
    lowest = min(p["metrics"]["tune"]["der"] for p in points)
    tied = [p for p in points if p["metrics"]["tune"]["der"] - lowest <= TIE_DER]
    return min(tied, key=lambda p: (p["metrics"]["tune"]["count_mae"], p["metrics"]["tune"]["der"]))


def worst_on_tune(points: list[dict]) -> dict:
    return max(points, key=lambda p: p["metrics"]["tune"]["der"])


def per_file() -> pd.DataFrame:
    return pd.read_csv(SWEEP_DIR / "all_points_per_file.csv")


def _der(rows: pd.DataFrame) -> float:
    errors = rows[list(COMPONENTS)].sum(axis=1)
    return float(errors.sum() / (errors / rows["der"]).sum())


def paired_bootstrap_delta(point_id: str, baseline_id: str = "VBX-baseline",
                           half: str = "report", draws: int = 2000, seed: int = 0) -> dict:
    """Delta DER (point - baseline) on `half`, with a 90% paired bootstrap
    interval: resample that half's meetings with replacement, recompute both
    half-level DERs on the same resampled meetings, take the 5th/95th
    percentiles of the difference."""
    table = per_file()
    rows = table[table["half"] == half]
    point = rows[rows["point_id"] == point_id].set_index("uri")
    base = rows[rows["point_id"] == baseline_id].set_index("uri")
    uris = sorted(point.index)
    assert uris == sorted(base.index) and len(uris) == 8, (point_id, uris)

    rng = np.random.default_rng(seed)
    deltas = np.empty(draws)
    for i in range(draws):
        sample = rng.choice(uris, size=len(uris), replace=True)
        deltas[i] = _der(point.loc[sample]) - _der(base.loc[sample])
    return {
        "delta": _der(point.loc[uris]) - _der(base.loc[uris]),
        "low": float(np.percentile(deltas, 5)),
        "high": float(np.percentile(deltas, 95)),
    }


# -- WS4 report ---------------------------------------------------------------

# Reference palette (dataviz skill, light surface); categorical slots in fixed order.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FIGURES = SWEEP_DIR / "figures"

METHODS = {
    # WS0 points are setup checks, not part of the method's sweep
    "agglomerative": ("AGG-1",),
    "hdbscan": ("HDB-",),
    "spectral": ("SPC-",),
}
COSINE_LINKAGES = ["average", "complete", "single", "weighted"]
EUCLIDEAN_LINKAGES = ["centroid", "median", "ward"]


def _baseline() -> dict:
    return json.loads((SWEEP_DIR / "baseline_reference.json").read_text())["metrics"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _unique_by_params(points):
    seen, out = set(), []
    for p in points:
        key = tuple(sorted(p["params"].items()))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _sweep_chart(path, title, panels, xlabel, baseline, logx=False):
    """panels: [(panel_title, {series: [(x, der_all, mae_all), ...]})]. Two rows
    sharing x: DER with the VBx baseline as a dashed line, then count MAE."""
    plt = _pyplot()
    fig, axes = plt.subplots(2, len(panels), figsize=(max(7.5, 5.4 * len(panels)), 6.2), squeeze=False,
                             sharex="col", gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor(SURFACE)
    for col, (panel_title, series) in enumerate(panels):
        der_ax, mae_ax = axes[0][col], axes[1][col]
        for ax in (der_ax, mae_ax):
            _style(ax)
            if logx:
                ax.set_xscale("log")
        der_ax.axhline(100 * baseline["all"]["der"], color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
        der_ax.text(0.99, 100 * baseline["all"]["der"], "VBx baseline 17.05",
                    transform=der_ax.get_yaxis_transform(), color=TEXT_2, fontsize=7.5,
                    va="bottom", ha="right")
        mae_ax.axhline(baseline["all"]["count_mae"], color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
        if logx:
            ticks = sorted({r[0] for rows in series.values() for r in rows})
            for ax in (der_ax, mae_ax):
                ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
                ax.minorticks_off()
        for i, (name, rows) in enumerate(series.items()):
            rows = sorted(rows)
            xs = [r[0] for r in rows]
            style = dict(color=SERIES[i % len(SERIES)], linewidth=2, marker="o", markersize=4.5,
                         markeredgecolor=SURFACE, markeredgewidth=1)
            der_ax.plot(xs, [100 * r[1] for r in rows], label=name, **style)
            mae_ax.plot(xs, [r[2] for r in rows], **style)
        der_ax.set_title(panel_title, color=TEXT, fontsize=10, loc="left")
        der_ax.set_ylabel("DER, all 16 meetings (%)", color=TEXT_2, fontsize=9)
        mae_ax.set_ylabel("speaker count MAE", color=TEXT_2, fontsize=9)
        mae_ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
        der_ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.suptitle(title, color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _xy(points, x_name):
    return [(float(p["params"][x_name]), p["metrics"]["all"]["der"], p["metrics"]["all"]["count_mae"])
            for p in points]


def charts() -> list:
    base = _baseline()
    made = []
    agg = [p for p in _unique_by_params(scored_points(METHODS["agglomerative"]))
           if p["params"].get("min_cluster_size") == "12"]
    if agg:
        panels = []
        for family, names in (("cosine linkages", COSINE_LINKAGES),
                              ("euclidean linkages (unit-normalised)", EUCLIDEAN_LINKAGES)):
            series = {n + (" (off its own scale, see notes)" if n == "ward" else ""):
                      _xy([p for p in agg if p["params"]["method"] == n], "threshold") for n in names}
            series = {n: rows for n, rows in series.items() if rows}
            if series:
                panels.append((family, series))
        path = FIGURES / "agglomerative_threshold.png"
        _sweep_chart(path, "Agglomerative (min_cluster_size 12): DER and counting vs threshold",
                     panels, "threshold", base)
        made.append(path)
    hdb = [p for p in scored_points(METHODS["hdbscan"]) if p["params"].get("min_samples") == "0"]
    if hdb:
        series = {sel: _xy([p for p in hdb if p["params"]["cluster_selection_method"] == sel], "min_cluster_size")
                  for sel in ("eom", "leaf")}
        path = FIGURES / "hdbscan_min_cluster_size.png"
        _sweep_chart(path, "HDBSCAN (min_samples default): DER and counting vs min_cluster_size",
                     [("cluster_selection_method", {k: v for k, v in series.items() if v})],
                     "min_cluster_size (log scale)", base, logx=True)
        made.append(path)
    spc = [p for p in scored_points(METHODS["spectral"]) if p["params"].get("max_speakers") == "10"]
    if spc:
        path = FIGURES / "spectral_pruning.png"
        _sweep_chart(path, "Spectral + eigengap (max_speakers 10): DER and counting vs pruning",
                     [("pruning = fraction of similarities kept per row", {"max_speakers 10": _xy(spc, "pruning")})],
                     "pruning (log scale)", base, logx=True)
        made.append(path)
    return made


def best_points() -> dict:
    return {m: best_on_tune(scored_points(prefixes)) for m, prefixes in METHODS.items()
            if scored_points(prefixes)}


def error_profile_chart(best: dict):
    plt = _pyplot()
    base = _baseline()["all"]
    short = {"agglomerative": "centroid, t=0.75", "hdbscan": "leaf, mcs=40", "spectral": "pruning=0.20"}
    labels = ["VBx\n(baseline)"] + [f"{m}\n{short.get(m, best[m]['point_id'])}" for m in best]
    rows = [base] + [best[m]["metrics"]["all"] for m in best]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    bottom = np.zeros(len(rows))
    for i, (key, name) in enumerate((("md", "missed speech"), ("fa", "false alarm"), ("conf", "confusion"))):
        values = np.array([r[key] for r in rows])
        ax.bar(range(len(rows)), values, bottom=bottom, color=SERIES[i], width=0.55,
               edgecolor=SURFACE, linewidth=2, label=name)
        if key == "conf":
            for x, (b, v) in enumerate(zip(bottom, values)):
                ax.text(x, b + v / 2, f"confusion {v:.0f} s", ha="center", va="center", color=TEXT, fontsize=7.5)
        bottom += values
    for x, total in enumerate(bottom):
        ax.text(x, total + 30, f"{total:.0f} s", ha="center", va="bottom", color=TEXT_2, fontsize=8)
    ax.set_xticks(range(len(rows)), labels, fontsize=8, color=TEXT_2)
    ax.set_ylabel("error, seconds (all 16 meetings)", color=TEXT_2, fontsize=9)
    ax.set_ylim(0, bottom.max() * 1.15)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, ncols=3, loc="upper left")
    ax.set_title("Error profile: what kind of mistakes each method's best setting makes",
                 color=TEXT, fontsize=10, loc="left")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "error_profile.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def _pct(x):
    return f"{100 * x:.2f}"


def results_table(best: dict) -> str:
    base = _baseline()
    a = base["all"]
    lines = [
        "| Method | Setting | DER tune | DER report | DER all | MD s | FA s | Conf s | Count MAE | Exact /16 | Δ DER report vs VBx [90% CI] |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        f"| VBx (baseline) | Fa=0.07, Fb=0.8, threshold=0.6 | {_pct(base['tune']['der'])} | {_pct(base['report']['der'])} | "
        f"{_pct(a['der'])} | {a['md']:.1f} | {a['fa']:.1f} | {a['conf']:.1f} | {a['count_mae']:.3f} | {a['exact']} | – |",
    ]
    for method, prefixes in METHODS.items():
        points = scored_points(prefixes)
        if not points:
            continue
        for tag, p in (("best", best[method]), ("worst", worst_on_tune(points))):
            m, a = p["metrics"], p["metrics"]["all"]
            ci = paired_bootstrap_delta(p["point_id"])
            lines.append(
                f"| {method} ({tag}, {p['point_id']}) | {p['params_text']} | {_pct(m['tune']['der'])} | "
                f"{_pct(m['report']['der'])} | {_pct(a['der'])} | {a['md']:.1f} | {a['fa']:.1f} | {a['conf']:.1f} | "
                f"{a['count_mae']:.3f} | {a['exact']} | "
                f"{100*ci['delta']:+.2f} [{100*ci['low']:+.2f}, {100*ci['high']:+.2f}] |")
    return "\n".join(lines)


def error_profile_table(best: dict) -> str:
    base = _baseline()
    entries = [("VBx (baseline)", "VBX-baseline", base)]
    entries += [(f"{method} ({p['point_id']})", p["point_id"], p["metrics"]) for method, p in best.items()]
    table = per_file()
    lines = [
        "| Method | Half | MD / FA / Conf s | MD / FA / Conf % of error | der_overlap_system | der_overlap_assigned | JER | "
        "overlap caught / missed / invented s | count MAE | signed bias | over / under / exact |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, pid, m in entries:
        for half in ("all", "report"):
            h = m[half]
            if half == "all":
                ovs, ova, jer = (f"{m[k]:.4f}" for k in ("der_overlap_system", "der_overlap_assigned", "jer"))
            else:
                rows = table[(table["point_id"] == pid) & (table["half"] == "report")]
                ovs, ova, jer = (f"{rows[k].mean():.4f}\\*" for k in ("der_overlap_system", "der_overlap_assigned", "jer"))
            lines.append(
                f"| {name} | {half} | {h['md']:.0f} / {h['fa']:.0f} / {h['conf']:.0f} | "
                f"{100*h['md_share']:.1f} / {100*h['fa_share']:.1f} / {100*h['conf_share']:.1f} | {ovs} | {ova} | {jer} | "
                f"{h['region_t_and_d']:.0f} / {h['region_t_minus_d']:.0f} / {h['region_d_minus_t']:.0f} | "
                f"{h['count_mae']:.3f} | {h['count_bias']:+.3f} | {h['over']} / {h['under']} / {h['exact']} |")
    lines += ["", "\\* Report-half overlap DER and JER are the mean of per-file values, not corpus-level (ticket §1)."]
    return "\n".join(lines)


def meetings_vs_vbx(best: dict, n: int = 3) -> str:
    table = per_file().set_index(["point_id", "uri"])
    base = table.loc["VBX-baseline"]["der"]
    lines = []
    for method, p in best.items():
        diff = (table.loc[p["point_id"]]["der"] - base).sort_values()
        better = ", ".join(f"{u} {100*d:+.2f}" for u, d in diff.head(n).items())
        worse = ", ".join(f"{u} {100*d:+.2f}" for u, d in diff.tail(n)[::-1].items())
        lines.append(f"- **{method}** ({p['point_id']}) per-meeting DER vs VBx, points. Best: {better}. Worst: {worse}.")
    return "\n".join(lines)


def hdbscan_outliers(point: dict) -> str:
    folder = SWEEP_DIR / "hdbscan" / point["point_id"]
    rows = pd.read_csv(folder / "outliers.csv")
    diagnostics = json.loads((folder / "diagnostics.json").read_text())
    outliers, inliers = rows[rows["is_outlier"]], rows[~rows["is_outlier"]]
    per_meeting = rows.groupby("uri")["is_outlier"].mean().sort_values()
    dropped = {u: (d["num_dropped_by_filter"], d["num_active"]) for u, d in diagnostics.items()}
    drop_rate = sum(v[0] for v in dropped.values()) / sum(v[1] for v in dropped.values())
    lines = [
        f"At {point['point_id']} ({point['params_text']}): {len(outliers)} of {len(rows)} training embeddings "
        f"({100*len(outliers)/len(rows):.1f}%) were outliers.",
        "",
        f"- Mean overlap_fraction: outliers {outliers['overlap_fraction'].mean():.3f}, "
        f"non-outliers {inliers['overlap_fraction'].mean():.3f}.",
        f"- Share with any overlap at all: outliers {100*(outliers['overlap_fraction'] > 0).mean():.1f}%, "
        f"non-outliers {100*(inliers['overlap_fraction'] > 0).mean():.1f}%.",
        f"- Outlier rate per meeting: {100*per_meeting.min():.1f}% ({per_meeting.index[0]}) to "
        f"{100*per_meeting.max():.1f}% ({per_meeting.index[-1]}).",
        f"- Before any clustering method runs, the pipeline's own filter (< 20% solo speech) held back "
        f"{100*drop_rate:.1f}% of active embeddings. Same for every method.",
        "",
        "| meeting | outlier rate | held back by filter / active |",
        "|---|---|---|",
    ]
    lines += [f"| {u} | {100*r:.1f}% | {dropped[u][0]} / {dropped[u][1]} |" for u, r in per_meeting.items()]
    return "\n".join(lines)


def report() -> str:
    best = best_points()
    made = charts()
    parts = [
        "# Clustering benchmark report (generated)", "",
        "Generated by `python -m harness.clustering_benchmark_report report`. Best = lowest tune-half DER.", "",
        "## Results", "", results_table(best), "",
        "## Error profiles", "", error_profile_table(best), "",
        f"![error profile](figures/{error_profile_chart(best).name})", "",
        "### Meetings vs VBx", "", meetings_vs_vbx(best), "",
        "## Charts", "", *[f"![{p.stem}](figures/{p.name})" for p in made], "",
    ]
    if "hdbscan" in best and (SWEEP_DIR / "hdbscan" / best["hdbscan"]["point_id"] / "outliers.csv").exists():
        parts += ["## HDBSCAN outliers", "", hdbscan_outliers(best["hdbscan"]), ""]
    text = "\n".join(parts)
    (SWEEP_DIR / "REPORT.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    best = sub.add_parser("best", help="best point on the tune half among point-id prefixes")
    best.add_argument("prefixes", nargs="+")
    sub.add_parser("report", help="write REPORT.md and figures/")
    args = parser.parse_args(argv)

    if args.command == "report":
        report()
        print(SWEEP_DIR / "REPORT.md")
    elif args.command == "best":
        points = scored_points(tuple(args.prefixes))
        chosen = best_on_tune(points)
        tune = chosen["metrics"]["tune"]
        print(json.dumps({
            "point_id": chosen["point_id"], "params": chosen["params"],
            "tune_der": tune["der"], "tune_mae": tune["count_mae"], "candidates": len(points),
        }))


if __name__ == "__main__":
    main()
