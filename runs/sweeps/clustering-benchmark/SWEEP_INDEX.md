# Clustering benchmark: sweep index

Main columns are all 16 meetings. DER is a percentage; seconds are summed. Tune half = meetings ending a/c, report half = b/d. Regenerated from `points.jsonl` by `python -m harness.clustering_benchmark index`; do not hand-edit.

## VBx baseline reference (run `20260925T101811Z-db977741`, not re-run)

| half | DER | MD s | FA s | Conf s | Conf % of error | count MAE | signed bias | exact | over / under |
|---|---|---|---|---|---|---|---|---|---|
| tune | 16.26 | 1408.2 | 446.4 | 460.9 | 19.9 | 0.250 | +0.250 | 6/8 | 2 / 0 |
| report | 17.73 | 1516.8 | 652.7 | 751.2 | 25.7 | 0.250 | +0.250 | 6/8 | 2 / 0 |
| all | 17.05 | 2925.0 | 1099.1 | 1212.2 | 23.1 | 0.250 | +0.250 | 12/16 | 4 / 0 |

All 16: der_overlap_system 0.3339, der_overlap_assigned 0.2195, JER 0.2219.

## Sweep points

| point_id | WS | method | params | run_id | DER tune | DER report | DER all | MD s | FA s | Conf s | Conf % of error | der_overlap_system | der_overlap_assigned | JER | overlap caught / missed / invented s | count MAE | signed count bias | exact/16 | outlier rate (HDBSCAN) | est. k range (spectral) | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AGG-0-01 | WS0 | agglomerative | method=centroid, min_cluster_size=12, threshold=0.7 | 20261003T002832Z-1c27ba23 | 17.98 | 17.89 | 17.94 | 2925.0 | 1099.1 | 1484.5 | 26.9 | 0.3585 | 0.2497 | 0.2390 | 2433 / 1394 / 383 | 0.500 | +0.250 | 11 |  |  | 3.1-like setting (unverified) |
| AGG-0-02 | WS0 | agglomerative | method=centroid, min_cluster_size=12, threshold=0.7 | 20261003T002918Z-1f640b87 | 17.98 | 17.89 | 17.94 | 2925.0 | 1099.1 | 1484.5 | 26.9 | 0.3585 | 0.2497 | 0.2390 | 2433 / 1394 / 383 | 0.500 | +0.250 | 11 |  |  | determinism re-run of AGG-0-01, fresh RTTM cache |
