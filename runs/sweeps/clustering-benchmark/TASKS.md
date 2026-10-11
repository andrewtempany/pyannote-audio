# Clustering benchmark: task checklist

Ticket: `Obsidian-Diarisation/Tickets/Open/clustering-benchmark-run.md`. Re-read this file after any context reset.
Driver: `python -m harness.clustering_benchmark {baseline|run|index}`. Ledger: `points.jsonl` -> `SWEEP_INDEX.md`.

## Kickoff
- [x] Read ticket + linked notes; contradictions reported to Andrew
- [x] Assumptions register first pass (A1, A4 verified; A7 unverified)
- [x] Questions answered by Andrew (see ticket Implementation Notes, 2026-10-03)
- [x] Branch `clustering-benchmark` (loose files committed on oracle-experiment first)
- [x] `clustering_*` run conditions added to vocabulary (+test)
- [x] Harness hook `--clustering-diagnostics-dir`
- [x] Sweep driver `harness/clustering_benchmark.py`

## WS0 setup and safety
- [x] Baseline reference values (tune / report / all) -> `baseline_reference.json`
- [x] AGG-0-01: centroid / 12 / 0.70 ("3.1-like setting, unverified"): 16 full hits, 0 writes; manifest config exact; DER != baseline
- [x] AGG-0-02: same point, fresh `--cache-dir` holding a COPY of the intermediates only (an empty dir would be a cold GPU run, a hard stop): DER matches to 1e-6 (A8)
- [x] Commit WS0

## WS1 agglomerative
- [x] 1a centroid x 12 thresholds
- [x] 1b average x 12 thresholds
- [x] 1c min_cluster_size 1/3/6/12/20 at best of 1a/1b (tune half)
- [x] 1d complete/single/weighted/median/ward x 0.4/0.6/0.8/1.0/1.4 (+ edge points if best at edge)
- [x] Record effective min_cluster_size (needs num_train per file)
- [x] Commit WS1

## WS2 HDBSCAN
- [x] Tests first (routing, final-key differs, no -1 labels) shown failing
- [x] `HDBSCANClustering` + enum + vocab; tests pass
- [x] 2a eom x 5, 2b leaf x 5, 2c min_samples 1/5 at best
- [x] Outlier rate per point; outliers.csv with overlap_fraction; dropped-by-filter count per meeting
- [x] Commit WS2

## WS3 spectral
- [x] Tests first (routing, final-key differs, eigengap k=3 on 3 blobs) shown failing
- [x] `SpectralEigengapClustering` + enum + vocab; tests pass
- [x] Speed check on first point (eigsh if > 60 s/file)
- [x] 3a pruning x 6, 3b max_speakers 6/15 at best
- [x] Commit WS3

## WS4 write-up
- [x] Results table (+ bootstrap 90% CI on report-half delta)
- [x] Charts per method
- [x] Error profiles table + stacked bar chart; strength/weakness lines; best/worst 3 meetings vs VBx
- [x] HDBSCAN outlier summary
- [x] Plain-language summary
- [x] Ticket -> Docs conversion (benchmark + three method notes); split written into clustering-vbx

Closed 2026-10-03. Ward on its own scale (thresholds ~4-20) proposed, not approved; documented as untested.
