# Assignment classifier — task checklist

Ticket: `Obsidian-Diarisation/Tickets/Open/overlap-assignment-classifier.md`
Branch: `assignment-classifier` (off `clustering-benchmark`). Don't push.
Re-read this file and the ticket's "Blocked on Andrew" + "Implementation Notes" after any context reset.

## Working decisions (pending Andrew; defaults in force)

- D1 Floor under VBx = `identity` (VBx assigns with a per-chunk Hungarian step by default,
  clustering.py:559/658). `nc_target` = plain argmax on target pairs; must equal `identity`
  only under HDBSCAN. Hard-fail 6 checked on HDBSCAN; on VBx it's reported as a real row.
- D2 Test-split parquet rows have `is_label = null`. Test labels live in a separate file
  written by the pair evaluator (`harness/assignment_eval.py`), the only non-scorer reader of test RTTMs.
- D3 Run conditions reused: `baseline` (VBx) and `clustering_hdbscan`; runs differ by
  `refinement_strategy`. Only `VALID_REFINEMENT_STRATEGIES` + CLI choices get new names.
- D4 Download/convert scripts live in `~/Code/AMI-diarization-setup/pyannote/`.

## Kickoff (§0)
- [x] Read ticket + 8 linked docs; contradictions logged in ticket Implementation Notes
- [x] Assumptions register pass (A1, A3, A5, A9 verified; A4 in WS0; A2 in WS1; A6 WS1; A7 write-up; A8 WS4)
- [x] Created this folder, TASKS.md, RUN_INDEX.md
- [x] settings.json allows run_harness.py, git add, git commit
- [x] Questions posted to Andrew (ticket "Blocked on Andrew")
- [x] status: in-progress

## WS0 Setup and checks
- [x] A4: centroids/soft_clusters non-None and aligned (VBx, HDBSCAN) — test on one cached meeting
- [x] Identity on test reproduces VBx 0.17048543579940637 (cold-equivalent) and HDBSCAN leaf/40 16.86%
- [x] Commit WS0

## WS1 Data
- [x] `download_ami_split.sh <split> <dest>` (generalised test script)
- [x] Download dev, train Mix-Headset
- [x] A2: check every WAV header; repair; list repaired files
- [x] convert_for_harness.sh train / dev; RTTM/UEM counts match audio
- [~] Cold VBx identity: dev 16.90% done; train running (GPU)
- [~] HDBSCAN identity: dev 17.19% (18/18 hits, A6 ok for dev); train pending
- [ ] Commit WS1

## WS2 Pair dataset
- [x] Predictions recorded in ticket before WS2 (already in §6; confirm unchanged)
- [x] `harness/assignment_dataset.py` + tests (incl. hard-fail 4 loader-patch test)
- [~] per split × method as .csv.gz (pyarrow missing): test, dev done; train pending
- [ ] dataset_stats.md: target counts, labelled share, floor pair accuracy by overlap bin
- [ ] Plain-language floor note to Andrew
- [ ] Commit WS2

## WS3 Non-learned
- [x] `nc_target`, `constrained_target`, `knn_target` (k by dev pair acc) + tests
- [x] Test runs VBx + HDBSCAN; RUN_INDEX rows
- [x] Check-in note to Andrew (WS2 floors + WS3 table)
- [x] Commit WS3

## WS4 Learned
- [ ] Grid on dev pair acc → dev_selection.csv
- [ ] Freeze best per type × method + pooled; models/<id>/ with sha256
- [x] `classifier:<model_id>` strategy; refinement_id includes sha256; key-distinctness test
- [ ] Test once per frozen model × method (with/without constraint)
- [ ] Commit WS4

## WS5 Checks
- [ ] Shift: NC acc + DER across splits; dev-only retrain → test once
- [ ] Permutation importance by group on dev; ablations (no temporal, no similarity) on dev
- [ ] Bootstrap 90% CIs (2000, seed 0)
- [ ] Commit WS5

## WS6 Write-up
- [ ] Results table, figures/, top-3 meetings, plain summary
- [ ] Convert ticket → Docs/Assignment Classifier Results.md; AMI Corpus Setup train/dev section
- [ ] Commit
