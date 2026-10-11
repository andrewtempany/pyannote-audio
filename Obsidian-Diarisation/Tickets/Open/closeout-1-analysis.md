---
status: done
created: 2026-10-08
---

# Closeout 1: analysis of results we already have

## The goal, plainly

Three pieces of analysis over data already on disk. No GPU, no new runs. All three feed named
sections of the report, and all three can start immediately.

**Run this ticket first.** It is cheap, it changes how the results get written, and one part of
it may change what is worth running afterwards.

## What we get

1. **The classifier numbers in report format.** Pair accuracy by overlap bin, oracle-ceiling
   shares and feature importance were all measured and are **already written up** in
   [[Assignment Classifier Results]]. This item only copies them into the results doc.
2. **A confidence interval on every result.** Several cited numbers, including the September
   nearest-centroid variant (16.79), are bare point estimates.
3. **An answer to: do all the clustering methods fail on the same meetings?** If they do, that
   is direct evidence the bottleneck sits upstream of clustering, which is the report's central
   claim, shown a second way with no oracle run.

## How we measure it

- Nothing new is computed from audio. Everything comes from existing per-file CSVs, the sweep
  index and the results docs.
- Intervals are 90% paired bootstrap over the 16 test meetings, 2000 draws, seed 0, matching
  what the benchmark already used.

## Done means

All three outputs exist as tables or short written sections in
`Obsidian-Diarisation/Docs/Closeout Results.md`, and the results table format in §4 below is fixed
there so the other tickets can write into it.

## Not in this ticket

No runs of any kind. No new strategies, no new methods. The 16.79 rerun is closeout 3 §5.

---

## For the agent

### 1. Copy the existing classifier results into report format

**Correction (8 Oct):** an earlier version of this ticket said these never reached the summary.
They did. All of them are already in [[Assignment Classifier Results]] (results table, "Share of
the oracle ceiling", "Which clues matter", "Predictions vs outcomes"), with the full table in
`runs/sweeps/assignment-classifier/REPORT.md`.

- Copy into the results doc: pair accuracy by overlap bin per strategy and clustering method,
  the oracle-ceiling shares (VBx only; state that no oracle exists for HDBSCAN), and permutation
  importance by feature group (dev).
- **Check, do not recompute:** spot-check three cells against `REPORT.md`. If the doc and
  `REPORT.md` disagree, report the disagreement.
- The nearest-centroid prediction (lower pair accuracy in high-overlap bins) is recorded there as
  **held** (VBx 78 → 60, HDBSCAN 84 → 62). Carry it across as is.

Budget: under 30 minutes. This is transcription, not analysis.

### 2. Bootstrap retrofit

Give every citable result a 90% paired interval over the 16 test meetings, computed from its
per-file CSV. 2000 draws, seed 0.

Priority order:

1. `32241285` — the September `nearest_centroid` run, DER 16.79. It has no interval.
   **It is a variant, not the project's best result.** It relabels **every** voice in every
   chunk by plain argmax over centroids (`harness/refinement.py:33-69`). `nc_target`
   (`703683ce`, 16.89) relabels only held-back voices. Write that scope difference next to the
   number, quoting the code lines. It was also produced before the 25 September cache-key change;
   closeout 3 §5 reruns it on the current cache.
2. Any other cited run lacking an interval.

Where a run's per-file CSV is missing, say so; do not re-run it. Say which CSV each interval was
computed from.

### 3. Convergent failure analysis

Zero GPU. Uses `runs/sweeps/clustering-benchmark/all_points_per_file.csv` and the per-file CSVs
from the baseline and the three best clustering points.

For VBx, agglomerative (`da03276c`), HDBSCAN (`b15caf86`) and spectral (`fda86985`):

- Per-meeting DER for each method, as one table of 16 rows by 4 columns.
- Rank correlation between the methods' per-meeting DERs. High correlation means they fail on
  the same meetings.
- The meetings in the worst quartile for all four methods, and the meetings where the methods
  disagree most.
- Correlate per-meeting baseline DER against meeting properties available from the reference:
  speaker count, total overlap fraction, duration, number of turns.

**Two readings, fixed in advance:**

- **Methods correlate strongly** → the bottleneck is shared and upstream of clustering. This
  supports the budget decomposition directly, without an oracle run.
- **Methods correlate weakly** → each has distinct failure modes that happen to average to a
  similar DER, and the ceiling is not a single shared limit. More interesting, and it would need
  saying carefully.

`IS1009b` is already known to be unusual — it carried most of HDBSCAN's apparent test gain. Check
whether it is an outlier for every method or only for HDBSCAN.

### 4. Fix the cross-condition results table format

The other two tickets write into this, so settle it here. Follow the shape of the pyannote model
card: one row per condition, one column per system.

| Condition | VBx (default) | Agglomerative | HDBSCAN | Spectral | Nearest centroid | kNN |
|---|---|---|---|---|---|---|
| AMI IHM (published) | 17.0 | — | — | — | — | — |
| AMI IHM (measured) | 17.05 | 16.87 | 16.86 | 17.16 | 16.89 | 16.85 |
| AMI SDM (published) | 19.9 | — | — | — | — | — |
| AMI SDM (measured) | | | | | | |

Rules:
- Every measured cell carries its 90% interval in the report version.
- The published row is the model card's figure, cited, not something we ran.
- Settings for each method are the ones selected on **IHM**, carried across unchanged. See
  closeout 2 for why.
- **"Nearest centroid" means `nc_target` (16.89)** in this table and everywhere else. The
  September all-voices variant (16.79) gets a footnote with its scope, not a column.

Create `Obsidian-Diarisation/Docs/Closeout Results.md` and write the empty table into it so the
SDM ticket has somewhere to put its numbers. All three analyses in this ticket go in the same
doc.

### Deliverable

The three analyses as tables plus a short written reading of each, and the empty cross-condition
table in place. No recommendations appended.

### Feeds

§4(c) and §5 of [[report-plan]] for item 1, §4 throughout for item 2, §5 for item 3.


## Implementation Notes

Output: [[Closeout Results]] (`Obsidian-Diarisation/Docs/Closeout Results.md`), all four sections plus empty headings for closeouts 2, 3, 4. Ticket left in `Tickets/Open/`, not converted.

Scripts (all read-only, in `C:\Users\Andrew\AppData\Local\Temp\claude\c--Users-Andrew-Code-pyannote-audio\f59f0102-59db-4e74-bf9b-fa2a90e8be3e\scratchpad`), run from the repo root with `diarisation-env/Scripts/python.exe` and `PYTHONIOENCODING=utf-8`:
- `closeout1_bootstrap.py` : section 2 intervals. Imports `harness.assignment_runs` and calls its `bootstrap`, `_der` unchanged. Writes `closeout1_bootstrap_out.csv`.
- `closeout1_convergent.py` : section 3 tables, Spearman, quartiles, IS1009b, reference-derived meeting properties (writes `closeout1_per_meeting_der.csv`, `closeout1_properties.csv`).
- `closeout1_confusion_only.py` : section 3 addendum (confusion-rate correlations; MD/FA identical across methods).

Decisions and findings:
- Bootstrap code to reuse: `harness/assignment_runs.py:150` (16 meetings, `rng.integers`, 2000 draws, seed 0, `_der` = total error / total reference speech). The clustering benchmark's own `harness/clustering_benchmark_report.py:57` bootstraps the 8-meeting report half only, so it was not reused; the Agglo/HDBSCAN/Spectral 16-meeting intervals are therefore new numbers.
- Reused function reproduces the ledger exactly (nc_target [-0.26,-0.06], kNN [-0.35,-0.06]).
- All headline DERs reproduce from per-file CSVs to 4 decimals and match manifest `summary.der`. Nothing failed to reproduce.
- `32241285` has no per-file CSV in its manifest or under runs/. Repo-root `per_file_nearest_centroid_t5.csv` identified by exact match of MD/FA/Conf sums and DER to the manifest (value match, no recorded link). Same for the two oracle CSVs, matched by DER only.
- Ticket cites `refinement.py:33-69`; the function is 33-66.
- Hook doc says `32241285` is post the `a4a3101a` (16-17 Sept) cache-key fix; ticket text says it predates the 25 Sept change. Not contradictory (two different cache-key changes) but not reconciled here; closeout 3 section 5 rerun settles it.
- MD and FA are identical across all four methods on every meeting, so the headline Spearman table partly reflects shared segmentation error; confusion-rate-only correlations added (mean off-diagonal 0.92 both ways).
- Spot-checks of the classifier doc against REPORT.md found no value mismatch (formatting differences only).
