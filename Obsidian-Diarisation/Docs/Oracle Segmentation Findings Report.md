---
status: done
created: 2026-09-20
---

# Oracle Segmentation — Findings Report

**Audience:** project management / planning
**Scope:** AMI IHM test split (16 meetings), `pyannote/speaker-diarization-community-1`
**Status:** headline result delivered and verified. Two follow-ups outstanding (§6).

---

## 1. Executive summary

A previously-reported oracle-segmentation result was **invalid** — the experiment silently
measured nothing. The cause was found, fixed, and the experiment re-run.

**The corrected result: DER falls from 17.05% to 3.96% when the system is given
ground-truth segmentation.**

Practical meaning: **roughly 13 of the 17 points of diarisation error are attributable to
the segmentation stage.** Everything downstream — clustering, speaker assignment, overlap
handling — competes for the remaining ~4 points.

This materially reframes the planned next decision (§5).

---

## 2. What went wrong with the original experiment

Run `3c8deb71` reported an oracle-segmentation DER of 17.05% — identical to baseline to
eleven decimal places across seven separate metrics.

**Root cause.** The mechanism for injecting ground-truth segmentation was gated behind a
pipeline flag (`training`). The harness set that flag, injected the data, then *unset the
flag before the pipeline actually ran*. The pipeline therefore ignored the injected data and
recomputed its own segmentation. The experiment ran to completion, produced a manifest, and
measured the baseline system.

**Why it was not caught.** Three compounding factors:

1. The failure was silent — no error, no warning, a plausible-looking number.
2. The test that existed specifically to catch this had **never executed** on this machine,
   blocked by an unrelated Windows filename-encoding bug in a dependency.
3. The result was *nearly* undetectable: the output differed from baseline only in speaker
   *names*, which do not affect any scored metric.

**Detection.** The number was too exact to be real. Two independent lines of reasoning —
the improbability of seven identical metrics, and an arithmetic prediction that a working
oracle run should land near 4% rather than 17% — identified it before any code was read.

---

## 3. The corrected result

| Metric | Baseline (`aa8fbfa4`) | Oracle segmentation (`18016c15`) | Change |
|---|---|---|---|
| **DER** | 17.05% | **3.96%** | **−13.09 pt** |
| Missed detection | 2925.05 s | **16.43 s** | −99.4% |
| False alarm | 1099.07 s | **141.13 s** | −87.2% |
| Confusion | 1212.15 s | 1057.51 s | −12.8% |
| JER | 22.19% | 7.98% | −14.21 pt |
| Overlap DER (system) | 33.39% | 10.43% | −22.96 pt |
| Region: reference-only | 1393.94 s | **5.80 s** | collapsed |

Run `18016c15`, 2026-09-18, 2839 s (~47 min) on GPU, full 16-meeting split,
`counts_toward_results: true`.

**The result was predicted before it was measured.** From baseline's error components, a
working oracle run should leave mostly confusion: 1212 / 30714 ≈ 4%. Measured: 3.96%. That
agreement is the strongest available evidence the experiment is now sound.

**Independent confirmation of the mechanism.** Before the scored run, a single-meeting check
confirmed the injected segmentation actually reaches the pipeline: segment count changed
1000 → 796 and total detected speech 2263 s → 2543 s. Before the fix, oracle and baseline
output had been *identical* in timing for all 16 meetings; after the fix, **0 of 16** match.
The injection demonstrably takes effect.

---

### 3a. Control: the fix has no side-effect of its own — verified

The fix works by holding a pipeline flag (`training`) on while the pipeline runs. That flag
unlocks reading the injected segmentation, but it also enables an internal embedding cache —
two effects where only one was wanted. Left unaddressed, the oracle and baseline conditions
would have differed in *two* ways, and the 13-point gap could not be attributed cleanly to
segmentation.

The fix applies the flag **uniformly to both conditions**, so the only remaining difference
is whether segmentation was injected. A control run then tested whether the flag alters
output at all: baseline was re-run with the flag on, forced to fully recompute against an
empty throwaway cache, and compared against the pre-fix baseline output.

**Result: 16/16 byte-identical RTTM files. 0 differences.** Metrics match to every digit
(DER 0.17048543579940637, missed detection 2925.049468749914, confusion 1212.152875000019).

The flag is inert with respect to output. **The 13-point effect is attributable to
segmentation alone — measured, not argued.** This was the last open caveat on the headline
figure and it is now closed.

(Method note: the throwaway cache was essential. Re-running against the normal cache would
have returned the existing files without invoking the pipeline, and the comparison would have
passed trivially while testing nothing.)

---

## 4. Secondary finding: speaker counting regressed

Not a defect, but it should not go unrecorded.

| | Baseline | Oracle |
|---|---|---|
| Exact speaker-count matches | 12/16 | **7/16** |
| Counting mean absolute error | 0.25 | 0.69 |

**Direction is consistent: the oracle run over-counts in every one of the 9 mismatching
meetings, and never under-counts.** With more true speech correctly detected, the clustering
stage splits speakers it previously merged.

Interpretation: with segmentation error removed, **clustering's weaknesses become the
binding constraint** — visible as speaker fragmentation. This does not undermine the 3.96%
headline (DER already accounts for it via the confusion term), but it is a concrete pointer
to where the residual error now lives, and is directly relevant to §5.

---

## 5. Implication for the next decision

The pending question is **clustering sweep vs. trained classifier** for post-clustering
improvement. Prior work measured the ceilings for the classifier route:

| Intervention | Recoverable DER | Source |
|---|---|---|
| Oracle assignment (all pairs) | +0.97 pt | `1db2ba34` |
| Oracle assignment (overlap-degraded only) | +0.47 pt | `c93fb9f9` |
| **Oracle segmentation** | **+13.09 pt** | `18016c15` |

**Segmentation dominates by more than an order of magnitude.** Both candidate interventions
are competing for a ~1 pt prize inside a budget where segmentation holds ~13.

This does not automatically redirect the work — a better segmentation model may be out of
scope, cost-prohibitive, or simply unavailable, whereas a post-clustering fix is tractable
and self-contained. But the decision should now be made *knowing* the relative sizes, rather
than assuming the clustering/assignment stage is where the error budget sits. The §4 finding
adds nuance: even with perfect segmentation, clustering fragments speakers, so clustering
work is not worthless — it is just capped far lower than segmentation work.

**Recommendation:** treat the 13-point figure as a scoping input before committing to either
arm. If segmentation improvement is viable at all, it is where the return is.

---

## 6. Outstanding items

| # | Item | Status | Risk if skipped |
|---|---|---|---|
| 1 | ~~**Baseline-equivalence check**~~ | **DONE — PASSED** (§3a) | Resolved. No longer a caveat. |
| 2 | **Manifest provenance** — record the oracle reference file; flag uncommitted-code runs | Ticketed, not started | Future runs remain hard to reproduce exactly |
| 3 | **Re-run the two assignment ceilings** under a clean committed tree | Ticketed, not started | Their +0.97/+0.47 figures are trusted but their exact code state is unrecoverable. Re-run before citing externally. |
| 4 | Two stale tests assert a renamed metric field | Known, deferred | Cosmetic; cannot affect results |

### Invalidated runs — do not cite

- **`3c8deb71`** (oracle_segmentation, 17.05%) — measured nothing. **Superseded by
  `18016c15`.**
- **`e9f25911`** (nearest_centroid) — invalidated by an earlier caching defect. Superseded
  by `32241285`.

Both remain in `runs/comparison.csv`. **`runs/comparison.csv` also carries a data trap:** an
overlap metric was renamed mid-project, so two columns exist and each is only partly
populated. Charting by a single column name silently drops runs.

### Runs that remain valid

`aa8fbfa4` (baseline), `32241285` (nearest_centroid), `1db2ba34` / `c93fb9f9` (oracle
assignment — valid mechanism, see item 3), `d3ef4437` and `a9c41646` (earlier baselines).

---

## 7. Confidence assessment

**High confidence in the 3.96% figure.** Five independent supports:

1. Matches a prediction made before measurement (3.96% vs ~4%).
2. Error components moved in the correct direction and magnitude — missed detection fell
   99.4%, which is the specific signature of working oracle segmentation.
3. Mechanism independently verified on real audio before the scored run.
4. **Control run confirms the fix has no side-effect of its own — 16/16 byte-identical
   baseline output (§3a).** The effect is isolated to segmentation by measurement.
5. The previously-never-executing test that guarded this behaviour now genuinely runs, along
   with the rest of the suite (211 passing).

**Known limitations, stated plainly:**

- Single corpus, single microphone condition, single checkpoint. The 13-point split is an
  AMI IHM result, not a universal one.
- The oracle reference is the same file used for scoring. This is correct and necessary for
  the prediction to hold, but it means the result is an upper bound: it shows what perfect
  segmentation *of this reference* buys.

---

## 8. Process note

The root cause was a correct mechanism applied at the wrong scope, and the guard against it
existed but had never run. The generalisable lesson: **a test that has never executed is not
a guard**, and a silent no-op that produces plausible numbers is more dangerous than a
crash. The test suite is now genuinely executing, which closes that specific gap.

Recommend items 6.2 and 6.3 (provenance) be prioritised on the same reasoning: they are the
difference between a result being trustworthy and a result being *demonstrably*
reproducible.
