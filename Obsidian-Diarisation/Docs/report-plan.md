# Report Plan — Speaker Diarisation Project

**Created:** 2026-10-08 | **Report due:** 11 October 2026
**Format:** IEEE conference template, no style modifications. Five pages.
Statement and references sit outside the page limit.
**Graded on:** methodological soundness, not on beating the baseline.

**Purpose:** the structure of the report, what each section needs, and which remaining
work feeds which section. Anything not mapped to a section here should not be run.

---

## Page budget

| § | Section | Pages |
|---|---|---|
| 1 | Introduction and problem | 0.5 |
| 2 | Related work | 0.5 |
| 3 | Method | 1.0 |
| 4 | Results | 1.5 |
| 5 | Discussion | 1.0 |
| 6 | Ethics | 0.25 |
| 7 | Conclusion | 0.25 |

Roughly 4,000 words plus three or four exhibits. Figures consume space quickly, so the
~70 sweep points become one figure rather than a table, and the 133 run manifests become
a repository link.

---

## §1 Introduction and problem

- The pipeline: segmentation → embedding → clustering, and which stages are frozen neural
  components.
- The central distinction: overlap **detection** is largely solved by the powerset
  segmentation head; overlap **assignment** is the weak point. This framing carries the
  whole paper.
- The retraining-free constraint, stated as an honest hardware limit (RTX 3060, 8 GB) rather
  than as a design preference.
- Research question: how much error is recoverable at the clustering stage without retraining,
  and what bounds it.
- Contribution, in three parts: an error budget decomposition, a method comparison measured
  against that budget, and a methodological account of how the measurements were validated.

**Status:** material held, needs writing.

---

## §2 Related work

- community-1's architecture and the powerset segmentation head.
- **VBx** — variational Bayes HMM over embeddings. This is the *actual shipped default*
  (`VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6)`), not agglomerative as earlier drafts
  assumed, and it is the standard strong baseline in DIHARD and VoxSRC. It needs prominence.
- Spectral clustering and the eigengap heuristic in diarisation.
- Density-based clustering (HDBSCAN), noting it is not standard for this problem.
- TS-VAD as a published overlap-handling ceiling, with its fixed-slot architectural limitation.
- EEND, and why community-1 is a hybrid rather than pure EEND.
- Prior oracle and ablation studies in diarisation, as methodological precedent for the
  ceiling analysis.

**Status:** not started. Needs no GPU. The only wholly unwritten section.

---

## §3 Method

- Harness design: injectable segmentation source, post-clustering refinement hook, two-tier
  cache (intermediates keyed without clustering config, final hypotheses keyed with it).
- Scoring protocol: collar 0, overlap not skipped, corpus-level accumulation, AMI IHM test
  split, 16 meetings.
- **Benchmark reproduction: 17.048% against a published 17.0%.** State this early — it is
  what licenses every other number in the paper.
- The 8/8 tune/report split, fixed before any result was inspected.
- Bootstrap protocol: 90% paired intervals over the 16 meetings.
- The 0.3-point resolution floor, derived from the sample size rather than asserted.
- Pre-registered predictions as a working practice.

**Status:** material held, needs writing.

---

## §4 Results

Three blocks, in this order. **(a) gets the most space.**

### (a) Error budget decomposition — the headline

| Budget | DER points |
|---|---|
| Total error | 17.05 |
| Attributable to segmentation | **13.09** |
| Assignment, given real segmentation | 0.47 (overlap-degraded) / 0.97 (all pairs) |
| Assignment, given clean segmentation | 1.76 |
| Captured by zero-training nearest centroid | 0.26 |

Component split: missed detection 2925.05 s (56% of error), confusion 1212.15 s (23%),
false alarm 1099.07 s (21%).

### (b) Clustering method comparison

| Run | Method, best setting | DER | Δ report half |
|---|---|---|---|
| `db977741` | VBx baseline (reference) | 17.05 | — |
| `da03276c` | Agglomerative, centroid, threshold 0.75, min size 12 | 16.87 | −0.20 [−0.65, +0.31] |
| `b15caf86` | HDBSCAN, leaf, min cluster size 40 | 16.86 | −0.55 [−1.08, −0.14] |
| `fda86985` | Spectral + eigengap, pruning 0.20 | 17.16 | +0.13 [−0.87, +1.45] |

HDBSCAN carries a caveat: worse than VBx on the tune half, on dev (17.19 vs 16.90) and on
train (17.07 vs 15.65), with one meeting (`IS1009b`) driving most of the test gain. Report as
"matched VBx; one split favoured it."

Ward linkage is reported as **untested** — its threshold grid was on the wrong scale.

### (c) Assignment strategies

| Run | Strategy | DER | Δ |
|---|---|---|---|
| `703683ce` | Nearest centroid on VBx held-back pairs | 16.89 | −0.16 [−0.26, −0.06] |
| `e2ef8a62` | kNN k=25 on VBx held-back pairs | 16.85 | −0.20 [−0.35, −0.06] |
| `d31bf9f4` | One-per-chunk constraint on VBx | 17.05 | identical |
| `32241285` | Nearest centroid (September) | **16.79** | no interval yet |

The one-per-chunk null is a finding, not a filler row: VBx already enforces one speaker per
voice per chunk by construction, so the constraint cannot change anything.

The kNN result carries a selection caveat: best of roughly thirteen strategies scored on the
same test meetings.

### Exhibits

1. Error budget decomposition.
2. One sensitivity curve across the clustering sweep.
3. The method comparison table with intervals.
4. The VBx `Fb` curve, if that sweep lands.

**Status:** most held. Missing: an interval on 16.79, the VBx sweep, SDM, ward.

---

## §5 Discussion

- Why the ceiling is narrow. The budget decomposition *explains* the null result rather than
  excusing it: clustering work was always competing for about 4 points of a 17-point budget,
  of which roughly 1 is reachable.
- Whether the methods fail on the **same meetings** (convergent failure analysis). If they do,
  that is direct evidence the bottleneck sits upstream of clustering — the thesis demonstrated
  a second way, without an oracle run.
- The counting behaviour and `Fb`, if the sweep resolves it. Higher `Fb` should give fewer
  speakers; the shipped 0.8 on a range of [0.01, 15.0] is weak regularisation, consistent with
  the over-splitting observed under clean segmentation.
- Limitations, stated plainly: 16 meetings, one corpus, one microphone condition, and any
  selection effect that survives the dev/train confirmation.
- What this implies for retraining-free clustering intervention as a class.
- The instrumentation thread: four documented cases where a measurement measured something
  other than what it claimed, framed as an observation about how diarisation results get
  produced and reported.

**Status:** depends on what the remaining runs return.

---

## §6 Ethics

Concrete and specific to this work, not generic:

- pyannote.audio ships OpenTelemetry usage tracking, opt-out by default, exporting to
  `otel.pyannote.ai`, unmentioned in the README quickstart. This project ran on it.
- AMI participants consented to a research corpus. Speaker identifiers (`MEE073`, `FEO072`)
  encode gender and role.
- Dual use: diarisation is an enabling technology for surveillance and for speaker
  identification without consent.
- Oracle conditions are methodological instruments, not deployable results. The 3.96% figure
  requires ground truth and cannot be achieved in deployment; reporting it as a ceiling rather
  than a result is itself an honesty requirement.

**Status:** material held, needs writing.

---

## §7 Conclusion

Short. What was measured, what bounds it, what the negative result means.

---

## Statement (outside the page limit)

- Tools used and own-work declaration.
- **Working code link.** Required — without it the code is not assessed.

---

## Mapping: does each remaining task serve a section?

| Remaining work | Section it feeds | Cut if it slips? |
|---|---|---|
| Related work, ethics, introduction | §1, §2, §6 | **No.** Nothing substitutes for these. |
| Bootstrap retrofit (no GPU, ~1 h) | §4, every row | No. Cheap and load-bearing. |
| Convergent failure analysis (no GPU) | §5 | No. Free, and it is the explanation. |
| VBx Fa/Fb/threshold sweep (~3 h warm) | §4 exhibit, §5 mechanism | Reluctantly. |
| Dev/train confirmation (~2 h) | §4, §5 limitations | Yes — becomes a stated caveat. |
| SDM + one oracle segmentation run | §4 second column, §5 | Yes — becomes future work. |
| Ward linkage redo (~30 min) | §4, one row | Yes. |
| Segmentation post-processing check (10 min) | §5 future work, or §4 if it exists | Yes. |

**Anything not in the left column should not be run.**

---

## Structural recommendation

The budget decomposition is the paper; the method comparison is the test of it.

Write §4(a) first and give it the space it deserves. If the clustering comparison reads as the
headline and the budget as context, the null result looks like a failure rather than a
prediction that held.

---

## Why the verification work is the science, not insurance

Worth stating explicitly in §3, because it is what the report has to argue.

The claim is "no clustering method or learned classifier clearly beats VBx." That claim is only
as strong as the ability to have detected a difference if one existed. Without that, "we found
nothing" and "we could not have found anything" are indistinguishable.

- **Bootstrap intervals** turn a null into a quantified statement: a 0.2-point improvement was
  not distinguishable from noise at this sample size.
- **The tune/report split** stops every method's best point being flattered by the number of
  settings tried.
- **Dev/train confirmation** separates a finding from a lucky draw — it is precisely why the
  HDBSCAN result does not survive.
- **Pre-registered predictions** (a dozen or more, recorded before measurement, three hit) are
  a method, not history.
- **The four instrumentation findings** are a contribution about reporting practice, not an
  apology.

---

## See also

- `claude/report-notes.md` — canonical numbers, the prediction ledger, limitations, scope
  decisions.
- The work-tracking board artifact — phase timeline and card status.
- `Obsidian-Diarisation/Docs/` — the per-component reference documentation.
