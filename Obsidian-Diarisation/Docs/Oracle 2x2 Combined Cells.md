---
status: done
created: 2026-09-20
---

# Oracle 2x2 — Combined Segmentation + Assignment Cells

**Audience:** project management / planning, and anyone extending the oracle refinement work
**Scope:** AMI IHM test split (16 meetings), `pyannote/speaker-diarization-community-1`,
collar 0, `skip_overlap` False
**Status:** both cells measured. The headline number came in outside its predicted range,
and the predicted *mechanism* turned out not to exist — §4 is the most load-bearing section
of this document.

---

## 1. Executive summary

The oracle 2x2 is now complete. Giving the system both ground-truth segmentation **and**
ground-truth speaker assignment lands DER at **2.19%**, against 3.96% for ground-truth
segmentation alone.

**The assignment budget under clean segmentation is 1.76 pt** — 1.8x the 0.97 pt measured
under baseline segmentation, confirming that the earlier ceiling was partly masked by
segmentation error, but well short of the ~3 pt that was predicted.

Two things are worth more than the headline:

1. **The result supports neither interpretation that was fixed in advance** (§3). Relabelling
   is not useless on clean input — it removes over half the confusion — but it is not the
   large win the prediction implied either.
2. **The predicted blocking mechanism fired exactly zero times** (§4). The instrumentation
   built to explain a surprise found a different explanation than the one the ticket
   proposed, and the real blocker is structural rather than learnable.

---

## 2. The completed grid

DER, AMI IHM test, 16 meetings:

|                             | baseline seg | oracle seg |
| --------------------------- | ------------ | ---------- |
| baseline assignment         | 17.05%       | 3.96%      |
| oracle assignment, degraded | 16.58%       | **2.75%**  |
| oracle assignment, all      | 16.08%       | **2.19%**  |

Confusion, in seconds:

|                             | baseline seg | oracle seg |
| --------------------------- | ------------ | ---------- |
| baseline assignment         | 1212.2       | 1057.5     |
| oracle assignment, degraded | n/a          | **686.1**  |
| oracle assignment, all      | n/a          | **516.0**  |

The two `n/a` cells are genuine gaps: those runs predate the DER component breakdown in the
summary, so their confusion was never recorded. They were not re-run, because the grid turns
on DER and the confusion deltas that matter are measured against oracle segmentation.

Confusion removed, against the 1057.51 s under oracle segmentation alone:

- `all_pairs`: 516.0 s — **541.5 s removed (51.2%)**
- `overlap_degraded`: 686.1 s — **371.4 s removed (35.1%)**

Missed detection (16.427 s) and false alarm (141.127 s) are identical to the
oracle-segmentation run to eight decimal places in **both** cells. Relabelling moved
confusion and nothing else, which is the strongest available evidence the refinement touches
only what it is supposed to.

| Run | scope | manifest | DER | Confusion |
| --- | --- | --- | --- | --- |
| 1 | `all_pairs` | `20260920T082056Z-2aada13d` | 2.193% | 516.00 s |
| 2 | `overlap_degraded` | `20260920T101248Z-5479daf2` | 2.747% | 686.09 s |

Both from commit `1dab5122`, oracle RTTM `harness-data/IHM/test.rttm`,
`counts_toward_results: true`. Run 1 took 35 min; Run 2 was faster because segmentation and
embeddings were already warm, but its pair counts are non-zero, so it genuinely executed
rather than replaying Run 1's cached output.

Scope ordering held — `overlap_degraded` (2.75%) is worse than `all_pairs` (2.19%), as
expected. This was deliberately held as an expectation rather than a pass/fail criterion,
because `_is_overlap_degraded` reclassifies over tracks that oracle segmentation
substantially changes; an inversion would have been a finding, not a failure. It did not
invert.

---

## 3. What was predicted, and what the miss means

The prediction was recorded before the runs, specifically so a surprise would be visible:

| | Predicted | Measured |
|---|---|---|
| DER (`all_pairs`) | 0.5% – 1.5% | **2.19%** |
| Confusion (`all_pairs`) | 100 – 350 s | **516.0 s** |

Both ranges were missed, and the ticket's standing instruction was explicit — *"anything
outside 0.5%–1.5% is a finding to report, not a bug to fix"* — so nothing was adjusted to
chase them. The confusion range was derived arithmetically from the DER range rather than
guessed independently, so the two misses are one miss seen twice, not two.

Two interpretations had been fixed in advance. **Neither is what happened:**

- *"Lands near 1%"* would have meant the baseline-segmentation ceilings badly understated a
  large recoverable budget. At 2.19% the budget is real and larger than previously measured,
  but 1.76 pt is not ~3 pt.
- *"Lands near 3.9%"* would have been a decisive negative for the classifier direction. At
  2.19%, relabelling removes over half the confusion under clean segmentation and is the
  single largest post-segmentation improvement measured on this project — emphatically not
  useless.

The honest reading is in between: **the assignment budget under clean segmentation is
roughly double what baseline-segmentation measurement suggested, and about 60% of what the
prediction implied.** The floor, if confusion went to zero, is 0.51%; the measured result
leaves a **1.68 pt gap to that floor**, and §4 explains what occupies it.

---

## 4. The predicted mechanism does not exist

This is the most transferable finding in this document.

The prediction did not merely guess a number — it named a mechanism. The oracle strategy
leaves a pair unchanged when its dominant reference speaker maps to no produced cluster, and
9 of 16 meetings over-cluster under oracle segmentation. The ticket reasoned that this
`unmapped_speaker` path would be what held DER off the floor, and called it "the most
diagnostically valuable of the four" exit paths.

**`unmapped_speaker` fired zero times. In both runs. Across 126,925 pairs.**

The inference from "meetings over-cluster" to "reference speakers are left unmapped" does not
hold, and this was verified directly against `optimal_mapping` rather than argued:

- With 3 reference speakers split across 5 hypothesis clusters, **all 3 speakers map**
  (`{A→0, B→2, C→4}`). The 2 surplus clusters go unmapped in the *other* direction, which
  costs nothing, because the strategy only ever reads `speaker_to_cluster`.
- A reference speaker is stranded only when it has **zero** hypothesis overlap across the
  whole file — confirmed as the single case that does produce an unmapped speaker.

Under oracle segmentation every reference speaker has speech by construction. So
`unmapped_speaker = 0` is **structural, not luck**, and over-clustering is simply not the
constraint. Over-clustering costs DER through confusion (a pair labelled with the wrong one
of several clusters belonging to the same speaker), not through unmappability.

### What actually blocks it — CORRECTED

> **This section previously drew a strong planning conclusion from a misread counter. The
> conclusion was wrong and is retracted in full.** The original text claimed the 56,998
> `no_reference_overlap` pairs were tracks over silence, that this reflected "the frame-level
> active-speaker mask marking speech in regions the reference scores as silence", and that "a
> learned assignment classifier inherits this ceiling exactly as the oracle does". All three
> claims are false. The boundary-distance analysis that falsified them is
> [[unreachable-pair-boundary-distance]].

`_dominant_reference_speaker` returned `None` for two unrelated conditions — an empty support
and a support overlapping no reference speaker — and the strategy funnelled both into
`no_reference_overlap`. Reconstructed over all 16 files:

- **Empty support — no active frames at all: 54,888 pairs, 44.1%** of the 124,576 reconstructed
  (chunk, speaker) slots, and **96.3% of the recorded 56,998**. These are unused slots on the
  fixed-width `local_num_speakers` axis: every chunk with fewer active speakers than slots
  contributes empty pairs. A property of the tensor layout, not of segmentation quality.
- **Genuinely silence-dwelling pairs: 6**, totalling **0.2024 s** across 16 meetings — 0.01% of
  the recorded figure.

(The two denominators differ because the reconstruction derived file duration from the
reference extent while the run used audio duration; see that analysis's reconciliation section.
The ~2,349-pair gap is itself trailing silence, which reinforces the finding.)

**Oracle segmentation does not produce meaningful tracks over silence.** The premise of the
original passage was false.

**The 1.68 pt gap to the floor is therefore unexplained.** No mechanism in this document
accounts for it. The leading candidate — that whole-pair relabelling cannot split a support
spanning a speaker change, so a pair covering two speakers gets one label whichever speaker
dominates — is **untested**, and is recorded here only to say what has not been ruled out. It
must not be cited as the explanation. Testing it is separate work.

### Full pair disposition

| Path | Run 1 (`all_pairs`) | Run 2 (`overlap_degraded`) |
| --- | --- | --- |
| `relabelled` | 69,927 (55.1%) | 28,384 (22.4%) |
| `no_reference_overlap` | 56,998 (44.9%) | 0 |
| `out_of_scope` | 0 | 98,541 (77.6%) |
| `unmapped_speaker` | **0** | **0** |
| total | 126,925 | 126,925 |

The four paths partition exactly in both runs, so this is a complete account of every pair
rather than a sample — which is what makes the zero trustworthy rather than merely unobserved.

> **Note on `no_reference_overlap`.** The counts above are correct as recorded; only their
> interpretation was wrong. This path conflated two dispositions — pairs with no active frames
> (96.3% of the 56,998) and pairs genuinely in ground-truth silence (6 pairs). The counter has
> since been split into `empty_support` and `no_reference_overlap`, so runs after that change
> report five paths rather than four. See [[Oracle Assignment Strategy]]. Expect
> `empty_support = 0` under `overlap_degraded` scope: the scope check runs before the
> dominant-speaker call, so empty-support pairs are absorbed into `out_of_scope` there.

`no_reference_overlap = 0` in Run 2 is structural too: any pair passing
`_is_overlap_degraded` necessarily has reference overlap, so the narrower scope cannot reach
that path.

### Scope populations are not nested

Only 22.4% of pairs classify as overlap-degraded under oracle segmentation. The ticket
suspected this and it is confirmed: `_is_overlap_degraded` runs over tracks that oracle
segmentation substantially changed (on the test meeting, segments fell 1000 → 796 and
reference-only region collapsed 1394 s → 5.8 s). The two scopes are **not** nested subsets of
the same pair population they were under baseline segmentation, so their DERs are not
comparable across segmentation conditions on a like-for-like basis.

---

## 5. How to run these conditions

The combined cell is condition `oracle_segmentation_assignment`, which requires the
refinement strategy and the oracle RTTM to be selected alongside it:

```bash
./run_experiment.sh oracle_segmentation_assignment "combined 2x2 cell, all_pairs" \
    --refinement-strategy oracle --oracle-scope all_pairs \
    --oracle-rttm path/to/only_words.rttm --counts-toward-results
```

Each run prints its pair disposition:

```
oracle pair disposition (scope=all_pairs, 126925 pairs over 16 files):
  no_reference_overlap=56998, out_of_scope=0, relabelled=69927, unmapped_speaker=0
```

**Two caveats when reading those counts.**

1. **A cache hit skips the pipeline entirely**, so the strategy never runs and its counts stay
   at zero. The totals describe uncached files only; re-running an already-cached condition
   reports zeros, which means "not measured", not "none occurred".
2. **`relabelled` counts every pair assigned a mapped cluster**, including pairs whose label
   was already correct. It is not a count of changed values. This keeps the four paths a clean
   partition; a "changed" counter would leave a fifth, unlabelled path.

---

## 6. Architecture: how the conditions compose

`OracleSegmentation` and the oracle refinement strategy were always independent of each other,
but the plumbing that *selected* them was not: three call sites treated the manifest's
`condition` field as a proxy for "which segmentation source". Composing the two required
fixing that conflation rather than writing new experimental code.

### The routing trap

Segmentation source selection previously tested `condition` for exact equality with
`"oracle_segmentation"`, so any new oracle-segmentation condition fell through to
`BaselineSegmentation()`. The run would have completed, written a manifest **claiming** oracle
segmentation, and silently scored baseline — landing near 16%, which reads as a plausible
experimental result rather than a bug.

Routing now keys on membership in `ORACLE_SEGMENTATION_CONDITIONS`
([run_manifest.py](../../harness/run_manifest.py)), declared beside the vocabulary it belongs
to so the next condition of this kind cannot silently miss it. `--run-condition`'s choices are
derived from `VALID_CONDITIONS` rather than retyped, removing the drift risk entirely.

This is the same failure shape as [[Segmentation Injection Seam]]: a silent no-op producing a
credible number. Worth treating as a recurring hazard in this harness rather than a one-off.

### Budget deltas need a per-condition reference

`_BUDGET_CONDITIONS` ([aggregate_runs.py](../../harness/aggregate_runs.py)) maps each
condition to `(reference_condition, label)`. The combined cell's reference is
**`oracle_segmentation`, not `baseline`** — a baseline-referenced delta would compute
`17.05 − 2.19 = ~15 pt` and read as a budget for the combined intervention, double-counting
the 13.09 pt segmentation budget already reported on its own line. The label names its
reference in the table so a reader never has to infer it, and a missing reference row emits an
explicit "cannot be computed" line rather than silently falling back to baseline.

A second route to the same wrong number was found only by rendering the table against **real
manifests** rather than the synthetic fixtures the unit tests use. `runs/` holds two
`oracle_segmentation` runs, both marked `counts_toward_results`: the pre-seam-fix one (17.05%,
which measured nothing — see [[Oracle Segmentation Findings Report]]) and the corrected re-run
(3.96%). Resolving the reference by *first match* picked the stale row and produced `+0.1605`
— precisely the double-counted figure the per-condition reference exists to prevent. References
now resolve to the **most recent** matching run by `created_at`.

The generalisable point: **single-row-per-condition test fixtures cannot express a
stale-duplicate bug.** Render against real data as well.

### Scope suppression

`oracle_scope` is blanked for conditions that do not refine, because `--oracle-scope` has a
default that is recorded whether or not anything chose it, and showing it would imply a
decision nobody made. The combined condition belongs in `_SCOPED_CONDITIONS`: its two runs
differ *only* by scope, so blanking it would render them as identical rows.

### Provenance

`oracle_rttm` is now recorded in `run_config` — additive, defaulting to `None`, and
deliberately **not** in `REQUIRED_RUN_CONFIG_FIELDS` so that manifests written before the
change stay readable. This closes Gap 1 of [[run-manifest-provenance]]. Its value was
immediate: the two earlier oracle-segmentation runs both record `oracle_rttm: None`, so which
reference built their ground truth is unrecoverable from the manifests.

---

## 7. Tests

`tests/test_oracle_combined_cells.py` — 24 tests covering all six work items. Built
test-first; 17 failed before implementation.

Two are worth knowing about specifically:

- **`test_combined_condition_requires_oracle_rttm`** asserts on argparse's *message*, not just
  `SystemExit`. It initially passed vacuously: argparse rejected the not-yet-valid condition
  as an "invalid choice" and exited, satisfying `pytest.raises(SystemExit)` without the
  required-argument check existing at all. **Any `SystemExit` assertion against an argparse CLI
  is suspect while the value under test is not yet a valid choice.**
- **`test_oracle_strategy_counts_every_pair_exactly_once`** asserts the four exit paths sum to
  the pair count. This is what makes `unmapped_speaker = 0` evidence rather than an absence of
  evidence.

Four routing tests pin the full matrix — combined → oracle, plain `oracle_segmentation` →
oracle, `baseline` → baseline, and `oracle_assignment`-alone → baseline. That last one matters:
widening the routing test must not capture the already-measured baseline-segmentation cell.

Full harness suite: 142 passed, 1 skipped (pre-existing).

---

## 8. Confidence assessment

**High confidence in the 2.19% and 2.75% figures.**

1. Missed detection and false alarm are unchanged from the oracle-segmentation run to eight
   decimal places, in both runs. Relabelling provably moved only confusion.
2. The manifests confirm `segmentation_source_id: oracle-v2` — the routing trap described in
   §6 did not fire; these runs genuinely used oracle segmentation.
3. Cache-key separation was verified *empirically before launching*, not assumed: `identity`,
   `oracle:all_pairs` and `oracle:overlap_degraded` produce three distinct keys, so neither
   run could have been served the earlier oracle-segmentation run's hypotheses.
4. The four pair-disposition paths partition exactly in both runs. (Partition-exactness is a
   completeness property only. It says every pair is accounted for once; it says nothing about
   whether each path means what its name suggests — which is exactly how §4's error survived
   this checklist.)
5. The `unmapped_speaker = 0` explanation was verified against `optimal_mapping` directly
   rather than inferred from the counts alone.

**Not confident in any mechanism for the 1.68 pt residual.** §4's original explanation was
falsified after this assessment was first written. The DER figures above are unaffected — they
were measured, not inferred from the counter — but the planning conclusion drawn from them was
withdrawn. See §4.

**Known limitations, stated plainly:**

- Single corpus, single microphone condition, single checkpoint. This is an AMI IHM result.
- The oracle reference is the same file used for scoring, so these are upper bounds: they show
  what perfect assignment *of this reference* buys.
- Pair counts cover uncached files only (§5).
- The two `oracle_assignment` cells have no recorded confusion (§2), so the confusion column
  of the grid is incomplete on the baseline-segmentation side.

**Known pre-existing wart in the cross-condition table.** `_build_delta_lines` emits one line
per row, so the superseded pre-seam-fix `oracle_segmentation` run still produces its own
`Downstream budget: +0.0000` line beside the real `+0.1309`, and both generations appear as
table rows. This predates this work and does not affect the combined delta now that references
resolve by recency. Fixing it properly means deciding whether superseded runs should be
de-marked, filtered by segmentation id, or de-duplicated by condition — a call for the
results-table owner. Candidate follow-up ticket.

---

## 9. Process note

The instrumentation in this work was built to explain a surprise in the *magnitude* of the
result. It instead falsified the *mechanism* the prediction rested on — and the mechanism had
been argued confidently enough that it justified building a dedicated counter to observe it.

The counter is why that is knowable. Had the four exit paths been collapsed into a single
"skipped" total (the obvious simplification), `unmapped_speaker = 0` and
`no_reference_overlap = 56,998` would have summed to one uninformative number, the prediction's
mechanism would have looked untested rather than refuted, and the conclusion that a learned
classifier inherits this ceiling would not have been available.

**A prediction that names a mechanism is worth more than one that only names a number,
because it can be shown to be wrong in a way that teaches something.** Here the number was
merely off; the mechanism was absent, and finding that out redirected the planning conclusion
more than the DER figure did.

**The second half of the lesson, learned later and more expensively: a counter is only as good
as the distinction it draws.** The same instrumentation that made `unmapped_speaker = 0`
knowable also produced `no_reference_overlap = 56,998`, and that number was read as "tracks
over silence" when 96.3% of it was empty tensor slots. §4's original conclusion — confidently
argued, and wrong — rested entirely on it.

Keeping four paths apart was right. But the granularity of a counter is a claim about which
distinctions matter, and that claim needs checking just as the mechanism did. The failure mode
is specific and worth naming: **a single counter whose name describes only one of the
conditions that reaches it.** `_dominant_reference_speaker` returned `None` for two unrelated
reasons; the name documented the interesting one and silently absorbed the common one.
Partition-exactness could not catch this, because the paths did partition — completeness and
correctness of meaning are different properties, and only the first was being verified.

---

## See also

- [[Oracle Segmentation Findings Report]] — the 3.96% run and its error breakdown; the
  reference condition for this grid's deltas.
- [[Oracle Assignment Strategy]] — the refinement strategy, its scopes and exit paths.
- [[Oracle Ceiling Metrics]] — the 0.47 / 0.97 pt baseline-segmentation ceilings this work
  reframes.
- [[Cross-Condition Deltas]] — the T5 table and budget-delta logic extended here.
- [[Run Manifest]] — manifest schema and controlled vocabularies.
- [[Segmentation Injection Seam]] — the earlier silent-no-op failure of the same shape as §6.
- [[run-manifest-provenance]] — the provenance ticket whose Gap 1 this closes.
