---
status: done
created: 2026-09-25
---

# Clustering model selection in the harness

## Goal

Let a run choose which clustering method the pipeline uses, and with which hyperparameters,
and record both in the manifest.

Prerequisite for every ticket under [[clustering-method-comparison]].

## Current state

[run_harness.py:257-260](../../../run_harness.py):

```python
# No real clustering-model selection exists yet (see the run-manifest
# ticket) -- hardcoded the same way `checkpoint` is above, until a
# future ticket implements clustering-model selection.
clustering_model = "pyannote-default"
```

The manifest field already exists and is already written
([run_harness.py:154](../../../run_harness.py)). Only the selection is missing. The help text at
line 202 even names "dbscan clustering" as an intended future value.

`Clustering` enum, [clustering.py:759-763](../../../src/pyannote/audio/pipelines/clustering.py):
`AgglomerativeClustering`, `KMeansClustering`, `VBxClustering`, `OracleClustering`.

## Coordinator finding, 2026-09-25 — `pyannote-default` is VBx, NOT agglomerative

**A premise repeated across this ticket, [[oracle-segmentation-seam-no-op]] and
[[Oracle Segmentation Seam]] is false.** Those notes state that community-1 uses
`AgglomerativeClustering` ("Community-1 uses `pyannote-default` = `AgglomerativeClustering`").
It does not.

Verified two ways. The shipped config at
`~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf.../config.yaml`:

```yaml
pipeline:
  params:
    clustering: VBxClustering
    plda: $model/plda
params:
  clustering:
    threshold: 0.6
    Fa: 0.07
    Fb: 0.8
```

And from the live instantiated pipeline:

```
klustering       : VBxClustering
clustering class : VBxClustering
expects_num_clus : False
_expects_num_spk : False
threshold / Fa / Fb : 0.6 / 0.07 / 0.8
powerset         : True
```

**What this changes for this ticket:**

1. `pyannote-default` must map to **`VBxClustering` with threshold 0.6, Fa 0.07, Fb 0.8** —
   these are the values that produced DER 0.17048543579940637. Mapping it to agglomerative
   would silently change the default condition and break the baseline reproduction.
2. §1's vocabulary is therefore misleading: `vbx` and `pyannote-default` are the *same class*,
   differing only in whether hyperparameters are explicit. Say so in the vocabulary, or a
   future reader will treat them as two conditions.
3. `VBxClustering.__init__` requires a positional `plda: PLDA` with no default
   ([clustering.py:555-563](../../../src/pyannote/audio/pipelines/clustering.py)). It cannot be
   constructed from a bare vocabulary name. The library already handles this at
   [speaker_diarization.py:290-291](../../../src/pyannote/audio/pipelines/speaker_diarization.py),
   special-casing `self.klustering == "VBxClustering"` to pass `self._plda`. Any selection path
   must go through that, not construct the class directly.
4. **The conclusion that the speaker-count leak is dormant still holds, but for a different
   reason than recorded.** `VBxClustering.expects_num_clusters = False`
   ([clustering.py:552](../../../src/pyannote/audio/pipelines/clustering.py)), so
   `_expects_num_speakers` is False — confirmed live above. The KMeans caveat is unaffected.
5. `SpeakerDiarization.__init__` **already** validates the clustering name against the enum and
   raises `ValueError` listing valid members
   ([speaker_diarization.py:283-288](../../../src/pyannote/audio/pipelines/speaker_diarization.py)).
   Criterion 1's "unknown value fails loudly" partly exists already — check what it does before
   building a second validation layer, and make sure the harness-level error is not weaker than
   the library's.
6. Clustering is chosen in `__init__`, so it is **not** mutable on an already-loaded pipeline by
   simple attribute assignment — `self.clustering`, `self.klustering` and
   `self._expects_num_speakers` are all set together. Decide deliberately whether selection
   happens via `from_pretrained` parameters or post-hoc re-instantiation, and state why.

### The two override paths are separate — verified in `Pipeline.from_pretrained`

Read from [core/pipeline.py:275-294](../../../src/pyannote/audio/core/pipeline.py). They are not
the same seam, and conflating them is the likeliest way this ticket goes wrong:

```python
params = config["pipeline"].get("params", {})   # :275  <- holds clustering: VBxClustering
pipeline = Klass(**params)                      # :278  <- CLASS is fixed here, at construction
...
if "params" in config:
    pipeline.instantiate(config["params"])      # :290-291 <- HYPERPARAMETERS applied here
if hparams_file is not None:
    pipeline.load_params(hparams_file)          # :293-294 <- a third route
```

Consequences:

- **`--clustering-param` does not need the pipeline rebuilt.** `instantiate()` is the sanctioned
  seam for hyperparameter values, and it is how `threshold: 0.6 / Fa: 0.07 / Fb: 0.8` already
  reach the default VBx instance. Prefer it over poking attributes directly.
- **`--clustering-model` DOES need construction-time intervention**, because the class comes from
  `config["pipeline"]["params"]["clustering"]` before `Klass(**params)` runs. Changing the class
  on a loaded pipeline means re-running the `__init__` logic at
  `speaker_diarization.py:283-295` (enum lookup, PLDA special-case, `_expects_num_speakers`), so
  re-instantiation is the honest approach, not attribute assignment.
- §2 of this ticket says to "check how `Pipeline.from_pretrained` sets them for `community-1`
  before designing the override path" — the above is that check; do not redo it, but DO verify
  that whatever you build actually reaches the instance (criterion 2 requires reading the value
  back off the clustering object).
- The §2 warning about `VBxClustering` overriding `__call__` rather than `cluster`
  ([clustering.py:572](../../../src/pyannote/audio/pipelines/clustering.py)) is now doubly
  relevant, since VBx is the *default* path, not an exotic one. Verify overrides reach it.

## Work

### 1. `--clustering-model` argument and vocabulary

Add a controlled vocabulary to `run_manifest.py` alongside the existing ones. Start with the
methods that are implemented and in scope:

- `pyannote-default` (agglomerative, as shipped) — keep as the default so existing behaviour is
  unchanged when the flag is omitted
- `agglomerative` (explicit, for sweeps)
- `vbx`
- `kmeans`

New methods add values as their tickets land. `OracleClustering` is **not** in the vocabulary;
oracle conditions are out of scope for this line of work.

### 2. Hyperparameter overrides

Every ticket under the epic depends on this. Needs a path from the command line to the
instantiated clustering object.

pyannote pipelines carry hyperparameters as `Uniform` / `Integer` / `Categorical` descriptors
that are resolved on instantiation, so check how `Pipeline.from_pretrained` sets them for
`community-1` before designing the override path. **`VBxClustering` overrides `__call__`
rather than `cluster`** ([clustering.py:572](../../../src/pyannote/audio/pipelines/clustering.py)),
so verify overrides reach it by the same route as the others rather than assuming they do.

Suggested interface: repeatable `--clustering-param name=value`, parsed against the target
class's declared descriptors so a typo fails loudly rather than being silently ignored. A
silently ignored hyperparameter would produce a flat sweep curve that looks like a real
negative result.

### 3. Cache key — co-owned with [[pre-clustering-embedding-cache]]

**This is the part that breaks the epic if it is missed.**

[runner.py:37-42](../../../harness/runner.py) builds the cache key from
`pipeline_config_id | segmentation_source.id | refinement_id | uri`, and
[run_harness.py:280](../../../run_harness.py) passes the bare checkpoint string as
`pipeline_config_id`.

So clustering configuration is absent from the key. Run a sweep today and every point after the
first is a cache hit returning the first point's RTTM: identical DER at every setting, valid
manifests, a flat curve, and no error anywhere. This is the `e9f25911` defect at sweep scale.

`pipeline_config_id` must incorporate the clustering class and every instantiated
hyperparameter value. The `Runner` docstring already describes this as an intended use ("or a
hash of its instantiated hyperparameters"), so no interface change is needed.

Record the readable form in the manifest as well as the hash. A hash alone makes a run
unreadable a month later.

Whichever of the two tickets lands first implements this. The other verifies it with its own
test. **Neither may assume the other did it.**

### 4. Routing regression tests, one per method

[[Oracle 2x2 Combined Cells]] item 2 found an exact-equality selection test that fell through
to a default branch: the run completed, wrote a manifest claiming one thing, and scored
another. Silent, plausible-looking, wrong.

Assert per vocabulary value that the pipeline actually holds an instance of the named class
after selection. Include a test that an unknown value fails loudly rather than falling back to
the default.

## Acceptance criteria

1. `--clustering-model` selects the named class; a test per vocabulary value asserts the
   pipeline holds that class. An unknown value exits with an error.
2. `--clustering-param` reaches the instantiated object. A test asserts a set value is
   readable off the clustering instance afterwards, and that an unknown parameter name fails
   loudly.
3. Two runs differing only in a clustering hyperparameter produce **different cache keys**.
   This test must fail on the current code.
4. The manifest records the clustering class and the full hyperparameter set in readable form.
5. Omitting `--clustering-model` reproduces current behaviour exactly: a cold-cache baseline
   run still yields DER 0.17048543579940637. **(Run by the coordinator after both tickets
   land, not by this ticket — see the brief. This ticket must additionally assert at unit
   level that omitting the flag yields the `pyannote-default` path — which resolves to
   **`VBxClustering`**, NOT `AgglomerativeClustering`; see the coordinator finding below —
   since that part CAN be tested without a GPU run.)**
6. **The pre-existing test-failure set is UNCHANGED.** Verified by capturing sorted
   `FAILED`/`ERROR` lines before and after the change and diffing them.

   **Rewritten by the coordinator (2026-09-25) — the original read "full test suite passes",
   which is unsatisfiable on this tree and so would have been silently reinterpreted.** 28
   tests fail before any work starts. An aggregate pass count is not acceptable evidence; the
   diff of the failure set is.

7. **Added by the coordinator (2026-09-25).** The cache-key restructure is owned by
   [[pre-clustering-embedding-cache]], which lands first. This ticket does **not** restructure
   `pipeline_config_id`; it adds clustering fields into a key that already has the right
   shape, and proves with its own test that two clustering configurations differ in the final
   key while sharing the intermediate key. If the restructure is absent when this ticket
   starts, stop and report — do not implement it a second time.

## Out of scope

- No new clustering implementations. Those are the epic's sub-tickets.
- `OracleClustering` is not added to the vocabulary.
- No sweeps. This ticket makes sweeps possible; it does not run any.

## Implementation Notes

_Append while the work happens._

### Scope as actually executed (2026-09-25)

The cache-key restructure (§3) was **already landed** by
[[pre-clustering-embedding-cache]] before this ticket started. Verified, not
rebuilt: `harness/cache_id.py` already exposes `pipeline_config_id`
(checkpoint + clustering class + every instantiated hyperparameter),
`intermediate_config_id` (checkpoint only) and
`clustering_config_description`, and `run_harness.py` already records
`clustering_config` / `intermediate_config_id` in the manifest. This ticket
added only its own sensitivity test over that existing key.

### Verified facts the work rests on

* `pyannote-default` is **`VBxClustering` with threshold 0.6 / Fa 0.07 /
  Fb 0.8**, not agglomerative. Re-verified from the shipped
  `config.yaml` (`pipeline.params.clustering: VBxClustering`,
  `params.clustering: {threshold: 0.6, Fa: 0.07, Fb: 0.8}`). Several docs and
  this ticket's own §1 said agglomerative; that is false.
* `vbx` and `pyannote-default` are the **same class**. They differ only in
  whether hyperparameters are stated explicitly, not in algorithm.

### Override seams chosen, and why

Two different seams, because the two overrides happen at different times in
`Pipeline.from_pretrained` (`core/pipeline.py:275-294`).

**Class selection -> construction time.** The class comes from
`config["pipeline"]["params"]["clustering"]` and is consumed by
`Klass(**params)` at :278. `SpeakerDiarization.__init__`
(`speaker_diarization.py:283-295`) does three things with it: the `Clustering`
enum lookup, the `VBxClustering`-only special case that passes `self._plda`,
and setting `self._expects_num_speakers` from
`clustering.expects_num_clusters`. So selection is implemented by passing
`clustering=<ClassName>` through to `from_pretrained`, letting `__init__` run
all three steps.

*Alternative rejected:* assigning `pipeline.clustering = SomeClustering(...)`
on a loaded pipeline. That leaves `klustering` and `_expects_num_speakers`
describing the old class while `clustering` is the new one — a silent,
plausible-looking inconsistent state, and exactly the class of bug §4 was
written to guard against. It also cannot construct `VBxClustering` at all,
whose `__init__` takes a positional `plda` with no default
(`clustering.py:555-563`).

**Hyperparameters -> `pipeline.instantiate()`.** This is the sanctioned seam
and already how 0.6/0.07/0.8 reach the default VBx instance (:290-291). It
needs no rebuild. It also gives loud failure for free: base
`Pipeline.instantiate` raises `ValueError: parameter '<name>' does not exist`
for an unknown name (site-packages `pyannote/pipeline/pipeline.py:443-444`).

### Unknown-value validation: do not weaken the library's

`SpeakerDiarization.__init__` already raises `ValueError` listing every valid
enum member for an unknown clustering name. The harness layer therefore does
**not** add a second, weaker check that could accept something the library
rejects. It maps its own short vocabulary to class names and rejects anything
outside that vocabulary at argparse level, so both layers fail loudly and the
harness's set is a strict subset of the library's.

### "If this silently did nothing, what would tell us?"

* **`--clustering-model` parsed but never reaching the pipeline** — the
  manifest would look perfect while the default VBx ran. Guarded by deriving
  the manifest's `clustering_model` from the LIVE pipeline object
  (`clustering_config_description(pipeline)`), never from the flag string, and
  by a test per vocabulary value asserting `isinstance(pipeline.clustering,
  <class>)` after selection.
* **`--clustering-param` silently ignored** — a flat sweep curve that reads as
  a real negative result. Guarded by reading the value back off the clustering
  instance after selection, and by the library's own unknown-name `ValueError`.
* **Clustering absent from the cache key** — every sweep point serves point
  1's RTTM. Guarded by the already-landed key plus this ticket's sensitivity
  test, which is shown failing against the old bare-checkpoint key logic.

### Files touched

* `run_harness.py` — `--clustering-model` / `--clustering-param` flags,
  `_parse_clustering_params`, `_clustering_class_name`, `build_pipeline`.
* `harness/run_manifest.py` — `VALID_CLUSTERING_MODELS` vocabulary,
  `CLUSTERING_MODEL_CLASSES` mapping.
* `tests/test_clustering_model_selection.py` — new.

### Findings made during the work (not anticipated by the ticket)

**1. `Pipeline.from_pretrained` has no seam for a construction parameter.** The
planned approach - pass `clustering=<name>` through `from_pretrained` - does not
work. Its signature is fixed (`checkpoint, revision, hparams_file, subfolder,
token, cache_dir`, core/pipeline.py:153-161) with no `**kwargs`, so it raises
`TypeError: unexpected keyword argument 'clustering'`. The ticket's analysis
correctly identified that the CLASS is fixed at construction, but assumed a
reachable seam existed there; none does.

Caught only by running against the real pipeline. The unit tests passed first
try because the fake `from_pretrained` accepted `**kwargs` - a fake more
permissive than the real thing hides exactly the bug it should catch. The fake
now mirrors the real fixed signature, and its docstring records why.

Resolved without touching `src/pyannote/`: `_from_pretrained_with_clustering`
reproduces `from_pretrained`'s own load sequence over the shipped config with
`pipeline.params.clustering` swapped. Rejected alternative: passing a config
dict to `from_pretrained`, which takes a dict checkpoint - it sets
`model_id = Path.cwd()` (:191) and breaks `$model/...` asset resolution.

**2. The default condition is a separate branch, deliberately.** `build_pipeline`
routes the shipped class through the *untouched* pre-ticket
`Pipeline.from_pretrained(checkpoint, token=...)` call and only non-default
classes through the reimplemented helper. This is what makes "the baseline
cannot have moved" structural rather than a claim - the reimplemented sequence
is not on the default path at all. A test asserts the helper is not called for
`pyannote-default`/`vbx`, and mutating the branch to route everything through
the helper fails 2 tests.

**3. Shipped hyperparameter defaults must be filtered per class.** The shipped
config instantiates VBx's `threshold`/`Fa`/`Fb`, and `Pipeline.instantiate`
raises on a parameter the target class does not declare - so handing
agglomerative VBx's `Fa` makes every non-default selection crash on load.
Defaults are filtered to declared names and the drop is PRINTED, not silent.

Gotcha worth knowing: `threshold` exists on BOTH VBx and agglomerative with
different meanings and ranges (`Uniform(0.5, 0.8)` vs `Uniform(0.0, 2.0)`), so
`agglomerative` inherits `threshold=0.6` from the VBx config rather than running
at its own default. Verified: `AgglomerativeClustering(threshold=0.6)`. An
explicit `--clustering-param` is applied afterwards and always wins.

**4. `kmeans` is in the vocabulary but NOT RUNNABLE by this harness.**
`KMeansClustering.expects_num_clusters` is True and `apply()` requires
`num_speakers` (speaker_diarization.py:600-607), which the harness never
supplies. The ticket put `kmeans` in the vocabulary without noting this.
`build_pipeline` refuses it at selection time.

Refused at selection rather than left to the library for two reasons. The
library raises part-way through the corpus after minutes of GPU work. And the
worse one: `Runner.run()` sets `pipeline.training = True`, and the library's
guard falls back to `len(file["annotation"].labels())` when an annotation is
present. Today the harness keeps `annotation` off the file dict so it cannot
fire - but if it ever did, a kmeans run would silently take k from ground truth
and report an oracle-count result under an ordinary baseline manifest.
`harness/segmentation.py` already guards the oracle path; this covers baseline.

Tests split "selectable" from "runnable" so criterion 1 is still proven for
`kmeans`, and one test asserts the refusal is not hiding a broken selection
(failure mode 3 - a raise is only as good as the distinction it draws).

**5. A bug I introduced and fixed, kept as a regression test.** The first version
of the num_speakers guard used plain truthiness. `getattr` on a `MagicMock`
returns a truthy Mock for any attribute name, so the guard refused every
mock-driven caller and broke 6 pre-existing tests
(`test_run_harness_cli.py`, `test_oracle_combined_cells.py`). Now `is True`.
Loud here, but the reverse - a guard that silently never fires - is how an
unrunnable configuration reaches a real run, so the case is a test.

### Verification approach

Every guard was proven to have teeth by mutation rather than by passing:
selection ignored (10 tests fail), hyperparameter override dropped (7),
manifest read from the flag instead of the live pipeline (2), unknown model
silently falling back (1), default routed through the reimplemented path (2),
truthiness guard (8 across three files). Criterion 3's sensitivity is shown
by an explicit test asserting the old bare-checkpoint key COLLIDES where the
current key differs.

Real-pipeline verification (model loading only - no GPU inference, no scored
run; `PYANNOTE_SKIP_DEPENDENCY_CHECK=1` for the dev-version check):

```
pyannote-default  VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6)  final=190b62c3 interm=c636fc56
vbx               VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6)  final=190b62c3 interm=c636fc56
agglomerative     AgglomerativeClustering(threshold=0.6)         final=610c4b8e interm=c636fc56
kmeans            REFUSED (expects_num_clusters True)
vbx threshold=0.7 readback=0.7                                   final=d12f731e interm=c636fc56
```

DEFAULT KEY UNCHANGED: final `190b62c3...` and intermediate `c636fc56...` both
identical to what the pre-ticket code produces, matching the coordinator's
independently recorded values. The baseline run will hit its existing cache.

## On completion

Fold into [[Evaluation Harness]] and [[Run Manifest]] rather than creating a new doc. Set
`status: done` and delete.

## See also

- [[clustering-method-comparison]] — the epic this unblocks.
- [[pre-clustering-embedding-cache]] — co-owns the cache-key fix in item 3.
- [[Run Manifest]] — the `clustering_model` field, currently hardcoded.
- [[Oracle 2x2 Combined Cells]] — the routing trap item 4 guards against.
