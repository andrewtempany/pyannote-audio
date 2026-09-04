# 03 — Dataset adapter

Part of the [eval harness epic](TICKET-eval-harness.md). Depends on: [02](TICKET-02-config-module.md).

## Scope

`harness/datasets.py` — given a config (split + data root + condition), yield `(uri, reference, uem)` tuples: `reference` a `pyannote.core.Annotation` built from ground-truth RTTM, `uem` a `pyannote.core.Timeline` from the official UEM. This is the only component that knows about file paths, RTTM parsing, and UEM parsing. IHM is the default condition to validate end to end; SDM must be selectable via config (ticket 02) without code changes here beyond reading the condition value.

Note the epic underspecified one thing worth resolving explicitly in this ticket rather than discovering it mid-implementation: AMI's IHM condition has multiple headset-mic channels per meeting, not one file per meeting. Decide and document the file-selection/aggregation strategy (e.g., one file per headset channel scored independently, vs. some per-meeting merge) as part of this ticket's design, since it changes what `uri` means.

## Non-goals

- No model loading, no pipeline invocation.
- No RTTM/UEM writing (read-only adapter).
- Do not attempt to download AMI — if the configured data root doesn't contain the expected files, fail with a clear message naming what's missing (per the epic's Environment section).

## Tests first

Add `tests/test_datasets.py` with small fixture RTTM/UEM files under `tests/fixtures/ami/` (synthetic, minimal — a couple of speakers, one overlap region, not real AMI data):

1. `test_iterates_expected_uris` — given a fixture split directory, assert the adapter yields the expected set of uris, no more, no fewer.
2. `test_reference_is_annotation_with_expected_labels` — for one fixture file, assert the returned `reference` is a `pyannote.core.Annotation` with the exact speaker labels and segment boundaries the fixture RTTM encodes.
3. `test_uem_matches_uem_file` — assert the returned `uem` is a `pyannote.core.Timeline` matching the fixture UEM's extent exactly.
4. `test_default_condition_is_ihm` — adapter constructed without an explicit condition selects IHM files.
5. `test_condition_param_switches_to_sdm` — adapter constructed with `condition="SDM"` selects the SDM fixture files instead of IHM.
6. `test_ihm_multi_channel_handling` — encodes whatever decision this ticket makes about multi-headset-channel IHM files (one test per channel-as-separate-uri, or one test proving a merge strategy — whichever was decided above).
7. `test_missing_rttm_for_expected_uri_raises_clear_error` — fixture UEM references a uri with no matching RTTM; assert a clear, specific error naming the missing file, not a bare `KeyError`/`FileNotFoundError`.
8. `test_missing_data_root_content_fails_clearly` — config points at a real-but-empty directory; assert a clear error distinct from a generic Python traceback.

## Acceptance criteria

- All tests pass against fixtures, no real AMI download or network access required to run the suite.
- The multi-channel IHM decision is written down (docstring or comment) as well as tested.

## Implementation notes (resolved during build — 2026-09-04)

- **RTTM/UEM line-parsing is delegated to `pyannote.database.util.load_rttm`/`load_uem`** (a sister package this repo already depends on) rather than hand-rolled — each returns a `{uri: Annotation}` / `{uri: Timeline}` dict. `harness/datasets.py` only owns directory-path resolution, condition/split selection, and error messages.
- **Directory convention adopted (a harness-specific choice, not a real-AMI given):** `{data_root}/{condition}/{split}.rttm` and `{data_root}/{condition}/{split}.uem` — one combined multi-uri file per (condition, split), mirroring this repo's own existing test fixture convention (`tests/data/debug.{split}.rttm`). If the user's real AMI layout differs, this directory-resolution logic is the only place that needs to change.
- **Multi-channel IHM resolution (as flagged in Scope):** the adapter's uri space comes entirely from what's present in the RTTM/UEM files — one `(uri, reference, uem)` per *meeting*, never per audio channel. It never enumerates or touches audio files at all. Resolving a uri to a specific audio path (e.g. a pre-mixed "Mix-Headset" file under IHM) is explicitly deferred to whatever loads audio later (TICKET-06's runner) — this adapter's output tuple has no audio path in it at all, matching the ticket's own `(uri, reference, uem)` contract.
- The adapter iterates the **UEM's** uris as the authoritative "files to score" list; a uri present in the UEM with no matching RTTM entry raises `DatasetError` naming the exact uri and both file paths involved.
- **Testing gotcha worth knowing for any future test involving `Annotation` equality:** `pyannote.core.Annotation.__eq__` also compares internal track identifiers, not just segment/label content. `load_rttm` assigns track ids from a pandas row index (`0`, `1`, `2`, ...), so a hand-built comparison `Annotation` using the default/auto track naming will compare unequal even when segments and labels match exactly. Fixed in this ticket's own test by comparing `itertracks(yield_label=True)` content instead of full `Annotation` equality — do the same in any later ticket that compares annotations built via different code paths (see also TICKET-05's note on constructing overlapping tracks).
