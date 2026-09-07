---
status: in-progress
created: 2026-09-04
---

# AMI corpus acquisition and harness data-layer wiring

**Blocks:** Baseline evaluation harness run, ceiling analysis
**Depends on:** Nothing

## Objective

Get the AMI corpus on disk in the standard benchmark configuration (IHM/Mix-Headset,
test split only, 16 meetings), and get it into the exact directory shape
[[Dataset Adapter]] and [[Orchestrator]] already expect, so `run_harness.py` can score
`community-1` end to end. Also register a `pyannote.database` protocol over the same raw
download, purely so `pyannote-audio benchmark` can be run as an independent cross-check.

SDM and the dev split are follow-on, not in scope here.

## Why this ticket exists in this shape

An earlier draft of this ticket (not checked into the repo, reviewed 2026-09-04) assumed
that registering a `pyannote.database` protocol was the whole job — get `get_protocol()`
returning 16 files with `annotation`/`audio` populated, cross-check DER against
`pyannote-audio benchmark`, done. That's necessary but not sufficient.

[[Evaluation Harness]] does not consume `pyannote.database` protocols at all. Its own doc
says so explicitly: *"Does not use the `pyannote-audio benchmark` CLI... has no oracle
seam."* Concretely, per [[Dataset Adapter]] and [[Orchestrator]]:

- `harness/datasets.py`'s `AMIDatasetAdapter` reads exactly
  `{data_root}/{condition}/{split}.rttm` and `{data_root}/{condition}/{split}.uem` —
  **one combined, multi-uri file per split**, parsed with `pyannote.database.util.load_rttm`
  / `load_uem`.
- `run_harness.py`'s `_audio_path()` resolves audio at
  `{data_root}/{condition}/audio/{uri}.wav` — note: `{uri}.wav`, not
  `{uri}.Mix-Headset.wav`.

Real AMI-diarization-setup data does not look like that. Confirmed directly against the
`pyannote/AMI-diarization-setup` repo (2026-09-04):

- `only_words/rttms/test/` holds **16 separate per-meeting RTTM files**, not one
  combined `test.rttm`.
- `uems/test/` holds **16 separate per-meeting UEM files**
  (`EN2002a.uem` … `TS3003d.uem`), not one combined `test.uem`.
- `download_ami.sh` downloads audio to `$DLFOLDER/{uri}/audio/{uri}.Mix-Headset.wav` —
  confirming the ticket's claim about the shipped `database.yml` path template, but also
  confirming the filename does **not** match what `_audio_path()` expects.

So a protocol-only ticket produces data that `pyannote-audio benchmark` can score, but
that `run_harness.py` still can't find. This version of the ticket adds the missing
conversion step (A6) and makes it part of the acceptance gate, and corrects the
cross-check step (old A6, now A7) to stop describing something that isn't literally
possible ("run the harness over the protocol" — the harness takes a data folder, not a
protocol object).

Everything else in the original draft was checked against the live repo and confirmed
accurate: the 16 test-split meeting names, the malformed-WAV file list (all four
`IS1009{a,b,c,d}` are genuinely on it, confirmed against the real 24-file
`fix_wav` call list in `download_ami.sh`), and the published `community-1` reference DER
(17.0% AMI IHM / 19.9% AMI SDM, matching pyannote.ai's own community-1 announcement).

## HUMAN TASKS

Do these first — they require a decision or manual action.

### H1. Confirm HuggingFace model conditions are accepted

- [ ] Log into HuggingFace and confirm user conditions are accepted on
      `pyannote/speaker-diarization-community-1`
- [ ] Confirm a token is available and current — no `.env` currently exists in this repo
      (checked 2026-09-04), so this is a fresh setup step, not just a staleness check

### H2. Decide corpus storage location

- [ ] Choose a path on the RTX 3060 machine (primary)
- [ ] Choose a path on the MacBook Pro (secondary), or decide to skip it for now
- [ ] Confirm free disk space at each location

| Scope | Meetings | Approx size |
|---|---|---|
| test, IHM only | 16 | ~500 MB |
| test + dev, IHM | 34 | ~1.1 GB |
| full corpus, IHM | 171 | ~5 GB |

Corpus lives **outside** the git repo. If it ends up nested anyway, add the path to
`.gitignore`.

### H3. Decide download scope

- [ ] Test split only (recommended)
- [ ] Test + dev
- [ ] Full corpus

Dev is only needed if threshold tuning happens later. Train is never needed — nothing is
being retrained.

### H4. Decide repo layout for the raw AMI-diarization-setup checkout

- [ ] **Option A (default):** sibling checkout. Clone `AMI-diarization-setup` next to the
      pyannote fork, point `PYANNOTE_DATABASE_CONFIG` at it.
- [ ] **Option B:** vendor RTTMs/UEMs/lists into this repo. Self-contained but duplicates
      upstream data and needs an attribution note.

This decision only affects the raw download + protocol-registration side (A1–A5). The
harness-facing converted data (A6) is a separate, harness-owned directory either way —
see A6.

### H5. Review the validation cross-check result

- [ ] Compare harness DER (`run_harness.py` against the converted data root, A6) with
      `pyannote-audio benchmark` DER (against the registered protocol, A5) on the same 16
      files
- [ ] Judge whether any discrepancy is a harness bug or a legitimate setup difference

This is the acceptance gate. See Reference values below.

## Agent tasks (CLI)

### A1. Clone the diarization setup

```bash
git clone https://github.com/pyannote/AMI-diarization-setup.git
```

### A2. Download audio

Scripts live in `AMI-diarization-setup/pyannote/`: `download_ami.sh` (full corpus),
`download_ami_mini.sh` (36-file smoke-test subset), `download_ami_sdm*.sh` (SDM,
follow-on). For test-split-only, filter the `wget` lines against
`lists/test.meetings.txt` rather than pulling all 171 files.

Test split (confirmed against `uems/test/`, 16 files): `IS1009{a,b,c,d}`,
`ES2004{a,b,c,d}`, `TS3003{a,b,c,d}`, `EN2002{a,b,c,d}`.

Run `download_ami_mini.sh` first as a smoke test before committing to the full
test-split download.

### A3. Run the WAV repair step

24 files across the corpus have malformed WAV chunk headers; the `fix_wav` function at
the bottom of the download scripts rewrites them. Confirmed (2026-09-04) the full list
of 24 includes all four test-split IS meetings: `IS1009a`, `IS1009b`, `IS1009c`,
`IS1009d`. Do not skip this step for those four files.

Constraints:

- Run with `bash`, not `sh` — the script uses the `function` keyword.
- The function shells out to `python`, not `python3`. Ensure `python` resolves in the
  active environment, or patch the script.

### A4. Configure the protocol

Shipped `database.yml` uses a relative audio path
(`amicorpus/{uri}/audio/{uri}.Mix-Headset.wav`) that only resolves when run from the
`pyannote/` directory — confirmed against `download_ami.sh`'s actual output layout. Set
an absolute path instead, given two machines with different layouts.

```bash
export PYANNOTE_DATABASE_CONFIG=/abs/path/to/AMI-diarization-setup/pyannote/database.yml
```

### A5. Verify the protocol loads

```python
from pyannote.database import get_protocol, FileFinder

protocol = get_protocol(
    'AMI.SpeakerDiarization.only_words',
    preprocessors={'audio': FileFinder()},
)

for file in protocol.test():
    print(file['uri'], file['audio'])
    assert file['annotation'] is not None
    assert file['annotated'] is not None
```

Expect 16 files, every audio path resolving to an existing file. Use `only_words`, not
`word_and_vocalsounds` — the BUT authors flag annotator inconsistency in the
vocal-sounds references, and `only_words` is what published benchmarks score against.

This step exists so **A7's cross-check tool** (`pyannote-audio benchmark`) has something
to run against. It does not by itself produce anything `run_harness.py` can read — see A6.

### A6. Convert into the harness's expected layout (new — was missing entirely)

[[Dataset Adapter]] needs, under one `{data_root}`:

```
{data_root}/IHM/test.rttm          # one combined file, all 16 uris
{data_root}/IHM/test.uem           # one combined file, all 16 uris
{data_root}/IHM/audio/{uri}.wav    # per-file, note: not .Mix-Headset.wav
```

Concretely:

- Concatenate the 16 per-meeting files under
  `AMI-diarization-setup/only_words/rttms/test/*.rttm` into one
  `{data_root}/IHM/test.rttm`.
- Concatenate the 16 per-meeting files under `AMI-diarization-setup/uems/test/*.uem`
  into one `{data_root}/IHM/test.uem`.
- For each of the 16 uris, symlink (don't copy — corpus stays outside git either way)
  `amicorpus/{uri}/audio/{uri}.Mix-Headset.wav` → `{data_root}/IHM/audio/{uri}.wav`.

Plain concatenation is sufficient for both formats — RTTM lines are self-delimited by
their own `uri` field, and per-meeting UEM files use the same tab/space-separated
`uri channel start end` shape `load_uem` expects across multiple uris. Confirm by
inspecting one converted file before the full run: line count should equal the sum of
the 16 source files' line counts, with no header/footer to strip.

Whether this conversion is a one-off shell script or a small checked-in helper is an
implementation detail — either is fine as long as it's reproducible on the second
machine (H2/H4) without hand-editing.

### A7. Run the cross-check (was A6 in the earlier draft — corrected)

Two independent runs against the same 16 files, compared:

1. `pyannote-audio benchmark pyannote/speaker-diarization-community-1 AMI.SpeakerDiarization.only_words <output-dir>` —
   against the protocol from A5. (Positional args confirmed against the actual CLI
   signature in `src/pyannote/audio/__main__.py`: pipeline, protocol, output directory.)
2. `python run_harness.py --data-root <data_root> --split test --condition IHM ...` —
   against the converted layout from A6.

Both should land near the same DER on the same 16 meetings.

## Acceptance criteria

- [ ] 16 test-split Mix-Headset WAV files on disk, WAV repair applied to all four
      `IS1009{a,b,c,d}`
- [ ] `AMI.SpeakerDiarization.only_words` protocol loads, all 16 audio paths resolve,
      every file yields non-null `annotation`/`annotated`
- [ ] `{data_root}/IHM/test.rttm` and `test.uem` exist as single combined files covering
      all 16 uris, and `{data_root}/IHM/audio/{uri}.wav` resolves for all 16
- [ ] `run_harness.py` runs to completion against `{data_root}` with no `DatasetError`
- [ ] Harness DER and `pyannote-audio benchmark` DER agree on the same 16 files (within
      roughly a point of each other, and both within roughly a point of the 17.0%
      published reference)
- [ ] Corpus path recorded in project config, corpus excluded from git
- [ ] Attribution details captured for the report Statement section

## Reference values

Published for `community-1`, scored with no forgiveness collar and no skipping of
overlapped speech — matches harness config (`collar=0`, `skip_overlap=False`), confirmed
against pyannote.ai's own community-1 announcement:

| Condition | community-1 DER |
|---|---|
| AMI (IHM) | 17.0% |
| AMI (SDM) | 19.9% |

Caveat carried over from the earlier draft: the published table doesn't explicitly state
it uses the BUT `only_words` test split. A small discrepancy may be a legitimate setup
difference rather than a bug; a large one is a bug.

## Data access notes

No access request needed for the corpus itself — AMI is CC BY 4.0, direct download from
the Edinburgh mirror (`groups.inf.ed.ac.uk/ami/AMICorpusMirror/`, reachable as of
2026-09-04), no registration. The only access gate in the stack is the HuggingFace model
conditions (H1).

## Rejected alternative

`diarizers-community/ami` on HuggingFace: easier to install but preprocessed/chunked for
fine-tuning segmentation models — wrong shape for full-file DER benchmarking. Worth a
one-line mention in the report as considered and rejected.

## Attribution for the Statement section

- AMI corpus, CC BY 4.0. Carletta et al., *The AMI Meeting Corpus: A Pre-Announcement*,
  MLMI 2006.
- Diarization setup and references: Landini, Profant, Diez, Burget, *Bayesian HMM
  clustering of x-vector sequences (VBx) in speaker diarization*. The BUT repository
  requests this citation specifically.
- Manual annotations version 1.6.2.

## Follow-on tickets

- Add SDM condition (`AMI-SDM.SpeakerDiarization.only_words`) once IHM is validated —
  same A6 conversion gap will apply there too
- Download dev split if threshold tuning is required
- Oracle segmentation injection interface (deferred; see [[Segmentation Source Interface]])

## Implementation Notes

_(append here as work happens — decisions made, gotchas hit, files touched)_
