---
status: done
created: 2026-09-04
---

# AMI Corpus Setup

Related: [[Evaluation Harness]], [[Dataset Adapter]], [[Orchestrator]], [[GPU Support]].

## What this covers

The AMI Meeting Corpus test split (IHM/Mix-Headset, 16 meetings) is downloaded, repaired,
and reshaped into the exact directory layout the evaluation harness expects, so
`run_harness.py` can score `community-1` end to end. A `pyannote.database` protocol is
also registered over the same raw download, purely so the official `pyannote-audio
benchmark` CLI can be run as an independent cross-check against the harness's own
result. Both were run to completion and agree with each other and with the published
reference DER — see Validation result below.

SDM and the dev split are not covered here; see Follow-ons.

## Why two separate data layouts

[[Evaluation Harness]] does not consume `pyannote.database` protocols at all — it reads
data directly off disk in its own layout. Concretely:

- `harness/datasets.py`'s `AMIDatasetAdapter` reads exactly
  `{data_root}/{condition}/{split}.rttm` and `{data_root}/{condition}/{split}.uem` — one
  combined, multi-uri file per split.
- `run_harness.py`'s `_audio_path()` resolves audio at
  `{data_root}/{condition}/audio/{uri}.wav` — note: `{uri}.wav`, not
  `{uri}.Mix-Headset.wav`.

The raw `pyannote/AMI-diarization-setup` repo doesn't ship data in that shape: its RTTM
and UEM files are one-per-meeting, not combined, and its downloaded audio keeps the
`.Mix-Headset` suffix. So getting a working `pyannote.database` protocol (for the
`pyannote-audio benchmark` cross-check) and getting the harness its own data are two
distinct steps — a conversion step exists specifically to bridge that gap.

## Where things live

- Raw checkout: `~/Code/AMI-diarization-setup` (sibling directory to the pyannote-audio
  repo, not nested inside it — no `.gitignore` entry was needed for the corpus itself).
- Downloaded audio: `~/Code/AMI-diarization-setup/pyannote/amicorpus/{uri}/audio/{uri}.Mix-Headset.wav`
  (nested one level inside `pyannote/`, matching where the shipped `database.yml` already
  expected it).
- Harness-ready converted data: `~/Code/AMI-diarization-setup/harness-data/IHM/` —
  `test.rttm`, `test.uem`, and `audio/{uri}.wav` for all 16 meetings.
- Registered protocol config: `~/Code/AMI-diarization-setup/pyannote/database.yml`
  (audio path rewritten to an absolute path so it resolves regardless of working
  directory).
- The corpus path is recorded in this repo's `.env` as `PYANNOTE_DATABASE_CONFIG` — a
  plain env var (not read via `dotenv_values` by any `harness/` code; it's only consumed
  by `pyannote.database`'s own `get_protocol()` for the cross-check path). `.env` is
  gitignored.

## How to reproduce

1. Clone `https://github.com/pyannote/AMI-diarization-setup.git` as a sibling of this
   repo.
2. Run `pyannote/download_ami_test_split.sh <dest-dir>` — a scoped version of the
   upstream `download_ami.sh` that fetches only the 16 test-split meetings (not all 171)
   and repairs the four `IS1009{a,b,c,d}` WAVs with malformed chunk headers. Uses `curl
   --fail --location` against `https://` URLs directly.
3. Point `database.yml`'s audio path at an absolute path to the downloaded `amicorpus/`
   directory.
4. Run `pyannote/convert_for_harness.sh <data-root> [split]` to produce the harness's
   combined RTTM/UEM files and per-uri `{uri}.wav` hardlinks.
5. Run `run_harness.py --data-root <data-root> --condition IHM` for the harness path, or
   `pyannote-audio benchmark ... --registry database.yml` for the independent
   cross-check.

## Gotchas hit building this

- **The Edinburgh mirror now redirects `http://` to `https://`.** The upstream
  `download_ami.sh` script's URLs are `http://groups.inf.ed.ac.uk/...`, which now
  301-redirect. A plain `curl`/`wget` call without following redirects silently saves a
  372-byte HTML redirect page instead of the actual WAV — for every single file, with no
  visible error. Fixed by using `curl --fail --location` against `https://` directly.
  `--fail` also means a bad HTTP status now aborts loudly instead of writing garbage.
- **`wget` isn't installed** in this Git-Bash-on-Windows environment; `curl` was used
  throughout instead.
- **`test.meetings.txt` has CRLF line endings.** Reading it naively in a bash loop left a
  trailing `\r` on every URI, silently corrupting every constructed file path (looked
  correct when printed, failed every `-f` file check). Fixed by stripping `\r` off each
  line.
- **Symlinks silently become copies on Windows without elevation.** Git Bash's `ln -s`
  on this machine, without admin rights or developer mode enabled, produces what looks
  like a symlink but is actually a full file copy (no `l` permission bit, separate
  inode). Used `ln` (hardlink) instead — same effect intended by the original plan (no
  duplicated disk usage, corpus stays outside git), and same-filesystem hardlinks don't
  need elevation.
- **The audio directory nests one level deeper than expected.** The download script's
  own convention puts `amicorpus/` inside `pyannote/`, not at the top of the raw
  checkout — `database.yml`'s relative path already assumed this, but it's easy to get
  wrong when first laying out where a download lands.
- **`pyannote-audio benchmark`'s CLI doesn't set `PYANNOTE_SKIP_DEPENDENCY_CHECK`** the
  way `HarnessConfig.load()` does internally for the harness path. Running the benchmark
  CLI directly against this fork's dev version string fails a pipeline-version
  compatibility check unless that env var is exported manually first.

## Validation result

Both tools were run against the same 16 test-split meetings, on GPU (see
[[GPU Support]] — the CPU version of this same run was taking 90+ minutes and was
stopped early before GPU support existed):

| Method | DER |
|---|---|
| `run_harness.py` (harness path) | 17.05% |
| `pyannote-audio benchmark` (protocol path) | 17.05% |
| Published `community-1` reference (AMI IHM) | 17.0% |

The two independent tools agree to two decimal places, and both sit within 0.05 points
of the published reference. This is a clean pass — no discrepancy needed investigating.

## Data access and attribution

No access request needed for the corpus itself — AMI is CC BY 4.0, downloaded directly
from the Edinburgh mirror, no registration required. The only access gate in the whole
setup is HuggingFace's model-conditions gate on `pyannote/speaker-diarization-community-1`.

For any report's Statement/attribution section:

- AMI corpus, CC BY 4.0. Carletta et al., *The AMI Meeting Corpus: A Pre-Announcement*,
  MLMI 2006.
- Diarization setup and references: Landini, Profant, Diez, Burget, *Bayesian HMM
  clustering of x-vector sequences (VBx) in speaker diarization*. The BUT repository
  requests this citation specifically.
- Manual annotations version 1.6.2.

`diarizers-community/ami` on HuggingFace was considered and rejected: easier to install,
but preprocessed/chunked for fine-tuning segmentation models — the wrong shape for
full-file DER benchmarking.

## Follow-ons

- Add the SDM condition (`AMI-SDM.SpeakerDiarization.only_words`) — the same
  per-meeting-to-combined conversion gap will apply there too.
- Download the dev split if threshold tuning is required later.
- Oracle segmentation injection interface (deferred; see
  [[Segmentation Source Interface]]).

## Key files

- `~/Code/AMI-diarization-setup/pyannote/download_ami_test_split.sh`
- `~/Code/AMI-diarization-setup/pyannote/convert_for_harness.sh`
- `~/Code/AMI-diarization-setup/pyannote/database.yml`
- This repo's `.env` — `PYANNOTE_DATABASE_CONFIG`
