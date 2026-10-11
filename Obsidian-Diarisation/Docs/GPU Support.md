---
status: done
created: 2026-09-08
---

# GPU Support for the Evaluation Harness

Part of the [[Evaluation Harness]]. Related: [[Orchestrator]], [[Run Manifest]].

## What it does

`run_harness.py`'s command-line entry point can now run the diarization pipeline on a
GPU. A `--device` flag selects the device explicitly (`cpu`, `cuda`, `cuda:0`, etc.);
left unset, it auto-detects — `cuda` if `torch.cuda.is_available()`, otherwise `cpu` —
so existing invocations on machines without a GPU keep working unchanged.

This closed a real gap: nothing previously moved the loaded pipeline off CPU, even on
GPU-equipped machines. A 16-meeting AMI test-split run took over 90 minutes on CPU
(stopped early, incomplete) versus a couple of minutes end-to-end on GPU — roughly two
orders of magnitude faster, since `pyannote.audio.core.pipeline.Pipeline` already had a
working `.to(device)` method that simply was never called anywhere in the harness.

## How it works

`_resolve_device(requested: Optional[str]) -> torch.device` in `run_harness.py`:

- `requested is None` → `torch.device("cuda")` if `torch.cuda.is_available()`, else
  `torch.device("cpu")`.
- `requested` given → `torch.device(requested)` directly, so any value `torch.device()`
  itself accepts (`"cpu"`, `"cuda"`, `"cuda:0"`, ...) works without extra parsing.

`run_harness.py`'s former `if __name__ == "__main__":` block is now a real
`main(argv: Optional[Sequence[str]] = None) -> Dict[str, float]` function, with
`if __name__ == "__main__": main()` at the bottom. This refactor exists specifically so
the CLI's actual device-placement wiring is testable end to end: `main()` can be called
with fake `argv` and a mocked `Pipeline` class, letting a test confirm
`pipeline.to(resolved_device)` really gets called and that the *moved* pipeline (not the
pre-move reference) is what proceeds to scoring. `argparse.parse_args(argv)` accepting
`None` preserves the original real-CLI behavior. `Pipeline` is now imported at module
level (was previously imported inline inside `__main__`) so it can be patched from tests.

`run_harness()` — the actual scoring function — is untouched. Device placement happens
entirely in `main()`, on the already-constructed pipeline, before it's passed in; there
was no need for a new parameter on `run_harness()` itself.

## Non-goals

- No multi-GPU / data-parallel support — one device for the whole run.
- No automatic fallback-on-OOM — if `--device cuda` is requested and fails, that's a
  real error to surface, not a silent retry on CPU.
- `Pipeline.to()` itself was not changed — it already did what was needed.

## Tests

`tests/test_run_harness_cli.py`, no real GPU required (`torch.cuda.is_available` is
mocked throughout):

- Device auto-detection resolves to `cuda` when available, `cpu` otherwise.
- An explicit `--device` value always overrides auto-detection, in both directions.
- An indexed device string (`cuda:0`) round-trips correctly.
- Driving `main()` with fake CLI args and a mocked `Pipeline` class confirms
  `pipeline.to()` is actually called with the resolved device, and that the object
  returned by `.to()` — not the pre-move pipeline — is what gets passed into
  `run_harness()` for scoring.

Re-running `tests/test_run_harness_integration.py`, `tests/test_run_manifest.py`, and
`tests/test_aggregate_runs.py` alongside this new file confirmed the `main()` refactor
introduced no regressions: the same 14 tests passed as before, and the same
pre-existing, unrelated `full_pipeline` fixture encoding bug (see [[Run Manifest]]'s doc)
still blocks the same 8 integration tests it always did.

## Real-hardware verification

Re-ran the AMI test-split cross-check from
[[AMI corpus acquisition and harness data-layer wiring]] with `--device cuda` against
the converted 16-meeting corpus:

- `nvidia-smi` showed 94% GPU utilization and ~3.5GB VRAM in use during the run,
  confirming the pipeline genuinely ran on GPU.
- The full run completed in a couple of minutes, versus 90+ minutes still incomplete
  when the same run was earlier stopped on CPU.
- Result: `der=0.1705` (17.05%), matching the published `community-1` AMI-IHM reference
  of 17.0% to within 0.05 points.

Full summary from that run:

```
{'der': 0.1705, 'overlap_der': 0.3339, 'jer': 0.2219, 'counting_mae': 0.25, 'counting_exact_match_percent': 75.0}
```

The independent `pyannote-audio benchmark` cross-check (comparing against the registered
`pyannote.database` protocol rather than the harness's own converted layout) was not run
in this session — the harness result alone already validates GPU support and lines up
closely with the published reference. Running the benchmark CLI as a second, independent
check remains open under the AMI ticket's own acceptance gate, not this one.

## Key files

- `run_harness.py` — `_resolve_device()`, `main()`, the `--device` CLI flag
- `tests/test_run_harness_cli.py`
