#!/usr/bin/env bash
# Wrapper around run_harness.py that always does the full tracked-run
# workflow: run the pipeline, confirm a manifest landed, refresh
# runs/comparison.csv. This is the same workflow the track-harness-run
# skill enforces -- this script exists so a human can run it directly
# without needing an agent to babysit each step.
#
# Usage:
#   ./run_experiment.sh <run-condition> <notes> [extra run_harness.py args...]
#
# Examples:
#   ./run_experiment.sh baseline "unmodified community-1 defaults"
#   ./run_experiment.sh nearest_centroid "T2 plumbing validation run"
#   ./run_experiment.sh oracle_assignment "oracle ceiling, all_pairs scope" \
#       --refinement-strategy oracle --oracle-scope all_pairs --counts-toward-results
#   ./run_experiment.sh oracle_segmentation "ground-truth segmentation" \
#       --oracle-rttm path/to/only_words.rttm --counts-toward-results
#   ./run_experiment.sh oracle_segmentation_assignment "combined 2x2 cell, all_pairs" \
#       --refinement-strategy oracle --oracle-scope all_pairs \
#       --oracle-rttm path/to/only_words.rttm --counts-toward-results
#
# Data root, condition, and split default to the full AMI IHM test split you
# already have downloaded -- override with DATA_ROOT / CONDITION / SPLIT env
# vars if needed.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <run-condition> <notes> [extra run_harness.py args...]" >&2
  echo "  run-condition: baseline | oracle_segmentation | oracle_assignment |" >&2
  echo "                 oracle_segmentation_assignment | nearest_centroid" >&2
  exit 1
fi

RUN_CONDITION="$1"
NOTES="$2"
shift 2
EXTRA_ARGS=("$@")

DATA_ROOT="${DATA_ROOT:-$HOME/Code/AMI-diarization-setup/harness-data}"
CONDITION="${CONDITION:-IHM}"
SPLIT="${SPLIT:-test}"
RUNS_DIR="${RUNS_DIR:-runs}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

echo "== Running: run-condition=${RUN_CONDITION}  notes=\"${NOTES}\" =="

BEFORE_COUNT=$(ls "${RUNS_DIR}"/*.json 2>/dev/null | wc -l)

python run_harness.py \
  --data-root "${DATA_ROOT}" \
  --condition "${CONDITION}" \
  --split "${SPLIT}" \
  --device cuda \
  --cache-dir .harness_cache \
  --dotenv .env \
  --per-file-csv "per_file_${RUN_CONDITION}_${TIMESTAMP}.csv" \
  --summary "summary_${RUN_CONDITION}_${TIMESTAMP}.json" \
  --run-condition "${RUN_CONDITION}" \
  --runs-dir "${RUNS_DIR}" \
  --notes "${NOTES}" \
  "${EXTRA_ARGS[@]}"

AFTER_COUNT=$(ls "${RUNS_DIR}"/*.json 2>/dev/null | wc -l)

if [[ "${AFTER_COUNT}" -le "${BEFORE_COUNT}" ]]; then
  echo "ERROR: no new manifest appeared in ${RUNS_DIR}/ -- run was NOT tracked." >&2
  exit 1
fi

NEW_MANIFEST=$(ls -t "${RUNS_DIR}"/*.json | head -1)
echo "== Manifest written: ${NEW_MANIFEST} =="

python -c "from harness.aggregate_runs import aggregate_runs; print('Refreshed:', aggregate_runs(runs_dir='${RUNS_DIR}', output_csv_path='${RUNS_DIR}/comparison.csv'))"

echo "== Done. See ${RUNS_DIR}/comparison.csv for the updated comparison table. =="
