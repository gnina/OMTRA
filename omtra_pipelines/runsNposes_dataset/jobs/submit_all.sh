#!/bin/bash
#
# Submit Zarr store jobs for Runs N' Poses (test split, no_links only).
#
# Output structure:
#   data/runs_n_poses/           (8.0 Å pocket crop)
#     └── no_links/test.zarr
#   data/runs_n_poses_wide/      (12.0 Å pocket crop)
#     └── no_links/test.zarr
#
# Usage:
#   bash submit_all.sh /path/to/ground_truth          # submit jobs
#   bash submit_all.sh /path/to/ground_truth --dry     # dry run
#

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash submit_all.sh <ground_truth_dir> [--dry]"
    exit 1
fi

GROUND_TRUTH="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/run_zarr_store.sh"
NUM_CPUS="${NUM_CPUS:-16}"
DRY_RUN=false

if [[ "${2:-}" == "--dry" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN (not submitting) ==="
    echo ""
fi

# 8.0 Å crop → runs_n_poses
# 12.0 Å crop → runs_n_poses_wide
declare -A CUTOFF_MAP
CUTOFF_MAP[runs_n_poses]=8.0
CUTOFF_MAP[runs_n_poses_wide]=12.0

count=0
for OUTDIR in runs_n_poses runs_n_poses_wide; do
    CUTOFF="${CUTOFF_MAP[$OUTDIR]}"
    JOB_NAME="rnp_no_links_test_${CUTOFF}"
    CMD="sbatch \
        --job-name=${JOB_NAME} \
        --export=SPLIT=test,CUTOFF=${CUTOFF},OUTDIR=${OUTDIR},GROUND_TRUTH=${GROUND_TRUTH},NUM_CPUS=${NUM_CPUS} \
        --cpus-per-task=${NUM_CPUS} \
        ${JOB_SCRIPT}"

    if [ "$DRY_RUN" = true ]; then
        echo "${CMD}"
    else
        echo "Submitting: ${JOB_NAME}"
        eval "${CMD}"
    fi
    count=$((count + 1))
done

echo ""
echo "Total jobs: ${count}"
