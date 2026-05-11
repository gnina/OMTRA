#!/bin/bash
# Creates eval directory structure and eval_commands.txt files for a model.
#
# Usage:
#   bash create_eval_commands.sh <model_name> <checkpoint_path> <partial_mode> [eval_modes...]
#
# Arguments:
#   model_name       Short name used for the output dir (e.g. mixed_frag_bernoulli)
#   checkpoint_path  Relative path to last.ckpt from OMTRA root
#   partial_mode     The partial mode the model was trained with
#   eval_modes       Optional list of additional partial modes to evaluate with
#                    (whole_fragments is always included by default)
#
# Examples:
#   bash create_eval_commands.sh mixed_frag_bernoulli \
#     outputs/2026-04-19/gnn_partial_mixed_frag_bernoulli_.../checkpoints/last.ckpt \
#     fragment_bernoulli_atom
#
#   bash create_eval_commands.sh mixed_whole_frag \
#     outputs/2026-04-20/gnn_partial_mixed_whole_frag_.../checkpoints/last.ckpt \
#     whole_fragments \
#     whole_fragment_plus_atoms

set -e

OMTRA="$(cd "$(dirname "$0")/../.." && pwd)"
CHUNKS_SRC="$OMTRA/outputs/eval/frag_bernoulli/partial_fixed_protein_ligand_denovo_condensed_fragment_bernoulli_atom/chunks"
PLINDER="/net/galaxy/home/koes/icd3/moldiff/OMTRA/data/plinder"

if [ $# -lt 3 ]; then
  echo "Usage: $0 <model_name> <checkpoint_path> <partial_mode> [extra_eval_modes...]"
  exit 1
fi

MODEL=$1
CKPT=$2
TRAINED_MODE=$3
shift 3
EXTRA_MODES=("$@")

# Always evaluate with the model's trained mode + whole_fragments
EVAL_MODES=("$TRAINED_MODE")
for extra in "${EXTRA_MODES[@]}"; do
  EVAL_MODES+=("$extra")
done
# Add whole_fragments if not already present
if [[ ! " ${EVAL_MODES[*]} " =~ " whole_fragments " ]]; then
  EVAL_MODES+=("whole_fragments")
fi

TASKS=(
  "partial_fixed_protein_ligand_denovo_condensed"
  "partial_rigid_docking_condensed"
)

for TASK in "${TASKS[@]}"; do
  for EVAL_MODE in "${EVAL_MODES[@]}"; do
    EVAL_DIR="$OMTRA/outputs/eval/$MODEL/${TASK}_${EVAL_MODE}"
    mkdir -p "$EVAL_DIR/chunks"

    # Symlink chunk files from reference eval
    for CHUNK in "$CHUNKS_SRC"/chunk_*.csv; do
      ln -sf "$CHUNK" "$EVAL_DIR/chunks/$(basename "$CHUNK")" 2>/dev/null || true
    done

    # Generate eval_commands.txt (100 lines, one per chunk)
    rm -f "$EVAL_DIR/eval_commands.txt"
    for i in $(seq 0 99); do
      CHUNK_FILE=$(printf "%s/chunks/chunk_%03d.csv" "$EVAL_DIR" "$i")
      OUT_DIR="$OMTRA/outputs/eval/$MODEL/${TASK}_${EVAL_MODE}/samples_${TASK}/chunk_${i}_rep_0"
      echo "python $OMTRA/omtra_pipelines/docking_eval/docking_eval.py --ckpt_path=$CKPT --task=$TASK --sys_idx_file=$CHUNK_FILE --n_replicates=100 --n_samples=1 --bs_per_gbmem=5 --output_dir=$OUT_DIR --plinder_path=$PLINDER --split=test --dataset=plinder --timeout 2700 --partial_mode $EVAL_MODE" >> "$EVAL_DIR/eval_commands.txt"
    done

    echo "Created: outputs/eval/$MODEL/${TASK}_${EVAL_MODE}  ($(wc -l < "$EVAL_DIR/eval_commands.txt") commands)"
  done
done

echo ""
echo "Submit with:"
for TASK in "${TASKS[@]}"; do
  for EVAL_MODE in "${EVAL_MODES[@]}"; do
    echo "  CMD_FILE=outputs/eval/$MODEL/${TASK}_${EVAL_MODE}/eval_commands.txt \\"
    echo "    sbatch --array=1-100 slurm_scripts/omtra_eval.slurm"
  done
done
