#!/bin/sh

# One-off preprocessing: pack the per-frame pose .h5 files into dense .pt
# tensors (see converter.py). Training then runs with `--use-pt`.
#
# Submit as an array — each task takes a disjoint, size-balanced slice of the
# videos, so wall time scales with the number of tasks:
#
#     cd slurm && mkdir -p logs
#     sbatch --array=0-7 convert.sh
#
# A plain `sbatch convert.sh` also works and converts everything in one task.
# Re-running skips videos that already have a .pt, so a job that hits the time
# limit can simply be resubmitted; add --overwrite to force a re-conversion.
#
# No GPU is used, so this asks for CPUs only and drops apptainer's --nv. If the
# partition refuses jobs without a GPU allocation, uncomment the --gres line.

#SBATCH --job-name="Convert BUTID"
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
##SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH -o logs/convert.%A_%a.out
#SBATCH -e logs/convert.%A_%a.err

set -eu

PROJECT_DIR="/mnt/fast/nobackup/users/bj134lq/BUTID"
SIF_IMAGE="${PROJECT_DIR}/docker/butid2.sif"
DATA_ROOT="/mnt/fast"

POSE_DIR="/mnt/fast/nobackup/scratch4weeks/bj134lq/data/BUTID_Mediapipe"
OUT_DIR="/mnt/fast/nobackup/scratch4weeks/bj134lq/data/BUTID_Mediapipe_pt"

# each worker holds one whole video in RAM (~830 bytes/frame, so ~0.5 GiB for a
# 5 GiB .h5) — keep --mem above WORKERS * 0.5 GiB
WORKERS="${SLURM_CPUS_PER_TASK:-4}"

# spot-check each converted shard against the .h5 it came from; set to 0 to skip
VERIFY_SAMPLES=5

# shard = this task's slice. Falls back to a single shard outside an array job.
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]; then
  NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT}"
elif [ -n "${SLURM_ARRAY_TASK_MAX:-}" ]; then
  NUM_SHARDS=$((SLURM_ARRAY_TASK_MAX - ${SLURM_ARRAY_TASK_MIN:-0} + 1))
else
  NUM_SHARDS=1
fi

# a sparse --array (0,2,5) would leave task ids above the shard count; the
# shards are only disjoint over a contiguous range
if [ "${SHARD_ID}" -ge "${NUM_SHARDS}" ]; then
  echo "error: task id ${SHARD_ID} exceeds shard count ${NUM_SHARDS};" \
       "use a contiguous --array range such as 0-$((NUM_SHARDS - 1))" >&2
  exit 1
fi

echo "shard ${SHARD_ID}/${NUM_SHARDS} on $(hostname) with ${WORKERS} workers"
echo "  ${POSE_DIR} -> ${OUT_DIR}"

apptainer exec \
  --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
  --bind "${DATA_ROOT}:${DATA_ROOT}" \
  --pwd "${PROJECT_DIR}" \
  "${SIF_IMAGE}" \
  python3 converter.py convert \
  --pose-dir "${POSE_DIR}" \
  --out-dir "${OUT_DIR}" \
  --workers "${WORKERS}" \
  --dtype float32 \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}"

if [ "${VERIFY_SAMPLES}" -gt 0 ]; then
  echo "verifying shard ${SHARD_ID} against the source .h5 files"

  # --start-pad / --end-pad must match what training passes, since they decide
  # which frames a clip covers
  apptainer exec \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${DATA_ROOT}:${DATA_ROOT}" \
    --pwd "${PROJECT_DIR}" \
    "${SIF_IMAGE}" \
    python3 converter.py verify \
    --pose-dir "${POSE_DIR}" \
    --out-dir "${OUT_DIR}" \
    --samples "${VERIFY_SAMPLES}" \
    --start-pad 16 \
    --end-pad 32 \
    --shard-id "${SHARD_ID}" \
    --num-shards "${NUM_SHARDS}"
fi

echo "shard ${SHARD_ID} done"
