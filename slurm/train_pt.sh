#!/bin/sh

# Same run as train.sh, reading the converted .pt pose files (slurm/convert.sh)
# instead of the raw .h5 ones. Only --pose-dir and --use-pt differ; every
# hyperparameter is left identical so the two are comparable.
#
#     cd slurm && mkdir -p logs
#     sbatch train_pt.sh

#SBATCH --job-name="BUTID SLT pt"
#SBATCH --partition=cogvis-project
#SBATCH --nodelist=aisurrey28
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=36G
#SBATCH --time=03-00:00:00
#SBATCH -o logs/slurm_pt.%a.out
#SBATCH -e logs/slurm_pt.%a.err

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

PROJECT_DIR="/mnt/fast/nobackup/users/bj134lq/BUTID"
SIF_IMAGE="${PROJECT_DIR}/docker/butid_blackwell.sif"
DATA_ROOT="/mnt/fast"

apptainer exec --nv \
  --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
  --bind "${DATA_ROOT}:${DATA_ROOT}" \
  --pwd "${PROJECT_DIR}" \
  "${SIF_IMAGE}" \
  deepspeed --include localhost:0 --master_port 29511 train.py \
  --batch-size 256 \
  --gradient-accumulation-steps 2 \
  --epochs 30 \
  --opt AdamW \
  --lr 3e-4 \
  --quick_break 2048 \
  --output_dir /mnt/fast/nobackup/scratch4weeks/bj134lq/data/train_output_pt \
  --dataset BUTID \
  --tasks SLT \
  --csv-dir /mnt/fast/nobackup/users/bj134lq/BUTID/data/split \
  --pose-dir /mnt/fast/nobackup/scratch4weeks/bj134lq/data/BUTID_Mediapipe_pt \
  --use-pt \
  --start-pad 16 \
  --end-pad 32
