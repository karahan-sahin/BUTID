#!/bin/sh

#SBATCH --job-name="BUTID SLT"
#SBATCH --partition=cogvis-project
#SBATCH --nodelist=aisurrey28
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=20G
#SBATCH --time=03-00:00:00
#SBATCH -o logs/slurm.%a.out
#SBATCH -e logs/slurm.%a.err

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

PROJECT_DIR="/mnt/fast/nobackup/users/bj134lq/BUTID"
SIF_IMAGE="${PROJECT_DIR}/docker/butid2.sif"
DATA_ROOT="/mnt/fast"

apptainer exec --nv \
  --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
  --bind "${DATA_ROOT}:${DATA_ROOT}" \
  --pwd "${PROJECT_DIR}" \
  "${SIF_IMAGE}" \
  deepspeed --include localhost:0,1 --master_port 29511 train.py \
  --batch-size 192 \
  --gradient-accumulation-steps 4 \
  --epochs 30 \
  --opt AdamW \
  --lr 3e-4 \
  --quick_break 2048 \
  --output_dir /mnt/fast/nobackup/scratch4weeks/bj134lq/data/train_output \
  --dataset BUTID \
  --tasks SLT \
  --csv-dir /mnt/fast/nobackup/users/bj134lq/BUTID/data/split \
  --pose-dir /mnt/fast/nobackup/scratch4weeks/bj134lq/data/BUTID_Mediapipe \
  --start-pad 16 \
  --end-pad 32
