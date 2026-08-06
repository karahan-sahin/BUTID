#!/bin/sh

#SBATCH --job-name="SLT 3"
#SBATCH --partition=a100
#SBATCH --nodelist=aisurrey24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=30G
#SBATCH --time=03-00:00:00
#SBATCH -o logs/slurm3.%a.out
#SBATCH -e logs/slurm3.%a.err

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

PROJECT_DIR="/mnt/fast/nobackup/users/bj134lq/BUTID"
SIF_IMAGE="/mnt/fast/nobackup/users/bj134lq/BUTID/docker/butid2.sif"
DATA_ROOT="/mnt/fast"

apptainer exec --nv \
  --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
  --bind "${DATA_ROOT}:${DATA_ROOT}" \
  --pwd "${PROJECT_DIR}" \
  "${SIF_IMAGE}" \
  deepspeed --include localhost:0,1 --master_port 29511 train.py \
  --finetune /mnt/fast/nobackup/scratch4weeks/bj134lq/data/train_output/SLT/best_checkpoint.pth \
  --batch-size 256 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 2e-4 \
  --quick_break 2048 \
  --output_dir /mnt/fast/nobackup/scratch4weeks/bj134lq/data/train_output3 \
  --dataset BUTID \
  --tasks SLT \
  --csv-dir /mnt/fast/nobackup/users/bj134lq/BUTID/data/split \
  --pose-dir /mnt/fast/nobackup/scratch4weeks/bj134lq/data/BUTID_Mediapipe \
  --start-pad 16 \
  --end-pad 32
