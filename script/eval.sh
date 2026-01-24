ckpt_path=out/stage3_finetuning/best_checkpoint.pth

# single gpu inference
# RGB-pose setting
deepspeed --include localhost:0 --master_port 29511 train.py \
   --batch-size 8 \
   --gradient-accumulation-steps 1 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --output_dir out/test \
   --finetune $ckpt_path \
   --dataset BUTID \
   --tasks SLT \
   --eval

deepspeed --include localhost:0 --master_port 29511 train.py \
   --batch-size 8 \
   --gradient-accumulation-steps 1 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --output_dir out/test \
   --finetune $ckpt_path \
   --dataset BSign22k \
   --tasks ISLR \
   --eval

deepspeed --include localhost:0 --master_port 29511 train.py \
   --batch-size 8 \
   --gradient-accumulation-steps 1 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --output_dir out/test \
   --finetune $ckpt_path \
   --dataset AUTSL \
   --tasks ISLR \
   --eval