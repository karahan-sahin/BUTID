output_dir=out/stage2_finetuning

# RGB-pose setting
ckpt_path=out/stage1_pretraining/best_checkpoint.pth

deepspeed --include localhost:0 --master_port 29511 train.py \
   --batch-size 8 \
   --gradient-accumulation-steps 1 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --output_dir $output_dir \
   --finetune $ckpt_path \
   --dataset BSign22k \
   --task ISLR 

# deepspeed --include localhost:0 --master_port 29511 train.py \
#    --batch-size 8 \
#    --gradient-accumulation-steps 1 \
#    --epochs 20 \
#    --opt AdamW \
#    --lr 3e-4 \
#    --output_dir $output_dir \
#    --finetune $ckpt_path \
#    --dataset AUTSL \
#    --task ISLR 
