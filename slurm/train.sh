EXPORT 

deepspeed --include localhost:0,1,2,3 --master_port 29511 train.py \
  --batch-size 32 \
  --gradient-accumulation-steps 8 \
  --epochs 25 \
  --opt AdamW \
  --lr 3e-4 \
  --quick_break 16 \
  --output_dir /home/onursefa/Desktop/temp/train_output \
  --dataset BUTID \
  --tasks SLT
