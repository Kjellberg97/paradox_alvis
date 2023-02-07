import os


# For RP_10X
os.system("bash scripts/5_train_bert.bash \
0,1,2,3 4 6006 \
 /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/PR_10X/BERT/ \
 --num_train_epochs 20.0 \
 --gradient_accumulation_steps 8 --per_gpu_train_batch_size=2S \
 --local_rank -1 \
 --train_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all.txt.balanced_rulenum_train \
 --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all.txt.balanced_rulenum_val" 
)