import os


# For RP_10X
os.system("bash scripts/5_train_bert.bash \
0,1,2,3 4 6006 \
 /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/PR_10X/BERT1/ \
 --num_train_epochs 20.0 \
 --gradient_accumulation_steps 8 --per_gpu_train_batch_size=2 \
 --train_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_train.txt \
 --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_val.txt" 
)