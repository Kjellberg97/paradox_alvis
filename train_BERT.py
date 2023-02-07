import os



"""
# For RP_10X
os.system("bash scripts/5_train_bert.bash \
0 4 6006 \
 /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/PR_10X/BERT/ \
 --num_train_epochs 20.0 \
 --gradient_accumulation_steps 8 --per_gpu_train_batch_size=1 \
 --local_rank 0 \
 --train_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all.txt.balanced_rulenum_train 
 --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all.txt.balanced_rulenum_val" \
)
"""


"""
# For RP
os.system("bash scripts/5_train_bert.bash \
0 4 8064 \
 /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/RP/BERT/ \
 --num_train_epochs 20.0 \
 --gradient_accumulation_steps 8 --per_gpu_train_batch_size=2 \
 --local_rank 0 \
 --per_gpu_train_batch_size 8\
 --train_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all.txt_train \
 --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all.txt_val"
)
"""



# For LP
os.system("bash scripts/5_train_bert.bash \
0 4 9602 \
 /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/LP/BERT/ \
 --num_train_epochs 20.0 \
 --gradient_accumulation_steps 8 --per_gpu_train_batch_size=2 \
 --local_rank 0 \
 --per_gpu_train_batch_size 8\
 --train_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all.txt_train \
 --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all.txt_val"
)



# bash /opt/local/bin/run_job.sh --script train_BERT.py --partition "cpu-shannon" --cpus-per-task 4 --env paradox-learning
#  --cache_dir '/home/2022/viktorkj/.cache/huggingface/hub' \
