import os
# For LP
os.system("bash scripts/5_train_bert.bash \
0,1,2,3 4 9602 \
 /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/LP/BERT/ \
 --num_train_epochs 20.0 \
 --gradient_accumulation_steps 8 --per_gpu_train_batch_size=2 \
 --per_gpu_train_batch_size 8\
 --train_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all.txt_train \
 --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all.txt_val"
)