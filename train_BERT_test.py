import os


# TEST
os.system("bash scripts/5_train_bert.bash \
0,1,2,3 4 9602 \
 /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/LP/TEST/ \
 --num_train_epochs 20.0 \
 --gradient_accumulation_steps 8 --per_gpu_train_batch_size=2 \
 --local_rank 0 \
 --per_gpu_train_batch_size 8\
 --train_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_0.txt \
 --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_1.txt"
)




# bash /opt/local/bin/run_job.sh --script train_BERT.py --partition "cpu-shannon" --cpus-per-task 4 --env paradox-learning
#  --cache_dir '/home/2022/viktorkj/.cache/huggingface/hub' \
