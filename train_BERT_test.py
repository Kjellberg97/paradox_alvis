import os


# TEST
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.system("bash scripts/5_train_bert.bash \
0,1,2,3 4 9602 \
 /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/TEST/ \
 --num_train_epochs 2.0 \
 --gradient_accumulation_steps 8 \
 --per_gpu_train_batch_size=2 \
 --train_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_0.txt \
 --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_1.txt"
)




# bash /opt/local/bin/run_job.sh --script train_BERT.py --partition "cpu-shannon" --cpus-per-task 4 --env paradox-learning
#  --cache_dir '/home/2022/viktorkj/.cache/huggingface/hub' \
