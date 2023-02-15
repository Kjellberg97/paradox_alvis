import os

#LP
"""
os.system("bash scripts/6_eval_bert.bash 0 \
    --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all.txt_test \
    --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/LP/BERT/checkpoint-19/pytorch_model.bin")

os.system("bash scripts/6_eval_bert.bash 0 \
    --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all.txt_test \
    --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/LP/BERT/checkpoint-19/pytorch_model.bin")
"""
os.system("bash scripts/6_eval_bert.bash 0 \
    --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all.txt.balanced_rulenum_test \
    --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/LP/BERT/checkpoint-19/pytorch_model.bin")


#RP
"""
os.system("bash scripts/6_eval_bert.bash 0 \
    --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all.txt_test \
    --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/RP/BERT/checkpoint-19/pytorch_model.bin")
"""
os.system("bash scripts/6_eval_bert.bash 0 \
    --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all.txt_test \
    --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/RP/BERT/checkpoint-19/pytorch_model.bin")

os.system("bash scripts/6_eval_bert.bash 0 \
    --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all.txt.balanced_rulenum_test \
    --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/RP/BERT/checkpoint-19/pytorch_model.bin")


#RP
"""
os.system("bash scripts/6_eval_bert.bash 0 \
    --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all.txt_test \
    --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/PR_10X/BERT/checkpoint-19/pytorch_model.bin")

os.system("bash scripts/6_eval_bert.bash 0 \
    --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all.txt_test \
    --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/PR_10X/BERT/checkpoint-19/pytorch_model.bin")
"""
os.system("bash scripts/6_eval_bert.bash 0 \
    --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all.txt.balanced_rulenum_test \
    --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/PR_10X/BERT/checkpoint-19/pytorch_model.bin")