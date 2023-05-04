import os

#LP

os.system("bash /cephyr/users/viktorkj/Alvis/paradox_alvis/code/classification_models/scripts/6_eval_bert.bash 0 \
    --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_brackets_test.txt \
    --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/LP/BERT1/checkpoint-19/pytorch_model.bin")

    #/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_brackets_test.txt

# os.system("bash scripts/6_eval_bert.bash 0 \
#     --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all_test.txt \
#     --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/LP/BERT2/checkpoint-19/pytorch_model.bin")

# os.system("bash code/classification_models/scripts/6_eval_bert.bash 0 \
#     --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_test.txt \
#     --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/LP/BERT1/checkpoint-19/pytorch_model.bin")


#RP

# os.system("bash scripts/6_eval_bert.bash 0 \
#     --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all_test.txt \
#     --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/RP/BERT1/checkpoint-19/pytorch_model.bin")

# os.system("bash scripts/6_eval_bert.bash 0 \
#     --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all_test.txt \
#     --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/RP/BERT1/checkpoint-19/pytorch_model.bin")

# os.system("bash scripts/6_eval_bert.bash 0 \
#     --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_test.txt \
#     --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/RP/BERT1/checkpoint-19/pytorch_model.bin")


#RP_10x

# os.system("bash scripts/6_eval_bert.bash 0 \
#     --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all_test.txt \
#     --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/PR_10X/BERT1/checkpoint-19/pytorch_model.bin")

# os.system("bash scripts/6_eval_bert.bash 0 \
#     --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all_test.txt \
#     --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/PR_10X/BERT1/checkpoint-19/pytorch_model.bin")

# os.system("bash scripts/6_eval_bert.bash 0 \
#     --val_file_path /mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_test.txt \
#     --custom_weight /mimer/NOBACKUP/groups/snic2022-22-744/OUTPUT/PR_10X/BERT1/checkpoint-19/pytorch_model.bin")
