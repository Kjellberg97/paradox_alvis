# Runs the proof checker
from proof_checker import Proof_Checker
import json
import numpy as np
import pickle

def read_file_lines(file_path):
    """
    ARG: path (str) in the form of /mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/prop_examples_train.txt
    RETURN: list of dicts or strings dependring on the formmating of the file
    """
    with open(file_path) as f:
        return json.load(f)

def reformat_files(checkpoint, model, test_on, type_of_data):

    path = "/mimer/NOBACKUP/groups/snic2022-22-744/"

    type_of_model = "/pretrained_BART/evaluation/" 

    checkpoint = checkpoint + "_" + model + "_" + test_on 

    if type_of_data == "val":
        t_checkpoint = checkpoint + "_VAL_output.txt"
    elif type_of_data == "test":
        t_checkpoint = checkpoint +  "_TEST_output.txt"

    test_preds_path = path + "MODELS/" + model + type_of_model + t_checkpoint

    if test_on == "RP_10X":
        labels_path = test_on + "/prop_examples_all_balanced_rulenum_cleaned_" + type_of_data + "_labels.txt"
        input_path = test_on + "/prop_examples_all_balanced_rulenum_cleaned_" + type_of_data + ".txt"
    else:
        labels_path = test_on + "/prop_examples_all_cleaned_" + type_of_data + "_labels.txt"
        input_path = test_on + "/prop_examples_all_cleaned_" + type_of_data + ".txt"

    test_truth_path = path + "DATA/" + labels_path
    input_data_path = path + "DATA/" + input_path

    save_stats_file = path + "MODELS/" + model + type_of_model + "proof_checker_stats/proof_checker_" + checkpoint  + "_" + type_of_data + ".txt"


    return test_preds_path, test_truth_path, input_data_path, save_stats_file   

def main():
    acc_by_rules = True
    #checkpoint = "checkpoint-????"

    models = ["LP", "RP", "RP_10X"]
    test_ons = ["LP", "RP", "RP_10X"]
    type_of_data = "test"

    for model in models:
        if model == "LP":
            checkpoint = "checkpoint-22392"
        else:
            checkpoint = "checkpoint-8750"
        for test_on in test_ons:
            print("\n\n\n##########################################################################")
            print(f"TRAIN DIST: {model}", f"{type_of_data.upper()} DIST: {test_on}", sep="\n")

            test_preds_path, test_truth_path, input_data_path, save_stats_file = reformat_files(checkpoint, model, test_on, type_of_data)


            PC = Proof_Checker(save_stats_file)
            input_data = read_file_lines(input_data_path)
            preds_data = read_file_lines(test_preds_path)
            truth_data = read_file_lines(test_truth_path)
            #PC.divide_data_into_depths(input_data, preds_data, truth_data)

            if acc_by_rules:
                acc_list = PC.divide_data_into_rules(input_data, preds_data, truth_data)
                print("Saving accuracies with pickle.")
                pkl_name = "accs_by_rules/ex2_rule_accs_" + model + "_" + test_on + "_" + type_of_data + ".pkl"
                with open(pkl_name, "wb") as f:
                    pickle.dump(acc_list, f)

            #PC.check_proof_for_errors(preds_data, input_data)

if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     pred_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/pretrained_BART/evaluation/checkpoint-8750_LP_RP_10X_TEST_output.txt"
#     target_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_cleaned_test_labels.txt"
#     input_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_cleaned_test.txt"
#     save_stats_file = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/pretrained_BART/evaluation/proof_checker/checkpoint-8750_LP_RP_10X_TEST_pc.txt"
#     main(pred_path, target_path, input_path, save_stats_file)

