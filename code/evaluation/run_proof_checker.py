# Runs the proof checker
from proof_checker import Proof_Checker
import json
import numpy as np

def read_file_lines(file_path):
    """
    ARG: path (str) in the form of /mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/prop_examples_train.txt
    RETURN: list of dicts or strings dependring on the formmating of the file
    """
    with open(file_path) as f:
        return json.load(f)


def main(pred_path, target_path, input_path, save_stats_file):
    
    print("predicted data: ",pred_path )

    predicted_strings = read_file_lines(pred_path)
    target_dicts = read_file_lines(target_path)
    input_dicts = read_file_lines(input_path)
    pf = Proof_Checker(save_stats_file)


    pf.divide_data_into_depths(input_dicts,predicted_strings, target_dicts)
    
    
    # ground_truth_bools = [ target_d['label'] for target_d in target_dicts ]

    # confusion_matrix = pf.create_confusion_matrix(predicted_strings, ground_truth_bools)
    # accuracy = pf.label_accuracy(confusion_matrix)
    # pf.stat_over_generated_data(predicted_strings,target_dicts,input_dicts)
    # print("Rates: TP, FP, TN, FN\n", np.sum(confusion_matrix, axis=0) / confusion_matrix.shape[0])
    # print(accuracy)


if __name__ == "__main__":
    pred_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/pretrained_BART/evaluation/checkpoint-22392_output_RP_test.txt"
    target_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all_cleaned_test_labels.txt"
    input_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all_cleaned_test.txt"
    save_stats_file = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/pretrained_BART/evaluation/checkpoint-22392_TLP_RRP.txt"
    main(pred_path, target_path, input_path, save_stats_file)