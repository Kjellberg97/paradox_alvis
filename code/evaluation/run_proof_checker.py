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


def main(pred_path, target_path):
    

    predicted_strings = read_file_lines(pred_path)
    target_dicts = read_file_lines(target_path)
    pf = Proof_Checker()

    ground_truth_bools = [ target_d['label'] for target_d in target_dicts ]

    confusion_matrix = pf.create_confusion_matrix(predicted_strings, ground_truth_bools)
    accuracy = pf.label_accuracy(confusion_matrix)
    print("Rates: TP, FP, TN, FN\n", np.sum(confusion_matrix, axis=0) / confusion_matrix.shape[0])
    print(accuracy)



if __name__ == "__main__":
    pred_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/pretrained_BART/evaluation/checkpoint-24880_output.txt"
    target_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/prop_examples_test_labels.txt"
    main(pred_path, target_path)