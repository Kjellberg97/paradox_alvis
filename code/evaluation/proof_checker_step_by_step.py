from proof_checker import Proof_Checker
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import json
import numpy as np

#test_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/gen_step_by_step/evaluation/checkpoint-8500_output_SMALL_DATA.txt"
test_preds_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/small_cleaned_test_step_labels.txt"
test_truth_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/small_cleaned_test_step_labels.txt"

class Proof_Checker_Step(Proof_Checker):
    def __init__(self):
        pass
 
    def read_file_lines(self, file_path):
        """
        ARG: path (str) in the form of /mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/prop_examples_train.txt
        RETURN: list of dicts or strings dependring on the formmating of the file
        """
        with open(file_path) as f:
            return json.load(f)

    def create_list_of_bool_labels(self, list_of_lists):
        return [ int(eval(x[-1])) for x in list_of_lists ]
        
def main():
    PC = Proof_Checker_Step()
    preds_data = PC.read_file_lines(test_preds_path)
    truth_data = PC.read_file_lines(test_truth_path)
    pred_bools = PC.create_list_of_bool_labels(preds_data)
    #print("Preds:", pred_bools)
    truth_bools = PC.create_list_of_bool_labels(truth_data)
    #print("truth:", truth_bools)

    cm_indices = PC.get_index_matrix(PC.create_confusion_matrix(pred_bools, truth_bools))
    #print("\nCM indices:\n", cm_indices)

    cm = confusion_matrix(truth_bools, pred_bools)
    acc = accuracy_score(truth_bools, pred_bools)
    f1 = f1_score(truth_bools, pred_bools)

    print("\nConfusion matrix:\n", cm)
    print("\nAccuracy:", acc)
    print("F1-score:", f1)

if __name__ == "__main__":
    main()