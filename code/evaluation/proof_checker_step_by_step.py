from proof_checker import Proof_Checker
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import json
import numpy as np

#test_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/gen_step_by_step/evaluation/checkpoint-8500_output_SMALL_DATA.txt"
test_preds_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/gen_step_by_step/evaluation/checkpoint-8500_output_LP_RPall.txt"
test_truth_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all_cleaned_test_step_labels.txt"
input_data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all_cleaned_test.txt"
save_stats_file = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/gen_step_by_step/evaluation/proof_cheker_stats/proof_checker_checkpoint-8500_LP_RP.txt"

class Proof_Checker_Step(Proof_Checker):
 
    def read_file_lines(self, file_path):
        """
        ARG: path (str) in the form of /mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/prop_examples_train.txt
        RETURN: list of dicts or strings dependring on the formmating of the file
        """
        with open(file_path) as f:
            return json.load(f)


    def create_list_of_bool_labels(self, list_of_lists):
        return [ int(eval(x[-1])) if x[-1] in ["True", "False"] else 0 for x in list_of_lists ]

    
    def find_non_bools(self, list_of_lists):
        non_bools = [ x for x in list_of_lists if x[-1] not in ["True", "False"] ]
        print("NUMBER OF OUTPUTS WITHOUT BOOLVALUES: ", len(non_bools))
        [ print(out) for out in non_bools ]
        return non_bools



    def divide_data_into_depths(self,input_data, predictions, ground_truth):
        """Divides the input data dependeing on the depths of each input data and 
        creates the confucion matrix and calculate basic stats about the lenght 
        of the rules in each group.

        ARGS:
            input_data (list) : all input data
            predictions (list) : the generated proofs and labels 
            ground_truth (list) : the true labels for each input
        
        RETURN:
            None
        """
        data_depths = [[],[],[],[],[],[],[]] 
        preds_depths = [[],[],[],[],[],[],[]] 
        ground_truth_depths = [[],[],[],[],[],[],[]] 
        pred_proof = [[],[],[],[],[],[],[]] 

        for i,data in enumerate(input_data):

            depths = int(data["depth"])

            pred = self.find_binary_label(predictions[i])
    

            data_depths[depths].append(data)
            preds_depths[depths].append(pred)
            ground_truth_depths[depths].append(ground_truth[i])
            pred_proof[depths].append(predictions[i])

        for depth in range(7):
            print()
            print("DEPTH: ",depth)
            print("NR. SAMPLES: ", len(ground_truth_depths[depth]))
            ground_truth_bools = [ self.find_binary_label(target_d) for target_d in ground_truth_depths[depth] ]
            confusion_matrix = self.create_confusion_matrix(preds_depths[depth], ground_truth_bools)
            accuracy = self.label_accuracy(confusion_matrix)
            with open(self.save_stats_file, "a") as file:
                file.write("\n#############################################################################")
                file.write("\nDEPTH: " + str(depth))
            self.stat_over_generated_data(preds_depths[depth] ,ground_truth_depths[depth] ,data_depths[depth],pred_proof[depth])
            print("Rates: TP, FP, TN, FN\n", np.round(np.sum(confusion_matrix, axis=0) / confusion_matrix.shape[0], 3))
            print("acc", accuracy)


    def find_binary_label(self, list_input):
        # Find the last occurence of False or True in the string, convert into corresponding int 0 or 1
        binary_digit = int(eval(list_input[-1])) if list_input[-1] in ["True", "False"] else 0 # Convert into int if a False or True is returned else convert to 0
        return binary_digit
        
def main():
    PC = Proof_Checker_Step(save_stats_file)
    input_data = PC.read_file_lines(input_data_path)
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
    f1 = f1_score(truth_bools, pred_bools, average='binary')
    #f1 = f1_score(truth_bools, pred_bools, average='micro')

    print("\nConfusion matrix:\n", cm)
    print("\nAccuracy:", acc)
    print("F1-score:", f1)
        
    PC.divide_data_into_depths(input_data, preds_data, truth_data)
    PC.find_non_bools(preds_data)

if __name__ == "__main__":
    main()