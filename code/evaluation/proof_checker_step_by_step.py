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


    def check_proof_for_errors(self, predicted_proofs, input_data):
        for pred_proof, in_data in zip(predicted_proofs, input_data):
            # From X-input remove everything after last '1'
            in_data["input"] = '1'.join(in_data["input"].split('1')[:-1]) + '1'
     
            label_is_correct = int(eval(pred_proof[-1])) == in_data["label"]
            proof_is_correct, inx, updated_input = self.correctness_of_proof(pred_proof, in_data["input"])
            
        
        print("Num of halluzinated rules:", self.hall_rule)
        print("Num of unfufilled rules:", self.hall_rule)
        print("Num of proofs ended too early:", self.ended_too_early)

                
            # If proof or and label incorrect analyze error


    def correctness_of_proof(self, pred_proof, input_string):
        # Check the correctness of the generated proofs
        inp = input_string.copy()
        # Loop through the predicted proof to see if the rules exist
        for i, pred_step in enumerate(pred_proof):
            if pred_step == "True" or pred_step == "False":
                return True, i

            # Find fact
            fact = pred_step.split(", ")[-1][:-1] # take after last ',' but do not include ':' at end of rule

            # Remove rule from input and add fact
            if pred_step in inp: 
                inp = inp.replace(pred_step, '')
                inp = inp + ' ' + fact + '1'
            else: # KONTROLLERA SÅ ATT DEN HAR TAGIT BORT NÅNTING??
                return False, i


            






            # find pred_step in rules_facts
            # if pred_step in inp:
            #     next_pred_step = inp.replace(pred_step, '')
            # else:
            #     return False, i

            # if i > 0 and i < len(pred_proof) - 1 and next_pred_step not in inp:
            #     return False, i



            # find if rule can be solved with known facts
            # if not correct send to function for evaluation of why 


            # if found rule and fact remove rule add fact 

            # Check so that the query is solved

                
            # 1. label rätt, proof rätt -> kommer gå igenom
            # 2. label rätt, proof fel -> cansat och tatt fel (slutat för tidigt?)
            # 3. label fel, proof "rätt" -> Har hittat på regler?
            # 4. label fel, proof fel -> 

            # Solve the proof by generating the next rule and remove it from the input and add the fact
            # Do until the proof is solved or until it does wrong.
    
        
    def find_fact(self, rule, input_d):
        # Check if rule can be solved with facts
        statements = rule.split()[:-1]
        found_fact = True

        for st in statements:
            st = st +"1"
            if not st in input_d:
                found_fact = False
        return found_fact



    def evaluate_error(self, pred_proof, inx, in_data):
        pass


        
        

        
def main():
    PC = Proof_Checker_Step(save_stats_file)
    input_data = PC.read_file_lines(input_data_path)
    preds_data = PC.read_file_lines(test_preds_path)
    truth_data = PC.read_file_lines(test_truth_path)
    pred_bools = PC.create_list_of_bool_labels(preds_data)
    truth_bools = PC.create_list_of_bool_labels(truth_data)

    cm_indices = PC.get_index_matrix(PC.create_confusion_matrix(pred_bools, truth_bools))

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