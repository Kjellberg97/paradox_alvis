from proof_checker import Proof_Checker
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import json
import numpy as np
import re





#test_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/gen_step_by_step/evaluation/checkpoint-8500_output_SMALL_DATA.txt"
# test_preds_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/RP_10X/gen_step_by_step_rule_sampling/evaluation/checkpoint-7500_RP_RP_10X_VAL_output.txt"
# test_truth_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_cleaned_val_step_labels.txt"
# input_data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_cleaned_val.txt"
# save_stats_file = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/RP_10X/gen_step_by_step_rule_sampling/evaluation/proof_checker_stats/proof_checker_checkpoint-7500_RP_RP_10X_VAL.txt"

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
        matrix_error = [[0,0], # X proof correct True False
                        [0,0]] # Y label correct True False
        pred_true_but_q_not_in_fact = 0
        
        for pred_proof, in_data in zip(predicted_proofs, input_data):
            self.input = in_data["input"]
            # Remove all 1s that stands alone
            
            # Remove all words that end with 0 and remove leading and trailing blanks.
            in_data["input"] = ' '.join(re.sub(r'\b\S*0\b', '', in_data["input"]).split())         

            try:
                pred_label = eval(pred_proof[-1])
            except SyntaxError:
                print("True/false not existing on", pred_proof[-1])
                pred_label = True if not in_data["label"] else False
            label_is_correct = int(pred_label) == in_data["label"]
            proof_is_correct, inx, updated_input = self.coherence_of_proof(pred_proof, in_data["input"])
                      
            pred_true_but_q_not_in_fact += 1 if pred_label and not self.query_in_facts(updated_input) else 0
                

            if label_is_correct and proof_is_correct:
                matrix_error[0][0] += 1
            elif label_is_correct and not proof_is_correct:
                matrix_error[0][1] += 1
            elif not label_is_correct and proof_is_correct:
                matrix_error[1][0] += 1
            elif not label_is_correct and not proof_is_correct:
                matrix_error[1][1] += 1
        
        n_samples = len(input_data)
        print("Errors in proof/label")
        print("Correct label, coherent proof:", round(matrix_error[0][0] / n_samples, 6) * 100, "%")
        print("Correct label, incoherent proof:", round(matrix_error[0][1] / n_samples, 6) * 100, "%")
        print("Incorrect label, correct proof:", round(matrix_error[1][0] / n_samples, 6) * 100, "%")
        print("Incorrect label, incoherent proof:", round(matrix_error[1][1] / n_samples, 6) * 100, "%")

        print("Share of True predictions where query is not in list of facts: ",
            round(pred_true_but_q_not_in_fact / n_samples, 6) * 100, "%")
        #print("Num of halluzinated rules:", self.hall_rule)
        #print("Num of unfufilled rules:", self.hall_rule)
        #print("Num of proofs ended too early:", self.ended_too_early)


    def coherence_of_proof(self, pred_proof, inp):
        """Check the coherence of the generated proofs"""
        # Loop through the predicted proof to see if the rules exist
        for i, pred_step in enumerate(pred_proof):
            if pred_step == "True" or pred_step == "False":
                return True, i, inp

            # Find fact
            fact = pred_step.split(", ")[-1][:-1] # take after last ',' but do not include ':' at end of rule
            # Remove rule from input and add fact
            self.last_fact = 'None'
            if pred_step in inp: 
                inp = inp.replace(pred_step, '')
                inp = inp + ' ' + fact + '1'
                self.last_fact = fact
            else: # KONTROLLERA SÅ ATT DEN HAR TAGIT BORT NÅNTING??
                print("Pred step not in input:", pred_step)
                return False, i, inp


            
    def query_in_facts(self, string):
        # Find query in string
        match = re.search(r"\b\S*\?", string)
        query = match.group(0)[:-1] + '1'
        
        # Take out a string of all facts into a list
        facts_list = re.findall(r'\b\S*1\b', string) # finds all words that end with '1'

        # Return bool depending on existence
        if query in facts_list:
            return True
        else:
            print(query)
            print(string)
            print(facts_list)
            #print("Last fact", self.last_fact)
            #print("Original input", self.input)
            return False
        
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



def reformat_files(checkpoint, model, test_on, type_of_data, rule_sampling=False):

    path = "/mimer/NOBACKUP/groups/snic2022-22-744/"

    if rule_sampling:
        type_of_model = "/gen_step_by_step_rule_sampling/evaluation/"
    else:
        type_of_model = "/gen_step_by_step/evaluation/" 

    checkpoint = checkpoint+ "_" + model + "_" + test_on 

    if type_of_data == "val":
        t_checkpoint = checkpoint + "_VAL_output.txt"
    elif type_of_data == "test":
        t_checkpoint = checkpoint +  "_TEST_output.txt"

    test_preds_path = path + "MODELS/" + model + type_of_model + t_checkpoint

    if test_on == "RP_10X":
        labels_path = test_on + "/prop_examples_all_balanced_rulenum_cleaned_" + type_of_data + "_step_labels.txt"
        input_path = test_on + "/prop_examples_all_balanced_rulenum_cleaned_" + type_of_data + ".txt"
    else:
        labels_path = test_on + "/prop_examples_all_cleaned_" + type_of_data + "_step_labels.txt"
        input_path = test_on + "/prop_examples_all_cleaned_" + type_of_data + ".txt"

    test_truth_path = path + "DATA/" + labels_path
    input_data_path = path + "DATA/" + input_path

    save_stats_file = path + "MODELS/" + model + type_of_model + "proof_checker_stats/proof_checker_" + checkpoint  + "_" + type_of_data + ".txt"


    return test_preds_path, test_truth_path, input_data_path, save_stats_file        
        
        
def main():


    rule_sampling = True
    checkpoint = "checkpoint-7500"
    model = "RP_10X"
    test_on = "RP_10X"
    type_of_data = "val"

    test_preds_path, test_truth_path, input_data_path, save_stats_file = reformat_files(checkpoint, model, test_on, type_of_data, rule_sampling)


    PC = Proof_Checker_Step(save_stats_file)
    input_data = PC.read_file_lines(input_data_path)
    preds_data = PC.read_file_lines(test_preds_path)
    truth_data = PC.read_file_lines(test_truth_path)
    pred_bools = PC.create_list_of_bool_labels(preds_data)
    truth_bools = PC.create_list_of_bool_labels(truth_data)

    print("\nPROOF CHECKED DATA: ",  test_preds_path)

    cm_indices = PC.get_index_matrix(PC.create_confusion_matrix(pred_bools, truth_bools))

    cm = confusion_matrix(truth_bools, pred_bools)
    acc = accuracy_score(truth_bools, pred_bools)
    f1 = f1_score(truth_bools, pred_bools, average='binary')
    #f1 = f1_score(truth_bools, pred_bools, average='micro')

    print("\nConfusion matrix:\n", cm)
    print("\nAccuracy:", acc)
    print("F1-score:", f1)
        
    PC.divide_data_into_depths(input_data, preds_data, truth_data)
    #PC.find_non_bools(preds_data)
    
    #PC.check_proof_for_errors(preds_data, input_data)

if __name__ == "__main__":
    main()