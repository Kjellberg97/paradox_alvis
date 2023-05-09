from proof_checker import Proof_Checker
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import json
import numpy as np
import re
import pickle
from collections import Counter





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
        return [ int(eval(x[-1])) if x[-1] in ["True", "False"] else -1 for x in list_of_lists ]

    
    def find_non_bools(self, list_of_lists):
        non_bools = [ x for x in list_of_lists if x[-1] not in ["True", "False"] ]
        print("NUMBER OF OUTPUTS WITHOUT BOOLVALUES: ", len(non_bools))
        [ print(out) for out in non_bools ]
        return non_bools

    def divide_data_into_rules(self,input_data, predictions, ground_truth):
        """Divides the input data dependeing on the rules of each input data and 
        creates the confucion matrix and calculate basic stats about the lenght 
        of the rules in each group.

        ARGS:
            input_data (list) : all input data
            predictions (list) : the generated proofs and labels 
            ground_truth (list) : the true labels for each input
        
        RETURN:
            None
        """
        data_rules = [ data["input"].count(":") for data in input_data ]
        max_rules = max(data_rules) + 1
        preds_rules = [ [] for _ in range(max_rules) ]
        ground_truth_rules = [ [] for _ in range(max_rules) ]
        pred_proof = [ [] for _ in range(max_rules) ]

        for i, data in enumerate(input_data):
            n_rules = data["input"].count(":")
            pred = self.find_binary_label(predictions[i])
            preds_rules[n_rules].append(pred)
            ground_truth_rules[n_rules].append(ground_truth[i])
            pred_proof[n_rules].append(predictions)
        
        acc_list = []
        for n_rules in range(max_rules):
            print()
            print("RULES: ", n_rules)
            print("NR. SAMPLES: ", len(ground_truth_rules[n_rules]))
            ground_truth_bools = [ self.find_binary_label(target_d) for target_d in ground_truth_depths[depth] ]
            #confusion_matrix = self.create_confusion_matrix(preds_depths[depth], ground_truth_bools)
            cm = confusion_matrix(ground_truth_bools, preds_depths[depth])
            #accuracy = self.label_accuracy(confusion_matrix)
            accuracy = accuracy_score(ground_truth_bools, preds_depths[depth])
            acc_list.append(accuracy)
            with open(self.save_stats_file, "a") as file:
                file.write("\n#############################################################################")
                file.write("\nRULES: " + str(n_rules))
            #self.stat_over_generated_data(preds_rules[n_rules] ,ground_truth_rules[n_rules] ,data_rules[n_rules],pred_proof[n_rules])
            print("Rates: TP, FP, TN, FN\n", np.round(np.sum(cm, axis=0) / cm.shape[0], 3))
            print("acc", accuracy)
        
        return acc_list
    

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

        acc_list = []
        acc_str_list = []
        for depth in range(7):
            print()
            print("DEPTH: ",depth)
            print("NR. SAMPLES: ", len(ground_truth_depths[depth]))
            ground_truth_bools = [ self.find_binary_label(target_d) for target_d in ground_truth_depths[depth] ]
            #confusion_matrix = self.create_confusion_matrix(preds_depths[depth], ground_truth_bools)
            cm = confusion_matrix(ground_truth_bools, preds_depths[depth])
            #accuracy = self.label_accuracy(confusion_matrix)
            accuracy = accuracy_score(ground_truth_bools, preds_depths[depth])
            with open(self.save_stats_file, "a") as file:
                file.write("\n#############################################################################")
                file.write("\nDEPTH: " + str(depth))
            self.stat_over_generated_data(preds_depths[depth], ground_truth_depths[depth], data_depths[depth], pred_proof[depth])
            print("Rates: TP, FP, TN, FN\n", np.round(np.sum(cm, axis=0) / cm.shape[0], 3))
            acc_round = str(round(accuracy * 100, 1))
            print("Accuracy:", acc_round)
            acc_str_list.append(str(acc_round))
            acc_list.append(accuracy)
            frac_consistent_proofs, _ = self.all_consistency(pred_proof[depth], data_depths[depth], preds_depths[depth], ground_truth_bools, ground_truth_depths[depth])
            print("Fractions of consistent chains of inference steps:", frac_consistent_proofs)
        mean_acc = round(np.mean(acc_list) * 100, 1)
        acc_str_list.append(str(mean_acc))
        print("Mean accuracy across depths:", mean_acc)
        

        return acc_str_list



    def find_binary_label(self, list_input):
        # Find the last occurence of False or True in the string, convert into corresponding int 0 or 1
        binary_digit = int(eval(list_input[-1])) if list_input[-1] in ["True", "False"] else -1 # Convert into int if a False or True is returned else convert to 0
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


    def check_consistency(self, pred_proof, input_data_whole, pred_bool, truth_bool, ground_proof):
        
        """Check that there are no missing rules in the predicted proof,
        and that all tules in proof exist in the original input."""
        
        # 0. Check true pred label, if true != pred, then mark as inconsistent
        # 1. Take the original input

        # if True = True:
        # Solve the problem with forward chaining using only the rules provided in the proof
        # 2. Check that all rules in pred-proof exist in original input /done
        # 3. Remove all rules from original input
        # 4. Add the generated rules from the pred-proof to the original input
        # 5. Solve with forward chaining, if solvable then mark as consistent

        # If FALSE = False:
        # 6. Solve original proof with forward chaining
        # 7. Compare so exactly the same rules are used in both pred and ground truth proof, if so mark as consistent

        input_raw = input_data_whole.copy()

        input_data = input_raw["input"]

        if pred_bool:
            for step in pred_proof[:-1]:
                # check if pred step is hallucinated by comp against input data
                if step not in input_data:
                    return False, "True"
            
            # See if problem is solvable with forward chaining using only pred proof
            solvable = self.solve_problem_with_gen_rules(pred_proof, input_data_whole)
            if solvable:
                return solvable , " "
            else:
                return solvable, "True"

        else: 

            # Remove predicted rules from input
            # Try to find applicable rules that can lead to new facts
            # If new facts can be found, then return False

            input_copy = (input_data_whole["input"]+ '.')[:-1]

            no_found_hall, hall_step, updated_input = self.check_hallucination(pred_proof, input_copy)

            if no_found_hall:
                if self.no_more_new_fact(updated_input):
                    return True, " "
                else:
                    return False, "False"
                
            else:
                return False, "False" 
    


    def no_more_new_fact(self, inp):
        
        # Try to find a new rule that is solvable and that leads to a new fact.
        # Use same function as from generate proof label
        facts = inp.split(":")[-1]
        facts = facts.split(" ")
        
        rules = inp.split("? ")[-1]
        rules = rules.split(":")[:-1]

        for i in range(len(rules)):
            if rules[i][-1] == ":":
                rules[i] = rules[i][:-1]
            
            rule = rules[i].split(", ")

            conditions = rule[:-1]
            concusion = rule[-1]

            if not concusion+"1" in facts:

                if all(condition+"1" in facts for condition in conditions):
                    return False
            
        return True



    def check_hallucination(self, pred_proof, inp):
        """Check if the generator hallucinates rules"""
        # Loop throsugh the predicted proof to see if the rules exist
        inp = (inp+'.')[:-1]

        for i, pred_step in enumerate(pred_proof):
            if pred_step == "True" or pred_step == "False":
                #print("All steps in ground_proof")
                return True, pred_step, inp

            # Find fact
            fact = pred_step.split(", ")[-1][:-1] # take after last ',' but do not include ':' at end of rule
            # Remove rule from input and add fact
            self.last_fact = 'None'
            if pred_step in inp and not pred_step == '':
                
                for req in pred_step.split()[:-1]:
                    req= req[:-1]+"1"
                    if not req[:-1]+"1" in inp:
                        return False, pred_step, inp

                # Check if the rule can be applied with the facts 
                inp = inp.replace(pred_step, '')
                inp = inp + ' ' + fact + '1'
                self.last_fact = fact
            else: # KONTROLLERA SÅ ATT DEN HAR TAGIT BORT NÅNTING??
                #print("Pred step not in input:", pred_step)
                return False, pred_step, inp



    def check_hallucination_batch(self, pred_proofs, inputs):
        """Finds hallucinations and returns the index, the hallucinated step and the updated input
        for that step. The step is the first hallucinated step in the pred"""

        all_hall_not_found = []
        num_hall = 0
        index=0
        for pred, inp in zip(pred_proofs, inputs):
            
            hall_not_found, step, updated_input = self.check_hallucination(pred, inp["input"])
            
            if not hall_not_found:
                all_hall_not_found.append({"Index":index, "Step": step, "Updated input": updated_input})
                num_hall+=1
            index +=1
        
        print("Fraction of examples with hallucinations: ", round(num_hall/len(pred_proofs)*100, 4), "%" )
        return all_hall_not_found
    


    def solve_problem_with_gen_rules(self, pred_proof, input_d):
        
        query = input_d["query"]
        facts = input_d["input"].split(":")[-1]

        facts = facts.split(" ")
        facts_reformat =[]
        for f in facts:
            f = f.replace("1","")
            f = f.replace(" ","")
            if f != "":
                facts_reformat.append(f)

        solvable_proof = self.forward_chain(pred_proof, facts_reformat, query)

        return solvable_proof


    def forward_chain(self, pred_proof, facts, query):
        
        for step in pred_proof:
            if query in facts:
                return True

            step_div = step.split(", ")[:-1]
            conclusion = step.split(", ")[-1][:-1]

            for req in step_div:
                if not req in facts:
                    return False
            facts.append(conclusion)

        if query in facts:
            return True
        else:
            return False

    
    def all_consistency(self, pred_data, input_data, pred_labels, truth_labels, ground_proofs):
        consistent = []
        on_true_or_false = []

        for i, pred_d in enumerate(pred_data):
            con, type_of_error = self.check_consistency(pred_d, input_data[i], pred_labels[i], truth_labels[i], ground_proofs[i])
            consistent.append(con)
            on_true_or_false.append(type_of_error)
        return consistent.count(True)/len(pred_data), on_true_or_false
            
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



def reformat_files(checkpoint, model, test_on, type_of_data, rule_sampling=True):

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
    acc_by_rules = False
    rule_sampling = True
    #checkpoint = "checkpoint-7500"

    models = ["LP", "RP", "RP_10X"]
    test_ons = ["LP", "RP", "RP_10X"]
    type_of_data = "test"

    proof_coherent = []

    latex_output = []
    for model in models:
        if model == "LP":
            checkpoint = "checkpoint-9328"
            if not rule_sampling:
                checkpoint = "checkpoint-8500"
        else:
            checkpoint = "checkpoint-7500"
        for test_on in test_ons:
            print("\n\n\n##########################################################################")
            print(f"TRAIN DIST: {model}", f"{type_of_data.upper()} DIST: {test_on}", sep="\n")

            test_preds_path, test_truth_path, input_data_path, save_stats_file = reformat_files(checkpoint, model, test_on, type_of_data, rule_sampling)


            PC = Proof_Checker_Step(save_stats_file)
            input_data = PC.read_file_lines(input_data_path)
            preds_data = PC.read_file_lines(test_preds_path)
            truth_data = PC.read_file_lines(test_truth_path)
            pred_bools = PC.create_list_of_bool_labels(preds_data)
            truth_bools = PC.create_list_of_bool_labels(truth_data)

            print("\nINPUT DATA: ",  input_data_path)
            print("\nPRED DATA: ",  test_preds_path)
            print("\nGROUND TRUTH: ",  test_truth_path)

            cm_indices = PC.get_index_matrix(PC.create_confusion_matrix(pred_bools, truth_bools))

            cm = confusion_matrix(truth_bools, pred_bools)
            acc = accuracy_score(truth_bools, pred_bools)
            #f1 = f1_score(truth_bools, pred_bools, average='binary')
            f1 = f1_score(truth_bools, pred_bools, average='micro')

            print("\nConfusion matrix:\n", cm)
            print("\nAccuracy:", round(acc * 100, 1))
            print("F1-score:", round(f1 * 100, 1))

            part_right, on_true_or_false = PC.all_consistency(preds_data, input_data, pred_bools, truth_bools, truth_data)
            print("\nTotal Fraction of consistent inference steps: ", part_right)
            print(Counter(on_true_or_false))

            proof_coherent.append({"Model":model, "Data":test_on, "Consistent proofs": part_right})
            acc_str_list = PC.divide_data_into_depths(input_data, preds_data, truth_data)
            latex_output.append(str(f'{model} & {test_on} & {" & ".join(acc_str_list)} \\\\ \\hline'))

            wrong_examples = PC.check_hallucination_batch(preds_data, input_data)

            for ex in wrong_examples:
                print(ex)        

            if acc_by_rules:
                acc_list = PC.divide_data_into_rules(input_data, preds_data, truth_data)
                print("Saving accuracies with pickle.")
                pkl_name = "accs_by_rules/ex3_rule_accs_" + model + "_" + test_on + "_" + type_of_data + ".pkl"
                with open(pkl_name, "wb") as f:
                    pickle.dump(acc_list, f)

    

            #PC.check_proof_for_errors(preds_data, input_data)

            break
    

    print("\nLATEX FORMATTING")
    [ print(x) for x in latex_output ]

    for i in proof_coherent:
        print(i)

if __name__ == "__main__":
    main()