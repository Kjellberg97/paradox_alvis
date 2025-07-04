"""This file is used to run the proof checker to analyze the accuracy and consistency of
the proofs generated by the models. This are used for the proofs that are generated step-by-step
and the proof here needs to be in the format of a list where each element are a step in the proof.

Before running, check if the main function uses the correct model, test_on and chechpoints.

"""



from proof_checker import Proof_Checker
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import json
import numpy as np
import re
import pickle
from collections import Counter, OrderedDict
from decimal import Decimal

# The number of decimals when creating the latex tables for accuracy and consistency
DECIMALS=3

class Proof_Checker_Step(Proof_Checker):
 
    def read_file_lines(self, file_path):
        """
        ARG: path (str) in the form of <path_to>/DATA/EXAMPLE/prop_examples_train.txt
        RETURN: list of dicts or strings dependring on the formmating of the file
        """
        with open(file_path) as f:
            return json.load(f)


    def create_list_of_bool_labels(self, list_of_lists):
        """Finds a bool in the input 

        Args:
            list_of_lists (list): Ech element is a input or generated output  

        Returns:
            bool of the True or False label that exist in the output. If none found return -1
        """
        return [ int(eval(x[-1])) if x[-1] in ["True", "False"] else -1 for x in list_of_lists ]

    
    def find_non_bools(self, list_of_lists):
        """checks if there are any inputs without a boolean element  

        Args:
            list_of_lists (list): Ech element is a input or generated output  

        Returns:
            non_bool (list): all inputs without any bool values
        """
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
        
        """
        data_rules = [ data["input"].count(":") for data in input_data ]
        max_rules = max(data_rules) + 1
        preds_rules = [ [] for _ in range(max_rules) ]
        ground_truth_rules = [ [] for _ in range(max_rules) ]
        pred_proof = [ [] for _ in range(max_rules) ]

        for i, data in enumerate(input_data):
            n_rules = data["input"].count(":")
            pred = self.find_binary_label(predictions[i])
            ground = self.find_binary_label(ground_truth[i])
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


    def dist_over_rules(self,input_data, predictions, ground_truth):
        """Distribution of inputs over the number of rules in them

        Args:
            input_data (list): inputs
            predictions (list): predicted proofs
            ground_truth (list): groud truth proof
        """

        data_rules = [ data["input"].count(":") for data in input_data ]
        max_rules = max(data_rules) + 1
        preds = [ [] for _ in range(max_rules) ]
        ground_truths = [ [] for _ in range(max_rules) ]


        for i, data in enumerate(input_data):
            n_rules = data["input"].count(":")
            preds[n_rules].append(predictions[i])
            ground_truths[n_rules].append(ground_truth)
        
        print(preds)
        print()
        print(ground_truths)
    

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
        self.errors = []
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



    def check_consistency(self, pred_proof, input_data_whole, pred_bool):
        """Check a single proof for inconsistencies. The checks are done differently depending on if 
        the predicted truth-value is True/False If the predicted truth-value is TRUE then the check 
        is done by checking if all the rules in the proof exist in the original problem definition, if 
        the proof include all rules necessary to solve the query. If the predicted thruth-value is FALSE
        then it checks if all the rules are in the original problem description and also updated the 
        input accordingly to the proof. Then a check is done to see if there is any more rules that could 
        be applied. Returns True and "None" if the proof is consistent. Returns False and the name of the 
        error if not consistent.   
        

        Args:
            pred_proof (list): One proof where each element is a step in the proof
            input_data_whole (str): The related input to the proof
            pred_bool (bool): The predicted truth-value of the proof
            truth_bool (bool): The ground truth-value of the input  

        Returns:
            bool, str: a bool and a string with the name of the error
        """
        

        input_raw = input_data_whole.copy()
        input_data = input_raw["input"]
        query, rules, facts = self.reformat_input_into_lists(input_data)

        if pred_bool:
            for step in pred_proof[:-1]:
                # check if pred step is hallucinated by comp against input data
                if step not in rules:
                    return False, "Hallucination"
            
            # See if problem is solvable with forward chaining using only pred proof
            solvable, type_of_error = self.forward_chain(pred_proof, facts, query)
            
            return solvable, type_of_error

        else: 

            # Remove predicted rules from input
            # Try to find applicable rules that can lead to new facts
            # If new facts can be found, then return False

            input_copy = (input_data_whole["input"]+ '.')[:-1]

            no_hallucination, hall_step, updated_rules, updated_facts, type_of_error = self.check_hallucination(pred_proof, query, rules, facts)

            if no_hallucination:
                if self.no_more_satisfiable_rules(updated_rules, updated_facts):
                    return True, "None"
                else:
                    return False, "Unexhausted Search Space"
            else:
                return False, type_of_error
    


    def no_more_satisfiable_rules(self, final_rules, final_facts):
        """Checks if there are more rules than the once in the proof that can be applied
        Returns True if no more rules can be applied, False otherwise

        Args:
            final_rules (list): A list of the rules
            final_facts (list): A list of the facts

        Returns:
            bool: True if no more rules, False otherwise 
        """

        for rule in final_rules:
            conditions = rule.split(", ")[:-1]
            concusion = rule.split(", ")[-1].replace(":", "1")

            if not concusion in final_facts:
                if all(condition+"1" in final_facts for condition in conditions):
                    return False
            
        return True



    def check_hallucination(self, pred_proof, query, rules, facts):
        """Checks so that all the rules are in the the original problem
        definition. The function also updates the input based on the 
        proof. If all the rules in the proof is correct the function returns
        a True and the rules from the input with the rules from the proof removed
        and the facts with the conclusios from the rules in the proof added

        Args:
            pred_proof (list): Each element are a step in the proof
            query (str): the qruery from the input
            rules (list): a list of the rules from the input
            facts (list): a list of the facts from the input

        Returns:
            False, pred_step, updated_rules, updated_facts, "Hallucination"
            bool, list, list, list: The bool is True if the rules in the proof 
            can be found in the input. The list are the predicted proof, the updated
            rules from the input and facts based on the proof
        """

        updated_rules = rules.copy()
        updated_facts = facts.copy()

        for i, pred_step in enumerate(pred_proof):
            if pred_step == "True" or pred_step == "False":
                return True, pred_step, updated_rules, updated_facts, "None"

            # Find fact
            # Remove rule from input and add fact
            if pred_step in updated_rules and not pred_step == '':

                conditions = pred_step.split(", ")[:-1]
                new_fact = pred_step.split(", ")[-1].replace(":", "1")
                
                for req in conditions:
                    req = req + "1"
                    if not req in updated_facts:
                        return False, pred_step, updated_rules, updated_facts, "Inapplicable Rule"

                # Check if the rule can be applied with the facts
                updated_rules.remove(pred_step) 
                updated_facts.append(new_fact)
            else: 
                return False, pred_step, updated_rules, updated_facts, "Hallucination"



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
    


    def forward_chain(self, pred_proof, facts, query):
        """Performes a forward chain on the proof to check if all steps necessary
        are present in the proof

        Args:
            pred_proof (list): The step in the proof
            facts (list): a list of all the facts from the input
            query (str): the query from the input

        Returns:
            bool, str: If the proof is correct and the type of error if any
        """
        query = query + '1'
        for step in pred_proof:
            if step == "True" or step == "False":
                continue
            
            conditions = step.split(", ")[:-1] # Take everything except conclusion
            conclusion = step.split(", ")[-1].replace(":", "1") # Take conclusion but change so ex. "happy1"

            for req in conditions:
                if not req + "1" in facts:
                    return False, "Inapplicable Rule"
            facts.append(conclusion)

        if query in facts:
            return True, "None"
        else:
            return False, "Spurious Match"


    
    def all_consistency(self, pred_data, input_data, pred_labels, truth_labels, ground_proofs):
        """Performes a consistency check on a batch of proofs

        Args:
            pred_data (list): each element is a proof
            input_data (list): each element is a input
            pred_labels (list): predicted truth-value from the predicted proof
            truth_labels (list): ground truth-value of the input
            ground_proofs (list): ground truth proof

        Returns:
            float: the fractions of correct proofs 
        """
        consistent = []
        on_true_or_false = []

        for i, pred_d in enumerate(pred_data):
            con, type_of_error = self.check_consistency(pred_d, input_data[i], pred_labels[i])
            consistent.append(con)
            on_true_or_false.append(type_of_error)
            self.save_label_consistency_errors(i, pred_d, ground_proofs[i], input_data[i], con, truth_labels[i], pred_labels[i], type_of_error )

        assert len(consistent) == len(pred_data)
        assert on_true_or_false.count("None") == consistent.count(True)
        print("ConsCount", consistent.count(True))
        print("LenPred", len(pred_data))
        print("LenTBools", len(truth_labels))
        return consistent.count(True)/len(pred_data), on_true_or_false



    def find_cm_value(self, pred_label, truth_label):
        """Finds the confusion matrix value for a single instance

        Args:
            pred_label (bool): The predicted truth-value
            truth_label (bool): the ground truth-value

        Returns:
            str: The Confusion matrix value
        """
        if pred_label == False:
            if truth_label == True:
                return "False Negative"
            elif truth_label == False:
                return "True Negative"
        
        elif pred_label == True:
            if truth_label == True:
                return "True Positive"
            elif truth_label == False:
                return "False Positive"
        
        else:
            return "No Label"



    def save_label_consistency_errors(self, index, pred_d, ground_proof, input_data, con, truth_label, pred_label, type_of_error):
        """Saves the consistency error in a simple way to be able to analyze it

        Args:
            index (int): Index of the input  
            pred_d (list): Predicted proof 
            ground_proof (list): ground truth proof
            input_data (list): input related to the proof
            con (bool): If the proof is consistent or not 
            truth_label (bool): ground truth-value 
            pred_label (bool): predicted truth-value
            type_of_error (str): type of error 
        """
        if not con or truth_label != pred_label:
            confusion_matrix_value = self.find_cm_value(pred_label, truth_label)


            error_data= {"Index": index, 
                        "Label error": confusion_matrix_value,
                        "Consistency": con,
                        "Type of consistency error": type_of_error,
                        "Predicted proof": pred_d, 
                        "Ground proof": ground_proof,
                        "Input Sting": input_data["input"]}

            self.errors.append(error_data)


    def reformat_input_into_lists(self, input_str):
        """Extract rules and facts from the input string and
        return them as seperated lists. The returning lists 
        will be nested lists were the elements are either single
        rules of facts.

        ARGS:
            input_string (str): The input string with all rules and facts 

        RETURN:
            list, list: A list od the rules and the facts
        """

        if input_str[-1] == '?': # special case where there are no rules or facts in input
            query, rules, facts = input_str[:-1], [], []
        else:
            query, rules_facts_str = input_str.split('?') # queryt är utan '?' så t.ex. 'old'
            query = query.strip()
            rules_facts_str = rules_facts_str.strip()
            facts = re.findall(r'\b\w+-?\w*1\b', rules_facts_str) # [apple1', 'banana1', 'orange1']
            rules = re.findall(r'(\w+[^:]*:)', rules_facts_str) # ['helpful, fearful, happy:', 'good, bad, ugly:']

        return query, rules, facts


    def save_errors(self, model, test_on, type_of_data):
        """Saves the types of error made in the proofs in a file 
        named after the model and the train and test data in a 
        predefined file

        Args:
            model (str): name of model
            test_on (str): name of the training data
            type_of_data (str): name of the test data
        """

        file_name = "type_of_errors/error_type_" + model + "_" + test_on + "_" + type_of_data + ".txt"

        sorted_errors = sorted(self.errors, key=lambda item: item["Label error"])
        
        cm_label = ""
        with open(file_name, "w") as file:
            for item in sorted_errors:
                if item["Label error"] != cm_label:
                    cm_label = item["Label error"]
                    file.write(f"\n##################\n{cm_label.upper()}\n###################\n")
                for key in item:
                    file.write(f"{key}: {item[key]}\n")
                file.write("\n")


def reformat_files(checkpoint, model, test_on, type_of_data, rule_sampling=True):
    """Creates the names of each of the file used to save the and input files based
    on which model and data that are choosen. 

    Args:
        checkpoint (str): The name of the chechpoint of the model
        model (str): model name
        test_on (str): name of the data to test on
        type_of_data (str): if the test data is validation or test data
        rule_sampling (bool, optional): rule sampling is the type of data that where the rules
        chosen in creating the training data was randomly selected. Defaults to True.

    Returns:
        str, str, str, str: The names of the files based on the input
    """

    path = ""

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
    """Main fuction of the proof checker. The function test all models and on all data. 
    All of the commands are hardcoded and need to be change here. The model, the testdata
    and the chechpoints are all hardcoded.  
    """
    acc_by_rules = False
    rule_sampling = True

    # NAME OF MODELS TO TEST
    models = ["LP", "RP", "RP_10X"]
    # NAME OF THE DATA TO TEST ON
    test_ons = ["LP", "RP", "RP_10X"]
    # TEST ON TEST-SET OR VAL-SET
    type_of_data = "test" # or "val"

    proof_coherent = []

    latex_output = []
    latex_errors = []
    latex_conf_prec_recall_f1 = []
    max_len_error_keys = 0
    
    for model in models:

        # NAME OF THE CHECHPOINTS TO USE 
        # CHANGE HERE IF NAME OF CHECHPOINTS NO LONGER ARE RELEVANT 
        if model == "LP":
            checkpoint = "checkpoint-9000" 
            # if not rule_sampling:
            #     checkpoint = "checkpoint-8500"
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
            percent_right = round(Decimal(part_right * 100), 2)
            print(Counter(on_true_or_false))

            PC.save_errors(model, test_on, type_of_data)

            # proof_coherent.append({"Model":model, "Data":test_on, "Consistent proofs": part_right})
            if model == "RP_10X": model_str =  "RP\_b"
            else: model_str = model
            if test_on == "RP_10X": test_on_str =  "RP\_b"
            else: test_on_str = test_on

            acc_str_list = PC.divide_data_into_depths(input_data, preds_data, truth_data)
            latex_output.append(str(f'{model_str} & {test_on_str} & {" & ".join(acc_str_list)} & {percent_right} \\\\ \\hline'))

            # PC.check_hallucination_batch(preds_data, input_data)        

            # if acc_by_rules:
            #     acc_list = PC.divide_data_into_rules(input_data, preds_data, truth_data)
            #     print("Saving accuracies with pickle.")
            #     pkl_name = "accs_by_rules/ex3_rule_accs_" + model + "_" + test_on + "_" + type_of_data + ".pkl"
            #     with open(pkl_name, "wb") as f:
            #         pickle.dump(acc_list, f)

            # Count errors and create latex table rows
            PC_error_counts = Counter(d["Type of consistency error"] for d in PC.errors)
            counts = Counter(on_true_or_false)
            #print("Ontruefalse", counts)
            #print("PC", PC_error_counts)

            # Set N and delete None from counts dict
            N = sum(counts.values())
            del counts["None"]
            counts["Consistency Errors"] = sum(counts.values())

            # Construct percent list
            headers = ["Hallucination", "Inapplicable Rule", "Spurious Match", "Unexhausted Search Space", "Consistency Errors"]
            percent = OrderedDict((key, str(round(Decimal((counts.get(key, 0) / N) * 100), 3))) for key in headers)
            
            # First row
            if not latex_errors:
                header_contents = " & ".join(f"{key}" for key in percent.keys())
                header_row = str(f'Train & Test & {header_contents} \\\\ \\hline')
                latex_errors.append(header_row)

            # All other rows
            table_row = " & ".join(percent.values())
            latex_errors.append(str(f'{model_str} & {test_on_str} & {table_row} \\\\ \\hline'))

            
            conf = PC.create_confusion_matrix(pred_bools, truth_bools)
            conf = np.array(conf)
            conf = conf.sum(axis=0)
            conf = [str(x) for x in conf]
            recall = round(recall_score(pred_bools, truth_bools, average="micro"), DECIMALS)
            precision = round(precision_score(pred_bools, truth_bools, average="micro"), DECIMALS)

            if not latex_conf_prec_recall_f1:
                h= ["True Positive", "False Positive", "True Negative", "False Negative", "Precision", "Recall","F1-Score"]
                latex_conf_prec_recall_f1.append(str(f'Train & Test & {" & ".join(h)} \\\\ \\hline'))

            latex_conf_prec_recall_f1.append(str(f'{model_str} & {test_on_str} & {" & ".join(conf)} & {precision} & {recall} & {round(f1, DECIMALS)}\\\\ \\hline'))
            #print(list(map(sum, conf)))

            #PC.dist_over_rules(input_data, pred_bools, truth_bools)


    

    print("\nLATEX ACCURACIES FORMATTING")
    [ print(x) for x in latex_output ]

    print("\nLATEX ERRORS FORMATTING")
    [ print(x) for x in latex_errors ]

    print("\nLATEX TP/FP FORMATTING")
    [ print(x) for x in latex_conf_prec_recall_f1 ]

    for i in proof_coherent:
        print(i)

if __name__ == "__main__":
    main()