import re
import numpy as np
from tqdm import tqdm
import json
import random



class Proof_Checker():

    def __init__(self, save_stats_file, seed=1):

        self.confusion_matrix = [0,0,0,0] # True Positive, False Positive, True Negative, False Negative
        self.accuracy = 0
        self.num_ex = 0

        self.corr_proofs = 0
        self.hall_rule = 0
        self.hall_fact = 0
        self.ended_too_early = 0

        self.temp_hal = []
        self.hallucination_list = []
        self.save_stats_file = save_stats_file
        open(self.save_stats_file, 'w').close()
        random.seed(seed)


    def save_result(self):
        # save the result of one of the functions in a file
        pass 


    def find_binary_label(self, string):
        # Find the last occurence of False or True in the string, convert into corresponding int 0 or 1
        match = re.search(r"True|False(?!.*True|False)", string) # Not followed by any characters (.* , and not followed by True|False
        binary_digit = int(eval(match.group())) if match else 0 # Convert into int if a False or True is returned else convert to 0
        return binary_digit


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

        


        #ground_truth_labels = [ target_d['label'] for target_d in ground_truth ]  



    def create_confusion_matrix(self, predictions, ground_truth):
        """
        Creates a confusion matrix based on predicted and ground truth labels.
        
        Args:
        predictions (list): A list of strings representing the predicted labels, containing a substring "'label': " followed by "0" or "1".
        ground_truth (list): A list of integers representing the true labels, where 0 indicates a negative example and 1 indicates a positive example.
        
        Returns:
        numpy.ndarray: A confusion matrix of shape (n_samples, 4) with columns for True Positive, False Positive, True Negative, and False Negative.
        """
        confusion_matrix = np.empty(shape=(len(ground_truth), 4), dtype=int) # True Positive, False Positive, True Negative, False Negative
        for i, (guess, truth) in enumerate(zip(predictions, ground_truth)):
            # Fill the confusion matrix
            confusion_matrix[i, 0] = 1 if guess == 1 and truth == 1 else 0 # True Positive
            confusion_matrix[i, 1] = 1 if guess == 1 and truth == 0 else 0 # False Positive
            confusion_matrix[i, 2] = 1 if guess == 0 and truth == 0 else 0 # True Negative
            confusion_matrix[i, 3] = 1 if guess == 0 and truth == 1 else 0 # False Negative
        self.confusion_matrix = confusion_matrix
        return confusion_matrix 

    

    def get_index_matrix(self, confusion_matrix):
        index_TP = np.argwhere((confusion_matrix == [1,0,0,0]).all(axis=1))
        index_FP = np.argwhere((confusion_matrix == [0,1,0,0]).all(axis=1))
        index_TN = np.argwhere((confusion_matrix == [0,0,1,0]).all(axis=1))
        index_FN = np.argwhere((confusion_matrix == [0,0,0,1]).all(axis=1))
    
        return index_TP, index_FP, index_TN, index_FN
        
    

    def stat_over_generated_data(self, predictions, ground_truth, input_dicts,pred_proof):
        """Check the stats over the different values in the conf. matrix.
        e.g. how many rules that exixst in each input to see any scatistical
        correlations between the different label values. And print the calculated stats

        ARGS:
            predictions (list) : a list where all element are dicts. Each dict is a generated proof and label
            ground_truth (list) : the true lable value
            input_dicts (list) : a list of all input values

        RETURN:
            None

        """
        #preds = [self.find_binary_label(p) for p in predictions ]
        ground_truth_labels = [ self.find_binary_label(target_d) for target_d in ground_truth ]
        confusion_matrix = self.create_confusion_matrix(predictions, ground_truth_labels)
        index_TP, index_FP, index_TN, index_FN = self.get_index_matrix(confusion_matrix)

        print("TP: ", self.len_rules(input_dicts, index_TP))
        print("FP: ", self.len_rules(input_dicts, index_FP))
        print("TN: ", self.len_rules(input_dicts, index_TN))
        print("FN: ", self.len_rules(input_dicts, index_FN))

        indexes = [index_TP, index_FP, index_TN, index_FN]
        

        self.save_proofs(pred_proof, ground_truth, indexes, input_dicts)
                

    def save_proofs(self, preds, ground_truth, index, input_dicts):
        sample_size = 10

        samples_idx = [None, None, None, None]
        
        for i, list_idx in enumerate(index):
            if len(list_idx)>= sample_size:
                nr_s = sample_size
                samples_idx[i] = random.sample(list(list_idx), nr_s)
            elif len(list_idx) > 0:
                samples_idx[i] = random.sample(list(list_idx), len(list_idx))

        cf_m = ["TP", "FP", "TN", "FN"]
        with open(self.save_stats_file, 'a') as file:
            for j, cf in enumerate(cf_m):
                file.write("\n\n"+str(cf)+ "-------------------")
                if samples_idx[j] == None:
                    file.write("\n\nNone\nNone")
                else:     
                    for x in samples_idx[j]:
                        x = x[0]
                        in_data = input_dicts[x]
                        file.write("\n\nIndex: " + str(x))
                        file.write("\nPREDICTED:    " + str(preds[x]))
                        file.write("\nGROUND TRUTH: " + str(ground_truth[x]))
                        file.write("\nINPUT:        " + str(in_data["input"]))
                


    
    def len_rules(self, data, indexes):
        """Calculates mean lenght, minimum and maximum lenght of the rules of the input data.

        ARGS:
            data (list) : a list of dict where each dict is a data point. Should be the input data
            indexes (numpy.array) : a array over the indexes of data input that are relevant

        RETURN:
            (list) : the mean lenght, min lenght and max lenght of the number of rules for rules 
                     in the data.
        """

        nr_ex = len(indexes)
        tot_len_rules = 0
        min_len=np.inf
        max_len=0
        for i in indexes:
            d = data[int(i)]
            depth = d["depth"]
            
            len_d = len(d["rules"])
            tot_len_rules +=len_d
            if len_d < min_len:
                min_len = len_d
            if len_d > max_len:
                max_len = len_d

        if nr_ex > 0:
            mean_len = round(tot_len_rules / nr_ex, 2)
            return mean_len, min_len, max_len
        
        else:
            return nr_ex, min_len, max_len







    def label_accuracy(self, confusion_matrix):
        """
        Calculates the label accuracy based on a given confusion matrix.
        
        Args:
        confusion_matrix (numpy.ndarray): A confusion matrix of shape (n_samples, 4) with columns for True Positive, False Positive, True Negative, and False Negative.
        
        Returns:
        float: The proportion of correctly labeled examples, as a decimal between 0 and 1.
        """
        correct_count = confusion_matrix[:, 0].sum() + confusion_matrix[:, 2].sum() # Sum True Positives and True Negatives

        accuracy = correct_count / confusion_matrix.shape[0] # count divided by number or rows
        self.accuracy = accuracy
        return accuracy



    def check_correctness(self, gen_proof):
        """INPUT
            gen_proof:
                list of the generated proof 
        """
        pass
    

    def check_syntax(self, proof):
        # Check if the syntax of the generated proof is correct
        # output the wrong part or none
        try:
            proof_d = dict(proof)
        except:
            # 
            pass    

        


    def check_order(self, gen_proof):

        # Chech if the order of the proof is correct
        pass

    
    def rule_fact_in_list(self,step, rules, facts):
        """INPUT
        step: 
            {dict} the step that will be taken with the condition for this step
        rules:
            [list] over the existing rules 
        facts:
            [list] over all the existing facts

        The function goes step by step in the proof and check so that all the rules and facts that was used 
        accually exist in the input data. The function counts all the 'hallucinations' or wrongly created
        facts and rules and saves a list over all wrong rules/facts and a counder of the number of
        hallucinations. 

        !Does NOT look at facts that are generated as not existing!
        """

        ground_truth_labels = [dict, dict]
        generated_labels = [str, str]

        try:
            gen = dict(gen)
        except:
            self.syntax_err +=1
            # maybe save the proofs with syntax error in a file!!
        
        for key in step.keys():
            rule_exist = False
            fact_exist = False
            if not (step[key] == 0 or step[key] == 1) :
                self.rule_fact_in_list(step[key], rules, facts)

            for r in rules :
                if str(key) == str(r):
                    rule_exist = True
            
            if not rule_exist:
                for f in facts:
                     if key == str(f):
                        fact_exist=True

            if not (rule_exist or fact_exist) and not (step[key] == 0):
                self.temp_hal.append(key)
                self.hallucination +=1

        pass

            

    def hallucination_rules_fact(self, gen_proof, rules, facts):
        """ Checks if the rules and facts are in the input list over rules and facts.
        
        ARGS
        step: 
            {dict} the first step in the proof
        rules:
            [list] over the existing rules 
        facts:
            [list] over all the existing facts

        Saves the result from rule_fact_in_list()
        """
        self.temp_hal = []
        self.rule_fact_in_list(gen_proof,rules, facts)
        if self.temp_hal:
            self.hallucination_list.append({self.num_ex: self.temp_hal})

        # save the list of hallucinations in a file



    def redundance():
        # Does all generated rules and fact exisit in the problem
        pass


    
    def print_result(self):
        # accuracy
        acc = (self.corr_labels[0] + self.corr_labels[2])/self.num_ex

        print("Accuracy: ",acc)
        print("Nr hallucinations:", self.hallucination)
        print("hallucinations: ", self.hallucination_list)



    def run_proof_check(self, indata, gen_anw):
        """
        ARGS:
            indata:
                a list of the input data of the model where each
                item is a [rules, facts, query, label] 
            gen_anw:
                list of generated answers for each input data
                [proof, label]
        """

        i=0
        for input_data in tqdm(indata):
            
            output= gen_anw[self.num_ex]

            
            true_label = input_data["label"]
            gen_label = output["label"]

            gen_proof = output["proof"]
            input_rules = input_data["rules"]
            input_facts = input_data["facts"]

            self.check_acc(true_label, gen_label)
            self.hallucination_rules_fact(gen_proof, input_rules, input_facts)

            self.num_ex +=1
        self.print_result()


