from tqdm import tqdm
class Proof_Checker():

    def __init__(self, flat=False):

        self.corr_labels = [0,0,0,0] # True Positive, False Positive, True Negative, False Negative
        self.corr_proofs = 0
        self.num_ex = 0 
        self.syntax_err = 0
        self.halluzination = 0
        self.temp_hal = []
        self.halluzination_list = []
        self.flat = flat


    def save_result(self):
        # save the result of one of the functions in a file
        pass 


    def check_acc(self, true_label, gen_label):
        """INPUT
            true_label: 
                the ground truth label either 1 or 0
            gen_label:
                the generated label either True or False
        """
        if true_label==1:
            if gen_label==1:
                # True Positive
                self.corr_labels[0] +=1
            else:
                # False Negative
                self.corr_labels[3] +=1
        
        else:
            if gen_label==1:
                # False Positive
                self.corr_labels[1] +=1
            else:
                # True Negative
                self.corr_labels[2] +=1

            


    def check_correctness(self, gen_proof):
        """INPUT
            gen_proof:
                list of the generated proof 
        """
        pass
    

    def check_syntax():
        # Check if the syntax of the generated proof is correct
        # output the wrong part or none 
        pass


    def check_order(self, gen_proof):

        # Chech if the order of the proof is correct
        pass

    
    def rule_fact_in_list(self, step, rules, facts):
        """INPUT
        step: 
            {dict} the step that will be taken with the condition for this step
        rules:
            [list] over the existing rules 
        facts:
            [list] over all the existing facts

        The function goes step by step in the proof and check so that all the rules and facts that was used 
        accually exist in the input data. The function counts all the 'halluzinations' or wrongly created
        facts and rules and saves a list over all wrong rules/facts and a counder of the number of
        halluzinations. 

        !Does NOT look at facts that are generated as not existing!
        """
        
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
                    self.halluzination +=1

            

    def halluzination_rules_fact(self, gen_proof, rules, facts):
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
            self.halluzination_list.append({self.num_ex: self.temp_hal})

        # save the list of halluzinations in a file



    def redundance():
        # Does all generated rules and fact exisit in the problem
        pass


    
    def print_result(self):
        # accuracy
        acc = (self.corr_labels[0] + self.corr_labels[2])/self.num_ex

        print("Accuracy: ",acc)
        print("Nr halluzinations:", self.halluzination)
        print("hallucinations: ", self.halluzination_list)



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
            self.halluzination_rules_fact(gen_proof, input_rules, input_facts)

            self.num_ex +=1
        self.print_result()


