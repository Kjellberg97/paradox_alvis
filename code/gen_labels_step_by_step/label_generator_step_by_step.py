"""This file is used to generate proofs as the training data for the models 
with the help of a forward-chaining algorihm. 
"""

import random
random.seed(10) 

class Find_next_rule():

    def __init__(self, data, random=False):
        self.rules = data["rules"]
        self.facts = data["facts"]
        self.query = data["query"]
        self.max_depth = data["depth"]
        self.dict = {"rules": [],
                     "facts": [],
                     "query": self.query}
        self.max_depth_FALSE = 40
        self.saved_path = []
        self.random = random
        self.label = data["label"]
        self.used_rules = []



    def create_proof(self):
        """
        Creates a proof for a given query using a set of rules and facts.
        Returns a list of steps taken to prove the query.
        """

        # Copy rules, facts and query to local variables
        rules = self.rules
        facts = self.facts
        query = self.query

        # Initialize variables
        end_flag = False
        steps = []

        # Loop until end flag is set
        while not end_flag: 
            
            # Check if query is already in facts
            if query in facts:
                steps.append("[True")
                end_flag = True 
            
            # If query is not in facts, try to find a new fact through fulfilling a rule
            else: 
                new_rule = self.find_rule(rules, facts)
                
                # If a new rule is found
                if new_rule != None:
                    # Remove the rule from the list of rules
                    rules = [ rule for rule in rules if rule != new_rule ]
                    # Add the new fact to the list of facts
                    conclusion = new_rule[1]
                    facts.append(conclusion)
                    # Add the new rule to the list of steps taken
                    steps.append(new_rule)
                
                # If no new rule is found
                else:
                    # Add False to the list of steps taken
                    steps.append("[False")
                    end_flag = True

        # Reformat steps taken and return
        steps = self.reformat(steps)    
        return steps



    def reformat(self, steps):

        for i in range(len(steps)):
            steps[i] = str(steps[i]).replace("'], [['", "': '").replace("'], '", ', ').replace("']]", ':').replace("'", '').replace("[", '').replace("]", ":")

        return steps



    def find_rule(self, rules, facts):
        """
        Find a rule that matches the given set of facts.

        Args:
        - rules (list): A list of rules in the format [[["angry", "comfortable"], "tedious"]]
        - facts (list): A list of facts to match against the rules in the format ["angry", "comfortable"]

        Returns:
        - rule (list): The first rule that matches the given facts, or None if no match is found.
        """
        # Create a set of the known facts for faster lookups
        known_facts = set(facts)

        if not self.random:
            for conditions, conclusion in rules:
                # Check if the conclusion is already known
                if conclusion in known_facts:
                    continue

                # Check if all the conditions are already known
                if all(condition in known_facts for condition in conditions):
                    return [conditions, conclusion]

        # If no new rule has been fulfilled and query has not been found
            return None
        
        else:
            
            possible_rules = []

            for conditions, conclusion in rules:
                # Check if the conclusion is already known
                if conclusion in known_facts:
                    continue

                # Check if all the conditions are already known
                if all(condition in known_facts for condition in conditions):
                    possible_rules.append([conditions, conclusion])

            if possible_rules:
                return random.choice(possible_rules)
            # If no new rule has been fulfilled and query has not been found
            return None
