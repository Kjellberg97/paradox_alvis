import numpy as np


class Find_Shortes_Path(): 

    def __init__(self, data, flat=False):
        self.rules = data["rules"]
        self.facts = data["facts"]
        self.query = data["query"]
        self.max_depth = data["depth"]
        self.dict = {"rules": [],
                     "facts": [],
                     "query": self.query}
        self.max_depth_FALSE = 40
        self.saved_path = []
        self.flat = flat
        self.label = data["label"]
        self.used_rules = []

        
    def check_rules(self, pred):
        # return the rules which the predicate we're looking for can be derived from
        rule_list = [ r for r in self.rules if pred == r[1] and r not in self.used_rules ]
        return rule_list


    def find_shortest_path_True(self, pred, depth):
        if depth > self.max_depth:
            # Base case, if we've reached the depth were the solution should've been found
            return None
        
        # Increment counter and initialize lists
        depth += 1
        
        shortest_path=[]

        # check if the predicate is in the list of facts, then return fact
        if pred in self.facts: 
            return {str(pred): 1}

        else:
            rules = self.check_rules(pred)

            # List containing all possible paths to return a query
            all_possible_paths = [] 
            for i, rule in enumerate(rules):
                whole_path = {}

                # Register that we are using the rule
                self.used_rules.append(rule)

                #branched_path = path.copy().append(rule)
                # Loop through the list with required predicates to qualify the query
                all_found = True
                for sub_pred in rule[0]:

                    fact_or_rule = self.find_shortest_path_True(sub_pred, depth)
                    if fact_or_rule is not None:

                        whole_path.update(fact_or_rule)
                        # if not self.flat:
                        #     whole_path.append(',\n' + depth * '\t')
                        # else:
                        #     whole_path.append(', ')
                    else:
                        all_found = False
                
                if whole_path and all_found:
                    temp = {str(rule):whole_path}
                    all_possible_paths.append(temp)
                
                # Remove the rule from used rule set as we move up the tree
                self.used_rules.remove(rule)
            
                
            # return shortest path (since no fact has been found)
            if all_possible_paths:
                shortest_path = min(all_possible_paths, key=len)
                #print("possible_paths  ",all_possible_paths,"\n#########################")
                #print("short_path  ",shortest_path,"\n#########################")

            else:
                shortest_path = None    
            
            
            return shortest_path

    

    def find_shortest_path_FALSE(self,pred,depth, prev_pred, break_flag=False):
        # If the label is false, then return the "best" closest answer
        # and maybe also Retun the missing facts?! Or just the closest right path
        # Prio number of missing facts
        # secondly lenght of proof
        
        if depth > self.max_depth_FALSE:
            # Base case, if we've reached the depth were the solution should've been found
            return None, break_flag
        
        # Increment counter and initialize lists
        depth += 1
        
        shortest_path=[]

        # check if the predicate is in the list of facts, if so return fact
        if pred in self.facts:
            return {str(pred):1}, break_flag

        else:
            rules = self.check_rules(pred)
            all_possible_paths = [] 
            fact_missing_list = []

            if depth==1 and not rules:
                return {str(pred):0}, break_flag


            for i, rule in enumerate(rules):
                if break_flag:
                    break
                fact_missing = 0
                whole_path = {}

                # Register that we are using the rule
                self.used_rules.append(rule)

                # Loop through the list with required predicates to qualify the query
                for sub_pred in rule[0]:
                    if break_flag:
                        break
                    
                    if sub_pred != pred and sub_pred != prev_pred: # If not pred is not looking for itself (loop)
                        fact_or_rule, break_flag = self.find_shortest_path_FALSE(sub_pred, depth, pred, break_flag)
                    else: # If pred is looking for itself (loop)
                        fact_or_rule = None
                        
                    if fact_or_rule is not None:
                        whole_path.update(fact_or_rule)
                    else:
                        whole_path.update({str(sub_pred):0})
                        fact_missing+=1
                        break_flag = True


                    # if not self.flat:
                    #     whole_path.append(',\n' + depth * '\t')
                    # else:
                    #     whole_path.append(', ')
            
                # Number of facts missing for each rule
                fact_missing_list.append(fact_missing)       

                # All complete routes to get to a specific rule
                temp = {str(rule):whole_path}
                all_possible_paths.append(temp)

                # Remove the rule from used rule set as we move up the tree
                self.used_rules.remove(rule)

            if all_possible_paths:
                # We choose the rule with the least facts missing
                idx = np.argmin(fact_missing)
                shortest_path = all_possible_paths[idx]
               
            else:
                shortest_path = None    
            
            return shortest_path, break_flag



    def find_best_path(self, pred, depth):
        best_path={}
        if self.label== True:

            best_path = self.find_shortest_path_True(pred, depth)
            reformat_path ={"proof": best_path}
            reformat_path["label"]=1
            
        else:
            best_path = self.find_shortest_path_FALSE(pred, depth, pred)[0] 
            reformat_path ={"proof": best_path}
            reformat_path["label"]=0

        return reformat_path