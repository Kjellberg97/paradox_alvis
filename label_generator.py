


class Find_Shortes_Path(): 

    def __init__(self, data):
        self.rules = data["rules"]
        self.facts = data["facts"]
        self.query = data["query"]
        self.max_depth = data["depth"]
        self.dict = {"rules": [],
                     "facts": [],
                     "query": self.query}
        self.saved_path = []


    def check_facts(self, pred):
        # check if the rule can be completed with the facts

        for fact in self.facts:
            if pred == fact:
                return True
        return False


        
    def check_rules(self, rule):
        # check if the rule can be completed with the rules
        rule_list=[]
        for i,r in enumerate(self.rules):
            if rule == r[1]:
                rule_list.append(r)
        return rule_list 



    def find_path(self, pred, depth):
        path=[]
        fact_rule = None
        print(depth)

        if depth > self.max_depth:
            return path
        depth += 1

        if self.check_facts(pred):
            pass
            path.append("fact "+pred)

        else:
            rules = self.check_rules(pred)
            if rules == None:
                return path
            
            found =False
            for rule in rules:
                print("rule: ",rule)
                if not found:
                    temp_fact = []
                    for sub_pred in rule[0]:
                        
                        fact_rule = self.find_path(sub_pred, depth)
                        print("fact_rule: ",fact_rule)
                        if fact_rule:
                            temp_fact.extend(fact_rule)
                        else:
                            break
                        
                if fact_rule:
                    path.extend(temp_fact)
                    path.append(rule)
                    break

        print("Full path: ",path, "\n-------------------------------------------")
        
        
        return path
    

    
def main():

    print("Start")

    ex ={"preds": ["talented", "friendly", "stubborn", "straightforward", "clumsy", "aggressive", "shiny", "fancy", "crowded", "fine", "naughty", "homely", "sincere", "impatient", "tired", "witty", "good", "pleasant", "pessimistic"],
    "rules": [
       
        [["aggressive", "naughty", "talented"], "pessimistic"],
        [["shiny", "pleasant", "friendly"], "witty"], 
        [["stubborn", "fine", "talented"], "naughty"], 
        [["friendly"], "crowded"], 
        [["clumsy", "witty", "shiny"], "good"], 
        [["sincere", "straightforward"], "clumsy"], 
        [["homely", "sincere"], "straightforward"], 
        [["tired", "naughty"], "impatient"], 
        [["fancy", "good"], "tired"], 
        [["naughty", "good", "witty"], "shiny"], 
        [["fine", "crowded", "stubborn"], "aggressive"], 
        [["talented", "witty", "tired"], "fine"], 
        [["good"], "crowded"], 
        [["fine"], "stubborn"],
        [["good"], "stubborn"]],
    "facts": ["friendly", "impatient", "fancy", "pessimistic", "good", "homely", "witty", "talented"],
    "query": "aggressive",
    "label": 1,
    "depth": 3}

    FP = Find_Shortes_Path(ex)

    path = FP.find_path(ex["query"], 0)

    

    print(path)




if __name__ == "__main__":
    main()
