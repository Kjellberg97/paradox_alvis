

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



    def find_steps(self):
        
        rules= self.rules
        facts = self.facts
        end_flag = False
        steps=[]

        while not end_flag: 

            print("\n\nRULES:",rules)
            print("FACTS:",facts)
            print("QUERY:",self.query)
            
            if self.query in self.facts:
                steps.append("True")
                end_flag = True 

            else: 
                step = self.find_rule(rules, facts)
                print("Step:", step)
                if step != None:
                    rules = [ rule for rule in rules if rule != step ]
                    steps.append(step)
                    fact = step[1]
                    print("fact:", fact)
                    facts.append(fact)
                else:
                    steps.append("False")
                    end_flag = True
            
            print("STEPS:",steps)
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

        for conditions, conclusion in rules:
            # Check if the conclusion is already known
            if conclusion in known_facts:
                continue

            # Check if all the conditions are already known
            if all(condition in known_facts for condition in conditions):
                return [conditions, conclusion]

        # If no new rule has been fulfilled and query has not been found
        return None