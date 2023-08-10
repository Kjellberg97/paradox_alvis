import json

def reformat_input(dict_in, include_non_facts=False):
    # Create list with the preds which are not facts
    non_facts_list = [ pred for pred in dict_in["preds"] if pred not in dict_in["facts"] ]
    
    # From list to string and add 1s at end
    facts = ' '.join([ f + '1' for f in dict_in["facts"] ])
    
    # From list to string and add 0s at end
    non_facts = ' '.join([ nf + '0' for nf in non_facts_list ])
    
    # Create a new string variable called new_query with the value of query followed by the symbol ?
    new_query = dict_in["query"] + "?"

    # From [["a", "b"], ["c"]] to a, b, c:
    new_rules = []
    for conditions, conclusion in dict_in["rules"]:
        new_rules.append(' '.join([ cond + ',' for cond in conditions ]) + ' ' + conclusion + ':')
    new_rules = ' '.join(new_rules)

    # Read the variable "new_query", and the keys "rules" and "facts" and "non-facts" as strings, combine them into one string and save it as a value with the key "input"
    dict_in["input"] = new_query 
    dict_in["input"] += ' ' + new_rules if new_rules else ''
    dict_in["input"] += ' ' + facts if facts else ''
    dict_in["input"] += ' ' + non_facts if include_non_facts and non_facts else ''
    
    return dict_in



def reformat_labels(dict_in):
    # Remake dict into a string
    dict_str = str(dict_in)    
    # Remove the following symbols {[]}"' and turn ': 0,' into '0' and turn ': 1,' into '1'
    symbols = ['{', '}', '[', ']', '"', "'"]
    for symbol in symbols:
        dict_str = dict_str.replace(symbol, '')
    dict_str = dict_str.replace(': 0,', '0')
    dict_str = dict_str.replace(': 1,', '1')
    # Remove "proof": 
    dict_str = dict_str.replace('proof: ', '')
    # Turn label: 0 into False and label: 1 into True
    dict_str = dict_str.replace('label: 0', 'False')
    dict_str = dict_str.replace('label: 1', 'True')
    # Return the reformatted string
    return dict_str


def read_file_lines(path):
    dictionaries=[]
    with open(path) as f:
        dictionaries = json.load(f)
    return dictionaries



def save_data(data, save_file_name, len_data):
    with open(save_file_name, 'w') as file:
        file.write("[")
        for d in data: 
            len_data -=1
            json.dump(d, file)

            if not len_data == 0:
                file.write(",\n")
        file.write("]")




def run_reformat(input_path, output_path, reformat_function):
    proof_list = read_file_lines(input_path)
    cleaned_list = [ reformat_function(proof_dict) for proof_dict in proof_list]
    save_data(cleaned_list, output_path, len(cleaned_list))


#THIS IS TO REFORMAT THE TARGET LABELS 

input_path_inputs = ["/<path>/<to>/DATA/RP/prop_examples_all_test.txt"]

output_path_inputs = ["/<path>/<to>/DATA/RP/prop_examples_all_test.txt"]
# For input

for inp, outp in zip(input_path_inputs, output_path_inputs):
    run_reformat(inp, outp, reformat_input)
    