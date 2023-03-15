import json

def reformat_input(dict_in):
    non_facts_list = [pred for pred in dict_in["preds"] if pred not in dict_in["facts"]]

    # Within the key 'facts':
        # Remove the following symbols ["]
        # Replace , with 1
    facts = "".join(str(dict_in["facts"])).replace('[', '').replace('"', '').replace(']', '1').replace(', ', '1 ').replace("'", "")

    # Within the key 'non-facts':
        # Remove the following symbols ["]
        # Replace , with 0
    non_facts = "".join(str(non_facts_list)).replace('[', '').replace('"', '').replace(']', '0').replace(', ', '0 ').replace("'", "")

    # Within the key 'query':
        # Create a new string variable called new_query with the value of query followed by the symbol ?
    new_query = dict_in["query"] + "?"

    # Within the key 'rules':
        # Convert '], [["' and '"], "' and ']], ' into ': '
    new_rules = str(dict_in["rules"]).replace("'], [['", "': '").replace("'], '", ', ').replace("']]", ':').replace("'", '').replace("[", '')
    # Read the variable "new_query", and the keys "rules" and "facts" and "non-facts" as strings, combine them into one string and save it as a value with the key "input"
    dict_in["input"] = new_query + ' ' + new_rules + ' ' + facts + ' ' + non_facts

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
input_path_labels =  "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_brackets_test_labels.txt"
output_path_labels = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_cleaned_test_labels.txt"

# For target labels
run_reformat(input_path_labels, output_path_labels, reformat_labels)



#input_path_inputs =  "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_brackets_val.txt"
#output_path_inputs = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_cleaned_val.txt"

# For input
#run_reformat(input_path_inputs, output_path_inputs, reformat_input)
    