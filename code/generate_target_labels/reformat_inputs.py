
import json


# test = {"preds": ["gleaming", "confident", "selfish", "clean", "tender", "smart", "dishonest", "old-fashioned", "gorgeous", "shy", "victorious", "diplomatic"], 
# "rules": [[["dishonest", "shy", "clean"], "victorious"], [["shy", "gorgeous"], "selfish"], [["tender", "old-fashioned"], "dishonest"], [["selfish", "gleaming"], "shy"], [["gleaming", "tender"], "victorious"], [["tender"], "clean"], [["smart", "selfish"], "victorious"], [["selfish", "diplomatic"], "old-fashioned"], [["old-fashioned", "selfish", "diplomatic"], "smart"], [["diplomatic"], "dishonest"], [["tender"], "gorgeous"], [["smart", "tender", "diplomatic"], "selfish"], [["dishonest", "shy"], "selfish"], [["confident", "selfish"], "gorgeous"], [["smart", "victorious", "selfish"], "gorgeous"], [["shy", "gleaming"], "victorious"], [["diplomatic", "clean"], "selfish"], [["clean", "shy", "victorious"], "tender"], [["tender", "smart"], "selfish"], [["shy", "confident", "gorgeous"], "diplomatic"], [["gorgeous"], "smart"], [["gorgeous", "old-fashioned", "diplomatic"], "smart"]],
# "facts": ["smart", "victorious", "gorgeous", "confident", "old-fashioned", "gleaming", "clean"], "query": "victorious", "label": 1, "depth": 0}


def reformat_input(dict):
    non_facts_list = [pred for pred in dict["preds"] if pred not in dict["facts"]]

    # Within the key 'facts':
        # Remove the following symbols ["]
        # Replace , with 1
    facts = "".join(str(dict["facts"])).replace('[', '').replace('"', '').replace(']', '1').replace(', ', '1 ').replace("'", "")

    # Within the key 'non-facts':
        # Remove the following symbols ["]
        # Replace , with 0
    non_facts = "".join(str(non_facts_list)).replace('[', '').replace('"', '').replace(']', '0').replace(', ', '0 ').replace("'", "")

    # Within the key 'query':
        # Create a new string variable called new_query with the value of query followed by the symbol ?
    new_query = dict["query"] + "?"

    # Within the key 'rules':
        # Convert '], [["' and '"], "' and ']], ' into ': '
    new_rules = str(dict["rules"]).replace("'], [['", "': '").replace("'], '", ', ').replace("']]", ':').replace("'", '').replace("[", '')
    # Read the variable "new_query", and the keys "rules" and "facts" and "non-facts" as strings, combine them into one string and save it as a value with the key "input"
    dict["input"] = new_query + ' ' + new_rules + ' ' + facts + ' ' + non_facts

    return dict



def read_file_lines(path):
    dictionaries=[]
    with open(path) as f:
        dictionaries = json.load(f)
    return dictionaries



def save_data(data, save_file_name, len_data):

    for d in data: 
        with open(save_file_name, 'a') as file:
            json.dump(d, file)

            if not len_data == 0:
                file.write(",\n")




def run_reformat_labels(input_path, output_path):
    proof_list = read_file_lines(input_path)
    cleaned_list = [ reformat_input(proof_dict) for proof_dict in proof_list]
    save_data(cleaned_list, output_path, len(cleaned_list))


# THIS IS TO REFORMAT THE INPUT  
input_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all_brackets_val.txt"
output_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all_cleaned_val.txt"
run_reformat_labels(input_path, output_path)
print("Done!")


