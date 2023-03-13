import json
test = '{"preds": ["gleaming", "confident", "selfish", "clean", "tender", "smart", "dishonest", "old-fashioned", "gorgeous", "shy", "victorious", "diplomatic"], \\
"rules": [[["dishonest", "shy", "clean"], "victorious"], [["shy", "gorgeous"], "selfish"], [["tender", "old-fashioned"], "dishonest"], [["selfish", "gleaming"], "shy"], [["gleaming", "tender"], "victorious"], [["tender"], "clean"], [["smart", "selfish"], "victorious"], [["selfish", "diplomatic"], "old-fashioned"], [["old-fashioned", "selfish", "diplomatic"], "smart"], [["diplomatic"], "dishonest"], [["tender"], "gorgeous"], [["smart", "tender", "diplomatic"], "selfish"], [["dishonest", "shy"], "selfish"], [["confident", "selfish"], "gorgeous"], [["smart", "victorious", "selfish"], "gorgeous"], [["shy", "gleaming"], "victorious"], [["diplomatic", "clean"], "selfish"], [["clean", "shy", "victorious"], "tender"], [["tender", "smart"], "selfish"], [["shy", "confident", "gorgeous"], "diplomatic"], [["gorgeous"], "smart"], [["gorgeous", "old-fashioned", "diplomatic"], "smart"]], "facts": ["smart", "victorious", "gorgeous", "confident", "old-fashioned", "gleaming", "clean"], "query": "victorious", "label": 1, "depth": 0}'

# Read it as a dictionary
# Read the keys 'preds' and 'facts' as strings, create a new key 'non-facts' which contains all the tokens in 'preds' that can not be found in 'facts'
# Within the key 'facts':
    # Remove the following symbols ["]
    # Replace , with 1
# Within the key 'non-facts':
    # Remove the following symbols ["]
    # Replace , with 0
# Within the key 'query':
    # Create a new string variable called new_query with the value of query followed by the symbol ?

# Read the variable "new_query", and the keys "rules" and "facts" and "non-facts" as strings, combine them into one string and save it as a value with the key "input"
# Within the key "input":
    # Convert '], [["' and '"], "' and ']], ' into ': '



'{ "input": "victorious? gorgeous, smart: gorgeous, old-fashioned, diplomatic, smart: smart1, victorious1, gorgeous1, confident1, old-fashioned1, gleaming1, clean1", "query": "victorious", "label": 1, "depth": 0}'


def reformat_input(dict):
    # PREDS: remove all preds that exist in facts and then add a 0 in the end of the preds that are left
    # FACTS: add a 1 in thje end of each fact andcombine all facts and preds
    # RULES: remove inner backets of all rules

    pass



def reformat_labels(dict):
    # Remake dict into a string
    dict_str = str(dict)    
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

    for d in data: 
        with open(save_file_name, 'a') as file:
            json.dump(d, file)

            if not len_data == 0:
                file.write(",\n")



def run_reformat_labels(input_path, output_path):
    proof_list = read_file_lines(input_path)
    cleaned_list = [ reformat_labels(proof_dict) for proof_dict in proof_list]
    save_data(cleaned_list, output_path, len(cleaned_list))



# #THIS IS TO REFORMAT THE TARGET LABELS 
# input_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all_test_labels_brackets.txt"
# output_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all_test_labels.txt"
# run_reformat_labels(input_path, output_path)
    