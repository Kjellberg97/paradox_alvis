
import json

def reformat(dict):
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


input_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all_test_labels_brackets.txt"
output_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all_test_labels.txt"
proof_list = read_file_lines(input_path)
cleaned_list = [ reformat(proof_dict) for proof_dict in proof_list]
save_data(cleaned_list, output_path, len(cleaned_list))
    