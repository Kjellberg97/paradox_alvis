import json
test = '{"preds": ["gleaming", "confident", "selfish", "clean", "tender", "smart", "dishonest", "old-fashioned", "gorgeous", "shy", "victorious", "diplomatic"], "rules": [[["dishonest", "shy", "clean"], "victorious"], [["shy", "gorgeous"], "selfish"], [["tender", "old-fashioned"], "dishonest"], [["selfish", "gleaming"], "shy"], [["gleaming", "tender"], "victorious"], [["tender"], "clean"], [["smart", "selfish"], "victorious"], [["selfish", "diplomatic"], "old-fashioned"], [["old-fashioned", "selfish", "diplomatic"], "smart"], [["diplomatic"], "dishonest"], [["tender"], "gorgeous"], [["smart", "tender", "diplomatic"], "selfish"], [["dishonest", "shy"], "selfish"], [["confident", "selfish"], "gorgeous"], [["smart", "victorious", "selfish"], "gorgeous"], [["shy", "gleaming"], "victorious"], [["diplomatic", "clean"], "selfish"], [["clean", "shy", "victorious"], "tender"], [["tender", "smart"], "selfish"], [["shy", "confident", "gorgeous"], "diplomatic"], [["gorgeous"], "smart"], [["gorgeous", "old-fashioned", "diplomatic"], "smart"]], "facts": ["smart", "victorious", "gorgeous", "confident", "old-fashioned", "gleaming", "clean"], "query": "victorious", "label": 1, "depth": 0}'

# Remove everything before "rules: " including  "rules: "
#

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


print(reformat(test))

# input_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all_test_labels_brackets.txt"
# output_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all_test_labels.txt"
# proof_list = read_file_lines(input_path)
# cleaned_list = [ reformat(proof_dict) for proof_dict in proof_list]
# save_data(cleaned_list, output_path, len(cleaned_list))
    