from label_generator import Find_Shortes_Path
import csv 
import json
import os
from tqdm import tqdm



def read_file_lines(path):
    dictionaries=[]
    with open(path) as f:
        dictionaries = json.load(f)
    
    return dictionaries



def save_data(data, save_file_name, len_data):
    
    FSP = Find_Shortes_Path(data, flat=False)
    path = FSP.find_best_path(data["query"], 0)

    with open(save_file_name, 'a') as file:
        json.dump(path, file)

        if not len_data ==0:
            file.write(",\n")



def main(home_path, dic_name, file_name):

    save_name = file_name + "_labels"
    
    file_path = home_path + "/" + dic_name + "/" + file_name + ".txt"
    save_path = home_path + "/" + dic_name + "/" + save_name + ".txt"
    data = read_file_lines(file_path)
    
    len_data = len(data)

    open(save_path, 'w').close()
    with open(save_path, 'a') as file:
        file.write("[")
    for example in tqdm(data):
        len_data -=1
        save_data(example,save_path, len_data)
    with open(save_path, 'a') as file:
        file.write("]")



if __name__ == "__main__":

    # Define the path to the data
    path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA"

    paths_to_train = [["TESTING_LABELS_DATA", "prop_examples_train"], ["TESTING_LABELS_DATA", "prop_examples_test"], ["TESTING_LABELS_DATA", "prop_examples_val"],
                      ["LP", "prop_examples_all_train"], ["LP", "prop_examples_all_test"], ["LP", "prop_examples_all_val"],
                      ["RP", "prop_examples_all_train"], ["RP", "prop_examples_all_test"], ["RP", "prop_examples_all_val"],
                      ["RP_10X", "prop_examples_all_balanced_rulenum_train"], ["RP_10X", "prop_examples_all_balanced_rulenum_test"], ["RP_10X", "prop_examples_all_balanced_rulenum_val"]]

    for p in paths_to_train:
        
        main(path,p[0],p[1])