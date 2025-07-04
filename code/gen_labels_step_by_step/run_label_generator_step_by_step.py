from label_generator_step_by_step import Find_next_rule
import csv 
import json
import os
from tqdm import tqdm



def read_file_lines(path):
    dictionaries=[]
    with open(path) as f:
        dictionaries = json.load(f)
    
    return dictionaries



def save_data(data, save_file_name, len_data, random):
    
    FNR = Find_next_rule(data,random)
    steps = FNR.create_proof()

    with open(save_file_name, 'a') as file:
        json.dump(steps, file)

        if not len_data ==0:
            file.write(",\n")



def main(home_path, dic_name, file_name, random=False):

    if random:
        save_name = file_name + "_step_random_labels"
    else:
        save_name = file_name + "_step_labels"
    
    file_path = home_path + "/" + dic_name + "/" + file_name + ".txt"
    save_path = home_path + "/" + dic_name + "/" + save_name + ".txt"
    
    data = read_file_lines(file_path)
    
    len_data = len(data)

    open(save_path, 'w').close()
    with open(save_path, 'a') as file:
        file.write("[")
    for example in tqdm(data):
        len_data -=1
        save_data(example,save_path, len_data, random)
    with open(save_path, 'a') as file:
        file.write("]")



if __name__ == "__main__":

    # Define the path to the data
    path = "DATA"

    # Each data 
    paths_to_train = [["RP", "prop_examples_all_cleaned_train"],
                  ["RP", "prop_examples_all_cleaned_test"],
                  ["RP", "prop_examples_all_cleaned_val"],
                  ["LP", "prop_examples_all_cleaned_train"],
                  ["LP", "prop_examples_all_cleaned_test"],
                  ["LP", "prop_examples_all_cleaned_val"],
                  ["RP_10X", "prop_examples_all_balanced_rulenum_cleaned_test.txt"],
                  ["RP_10X", "prop_examples_all_balanced_rulenum_cleaned_train.txt"],
                  ["RP_10X", "prop_examples_all_balanced_rulenum_cleaned_val.txt"]]

    for p in paths_to_train:
        
        main(path,p[0],p[1], random=True)