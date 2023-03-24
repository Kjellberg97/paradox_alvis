import random
import json
from tqdm import tqdm


def reduce_data(data, labels):

    random.seed(10)

    len_d = len(data)
    number_of_ex = len_d//100
    #number_of_ex = 1000
    indexes = random.sample(range(len_d),number_of_ex)

    reduced_data, reduced_labels = pick_data_from_index(indexes, data, labels)

    return reduced_data, reduced_labels



def pick_data_from_index(index, data, labels):

    reduced_data = []
    reduced_labels = []

    for i in index:
        reduced_data.append(data[i])
        reduced_labels.append(labels[i])

    return reduced_data, reduced_labels


def save_data(data, save_file_name, len_data):

    open(save_file_name, 'w').close()

    with open(save_file_name, 'a') as file:
        file.write("[")

    for example in tqdm(data):
        len_data -=1
        with open(save_file_name, 'a') as file:
            json.dump(example, file)

            if not len_data ==0:
                file.write(",\n")

    with open(save_file_name, 'a') as file:
        file.write("]")


def check_stats(data, labels):

    f=0
    t=0

    depths=[0,0,0,0,0,0,0]
    len(data)

    for d,l in zip(data, labels):

        depth = int(d["depth"])
        lab = int(d["label"])

        if lab == 1:
            t+=1
        elif lab == 0:
            f+=1
        
        depths[depth] += 1
    
    print("Labels TRUE/FALSE:     ", round(t/len(data),2), round(f/len(data),2))
    print("Num ex for each depth: ", depths)
    

def read_file_lines(path):
    dictionaries=[]
    with open(path) as f:
        dictionaries = json.load(f)
    
    return dictionaries


if __name__ == "__main__":

    file_name_data  = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all_cleaned_test.txt"
    file_name_label = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all_cleaned_test_step_labels.txt"
    save_path_data  = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/all_cleaned_reduced_test.txt"
    save_path_label = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/all_cleaned_reduced_test_step_labels.txt"

    data = read_file_lines(file_name_data)
    labels = read_file_lines(file_name_label)

    reduced_data, reduced_labels = reduce_data(data, labels)
    len_data = len(reduced_data)

    save_data(reduced_data, save_path_data, len_data)
    save_data(reduced_labels, save_path_label, len_data)

    check_stats(reduced_data,reduced_labels)

