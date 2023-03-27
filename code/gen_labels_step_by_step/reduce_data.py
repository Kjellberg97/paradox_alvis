import random
import json
from tqdm import tqdm


def reduce_data(data, labels):
    """Reduce the data to a specific amount or part of the data.
    Samples randomly in the indexes

    ARGS:
        data (list): 
            The data that should be reduced
        labels (list):
            The related true labels to the data 

    RETURN:
        reduced_data (list):
            the reduced data
        reduced_labels (list):
            the labels related to the reduced data
    """

    random.seed(10)

    len_d = len(data)
    number_of_ex = len_d//100
    #number_of_ex = 100
    indexes = random.sample(range(len_d),number_of_ex)

    reduced_data, reduced_labels = pick_data_from_index(indexes, data, labels)

    return reduced_data, reduced_labels



def pick_data_from_index(index, data, labels):

    """Reduce the data based on the choosen indexes

    ARGS:
        index (list):
            A list over the indexes that should be picked
        data (list): 
            The data that should be reduced
        labels (list):
            The related true labels to the data 

    RETURN:
        reduced_data (list):
            the reduced data
        reduced_labels (list):
            the labels related to the reduced data
    """

    reduced_data = []
    reduced_labels = []

    for i in index:
        reduced_data.append(data[i])
        reduced_labels.append(labels[i])

    return reduced_data, reduced_labels


def save_data(data, save_file_name, len_data):
    """Save the data into a choosen file

    ARGS:
        data (list):
            The reduced data
        save_file_name(str):
            the path where the data should be saved
        len_data (int):
            the len of the data
    """

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
    """Function for seeing if all the data is balanced.
    If it is equally many true as false and if it is about the same number 
    on each depth.

    ARGS:
        data (list):
            the data
        labels (list):
            the related labels
    """
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

def data_paths(file_name, save_name):

    input_suffix=["_train.txt", "_val.txt", "_test.txt"]
    label_suffix=["_train_step_labels.txt", "_val_step_labels.txt", "_test_step_labels.txt"]

    file_names_data = []
    file_names_labels = []
    save_names_data = []
    save_names_labels = []


    for ds, ls in zip(input_suffix, label_suffix):
        file_names_data.append(file_name+ds)
        file_names_labels.append(file_name+ls)
        save_names_data.append(save_name+ds)
        save_names_labels.append(save_name+ls)

    return file_names_data, file_names_labels, save_names_data, save_names_labels



if __name__ == "__main__":

    # file_name_data  = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all_cleaned_train.txt"
    # file_name_label = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all_cleaned_train_step_labels.txt"
    # save_path_data  = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/all_small_cleaned_reduced_train.txt"
    # save_path_label = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/all_small_cleaned_reduced_train_step_labels.txt"

    file_name_data  = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all_cleaned"
    save_path_data  = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/all_small_cleaned_reduced"

    file_names_data, file_names_labels, save_names_data, save_names_labels = data_paths(file_name_data, save_path_data)

    for i in range(3):
        print(file_names_data[i], file_names_labels[i])
        data = read_file_lines(file_names_data[i])
        labels = read_file_lines(file_names_labels[i])

        reduced_data, reduced_labels = reduce_data(data, labels)
        len_data = len(reduced_data)

        print(save_names_data[i],save_names_labels[i])
        save_data(reduced_data, save_names_data[i], len_data)
        save_data(reduced_labels, save_names_labels[i], len_data)

        check_stats(reduced_data,reduced_labels)

