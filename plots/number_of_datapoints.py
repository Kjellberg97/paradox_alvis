import json

# Load data


def read_file_lines( file_path):
    """
    ARG: path (str) in the form of /mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/prop_examples_train.txt
    RETURN: list of dicts or strings dependring on the formmating of the file
    """

    file_path = "C:/Users/vikto/OneDrive/Dokument/kurser/MASTERTHESIS/Data/DATA/" + file_path
    with open(file_path) as f:
        return json.load(f)
    


def count_data_points(files):

    proofs =[]

    for f in files:
        proofs.extend(read_file_lines(f))
    total_number_of_data = 0 

    for proof in proofs:
        lenght_proof = len(proof)
        total_number_of_data += lenght_proof

    print(files)
    print("NUMBER OF TRAINING INTANCES: ", total_number_of_data)
    print()



files = [["LP/prop_examples_all_cleaned_test_step_random_labels.txt", "LP/prop_examples_all_cleaned_train_step_random_labels.txt", "LP/prop_examples_all_cleaned_val_step_random_labels.txt"],
         ["RP/prop_examples_all_cleaned_test_step_random_labels.txt", "RP/prop_examples_all_cleaned_train_step_random_labels.txt", "RP/prop_examples_all_cleaned_val_step_random_labels.txt"],
         ["RP_10X/prop_examples_all_balanced_rulenum_cleaned_val_step_random_labels.txt", "RP_10X/prop_examples_all_balanced_rulenum_cleaned_test_step_random_labels.txt", "RP_10X/prop_examples_all_balanced_rulenum_cleaned_train_step_random_labels.txt"]]

for f in files:
    count_data_points(f)