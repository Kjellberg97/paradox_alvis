# Runs the proof checker
from proof_checker import Proof_Checker
import json


def read_input_data(path):
    dictionaries=[]
    with open(path) as f:
        dictionaries = json.load(f)
    
    return dictionaries



def read_gen_data(path):
    dictionaries=[]
    with open(path) as f:
        dictionaries = json.load(f)
    
    return dictionaries



def read_data(path, file):

    input_path = path + file +".txt"
    input_data = read_input_data(input_path)

    gen_path = path + file +"_labels.txt"
    output_data = read_gen_data(gen_path)

    return input_data, output_data



def main(path, file_name):

    input_data, output_data = read_data(path, file_name)

    pf = Proof_Checker()
    pf.run_proof_check(input_data, output_data)



if __name__ == "__main__":

    path = "/Users/vikto/OneDrive/Dokument/kurser/MASTERTHESIS/LLMforReasoning/reasoning-language-models/data/simple_logic/LP/"
    file = "prop_examples_3"
    main(path, file)