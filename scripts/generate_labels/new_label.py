
import json
from tqdm import tqdm

def read_file_lines(path):
    dictionaries=[]
    with open(path) as f:
        dictionaries = json.load(f)
    
    return dictionaries






def main(input_file, gen_labels_file):

    input_data = read_file_lines(input_file)
    lables = read_file_lines(gen_labels_file)

