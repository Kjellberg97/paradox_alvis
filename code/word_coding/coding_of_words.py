
import json

def read_words(file):
    with open(file) as f:
            lines = f.readlines()
            return lines
    
def read_file_lines( file_path):
    """
    ARG: path (str) in the form of /mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/prop_examples_train.txt
    RETURN: list of dicts or strings dependring on the formmating of the file
    """
    with open(file_path) as f:
        return json.load(f)


def save_coding(data, save_file_name= "coded_vocab.txt"):
    with open(save_file_name, 'w') as file:
        json.dump(data, file)


def coding():

    chars = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    for c in chars:
        for n in chars:
            yield c+str(n)


def main():
    words = read_words("vocab.txt")
    words = [ w.replace("\n", "") for w in words]

    coded_gen=coding()

    coded_dict = {}
    for w in words:
        coded_dict[w] = next(coded_gen)
      
    save_coding(coded_dict)

    coding_of_input(coded_dict, "/Users/vikto/OneDrive/Dokument/kurser/MASTERTHESIS/Data/DATA/LP/prop_examples_all_cleaned_test.txt")


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


def coding_of_input(coded_dict, input_file):
    data = read_file_lines(input_file)
    all_coded_input =[]

    for d in data:
        input_text = d["input"]
        coded_input = replace_all(input_text, coded_dict)
        
        all_coded_input.append(coded_input)
        
    save_coding(all_coded_input, save_file_name= "coded_prop_examples_all_cleaned_test.txt")

        
if __name__ == "__main__":
    main()