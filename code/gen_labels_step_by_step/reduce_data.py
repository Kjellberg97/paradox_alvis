
def reduce_data(data):

    

    return data


def read_file_lines(path):
    dictionaries=[]
    with open(path) as f:
        dictionaries = json.load(f)
    
    return dictionaries