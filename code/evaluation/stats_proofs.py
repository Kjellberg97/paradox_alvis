
from transformers import BartTokenizer, GPT2Tokenizer
import json
from tqdm import tqdm

class Cal_tonkenize():



    def read_input_data(self, path):
        dictionaries=[]
        with open(path) as f:
            dictionaries = json.load(f)
        
        return dictionaries



    def tokenizer_stat(self,data, model):

        tokenizer = None

        if model == "BART":
            tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
            max_seq_len = 1024

        elif model == "GPT2":
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            max_seq_len = 1024
        else:
            return 

        total_len = 0
        max_len = 0
        nr_ex = 0
        nr_over_max_seq = 0

        for d in tqdm(data):
            d_str = str(d)
            tokenized_data = tokenizer(d_str)
            len_d = len(tokenized_data['input_ids'])
            tokenized_data['input_ids']
            total_len += len_d
            
            if len_d > max_len:
                max_len = len_d

            if len_d > max_seq_len:
                nr_over_max_seq +=1
            
            nr_ex +=1

        return "\n"+str(model)+" TOKENIZER\n\tmax len: "+str(max_len) +"\n\tavg len: "+ str(total_len/nr_ex) +"\n\tnr over max seq len: "+str(nr_over_max_seq)



    def main(self, path, file):

        models=["BART","GPT2"]

        output=[]

        path = path + file +"_labels.txt"
        data = self.read_input_data(path)
        for m in models:
           output.append(self.tokenizer_stat(data,m))
        
        for out in output:
            print(out)





if __name__ == "__main__":

    path = "/Users/vikto/OneDrive/Dokument/kurser/MASTERTHESIS/LLMforReasoning/reasoning-language-models/data/simple_logic/LP/"
    file = "prop_examples_3"
    CT = Cal_tonkenize()
    CT.main(path, file)
