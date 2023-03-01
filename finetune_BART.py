from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import DatasetDict, Dataset, load_dataset
import json
import ast


"""
checkpoint = "facebook/bart-large"
model_path = 
tokenizer_path =

raw_inputs_path = 'data/simple_logic/LP/prop_examples_0.txt' # 'data/test_inputs.txt'
raw_labels_path = 'data/simple_logic/LP/prop_examples_0_labels.txt' # 'data/test_proofs.txt'

def read_file_lines(path):
    with open(path) as f:
        dicts = json.load(f)
    return dicts


raw_inputs = read_file_lines(raw_inputs_path)
raw_labels = read_file_lines(raw_labels_path)

dict_list = []
for input, target, in zip(raw_inputs, raw_labels):
    del input['depth']
    del input['label']

    d = {
            "input": str(input),
            "target": str(target),
            "label": target['label']
        }
    dict_list.append(d)

#print(dict_list[0])


raw_ds = Dataset.from_list(dict_list)
#print(raw_ds)
#print(raw_ds['target'])

# Test if we can load a generated label into a dictionary
test = ast.literal_eval(raw_ds['target'][0])
#print(test)




tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize(dset):
    return tokenizer(text=dset["input"], text_target=dset["target"], truncation=True)


ds_tokenized = raw_ds.map(tokenize, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# print(ds_tokenized)

# for i in ['input', 'target', 'label', 'input_ids', 'attention_mask', 'labels']:
#     print("\n",i)
#     print(ds_tokenized[i][0])

# print(tokenizer.convert_ids_to_tokens(ds_tokenized[i][0]))

# 90% train, 10% test + validation
train_testvalid = ds_tokenized.train_test_split(test_size=0.2)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
ds = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

print(ds)

# Training
training_args = Seq2SeqTrainingArguments(checkpoint)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()


"""



class Train_model():


    def __init__(self, model_path, model_name, data_path):
        self.data_path = data_path
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path + model_name) # download bart to local and change path here self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart") # download bart to local and change path here 
        self.tokenizer = AutoTokenizer.from_pretrained(model_path + model_name) # download bart to local and change path here 
        self.training_args = Seq2SeqTrainingArguments(model_path + model_name) # download bart to local and change path here
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

    def read_file_lines(self, path):
        """INPUT:
        path:
            str  the path to the file 

        OUTPUT:
        dicts:
            {} a dict of the data in the file
 
        """
        with open(path) as f:
            dicts = json.load(f)
        return dicts
    

    def format_data(self, raw_inputs_path, raw_labels_path):
        """INPUT:
        raw_inputs_path:
            str the path to the data that will be used as input data
        raw_labels_path: 
            str the path to the file with the genreated proofs and labels
        
        OUTPUT:
        raw_ds:
            a dataset with the input together with the related generated labels
        """
        raw_inputs = self.read_file_lines(raw_inputs_path)
        raw_labels = self.read_file_lines(raw_labels_path)

        dict_list = []
        for input, target, in zip(raw_inputs, raw_labels):
            del input['depth']
            del input['label']

            d = {
                    "input": str(input),
                    "target": str(target),
                    "labels": target['label']
                }
            dict_list.append(d)

        raw_ds = Dataset.from_list(dict_list)
        return raw_ds
    

    def tokenize(self, dset):
        return self.tokenizer(text=dset["input"], text_target=dset["target"], truncation=True)


    def tokenize_data(self, raw_inputs_path, raw_labels_path):

        raw_ds = self.format_data(raw_inputs_path, raw_labels_path)
        tokenized_ds = raw_ds.map(self.tokenize, batched=True)

        return tokenized_ds


    def load_data(self):

        # self.data_path in ex format "LP/prop_exampels_all"

        train_data = self.tokenize_data(self.data_path + '_train.txt',  self.data_path + '_train_labels.txt')
        test_data = self.tokenize_data(self.data_path + '_test.txt',  self.data_path + '_test_labels.txt')
        val_data = self.tokenize_data(self.data_path + '_val.txt',  self.data_path + '_val_labels.txt')

        ds = DatasetDict({
            'train': train_data,
            'test': test_data,
            'valid': val_data})

        return ds


    def fine_tune_model(self, ds):

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["valid"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        trainer.train()


    def run_training(self):
        ds = self.load_data()
        self.fine_tune_model(ds)


    def run_evaluation(self):
        pass




if __name__ == "__main__":
    model_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/MODELS/EXAMPLE/"
    model_name = "pretrained_BART/"

    data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/prop_examples"

    TM = Train_model(model_path, model_name, data_path)
    TM.run_training()