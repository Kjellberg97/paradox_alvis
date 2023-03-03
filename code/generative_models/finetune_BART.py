from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import DatasetDict, Dataset
from tqdm import tqdm
import torch
import json
import ast




class ProofGenerationModel():
    def __init__(self, model_path, model_name):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path + model_name) # download bart to local and change path here self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart") # download bart to local and change path here 
        self.tokenizer = AutoTokenizer.from_pretrained(model_path + model_name) # download bart to local and change path here 
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        self.training_args = Seq2SeqTrainingArguments(
            output_dir=model_path + model_name + 'OUTPUT',
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                weight_decay=0.01,
                save_total_limit=3,
                num_train_epochs=4,
            ) # download bart to local and change path here


    def read_file_lines(self, path):
        """ARGS:
        path (str):
            the path to the file 

        RETURN:
        dicts (dict):
             a dict of the data in the file
 
        """
        with open(path) as f:
            dicts = json.load(f)
        return dicts
    

    def format_data(self, raw_inputs_path, raw_labels_path):
        """ARGS:
        raw_inputs_path:
            str the path to the data that will be used as input data
        raw_labels_path: 
            str the path to the file with the genreated proofs and labels
        
        RETURN:
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

    def tokenize_inference(self, dset):
        return self.tokenizer(dset["input"], truncation=True, padding=True, return_tensors="pt")


    def tokenize_data(self, raw_inputs_path, raw_labels_path):
        """
        ARGS
        raw_input_path:
            str of the path to the input data
        raw_labels_path:
            str of the path to the labels of the input data
        
        RETURNS
        tokenized_ds:
            object of the tokenize data
        """
        raw_ds = self.format_data(raw_inputs_path, raw_labels_path)
        print("Formatting complete.\n")
        tokenized_ds = raw_ds.map(self.tokenize, batched=True, writer_batch_size=500)
        print("Mapping complete.")

        return tokenized_ds


    def load_all_data(self, data_path):
        """Loads and tokenizes data from the given file paths, and returns a Hugging Face DatasetDict object.

        ARGS:
        self (object): 
            An instance of the class that contains the `data_path` attribute and `tokenize_data()` method.

        RETURNS:
        ds (DatasetDict): 
            A Hugging Face DatasetDict object containing the tokenized train, test, and validation data.
        """
        # self.data_path in ex format "LP/prop_exampels_all"
        train_data = self.tokenize_data(data_path + '_train.txt',  data_path + '_train_labels.txt')
        val_data = self.tokenize_data(data_path + '_val.txt',  data_path + '_val_labels.txt')
        test_data = self.tokenize_data(data_path + '_test.txt',  data_path + '_test_labels.txt')

        print("Converting to dictionary.")
        ds = DatasetDict({
            'train': train_data,
            'test': test_data,
            'valid': val_data})
        
        print("Data loading complete.")

        return ds


    def load_one_data(self, data_path):
        data = self.tokenize_data(data_path + '.txt',  data_path + '_labels.txt')
        return data
        
    
    def load_checkpoint(self):
        pass


    def save_output(self, save_folder, output):
        save_path = save_folder + '/output.txt'
        print("Saving to ", save_path, "...", sep="")
        with open(save_path, 'a') as file:
            (json.dump(str(item) + '\n', file) for item in output)


    def run_training(self, ds):

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["valid"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            
        )
        trainer.train()


    def run_inference(self, test_data):
        # Generate outputs
        print("Inputs")
        try:
            print(test_data.keys())
        except:
            print(test_data)

        # Without batching
        inputs = self.tokenizer(test_data["input"], truncation=True, padding=True, return_tensors="pt").input_ids

        # With batching STILL NOT WORKING
        # ds = test_data.map(
        #     self.tokenize_inference, batched=True, writer_batch_size=500,
        #     batch_size=16, keep_in_memory=False, #drop_last_batch=True
        #     )
        # inputs = torch.tensor(ds["input_ids"])

        print(type(inputs))
        print(inputs)
        print(inputs.shape)

        print("Generating output...")
        outputs = []
        BATCH_SIZE = 16
        for i in tqdm(range(inputs.shape[0] // BATCH_SIZE + 1)):
            # Set the left and right slice
            idx_slice_left = i * BATCH_SIZE
            idx_slice_right = idx_slice_left + BATCH_SIZE
            if idx_slice_right > inputs.shape[0]:
                idx_slice_right = inputs.shape[0]
            
            # Generate batch and add to list
            generated_batch = self.model.generate(inputs[idx_slice_left:idx_slice_right], max_new_tokens=500, do_sample=True)
            outputs.append(generated_batch)
            if idx_slice_right == inputs.shape[0]: # break when we've generated last batch
                break

        print(outputs)
        # Decode into text
        print("Decoding")
        raw_output_text_list = [ self.tokenizer.batch_decode(out, skip_special_tokens=True) for out in outputs ] 
        print(raw_output_text_list)
        return raw_output_text_list
        
        # Converted the list of raw string generated labels into a list of dictionaries
        #text_dict_list = [ ast.literal_eval(sample) for sample in raw_output_text ]
        #return text_dict_list
        




if __name__ == "__main__":
    model_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/EXAMPLE/"
    model_name = "pretrained_BART/"
    data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/prop_examples"

    PGM = ProofGenerationModel(model_path, model_name)
    ds = PGM.load_all_data(data_path)
    PGM.run_training(ds)
