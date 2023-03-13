from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import DatasetDict, Dataset, load_metric
from tqdm import tqdm
import numpy as np
import re
import torch
import json
import ast
import os
import torch.multiprocessing as mp
import torch.distributed as dist
#from accelerate import Accelerator

                

class ProofGenerationModel():
    def __init__(self, model_path, model_name, checkpoint=None):

        self.checkpoint = checkpoint
        self.load_from_checkpoint = self.load_checkpoint(checkpoint)
        self.model_path= model_path
        self.model_name = model_name 
        if checkpoint:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path + model_name +"/OUTPUT/" + checkpoint + "/") # download bart to local and change path here self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart") # download bart to local and change path here 
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path + model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path + model_name) # download bart to local and change path here 
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        self.metric_acc = load_metric("accuracy")
        self.metric_f1 = load_metric("f1")
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=model_path + "pretrained_BART/" + 'OUTPUT',
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=10,
                per_device_eval_batch_size=10, # 10 innan
                gradient_accumulation_steps=6, # 32 innan och ingen prediction_loss_only
                prediction_loss_only=False, # Saving less information during evaluation, perhaps less memory usage
                fp16=True, # Less accurace floats when training
                #logging_steps=500,
                save_strategy="epoch",
                load_best_model_at_end=True,
                warmup_steps =200,
                weight_decay=0.01,
                save_total_limit=5,
                num_train_epochs=2,
                predict_with_generate=True,
                generation_max_length=1024, # generated tokens if predict_with_generate=True
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
  
        
    # Define a function that takes a file name as an argument
    # def read_info(file_name):
    #     # Use list comprehension to read and process each line in the file
    #     info_list = [line.strip().split(",") for line in open(file_name, "r")]
    #     # Return the info list
    #     return info_list
        

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
        print(raw_inputs_path)
        raw_inputs = self.read_file_lines(raw_inputs_path)
        print(raw_labels_path)
        raw_labels = self.read_file_lines(raw_labels_path)

        dict_list = []
        for input, target, in zip(raw_inputs, raw_labels):

            d = {
                    "input": str(input["input"]),
                    "target": str(target),
                    "labels": input["label"]
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
            'valid': val_data,
            'test': test_data
            })
        
        print("Data loading complete.")

        return ds


    def load_one_data(self, data_path):
        data = self.tokenize_data(data_path + '.txt',  data_path + '_labels.txt')
        return data
        
    
    def load_checkpoint(self, checkpoint):
        if checkpoint == None:
            return False
        else:
            return True


    def save_output(self, output):
        # Get correct path
        if self.checkpoint:
            save_path = self.model_path + self.model_name + "/evaluation/" + self.checkpoint + '_output.txt'
        else:
            # checkpoint_0 is when we dont use any checkpoint
            save_path = self.model_path + self.model_name + "/evaluation/checkpoint_0_output.txt"
           
        print("Saving to ", save_path, "...", sep="")

        # Remove any previous file¨
        open(save_path, 'w').close()

        print("Output length", len(output))

        # Save output
        with open(save_path, 'a') as file:
            file.write("[")
            for i, item in tqdm(enumerate(output)):
                print("Item length", len(item))
                json.dump(item, file)
                if i < len(output) - 1:
                    file.write(",\n")
            file.write("]")

    def find_binary_label(self, string):
        # Find the last occurence of False or True in the string, convert into corresponding int 0 or 1
        match = re.search(r"True|False(?!.*True|False)", string) # Not followed by any characters (.* , and not followed by True|False
        binary_digit = int(eval(match.group())) if match else 0 # Convert into int if a False or True is returned else convert to 0
        return binary_digit

    def compute_metrics(self, eval_pred):
        """
        Computes the accuracy and F1 score for binary classification based on the predictions and labels.
            Args: eval_pred: an EvalPrediction object that contains the model predictions and labels.
            Returns: A dictionary with the keys "accuracy" and "f1" and their corresponding values.
        """
        predictions, label_ids = eval_pred
        label_ids = np.where(label_ids != -100,  label_ids, self.tokenizer.pad_token_id) # remove padding

        # Decode
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_label_ids = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Extract the binary digit labels
        preds = [ self.find_binary_label(pred_str) for pred_str in decoded_preds ]
        truths = [ self.find_binary_label(truth_str) for truth_str in decoded_label_ids ]
        
        # Compute the results
        acc = self.metric_acc.compute(predictions=preds, references=truths)["accuracy"]
        f1 = self.metric_f1.compute(predictions=preds, references=truths)["f1"]
        result = {"accuracy": acc, "f1": f1}
        return result

    def run_training(self, ds):
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["valid"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            
        )
        if self.load_from_checkpoint:
            trainer.train(self.model_path + self.model_name)
        else:
            trainer.train()



    def run_inference(self, test_data):

        # Generate outputs
        print("Inputs")
        device = torch.device("cuda")
        # Without batching
        
        inputs = self.tokenizer(test_data["input"], truncation=True, padding=True, return_tensors="pt").input_ids.to(device)

        # With batching STILL NOT WORKING
        # ds = test_data.map(
        #     self.tokenize_inference, batched=True, writer_batch_size=500,
        #     batch_size=16, keep_in_memory=False, #drop_last_batch=True
        #     )
        # inputs = torch.tensor(ds["input_ids"])

        #print(type(inputs))
        #print(inputs)
        #print(inputs.shape)

        print("Generating output...")
        outputs = []
        
        # ONLY RUNS ON ONE GPU!!!!!!!!!

        self.model = self.model.to(device)
        
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
        
        outputs = [item for sublist in outputs for item in sublist]

        #print(outputs)
        # Decode into text
        print("Decoding")
        raw_output_text_list = [ self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs ] 
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
