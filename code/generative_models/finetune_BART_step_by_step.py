from finetune_BART import ProofGenerationModel
from datasets import Dataset, DatasetDict
import numpy as np
from time import time
import torch
from tqdm import tqdm


class StepsGenerationModel(ProofGenerationModel):

    def divide_step_by_step(self, inputs, labels):
        # Divide the input and labels into steps
        flat_list = []
        new_inputs, new_labels = [], []
        for inp, step_labels in zip(inputs, labels):
            inp = '1'.join(inp.split('1')[:-1]) + '1' # Remove everything after last '1'.
            new_inputs.append(inp)

            if self.use_divide_step_by_step:
                for lab in step_labels:
                    new_labels.append(lab)            

                    # The last lab is True/False which does not exist in the input
                    if lab != step_labels[-1]:
                        # Remove the fulfilled rule from next input
                        inp = inp.replace(lab, "")
                        # Remove leading and trailing whitespaces along with never having more than one ' ' in a row.
                        inp = ' '.join(inp.strip().split())
                        
                        # Split on comma to retrieve the fact, but not the last character which is a ':'
                        fact = lab.split(", ")[-1][:-1]

                        inp = inp + ' ' + fact + '1'
                        new_inputs.append(inp)
            
            else:
                new_labels.append(str(step_labels))

        return new_inputs, new_labels


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
        step_labels = self.read_file_lines(raw_labels_path)

        # Create list of strings
        step_inputs =  [ str(input["input"]) for input in raw_inputs ]

        # Process so that each step is one input, target pair
       
        step_inputs, step_labels = self.divide_step_by_step(step_inputs, step_labels)

        # Create Dataset from a list of dictionaries
        dict_list = []
        for step_input, step_target, in zip(step_inputs, step_labels):

            d = {
                    "input": step_input,
                    "target": step_target,
                }
            dict_list.append(d)

        raw_ds = Dataset.from_list(dict_list)
        return raw_ds



    def load_all_data(self, data_path, generate_on="test"):
        """Loads and tokenizes data from the given file paths, and returns a Hugging Face DatasetDict object.

        ARGS:
        self (object): 
            An instance of the class that contains the `data_path` attribute and `tokenize_data()` method.

        RETURNS:
        ds (DatasetDict): 
            A Hugging Face DatasetDict object containing the tokenized train, test, and validation data.
        """
        # self.data_path in ex format "LP/prop_exampels_all"

        self.data_path = data_path
    
        train_data = self.tokenize_data(data_path + '_train.txt',  data_path + '_train_step_labels.txt')
        if generate_on == "val":
            self.use_divide_step_by_step = False
        val_data = self.tokenize_data(data_path + '_val.txt',  data_path + '_val_step_labels.txt')
        if generate_on == "test":
            self.use_divide_step_by_step = False
        test_data = self.tokenize_data(data_path + '_test.txt',  data_path + '_test_step_labels.txt')


        print("Converting to dictionary.")
        ds = DatasetDict({
            'train': train_data,
            'valid': val_data,
            'test': test_data
            })
        
        print("Data loading complete.")

        return ds

    def compute_metrics(self, eval_pred):
        """
        Computes the accuracy for binary classification based on the predictions and labels.

        Args:
            eval_pred (EvalPrediction): An EvalPrediction object that contains the model predictions and labels.

        Returns:
            A dictionary with the key "accuracy" and its corresponding value.
        """
        preds, label_ids = eval_pred

        # Replace padding tokens with the pad_token_id and remove last token
        truths = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)[:, :-1]

        # Remove the start-of-sequence token
        preds = preds[:, 1:]

        # Replace end-of-sequence tokens with the pad_token_id
        preds = np.where(preds != self.tokenizer.eos_token, preds, self.tokenizer.pad_token_id)

        # Calculate accuracy by checking if each row of preds matches the corresponding row of truths
        acc = np.mean(np.all(preds == truths, axis=1))

        # Return a dictionary with the accuracy value
        result = {"accuracy": acc}
        return result


    def run_inference(self, data_path, generate_on="test",  beams=1, sample=False, 
                      penalty_alpha=0, top_k=1,num_beam_groups=1, 
                      constraints=None, force_words_ids=None  ):
        """Generates output of in testing data. The generation is done in fully based on 
        the input. The generation will loop through the input until a True or False label is generated.
        The function will also remove whatever the model genrates each step from the input and add the 
        new fact to the input

        ARGS: 
            test_data (list):
                the data that should be generated on
            beams (int):
                number of beams that are used in the generation should be 1 if not intended to use beam search
            sample (bool):
                True if you want to use smaple to generate
            penalty_alpha (float):
                used for contrastive search
            top_k (int):
                >1 if contrastive search
            num_beams_groups (int):
                >1 if group beam search should be used
            constrains (str):
                used for constrained beam search
            force_words_ids (str):
                used for constrained beam search
        
        RETURN:
            outputs(list):
                a list over the completed proofs for each input
        """

        """
        THE ARGS FOR GENERATE
        greedy decoding by calling greedy_search() if num_beams=1 and do_sample=False
        contrastive search by calling contrastive_search() if penalty_alpha>0. and top_k>1
        multinomial sampling by calling sample() if num_beams=1 and do_sample=True
        beam-search decoding by calling beam_search() if num_beams>1 and do_sample=False
        beam-search multinomial sampling by calling beam_sample() if num_beams>1 and do_sample=True
        diverse beam-search decoding by calling group_beam_search(), if num_beams>1 and num_beam_groups>1
        constrained beam-search decoding by calling constrained_beam_search(), if constraints!=None or force_words_ids!=None
        """

        data = SGM.load_all_data(data_path, generate_on)

        if generate_on == "test":
            test_data = data["test"]
        elif generate_on == "val":
            test_data = data["val"]

        # Generate outputs
        print("Inputs")
        device = torch.device("cuda")
        # Without batching
        
        inputs = self.tokenizer(test_data["input"], truncation=True, padding=True, return_tensors="pt").input_ids.to(device)
        self.model = self.model.to(device)
        print("Generating output...")
        outputs = []

        times_gen=[]
        times_tok_gen=[]
        times_tok_input=[]
        
        for i in tqdm(range(1, inputs.shape[0]+1)):
            inp = inputs[i-1:i]
            gen_steps = []
            # Generate batch and add to list
            ite = 0
            complete_proof = False
            #n_rules = inp.count(tok_colon)
            while not complete_proof and ite < 100:

                # Time for generating output
                tgs = time()
                gen_step = self.model.generate(inp, max_new_tokens=500, 
                                                do_sample=sample, num_beams=beams, penalty_alpha=penalty_alpha, 
                                                top_k=top_k, num_beam_groups=num_beam_groups, constraints=constraints, 
                                                force_words_ids=force_words_ids)
                times_gen.append(time() - tgs)

                # Take time for decoding output
                tgd = time()
                decoded_gen_step = self.tokenizer.batch_decode(gen_step, skip_special_tokens=True)
                times_tok_gen.append(time()-tgd)

                gen_steps.append(decoded_gen_step[0])

                if decoded_gen_step[0] == "True" or decoded_gen_step[0] == "False":
                    complete_proof = True

                   
                
                fact = decoded_gen_step[0].split(", ")[-1][:-1] # take after last ',' but do not include ':' at end of rule
                tdc_inp = time()

                # Removes the generated rule and adds the fact to the input
                decoded_inp = self.tokenizer.batch_decode(inp, skip_special_tokens=True)
                decoded_inp[0] = decoded_inp[0].replace(decoded_gen_step[0], '')
                decoded_inp[0] = decoded_inp[0] + ' ' + fact + '1'

                inp = self.tokenizer.encode(decoded_inp[0], return_tensors="pt").to(device)
                times_tok_input.append(time()- tdc_inp)
                
                ite+=1
            
            outputs.append(gen_steps)
        
        #outputs = [item for sublist in outputs for item in sublist]
        
        print("Avg times:")
        print("\ttime generate output:          ", np.mean(times_gen))
        print("\ttime to decode output:         ", np.mean(times_tok_gen))
        print("\ttimes to decode and code input:", np.mean(times_tok_input))
        
        print("Decoding")
        #raw_output_text_list = [ self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs ] 
        return outputs

   