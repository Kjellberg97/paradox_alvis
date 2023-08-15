# README
This the instructional file to reproduce the code used in the article "Transformers for logical reasoning - Finding the causes of spurious correlations".

This readme provides instructions for generating the augmented dataset created in the paper and for performing inference with the SIP-BART Language Model. The augmented dataset are based on the original SimpleLogic dataset created by [Zhang et al. (2022)](https://arxiv.org/abs/2205.11502v2). The SIP-BART architecture is based on the BART transformer, originally concieved by [Lewis et al. (2019)](https://arxiv.org/abs/1910.13461).

The full dataset used for the training can be found in the map DATA and the trained models used in the report can be found in the map MODELS. If you want to skip data creation and training and instead use the provided trained models and the already created datasets you can jump to STEP 3 after completing STEP 0. From STEP 3 and forward it is described how to generate the inference steps for the proofs and how to evaluate the models.

## 0 USING THE CORRECT PYTHON VERSION AND LIBRARIES
To run this code Python 3.8.13 was used. All the libraries needed are provided in the "requirements/requirements.txt" file.

## 1 CREATING THE DATASETS
Note - this is only relevant if you want to recreate the data from scrach!  

To create the dataset we need to first generate the SimpleLogic LP, RP and RP_b (also referred to as RP_10X) dataset and then augment them with step-by-step proofs.

### 1.1 GENERATE SIMPLELOGIC
This will create the folder DATA with LP, RP and RP_10X  

- Create Rule Priority (RP)
    - `bash 1_generate_rp.bash`
- Create Label Priority (LP)
    - `bash 2_generate_lp.bash`
- Create Balanced Rule Priority (RP_b) 
    - `bash 4_generate_rp_balanced.bash`

After running these scripts you chould have a folder called DATA with three sub-folders called LP, RP and RP_10X. In each folder there should be a _train, _test and _val file. These should be renamed to "prop_examples_all_train.txt", "prop_examples_all_test.txt" and "prop_examples_all_val.txt". The train, test and validation files need to have identical names exept for the "_train.txt" / "_test.txt" / "_val.txt" extentions. 

For the rest of the code to work without a need to change varable names the name for LP and RP should be:

    prop_examples_all_cleaned_train.txt
    prop_examples_all_cleaned_test.txt
    prop_examples_all_cleaned_val.txt

And for RP_10X (named RP_b in the paper):

    prop_examples_all_balanced_rulenum_cleaned_train.txt 
    prop_examples_all_balanced_rulenum_cleaned_test.txt
    prop_examples_all_balanced_rulenum_cleaned_val.txt

The renaming needs to be done manualy!!!! 

### 1.2 REFORMAT THE INPUT
The input files need to be reformatted into the right format. This need to be done on all train, test and validation files for all datasets. 

In the file code\generate_target_labels\reformat.py the variables input_path_inputs and output_path_inputs should be equal to a list of the files that should be reformatted. Example for reformatting the RP dataset:

    input_path_inputs = [
        "DATA/LP/prop_examples_all_cleaned_train.txt", 
        "DATA/LP/prop_examples_all_cleaned_test.txt", 
        "DATA/LP/prop_examples_all_cleaned_val.txt",
        "DATA/RP/prop_examples_all_cleaned_train.txt", 
        "DATA/RP/prop_examples_all_cleaned_test.txt", 
        "DATA/RP/prop_examples_all_cleaned_val.txt",
        "DATA/RP_10X/prop_examples_all_balanced_rulenum_cleaned_train.txt", 
        "DATA/RP_10X/prop_examples_all_balanced_rulenum_cleaned_test.txt", 
        "DATA/RP_10X/prop_examples_all_balanced_rulenum_cleaned_val.txt"
        ]
    output_path_inputs = [
        "DATA/LP/prop_examples_all_cleaned_train.txt", 
        "DATA/LP/prop_examples_all_cleaned_test.txt", 
        "DATA/LP/prop_examples_all_cleaned_val.txt",
        "DATA/RP/prop_examples_all_cleaned_train.txt", 
        "DATA/RP/prop_examples_all_cleaned_test.txt", 
        "DATA/RP/prop_examples_all_cleaned_val.txt",
        "DATA/RP_10X/prop_examples_all_balanced_rulenum_cleaned_train.txt", 
        "DATA/RP_10X/prop_examples_all_balanced_rulenum_cleaned_test.txt", 
        "DATA/RP_10X/prop_examples_all_balanced_rulenum_cleaned_val.txt"
        ]
                        

Then run 
    
    python code\generate_target_labels\reformat.py

Note that the reformatting needs to be done for all datasets. 

### 1.3 GENERATE PROOFS FOR SIMPLELOGIC
In the file "code\gen_labels_step_by_step\run_label_generator_step_by_step.py"
Change the path_to_train to the subset and the file name of a file with all the data for that subset. The file name is without the .txt extention. Make sure the file names are correct before running. Can be left unchanged to replicate the result from the paper. This chould be done on the train, test and validation file for LP, RP and RP_b:

    paths_to_train = [["RP", "prop_examples_all_cleaned_train"],
                  ["RP", "prop_examples_all_cleaned_test"],
                  ["RP", "prop_examples_all_cleaned_val"],
                  ["LP", "prop_examples_all_cleaned_train"],
                  ["LP", "prop_examples_all_cleaned_test"],
                  ["LP", "prop_examples_all_cleaned_val"],
                  ["RP_10X", "prop_examples_all_balanced_rulenum_cleaned_test.txt"],
                  ["RP_10X", "prop_examples_all_balanced_rulenum_cleaned_train.txt"],
                  ["RP_10X", "prop_examples_all_balanced_rulenum_cleaned_val.txt"]]

Then run 

    python code\gen_labels_step_by_step\run_label_generator_step_by_step.py

This will create a file with the ground truth proof for the chosen files.  

## 2 TRAINING THE MODEL
In the file "code\generative_models\training_run_step_by_step.py" change the model_path to where you want to save the model.
Change the data_path to the path to the data. EXCLUDING the "_train.txt" / "_test.txt" / "_val.txt" extension. Train/test/val additions are provided within the .py file itself. The should be saved in a map with the same name as the dataset it has been trained on. To get the result from our paper; train a separate model for each of the datasets (LP, RP, RP_b).  

The model_path is where the trained model will be saved. Here is the settings for training a model on the LP dataset:

    model_path = "MODELS/LP/"
    data_path = "DATA/LP/prop_examples_all_cleaned"

Then run 

    python code\generative_models\training_run_step_by_step.py

For training a model on RP, change the following and run the script again:

    model_path = "MODELS/RP/"
    data_path = "DATA/RP/prop_examples_all_cleaned"

And for RP_b change to the following and run the script:

    model_path = "MODELS/RP_10X/"
    data_path = "DATA/RP/prop_examples_all_balanced_rulenum_cleaned"

## 3 RUN INFERENCE WITH CHOSEN MODEL
The file "code/generative_models/run_inference_step_by_step.py" is generating the proofs using the SIP-BART model 

You need to change the checkpoints variables for the script to be able to find the right checkpoint. This is doe by changing the variable:

    args["Checkpoint"] = needs to be equal to the checkpoint of the model you want to use. Note that the checkpoint for each of the models can vary. If the already trained models are used, this variable can be left unchanged.

In the main function change the variables:

    list_of_models = to the model you want to use. Leave unchanged if you want to run inference on all three models

    list_of_data = to the data you want to generate proofs to. The file name should not include the _train.txt/_test.txt/_val.txt extentions. Leave unchanged if you want to run inference on all test sets in all three models. 

Then run 
    
    python code\generative_models\finetune_BART_step_by_step.py

This will run inference on all data in list_of_data for all models in list_on_models. So if the arguments are left unchanged, this will create nine new outputs (three for each model). These will be stored in the OUTPUT folder for each model. Example for LP these will be stored under:
    
    MODELS\LP\gen_step_by_step_rule_sampling\OUTPUT

## 4 EVALUATE THE RESULTS
The evaluation of the accuracy and the consistency of the generated proofs from the models is done in the code\evaluation\proof_checker_step_by_step.py file. To run this file you first need to change the variable path in the function reformat_files to wherever you have the output from the inference. 

    path = path to where you saved the output from running inference. Leave unchanged if you have not changed any other path variables before.

Then in the main function change the variables

    models = A list of the name of the models you want to run. Leave unchanged if you want to reproduce the results from the paper. 

    test_on = A list of the names of the data the model has generated inference steps on. Leave unchanged if you want to reproduce the results from the paper. 

Then run:

    python code\evaluation\proof_checker_step_by_step.py


