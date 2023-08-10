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

### 1.2 REFORMAT THE INPUT
The input files need to be reformatted into the right format. This need to be done on all train, test and validation files for all datasets. 

In the file reformat.py the variables input_path_inputs and output_path_inputs should be equal to a list of the files that should be reformatted. Example for reformatting the RP dataset:

    input_path_inputs = [
        "<path_to>/DATA/RP/prop_examples_all_train.txt", 
        "<path_to>/DATA/RP/prop_examples_all_test.txt", 
        "<path_to>/DATA/RP/prop_examples_all_val.txt"]
    output_path_inputs = [
        "<path_to>/DATA/RP/prop_examples_all_train.txt", 
        "<path_to>/DATA/RP/prop_examples_all_test.txt", 
        "<path_to>/DATA/RP/prop_examples_all_val.txt"
                        ]

Then run 
    
    python code\generate_target_labels\reformat.py

Note that the reformatting needs to be done for all datasets (RP, LP and RP_10X), not just for RP as in this example!

### 1.3 GENERATE PROOFS FOR SIMPLELOGIC
In the file "code\gen_labels_step_by_step\run_label_generator_step_by_step.py"
Change path to the DATA folder and change the path_to_train to the subset and the file name of a file with all the data for that subset. Example 
path = `"<path>/<to>/DATA"`:

    paths_to_train = [["RP", "prop_examples_all_train"],
                  ["RP", "prop_examples_all_test"],
                  ["RP", "prop_examples_all_val"]]

Then run 

    python code\gen_labels_step_by_step\run_label_generator_step_by_step.py

This will create a file with the ground truth proof for the chosen files.  

## 2 TRAINING THE MODEL
In the file "code\generative_models\training_run_step_by_step.py" change the model_path to where you want to save the model.
Change the data_path to the path to the data. EXCLUDING the "_train.txt" / "_test.txt" / "_val.txt" extension. Train/test/val additions are provided within the .py file itself. The should be saved in a map with the same name as the dataset it has been trained on.  

The model_path is where the trained model will be saved.

    model_path = "/<to>/<chosen>/<folder>/MODELS/LP/"
    data_path = "/<to>/<folder>/<with>/<data>/DATA/LP/prop_examples_all_cleaned"

Then run 

    python code\generative_models\training_run_step_by_step.py

## 3 RUN INFERENCE WITH CHOSEN MODEL
The file "code/generative_models/run_inference_step_by_step.py" is generating the proofs using the SIP-BART model 

In the model_data_checkpoint_names function change:
    model_path_name =  to the path to the map with the model. For example "MODELS/"

    data_path_name = to the map DATA. For example "DATA/"

You also need to change the checkpoints variables for the script to be able to find the right checkpoint. This is doe by changing the variable:

    args["Checkpoint"] = needs to be equal to the checkpoint of the model you want to use. Note that the checkpoint for each of the models can vary. If the already trained models are used, this variable can be left unchanged.

In the main function change the variables:

    list_of_models = to the model you want to use. Leave unchanged if you want to use the pre-trained models included in the repo

    list_of_data = to the data you want to generate proofs to. The file name should not include the _train.txt/_test.txt/_val.txt extentions. Leave unchanged if you want to use the data included in the repo.

Then run 
    
    python code\generative_models\finetune_BART_step_by_step.py



## 4 EVALUATE THE RESULTS
The evaluation of the accuracy and the consistency of the generated proofs from the models is done in the proof_checker_step_by_step.py file. To run this file you first need to change the variable path in the function reformat_files to whereever you vant to save the result. 

    path = where you want to save the result.

Then in the main function change the variables

    models = A list of the name of the models you want to run. Leave unchanged if you want to use the generated results in the repo. 

    test_on = A list of the names of the data the model has generated inference steps on. Leave unchanged if you want to use the generated results in the repo. 


