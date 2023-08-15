from finetune_BART_step_by_step import StepsGenerationModel


def model_data_checkpoint_names(list_of_models, list_of_data, rule_sampling=True ):

    model_path_name = "MODELS/"
    data_path_name = "DATA/"
    args_list = []
    model_name = "gen_step_by_step"
    model_name += "_rule_sampling" if rule_sampling else ""

    for model in list_of_models:
        for data_path in list_of_data:
            args = {}

            args["Model path"] = model_path_name + model +"/"
            args["Data path"] = data_path_name + data_path 
            args["Model name"] = model_name

            if model == "LP":
                args["Checkpoint"] = "checkpoint-9000"
            elif model == "RP":
                args["Checkpoint"] = "checkpoint-7500"
            elif model == "RP_10X":
                args["Checkpoint"] = "checkpoint-7500"
            else:
                raise ValueError

            args_list.append(args)

    return args_list



if __name__ == "__main__":
    
    generate_on = "test"
    list_of_models = ["LP", "RP", "RP_10X"]

    list_of_data = ["LP/prop_examples_all_cleaned", 
                    "RP/prop_examples_all_cleaned", 
                    "RP_10X/prop_examples_all_balanced_rulenum_cleaned"]
    args_list = model_data_checkpoint_names(list_of_models, list_of_data)

    for args in args_list:
        print()
        for a in args:
            print(a, args[a])
        
        print()


        SGM = StepsGenerationModel(args["Model path"], args["Model name"], args["Checkpoint"])
        print("Running inference...")
        predictions = SGM.run_inference(args["Data path"], generate_on=generate_on)
        print("Saving output...")
        SGM.save_output(predictions)