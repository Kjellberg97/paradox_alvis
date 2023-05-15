from finetune_BART_step_by_step import StepsGenerationModel

# # model_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/"
# # model_name = "pretrained_BART"
# # checkpoint = "checkpoint-22392"
# # data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all"

# model_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/RP_10X/"
# model_name = "gen_step_by_step"
# rule_sampling = False 
# model_name += "_rule_sampling" if rule_sampling else ""
# checkpoint = "checkpoint-7500"
# #data_path ="/mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/small_cleaned"
# #data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all_cleaned"
# data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_cleaned"
# #data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/small1000_cleaned_reduced"


# SGM = StepsGenerationModel(model_path, model_name, checkpoint)
# print("Loading data...")
# #data = SGM.load_all_data(data_path)
# print("Running inference...")

# # BEAM SEARCH: beams > 1
# # GREEDY SEARCH: only data 
# # SAMPLE: sample = True


# """
# THE ARGS FOR GENERATE
# greedy decoding by calling greedy_search() if num_beams=1 and do_sample=False
# contrastive search by calling contrastive_search() if penalty_alpha>0. and top_k>1
# multinomial sampling by calling sample() if num_beams=1 and do_sample=True
# beam-search decoding by calling beam_search() if num_beams>1 and do_sample=False
# beam-search multinomial sampling by calling beam_sample() if num_beams>1 and do_sample=True
# diverse beam-search decoding by calling group_beam_search(), if num_beams>1 and num_beam_groups>1
# constrained beam-search decoding by calling constrained_beam_search(), if constraints!=None or force_words_ids!=None
# """


# predictions = SGM.run_inference(data_path, generate_on="test")
# print("Saving output...")
# SGM.save_output(predictions)


def model_data_checkpoint_names(list_of_models, list_of_data, rule_sampling=True ):

    model_path_name = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/"
    data_path_name = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/"
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
            elif model == "RP" or model == "RP_10X":
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
        print(args)

        SGM = StepsGenerationModel(args["Model path"], args["Model name"], args["Checkpoint"])
        print("Running inference...")
        predictions = SGM.run_inference(args["Data path"], generate_on=generate_on)
        print("Saving output...")
        SGM.save_output(predictions)