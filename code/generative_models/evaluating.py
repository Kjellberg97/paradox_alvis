from finetune_BART import ProofGenerationModel

# model_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/"
# model_name = "pretrained_BART"
# checkpoint = "checkpoint-22392"
# data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all"

model_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/"
model_name = "pretrained_BART"
checkpoint = "checkpoint-22392"

#data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all_cleaned"
data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP_10X/prop_examples_all_balanced_rulenum_cleaned"
#data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all_cleaned"

PGM = ProofGenerationModel(model_path, model_name, checkpoint)
print("Loading data...")
data = PGM.load_all_data(data_path)
print(data)
print("Running inference...")



# BEAM SEARCH: beams > 1
# GREEDY SEARCH: only data 
# SAMPLE: sample = True

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

predictions = PGM.run_inference(data_path, generate_on="val")
print("Saving output...")
PGM.save_output(predictions)

