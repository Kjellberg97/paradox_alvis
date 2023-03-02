from finetune_BART import ProofGenerationModel

model_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/EXAMPLE/"
model_name = "pretrained_BART/OUTPUT/checkpoint-4000/"
#data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/small"
data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/prop_examples"
output_path = "/output"

PGM = ProofGenerationModel(model_path, model_name)
print("Loading data...")
data = PGM.load_all_data(data_path)
print("Running inference...")
list_dict_output = PGM.run_inference(data["test"])
#print("Saving output...")
#PGM.save_output(output_path, list_dict_output)

