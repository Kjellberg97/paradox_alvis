from finetune_BART import ProofGenerationModel

# model_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/"
# model_name = "pretrained_BART"
# checkpoint = "checkpoint-22392"
# data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all"

model_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/"
model_name = "pretrained_BART"
checkpoint = "checkpoint-22392"
data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/RP/prop_examples_all_cleaned"

PGM = ProofGenerationModel(model_path, model_name, checkpoint)
print("Loading data...")
data = PGM.load_all_data(data_path)
print(data)
print("Running inference...")
predictions = PGM.run_inference(data["test"])
print("Saving output...")
PGM.save_output(predictions)

