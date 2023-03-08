from finetune_BART import ProofGenerationModel

# model_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/"
# model_name = "pretrained_BART"
# checkpoint = "checkpoint-22392"
# data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all"

model_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/"
model_name = "pretrained_BART"
checkpoint = "checkpoint-24880"
data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all"

PGM = ProofGenerationModel(model_path, model_name, checkpoint)
print("Loading data...")
data = PGM.load_all_data(data_path)
print(data)
print("Running inference...")
predictions, label_ids, metrics = PGM.run_inference(data["test"])
print("Saving output...")
print(predictions)
PGM.save_output(predictions)

