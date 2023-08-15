from finetune_BART import ProofGenerationModel

model_path = "/MODELS/RP/"
model_name = "pretrained_BART"
checkpoint = "checkpoint-3500"
data_path = "/DATA/RP/prop_examples_all_cleaned"

PGM = ProofGenerationModel(model_path, model_name, checkpoint)
ds = PGM.load_all_data(data_path)
PGM.run_training(ds)
