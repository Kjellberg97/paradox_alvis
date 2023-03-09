from finetune_BART import ProofGenerationModel

model_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/"
model_name = "pretrained_BART"
checkpoint = "checkpoint-22"
data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/small"

PGM = ProofGenerationModel(model_path, model_name, checkpoint)
ds = PGM.load_all_data(data_path)
PGM.run_training(ds)

