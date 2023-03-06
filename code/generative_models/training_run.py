from finetune_BART import ProofGenerationModel

model_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/LP/"
model_name = "pretrained_BART/OUTPUT/checkpoint-28000"
data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/LP/prop_examples_all"

PGM = ProofGenerationModel(model_path, model_name)
ds = PGM.load_all_data(data_path)
PGM.run_training(ds)

