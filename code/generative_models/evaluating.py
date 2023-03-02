from finetune_BART import ProofGenerationModel

model_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/EXAMPLE/"
model_name = "pretrained_BART/OUTPUT/checkpoint-4000/"
data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/prop_examples"
output_path = "/output"

PGM = ProofGenerationModel(model_path, model_name)
ds = PGM.load_data(data_path)
list_dict_output = PGM.run_inference(ds["valid"])
PGM.save_output()
