from finetune_BART_step_by_step import StepsGenerationModel


model_path = "/mimer/NOBACKUP/groups/snic2022-22-744/MODELS/EXAMPLE/"
model_name = "gen_step_by_step"
checkpoint = None
data_path = "/mimer/NOBACKUP/groups/snic2022-22-744/DATA/EXAMPLE/small1000_cleaned_reduced"

SGM = StepsGenerationModel(model_path, model_name, checkpoint, num_epochs=10,
                          evaluation_strategy="steps", save_strategy="steps", 
                          logging_steps=50, random_sampling=True)
ds = SGM.load_all_data(data_path)
SGM.run_training(ds)