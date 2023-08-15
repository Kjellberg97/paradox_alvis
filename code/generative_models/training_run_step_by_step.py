from finetune_BART_step_by_step import StepsGenerationModel

model_path = "MODELS/LP/"
model_name = "gen_step_by_step"
checkpoint = None
data_path = "DATA/LP/prop_examples_all_cleaned"

SGM = StepsGenerationModel(model_path, model_name, checkpoint, num_epochs=1,
                          evaluation_strategy="epoch", save_strategy="epoch", 
                          logging_steps=1000, rule_sampling=True)
ds = SGM.load_all_data(data_path)
SGM.run_training(ds)