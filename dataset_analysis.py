import pyarrow.feather as feather
from pathlib import Path
from datasets import load_from_disk
from settings import run_settings
from src.FineTuning.FunctionGemma.setup import finetuning_setup
from src.FineTuning.tools.LoadModel import Load_Hugging_Face_Model

from src.FineTuning.tools.LoadDataset import MobileActionsDS


PROJECT_DIR = Path('.')
DATASET_DIR = PROJECT_DIR / "datasets"
if __name__ == "__main__":
    dataset_name = "google/mobile-actions"
    
    dataset_file_path = DATASET_DIR / dataset_name

    # To read .arrow files and store them in csv format
    if(not (dataset_file_path / "dataset.csv").exists()):
        # load Hugging Face datasets.Dataset object to csv
        loaded_dataset = load_from_disk(dataset_file_path)

        # You can then convert it to pandas if needed:
        df = loaded_dataset.to_pandas()
    
        print(f"file: {dataset_file_path} \n df: {df}")
    
        df.to_csv(str(dataset_file_path / "dataset.csv"))

    # access to configs:
    import yaml

    with open("run_config.yaml", "r") as file:
        run_config = yaml.safe_load(file)
    # run settings
    run_config = run_settings(run_config)

    # with open(EVAL_DIR / "eval_config.yaml", "r") as file:
    #     eval_config = yaml.safe_load(file)
    #     run_config.update(eval_config)

    run_config_gen = finetuning_setup(run_config)  # {"access_token":, "model_name": ,"model_path":, "store_model_path":}
    run_config = next(run_config_gen)
    run_config_gen.send(None)

    FT_Settings = run_config
    # load tokenizer of the base model
    try:
        # self.FT_Settings = finetuning_setup(run_config, logger)  # {"access_token":, "model_name": ,"model_path":, "store_model_path":}

        base_model, processor, tokenizer, device = Load_Hugging_Face_Model(
            access_token=FT_Settings["access_token"], model_name=FT_Settings["model_hug_name"],
            model_path=FT_Settings["model_path"],
            AutoModelForCausalLM_setting=FT_Settings["AutoModelForCausalLM"])
        base_model.config.pad_token_id = tokenizer.pad_token_id

        message = f"tokenizer loaded."
        print(message)
        # logger.info(f"evaluation_small.py: {message}")
    except Exception as e:
        # logger.error(f"evaluation_small.py: error in loading tokenizer. {e}")
        raise e

    # load database to let us access to the tools and max_token_count.
    try:
        mobile_action = MobileActionsDS(run_config, tokenizer)
        dataset = mobile_action.Load_Data()
        max_token_count = mobile_action.max_sequence_length()
        # train, valid
        train_dataset, eval_dataset = mobile_action.train_eval_split()
        print(f"dataset: {dataset.shape} - type {type(dataset)}")
        print(f"train_dataset: {train_dataset.shape} - type {type(train_dataset)}")
        print(f"train_dataset: {train_dataset.shape} - type {type(eval_dataset)}")

        # sample:
        sample = dataset[0]
        print("touch first data as a sample")
        print(f"sample: {sample}")
        print("apply template:")
        [print(f"{key}: {value}") for key, value in mobile_action.apply_format(sample).items()]


        print(f"random sample: {mobile_action.Sample_Data()}")



        # logger.info(f"evaluation_small.py: mobile action dataset loaded and split to train and eval.")
    except Exception as e:
        # logger.error(f"evaluation_small.py: error in loading mobile action dataset. {e}")
        raise e