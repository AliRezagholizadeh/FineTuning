from src.FineTuning.FunctionGemma.FullWeights.Mobile_Action import FuncGemma_MobileAction_FT
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.Evaluation.SmallEvaluations.evaluate import gemma_small_test
from src.Evaluation.SmallEvaluations.google_helper import get_scored_data_frame
from src.FineTuning.tools.LoadModel import Load_Hugging_Face_Model
from src.FineTuning.FunctionGemma.setup import finetuning_setup
from src.FineTuning.tools.LoadDataset import MobileActionsDS
from src.FineTuning.tools.utils import get_model_dir_path
from settings import run_settings
from datetime import datetime
from zoneinfo import ZoneInfo
from random import randint
from pathlib import Path
import logging
import json
import re



LOG_DIR = "logs/small_evaluation"
PROJECT_DIR = Path('.').absolute()
EVAL_DIR = PROJECT_DIR / "src" / "Evaluation"


def main():
    logdir = Path('.') / LOG_DIR
    logdir.mkdir(parents=True, exist_ok=True)
    datetime_now = f"{datetime.now(ZoneInfo('America/Toronto')).strftime('%Y-%m-%dT%H:%M:%S %Z')}"
    logging.basicConfig(filename=f'{logdir}/FineTuned_and_PreFineTuned_FuncGemm_MobileAct_{datetime_now}.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    # Create a logger for the main module
    logger = logging.getLogger(__name__)
    logger.info("main.py: To Fine-Tune Function Gemma with Mobile Action dataset. ")

    # load config
    import yaml
    with open("run_config.yaml", "r") as file:
        run_config = yaml.safe_load(file)
    # run settings
    run_config = run_settings(run_config)

    with open(EVAL_DIR / "eval_config.yaml", "r") as file:
        eval_config = yaml.safe_load(file)
        run_config.update(eval_config)

    run_config_gen = finetuning_setup(run_config,
                                       logger)  # {"access_token":, "model_name": ,"model_path":, "store_model_path":}
    run_config = next(run_config_gen)
    run_config_gen.send(None)

    FT_Settings = run_config
    # load tokenizer of the base model
    try:
        # self.FT_Settings = finetuning_setup(run_config, logger)  # {"access_token":, "model_name": ,"model_path":, "store_model_path":}

        base_model, processor, tokenizer, device = Load_Hugging_Face_Model(
            access_token=FT_Settings["access_token"], model_name=FT_Settings["model_hug_name"],
            model_path=FT_Settings["model_path"],
            AutoModelForCausalLM_setting=FT_Settings["AutoModelForCausalLM"], logger=logger)
        base_model.config.pad_token_id = tokenizer.pad_token_id

        message = f"tokenizer loaded."
        print(message)
        logger.info(f"evaluation_small.py: {message}")
    except Exception as e:
        logger.error(f"evaluation_small.py: error in loading tokenizer. {e}")
        raise e

    # load database to let us access to the tools and max_token_count.
    try:
        mobile_action = MobileActionsDS(run_config, tokenizer, logger)
        dataset = mobile_action.Load_Data()
        max_token_count = mobile_action.max_sequence_length()
        # train, valid
        train_dataset, eval_dataset = mobile_action.train_eval_split()
        logger.info(f"evaluation_small.py: mobile action dataset loaded and split to train and eval.")
    except Exception as e:
        logger.error(f"evaluation_small.py: error in loading mobile action dataset. {e}")
        raise e
    # try:
    #     funcGem_mobileAction_ft = FuncGemma_MobileAction_FT(run_config, logger)
    # except Exception as e:
    #     logger.error(f"Failed to instantiate the Function Gemma Fine Tuning: {e}")
    #     raise e

    # FT_Settings = funcGem_mobileAction_ft.FT_Settings



    # trained_gemma_model_dir = FT_Settings["store_model_path"]
    trained_gemma_model_dir = get_model_dir_path(FT_Settings, eval_fine_tuned = True, to_save=False)
    preFineTuned_gemma_model_dir = get_model_dir_path(FT_Settings,pre_finetuned_model = True, to_save=False)
    logger.info(f"main: fine-tuned model dir: {trained_gemma_model_dir} \n pre_fine-tuned model dir: {preFineTuned_gemma_model_dir}")

    # Reuse the tools from the sample
    # tools = json.loads(funcGem_mobileAction_ft.dataset[0]['text'])['tools']
    tools = json.loads(dataset[0]['text'])['tools']
    logger.info(f"main: tools loaded: {tools}")


    # load model and create pipeline
    # Fine-tuned pipeline
    trained_model = AutoModelForCausalLM.from_pretrained(trained_gemma_model_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(trained_gemma_model_dir)
    # new_pipe = pipeline("text-generation", model=trained_model, tokenizer=tokenizer)
    # load baseline
    base_pre_trained_model = AutoModelForCausalLM.from_pretrained(preFineTuned_gemma_model_dir, device_map="auto")
    base_tokenizer = AutoTokenizer.from_pretrained(preFineTuned_gemma_model_dir)
    # pipe_base = pipeline("text-generation", model=base_pre_trained_model, tokenizer=base_tokenizer, device_map="auto")


    # evaluate
    trained_scored = get_scored_data_frame(
        eval_dataset,
        pipeline("text-generation", model=trained_model, tokenizer=tokenizer, temperature=0.001)
    )
    base_scored = get_scored_data_frame(
        eval_dataset,
        pipeline("text-generation", model=base_pre_trained_model, tokenizer=base_tokenizer, device_map="auto", temperature=0.001),
    )



    # Compare the score of the base and fine-tuned models
    # Optional: save the score in json file
    comparison_dir = "comparison"
    trained_scored.to_json(f'{comparison_dir}/scored_df_20251215_trained.json')
    base_scored.to_json(f'{comparison_dir}/scored_df_20251215_base.json')

    print(f"\033[1mBase model score\033[0m       : {base_scored['correct'].mean()}")
    print(f"\033[1mFine-tuned model score\033[0m : {trained_scored['correct'].mean()}")



if __name__ == "__main__":
    main()