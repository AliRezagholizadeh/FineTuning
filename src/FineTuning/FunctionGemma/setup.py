# Make a large settings using configs, and env variable.
import copy
import logging
from src.FineTuning.tools.utils import get_model_dir_path
from dotenv import load_dotenv
from pathlib import Path
import os
import yaml

import os

MODULE_PATH = Path(os.path.abspath(__file__)).parent
print(f"FunctionsGemma_Path: {MODULE_PATH}")


# def find_model()


def finetuning_setup(run_config:dict, logger: logging = None):
    """

    Parameters:
        run_config:
        max_length:
        logger:
    Returns:

    """
    try:
        # load env variables within env.sample
        env_path = run_config["SolidConfig"]["ENV_DIR"]
        load_dotenv(dotenv_path = env_path)

        # use Hugging Face Access Token and login
        HUG_ACCESS_TOKEN_NAME = run_config["SolidConfig"]["HUGGINGFACE_ACCESS_TOKEN_ENV_NAME"]
        access_token = os.environ.get(HUG_ACCESS_TOKEN_NAME)
        # print("access_token: ", access_token)
        # run_config related parameters
        # determine model name
        # model_name = "google/functiongemma-270m-it"
        # hug_model_name = "litert-community/FunctionGemma_270M_Mobile_Actions"
        # hug_base_model_name = run_config["FineTune"]["Model"]["original"]["hug_base_model_name"]
        # 
        # 
        # # model path & save path
        # model_base_dir_path = run_config["MODEL_BASE_DIR_PATH"]

        

        # update with run_config
        ft_config_setup = copy.deepcopy(run_config)



        # get some parameters from fine-tune related configs: finetune_config.yaml
        with open(MODULE_PATH / "finetune_config.yaml", "r") as file:
            finetune_config = yaml.safe_load(file)

        if ("NumEpochs" in run_config["FineTune"]):
            finetune_config["SFT"].update({"num_train_epochs": run_config["FineTune"]["NumEpochs"]})

        if("report_to" in finetune_config["SFT"]):
            finetune_config["SFT"]["report_to"] = None
        finetune_config["SFT"]["learning_rate"] = float(finetune_config["SFT"]["learning_rate"])
        assert isinstance(finetune_config["SFT"]["completion_only_loss"], bool), "completion_only_loss assumed to be bool."

        if("PYTORCH_MPS_HIGH_WATERMARK_RATIO" in finetune_config["SFT"]):
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(finetune_config["SFT"]["PYTORCH_MPS_HIGH_WATERMARK_RATIO"])
            del finetune_config["SFT"]["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]


        # update with finetune_config
        # ft_setup.update({"SFT": finetune_config["SFT"]})
        # ft_setup.update({"AutoModelForCausalLM": finetune_config["AutoModelForCausalLM"]})
        ft_config_setup.update(finetune_config)


        # get stored model path
        # model_path = get_model_dir_path(hug_base_model_name, model_base_dir_path, pre_trained = True, to_save=False)
        model_path = get_model_dir_path(ft_config_setup, to_save=False)
        # store_model_path = get_model_dir_path(hug_base_model_name, model_base_dir_path, pre_trained=False, to_save=True)
        store_model_path = get_model_dir_path(ft_config_setup, to_save=True)

        # initiate ft_setup
        fine_tune_from = run_config["FineTune"]["FROM"]
        ft_config_setup.update({
            "access_token": access_token,
            "model_hug_name": run_config["FineTune"]["Model"][fine_tune_from]["hug_base_model_name"],
            "model_path": model_path,
            "store_model_path": store_model_path
        })

        ft_config_setup["SFT"]["output_dir"] = str(store_model_path)

        max_length = yield ft_config_setup
        
        if(max_length):
            print("Now adding max length")
            ft_config_setup["SFT"]["max_length"] = max_length



        yield ft_config_setup
    except Exception as e:
        if logger:
            logging.error(f"setup_finetuning unable to restore setting: {e}")
        raise e

    return "COMPLETED"
# FuncGemm_FT_Settings = finetuning_setup()

if __name__ == "__main__":
    project_path = Path(".").absolute().parent.parent.parent
    print(f"project_path: {project_path}")
    import sys
    sys.path.append(str(project_path))
    from settings import settings
    settings = finetuning_setup(settings, None)
    print(f"settings: {settings}")