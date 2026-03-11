"""
The main script to manage fine-tuning. You might instantiate the relevant module wrapped over a model and a dataset.
By now, a module to handle the fine-tuning of Function Gemma on Mobile Actions dataset, has been deployed (by inspiring from
[Fine-tune FunctionGemma 270M for Mobile Actions Cookbook](https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb#scrollTo=vPN-DTopaUIy).
"""

from settings import run_settings
from src.FineTuning.FunctionGemma.FullWeights.Mobile_Action import FuncGemma_MobileAction_FT
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

LOG_DIR = "logs"



def FuncGemma_Finetuning():
    logdir = Path('.') / LOG_DIR
    logdir.mkdir(parents=True, exist_ok=True)
    datetime_now = f"{datetime.now(ZoneInfo('America/Toronto')).strftime('%Y-%m-%dT%H:%M:%S %Z')}"
    logging.basicConfig(filename=f'{logdir}/FineTune_FuncGemm_MobileAct_{datetime_now}.log', level=logging.INFO,
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

    print(f"Main: run_config: {run_config}")
    # instantiate the fine-tune class: Function Gemma with Mobile Action dataset
    funcGem_mobileAction_ft = FuncGemma_MobileAction_FT(run_config, logger)

    funcGem_mobileAction_ft.run()


if __name__ == "__main__":
    FuncGemma_Finetuning()
    # print(f"settings: {settings}")

