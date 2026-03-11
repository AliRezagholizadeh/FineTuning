
from pathlib import Path
import os

PROJECT_DIR_PATH = Path(os.path.abspath(__file__)).parent
print(f"module_path: {PROJECT_DIR_PATH}")

def run_settings(run_config):
    # dirs
    assert "FineTuning" == str(PROJECT_DIR_PATH).split("/")[
        -1] and "str" != str(PROJECT_DIR_PATH).split("/")[
               -2], f"tools.py module couldn't find the main project Path. What found as project dir: {PROJECT_DIR_PATH}"

    MODEL_BASE_DIR_PATH = PROJECT_DIR_PATH / run_config["SolidConfig"]["MODEL_BASE_DIR_NAME"]
    FINETUNE_BASE_DIR_PATH = PROJECT_DIR_PATH / run_config["SolidConfig"]["FINETUNE_BASE_DIR_NAME"]
    DATA_BASE_DIR_PATH = PROJECT_DIR_PATH / run_config["SolidConfig"]["DATA_BASE_DIR_NAME"]
    ENV_BASE_DIR_PATH = PROJECT_DIR_PATH / run_config["SolidConfig"]["ENV_NAME"]
    EVAL_RESULTS_DIR_PATH = PROJECT_DIR_PATH / run_config["SolidConfig"].get("EVAL_RESULTS_DIR_NAME", "evaluation_results")

    settings_dict = {
        "PROJECT_DIR_PATH": str(PROJECT_DIR_PATH),
        "MODEL_BASE_DIR_PATH": str(MODEL_BASE_DIR_PATH),
        "FINETUNE_BASE_DIR_PATH": str(FINETUNE_BASE_DIR_PATH),
        "DATA_BASE_DIR_PATH": str(DATA_BASE_DIR_PATH),
        "EVAL_RESULTS_DIR_PATH": str(EVAL_RESULTS_DIR_PATH),
        "ENV_DIR": str(ENV_BASE_DIR_PATH)
    }

    run_config["SolidConfig"].update(settings_dict)
    return run_config




