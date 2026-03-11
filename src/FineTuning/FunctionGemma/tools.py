from dotenv import load_dotenv
from pathlib import Path
import sys

PROJECT_DIR_PATH = Path(".").absolute().parent.parent.parent
assert "FineTuning" == str(PROJECT_DIR_PATH).split("/")[
    -1] and "str" != str(PROJECT_DIR_PATH).split("/")[-2], f"tools.py module couldn't find the main project Path. What found as project dir: {PROJECT_DIR_PATH}"

PACKAGE_DIR_PATH = PROJECT_DIR_PATH / "str" / "FineTuning"

print(f"PACKAGE_DIR_PATH: {PACKAGE_DIR_PATH}")
sys.path.append(str(PROJECT_DIR_PATH))
sys.path.append(str(PACKAGE_DIR_PATH))

from tools.LoadModel import Load_Hugging_Face_Model


def get_FuncGemma_model(model_name: str = None, model_base_dir: str = None, pre_trained: bool = True):
    '''
    Parameters:
        model_name: Function Gemma specific model name
        model_base_dir: base dirs from the current dir to store and load the model and its dependencies.
        pre_trained: if we want to store or load a pre-trained model, or a fine-tuned one
    Returns:

    '''



    # get stored model path
    model_path = get_model_dir_path(model_name, model_base_dir, pre_trained, to_save = False)
    store_model_path = get_model_dir_path(model_name, model_base_dir, pre_trained, to_save = True)


if __name__ == "__main__":
    poject_dir = Path(".").absolute().parent.parent.parent
    assert "FineTuning" == str(poject_dir).split("/")[-1]

    print(Path(".").absolute().parent.parent.parent)