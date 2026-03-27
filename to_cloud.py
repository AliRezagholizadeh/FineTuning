from huggingface_hub import ModelCard, ModelCardData, whoami, login
from settings import run_settings
from src.FineTuning.FunctionGemma.setup import finetuning_setup
from src.FineTuning.tools.utils import get_model_dir_path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from pathlib import Path

LOG_DIR = "logs/small_evaluation"
PROJECT_DIR = Path('.').absolute()
EVAL_DIR = PROJECT_DIR / "src" / "Evaluation"




def to_hugging_face(output_dir, base_model, trained_model_name):
    trained_model = AutoModelForCausalLM.from_pretrained(output_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(output_dir)



    username = whoami()['name']
    hf_repo_id = f"{username}/functiongemma-270m-it-{trained_model_name}"

    model_name = f"functiongemma-270m-it-{trained_model_name}"
    repo_url = trained_model.push_to_hub(hf_repo_id, create_repo=True, commit_message="Upload model")
    # repo_url = trained_model.push_to_hub(hf_repo_id, commit_message="Upload model")
    # repo_url = trained_model.push_to_hub(model_name)
    tokenizer.push_to_hub(hf_repo_id)
    # tokenizer.push_to_hub(model_name)

    card_content = f"""
    ---
    base_model: {base_model}
    tags:
    - function-calling
    - mobile-actions
    - gemma
    ---
    A fine-tuned model based on `{base_model}`."""
    card = ModelCard(card_content)

    card.push_to_hub(hf_repo_id)

    print(f"Uploaded to {repo_url}")


if __name__ == "__main__":
    # load config
    import yaml

    with open("run_config.yaml", "r") as file:
        run_config = yaml.safe_load(file)
    # run settings
    run_config = run_settings(run_config)

    with open(EVAL_DIR / "eval_config.yaml", "r") as file:
        eval_config = yaml.safe_load(file)
        run_config.update(eval_config)

    run_config_gen = finetuning_setup(run_config)  # {"access_token":, "model_name": ,"model_path":, "store_model_path":}
    run_config = next(run_config_gen)
    run_config_gen.send(None)

    FT_Settings = run_config

    login(run_config["access_token"])
    print("logged in")
    # trained_gemma_model_dir = FT_Settings["store_model_path"]
    trained_gemma_model_dir = get_model_dir_path(FT_Settings, eval_fine_tuned = True, to_save=False)
    print(f"main: fine-tuned model dir: {trained_gemma_model_dir}")

    # to store the fine-tuned model on hugging face
    # @markdown Name your model
    trained_model_name = "mobile-actions"  # @param {type:"string"}
    based_model = "google/functiongemma-270m-it"
    to_hugging_face(trained_gemma_model_dir, based_model, trained_model_name)