from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
import sys
import os

PROJECT_DIR = Path(os.path.dirname(__file__)).parent.parent.parent
print(f"PROJECT_DIR: {PROJECT_DIR}")
SRC_DIR = PROJECT_DIR / "src"
sys.path.append(str(SRC_DIR))
from FineTuning.tools.LoadModel import Load_Hugging_Face_Model


weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

developer_content = f"""You are a model that can do function calling with the following functions. 
    Current date and time given in YYYY-MM-DDTHH:MM:SS format: {datetime.now(ZoneInfo("America/Toronto")).strftime('%Y-%m-%dT%H:%M:%S %Z')}.
    Day of week is {weekdays[datetime.now().weekday()]}"""


def gemma_small_test(FT_Settings, trained_model_dir, base_gemma_model_name_dir, user_prompt: str, tools: list, max_token_count: int, logger: logging):
    assert user_prompt, "user_prompt should not be empty."
    assert tools, "tools should not be empty."
    print(f"To compare trained model located in {trained_model_dir} with the model located in {base_gemma_model_name_dir}.")


    # Create Transformers inference pipeline
    trained_model = AutoModelForCausalLM.from_pretrained(trained_model_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(trained_model_dir)
    pipe = pipeline("text-generation", model=trained_model, tokenizer=tokenizer)

    #load baseline
    # base_pre_trained_model = AutoModelForCausalLM.from_pretrained(base_gemma_model_dir, device_map="auto")
    # base_tokenizer = AutoTokenizer.from_pretrained(base_gemma_model_dir)
    # load model from local dir or download it from huggingface
    try:
        # self.FT_Settings = finetuning_setup(run_config, logger)  # {"access_token":, "model_name": ,"model_path":, "store_model_path":}

        # print(f"FT_Settings: {FT_Settings}")
        base_pre_trained_model, base_processor, base_tokenizer, base_device = Load_Hugging_Face_Model(
            access_token=FT_Settings["access_token"], model_name=base_gemma_model_name_dir[0],
            model_path=base_gemma_model_name_dir[1],
            AutoModelForCausalLM_setting=FT_Settings["AutoModelForCausalLM"], logger=logger)
        base_pre_trained_model.config.pad_token_id = tokenizer.pad_token_id

        message = f"Setup parameters and Model loaded. Device: {base_pre_trained_model.device} - DType:  {base_pre_trained_model.dtype}"
        print(message)
        logger.info(f"Mobile_Action.py: {message}")
    except Exception as e:
        logger.error(f"Mobile_Action.py: error in loading setups or loading the model. {e}")
        raise e

    if(logger):
        logger.info("gemma_small_test: pipeline set for both models (base and target model).")

    # build pipeline
    pipe_base = pipeline("text-generation", model=base_pre_trained_model,tokenizer = base_tokenizer, device_map="auto")


    # Test a prompt
    messages = [
        {"role": "developer",
         "content": "Current date and time given in YYYY-MM-DDTHH:MM:SS format: 2024-11-15T05:59:00. You are a model that can do function calling with the following functions"},
        {"role": "user", "content": user_prompt}
    ]



    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True)

    print(f"\n\033[1mPrompt:\033[0m {prompt}")
    output = pipe(prompt, max_new_tokens=max_token_count)
    output_base = pipe_base(prompt, max_new_tokens=max_token_count)
    model_output = output[0]['generated_text'][len(prompt):].strip()
    model_output_base = output_base[0]['generated_text'][len(prompt):].strip()


    print(f"\n\033[1mFine-tuned model output:\033[0m {model_output}")
    if(logger):
        logger.info(f"evaluate.py: gemma_small_test: \n\033[1mFine-tuned model output:\033[0m {model_output}")
        logger.info(f"evaluate.py: gemma_small_test: \n\033[1mBase model output:\033[0m {model_output_base}")

    print(f"\n\033[1mBase model output:\033[0m {model_output_base}")