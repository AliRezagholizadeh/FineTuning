from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datetime import datetime
from zoneinfo import ZoneInfo
import logging

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

developer_content = f"""You are a model that can do function calling with the following functions. 
    Current date and time given in YYYY-MM-DDTHH:MM:SS format: {datetime.now(ZoneInfo("America/Toronto")).strftime('%Y-%m-%dT%H:%M:%S %Z')}.
    Day of week is {weekdays[datetime.now().weekday()]}"""


def gemma_small_test(trained_model_dir, base_gemma_model_dir, user_prompt: str, tools: list, max_token_count: int, logger: logging):
    assert user_prompt, "user_prompt should not be empty."
    assert tools, "tools should not be empty."

    # Create Transformers inference pipeline
    trained_model = AutoModelForCausalLM.from_pretrained(trained_model_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(trained_model_dir)
    pipe = pipeline("text-generation", model=trained_model, tokenizer=tokenizer)
    #load baseline
    base_pre_trained_model = AutoModelForCausalLM.from_pretrained(base_gemma_model_dir, device_map="auto")
    base_tokenizer = AutoTokenizer.from_pretrained(base_gemma_model_dir)
    pipe_base = pipeline("text-generation", model=base_pre_trained_model,tokenizer = base_tokenizer, device_map="auto")

    
    if(logger):
        logger.info("gemma_small_test: pipeline set for both models (base and target model).")

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