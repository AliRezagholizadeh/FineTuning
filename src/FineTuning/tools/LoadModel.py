from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
from pathlib import Path
import os
import logging



def Load_Hugging_Face_Model(access_token :str, model_name: str = None, model_path:str = None, AutoModelForCausalLM_setting: dict = {}, logger: logging = None):
    """
    To get the model, processor, and tokenizer of specific Function Gemma model, as well as device.

    Parameters:
        access_token: Hugging face API access key
        model_name: Function Gemma specific model name
        model_path: Path to the directory to store or load the pre-stored model
        logger: Logging instance to store the logs.
    Returns:
        model: Function Gemma model
        processor: Function Gemma processor
        tokenizer: Function Gemma tokenizer
        device: available device: cuda, mps, or cpu
    """

    # initialize
    processor = None
    model = None
    tokenizer = None

    # Load pre stored model. Load tokenizer first with use_fast=False to avoid
    # "Couldn't instantiate the backend tokenizer" (sentencepiece/tiktoken) errors.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        try:
            processor = AutoProcessor.from_pretrained(model_path, device_map="auto", tokenizer=tokenizer)
        except (TypeError, ValueError) as e:
            if isinstance(e, ValueError) and "tokenizer" not in str(e).lower() and "backend" not in str(e).lower():
                raise
            processor = tokenizer  # processor load failed (no tokenizer= support or backend error); use tokenizer as fallback
        model = AutoModelForCausalLM.from_pretrained(model_path, **AutoModelForCausalLM_setting)

    except Exception as e:
        if logger:
            logger.info(f"get_model: >> Loading the saved model from the path: {model_path}failed. message: {e}")
            logger.info(">> To load from Hugging Face repository... <<")

    if(not model or not processor):
        # CONNECT TO HUGGING FACE

        # login Hugging Face
        login(access_token)
        if logger:
            logger.info(">> Logged in Hugging Face. <<")

        # DOWNLOAD: tokenizer first with use_fast=False, then processor, then model
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        tokenizer.save_pretrained(model_path)
        try:
            processor = AutoProcessor.from_pretrained(model_name, device_map="auto", fix_mistral_regex=True, tokenizer=tokenizer)
        except (TypeError, ValueError) as e:
            if isinstance(e, ValueError) and "tokenizer" not in str(e).lower() and "backend" not in str(e).lower():
                raise
            processor = tokenizer
        processor.save_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_name, **AutoModelForCausalLM_setting)
        model.save_pretrained(model_path)

        if logger:
            logger.info(f">> Model and Processor ({model_name}) downloaded from Hugging Face and stored locally. <<")

    # set device
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    assert tokenizer!= None, "tokenizer not recognized"
    if logger:
        logger.info(f"model, processor, and device ({device}) found.")
    return model, processor, tokenizer, device
