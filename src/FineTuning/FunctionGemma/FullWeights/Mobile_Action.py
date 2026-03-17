"""
This Python script is going to fine tune the original [Function Gemma 270M](https://huggingface.co/google/functiongemma-270m-it)
with Mobile Action Datasets from [google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions).
"""
from src.FineTuning.FunctionGemma.setup import finetuning_setup
from src.FineTuning.tools.FineTune import FullFineTune
from src.FineTuning.tools.LoadDataset import MobileActionsDS
from src.FineTuning.tools.LoadModel import Load_Hugging_Face_Model
from transformers import pipeline
from random import randint
import re
import logging
import json


class FuncGemma_MobileAction_FT:
    def __init__(self, run_config, logger:logging = None):
        self.logger = logger
        self.fft = None

        FT_Settings_gen = finetuning_setup(run_config,
                                           logger)  # {"access_token":, "model_name": ,"model_path":, "store_model_path":}
        self.FT_Settings = next(FT_Settings_gen)

        # load model and relatives
        try:
            # self.FT_Settings = finetuning_setup(run_config, logger)  # {"access_token":, "model_name": ,"model_path":, "store_model_path":}

            print(f"FT_Settings: {self.FT_Settings}")
            self.base_model, self.processor, self.tokenizer, self.device = Load_Hugging_Face_Model(
                access_token=self.FT_Settings["access_token"], model_name=self.FT_Settings["model_hug_name"],
                model_path=self.FT_Settings["model_path"],
                AutoModelForCausalLM_setting=self.FT_Settings["AutoModelForCausalLM"], logger=self.logger)
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id

            message = f"Setup parameters and Model loaded. Device: {self.base_model.device} - DType:  {self.base_model.dtype}"
            print(message)
            logger.info(f"Mobile_Action.py: {message}")
        except Exception as e:
            logger.error(f"Mobile_Action.py: error in loading setups or loading the model. {e}")
            raise e


        # load database
        try:
            mobile_action = MobileActionsDS(run_config, self.tokenizer, logger)
            self.dataset = mobile_action.Load_Data()
            self.max_sequence_length = mobile_action.max_sequence_length()
                # train, valid
            self.train_dataset, self.eval_dataset = mobile_action.train_eval_split()
            logger.info(f"Mobile_Action.py: mobile action dataset loaded and split to train and eval.")
        except Exception as e:
            logger.error(f"Mobile_Action.py: error in loading mobile action dataset. {e}")
            raise e


        # get fine-tuning settings
        try:
            # self.FT_Settings = finetuning_setup(run_config, self.max_sequence_length,
            #                                     logger)  # {"access_token":, "model_name": ,"model_path":, "store_model_path":}
            self.FT_Settings = FT_Settings_gen.send(self.max_sequence_length)
            # instantiate FullFineTune
            self.fft = FullFineTune(base_model = self.base_model, train_dataset = self.train_dataset, eval_dataset= self.eval_dataset, tokenizer= self.tokenizer, FT_Settings= self.FT_Settings, logger= logger)
            logger.info(f"Mobile_Action.py: FullFineTune instantiated.")

            try:
                next(FT_Settings_gen)
            except StopIteration as e:
                logger.info(f"Mobile_Action.py: FT_Settings has been loaded completely: {self.FT_Settings}")
        except Exception as e:
            logger.error(f"Mobile_Action.py: error in instantiating the FullFineTune located in tools/FineTune.py. {e}")
            raise e


    def run(self):
        self.logger.info("Mobile_Action.py: to run FullFineTune.")
        if(self.fft):
            self.fft.run()
        else:
            raise ValueError("Mobile_Action.py: FullFineTune has not be instantiated.")


    def test_base_trian(self):
        # Create a transformers inference pipeline
        pipe = pipeline("text-generation", model=self.base_model, tokenizer=self.tokenizer)

        # Select a random sample from the test dataset
        rand_idx = randint(0, len(self.train_dataset) - 1)
        test_sample = self.train_dataset[rand_idx]

        input_prompt = test_sample['prompt']
        expected_output = test_sample['completion']

        # Generate the output
        output = pipe(input_prompt, max_new_tokens=self.max_sequence_length, skip_special_tokens=False)
        actual_output = output[0]['generated_text'][len(input_prompt):].strip()

        print(f"\n\033[1mInput prompt\033[0m   : {input_prompt}")
        print(f"\n\033[1mExpected output\033[0m: {expected_output}")
        print(f"\n\033[1mActual output\033[0m  : {actual_output}")