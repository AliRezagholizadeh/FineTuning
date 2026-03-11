"""
This module will use TRL (Transformers Reinforcement Learning) back source to fine-tuning.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from transformers.trainer_utils import get_last_checkpoint
from src.FineTuning.tools.LoadModel import Load_Hugging_Face_Model

import logging


class FullFineTune:
    def __init__(self, base_model: nn.Module, train_dataset, eval_dataset, tokenizer, FT_Settings: dict = None, logger: logging = None):
        """
        Parameters:
            dataset:
            FT_Settings: A settings obtained from setup.py
            logger:
        """
        self.base_model = base_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.settings = FT_Settings
        self.tokenizer = tokenizer
        self.logger = logger

        self.configure()




    def configure(self):
        self.args = SFTConfig(**self.settings["SFT"])

        print("Training configured")

    def run(self):
        # Train and save the fine-tuned model
        trainer = SFTTrainer(
            model=self.base_model,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )

        last_checkpoint = get_last_checkpoint(self.settings["SFT"]["output_dir"])

        # 2. Check if a checkpoint was found
        if last_checkpoint is not None:
            print(f"Found a checkpoint: {last_checkpoint}")
            # You can now use this path to resume training
            resume_from_checkpoint = True
        else:
            resume_from_checkpoint = False

        # trainer.train(**self.settings["SFT_Train"])
        trainer.train(resume_from_checkpoint = resume_from_checkpoint)

        trainer.save_model(self.settings["SFT"]["output_dir"])
        self.tokenizer.save_pretrained(self.settings["SFT"]["output_dir"])

        print(f"Fine-tuned model saved to {self.settings['SFT']['output_dir']}")
