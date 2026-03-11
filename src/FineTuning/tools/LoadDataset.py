import json
from random import randint
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import json
import logging
from pathlib import Path

class MobileActionsDS:
    def __init__(self, setting, tokenizer, logger: logging = None):
        self.setting = setting
        self.tokenizer = tokenizer
        self.dataset = None



    def Load_Data(self):
        # repo_id = "google/mobile-actions"
        repo_id = self.setting["FineTune"]["Data"]["hug_base_data_name"]
        data_path = Path(self.setting["SolidConfig"]["DATA_BASE_DIR_PATH"]) / repo_id

        # load local pre-stored dataset or load it from remote repo
        if(data_path.is_dir()):
            self.tr_dataset = load_from_disk(data_path)
        else:
            data_file = hf_hub_download(repo_id= repo_id, filename="dataset.jsonl", repo_type="dataset")
            self.tr_dataset = load_dataset("text", data_files=data_file, encoding="utf-8")["train"].shuffle()

            self.tr_dataset.save_to_disk(data_path)

        # apply format
        self.processed_dataset = self.tr_dataset.map(self.apply_format)
        return self.tr_dataset

    def Sample_Data(self):
        if(self.dataset):
            return f"\n\033[1mHere's an example from your dataset:\033[0m \n{json.dumps(json.loads(self.dataset[randint(0, len(self.dataset) - 1)]['text']), indent=2)}"
        else:
            return f"First load data by calling Load_Data method."


    def max_sequence_length(self):
        if(self.processed_dataset):
            longest_example = max(self.processed_dataset, key=lambda example: len(example['prompt'] + example['completion']))
            longest_example_token_count = len(self.tokenizer.tokenize(longest_example['prompt'] + longest_example['completion']))
        else:
            raise ValueError("processed_dataset is empty. First call Load_Data.")

        max_token_count = longest_example_token_count + 100

        return max_token_count


    def apply_format(self, sample):
        template_iputs = json.loads(sample['text'])

        prompt_and_completion = self.tokenizer.apply_chat_template(
            template_iputs['messages'],
            tools=template_iputs['tools'],
            tokenize=False,
            # add_generation_prompt is False since we don't need model output after all
            # messages.
            add_generation_prompt=False)

        prompt = self.tokenizer.apply_chat_template(
            template_iputs['messages'][:-1],
            tools=template_iputs['tools'],
            tokenize=False,
            # add_generation_prompt is True since we would like to include
            # "<start_of_turn>model" in the prompt, if needed.
            add_generation_prompt=True)

        completion = prompt_and_completion[len(prompt):]

        return {
            "prompt": prompt,
            "completion": completion,
            "split": template_iputs["metadata"],
        }
    def train_eval_split(self):
        if (not self.processed_dataset):
            raise ValueError("processed_dataset is empty. First call Load_Data.")

        train_dataset = self.processed_dataset.filter(lambda example: example['split'] == 'train')
        eval_dataset = self.processed_dataset.filter(lambda example: example['split'] == 'eval')

        return train_dataset, eval_dataset