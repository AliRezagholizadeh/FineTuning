import json
from random import randint
from datasets import load_dataset, load_from_disk, Dataset
import datasets
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
        self.logger = logger



    def Load_Data(self):
        # repo_id = "google/mobile-actions"
        repo_id = self.setting["FineTune"]["Data"]["hug_base_data_name"]
        data_path = Path(self.setting["SolidConfig"]["DATA_BASE_DIR_PATH"]) / repo_id

        # load local pre-stored dataset or load it from remote repo
        if(data_path.is_dir()):
            self.dataset = load_from_disk(data_path)
        else:
            data_file = hf_hub_download(repo_id= repo_id, filename="dataset.jsonl", repo_type="dataset")
            # data_file = hf_hub_download(repo_id= repo_id, filename="data/train-00000-of-00001.parquet", repo_type="dataset")
            # data_file = hf_hub_download(repo_id= repo_id, repo_type="dataset")
            # self.dataset = load_dataset("text", data_files=data_file, encoding="utf-8")
            self.dataset = load_dataset("text", data_files=data_file)
            # self.dataset = load_dataset(repo_id)
            print(f"dataset type: {type(self.dataset)}")
            self.dataset = self.dataset["train"].shuffle()

            self.dataset.save_to_disk(data_path)

        # apply format
        print(f"dataset type: {type(self.dataset)}")
        print(f"dataset: {self.dataset}")
        # print(f"len process dataset text: {len(self.dataset['text'])}")
        self.processed_dataset = self.dataset.map(self.apply_format)
        # return self.dataset

    def Sample_Data(self):
        if(self.dataset):
            # return f"\n\033[1mHere's an example from your dataset:\033[0m \n{json.dumps(json.loads(self.dataset[randint(0, len(self.dataset) - 1)]['text']), indent=2)}"
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
        # print(f"sample: {sample}")
        template_iputs = json.loads(sample['text'])
        # print("sample: ", sample)
        # print("sample type: ", type(sample))

        # if(isinstance(sample, str)):
        #     template_iputs = json.loads(sample)
        # elif(isinstance(sample, dict)):
        #     if ("text" in sample):
        #         template_iputs = json.loads(sample['text'])
        #     else:
        #         raise Exception(f"Dict sample is not in a right format. 'text' expected to be in it, but is: {sample}")
        # elif(isinstance(sample, datasets.formatting.formatting.LazyRow)):
        #     # template_iputs = sample.to_dict()
        #     sample = dict(sample)
        #
        #     # template_iputs = json.loads(json.dumps(sample))
        #     # sample = json.loads(json.dumps(sample))
        #
        #     assert "messages" in sample, f"messages is not in sample: {sample}"
        #     assert "tools" in sample, f"tools is not in sample: {sample}"
        #
        #     # convert string elements of tool to dict
        #     sample["tools"] = list(sample["tools"] )
        #     if(isinstance(sample["tools"][0], str)):
        #         sample["tools"] = [json.loads(tool) for tool in sample["tools"]]
        #
        #     sample["messages"] = list(sample["messages"])
        #     if (isinstance(sample["messages"][0], str)):
        #         sample["messages"] = [json.loads(message) for message in sample["messages"]]
        #
        #     # sample["messages"] = json.loads(sample["messages"])
        #     # sample["tools"] = json.loads(sample["tools"])
        #     # template_iputs = sample
        # else:
        #
        #     print(f"Sample type Error: {type(sample)}")
        #     # template_iputs = sample
        #     raise Exception(f"Unrecognized sample type: {type(sample)}")

        assert isinstance(template_iputs, dict), f"Error: template_iputs type: {type(template_iputs)}"
        assert "messages" in template_iputs, f"messages is not in template_iputs: {template_iputs.keys()}"
        assert "tools" in template_iputs, f"tools is not in template_iputs: {template_iputs.keys()}"

        # print(f"template_iputs: {len(template_iputs)}")
        # print(f"messages type: {type(template_iputs['messages'])}")
        # print(f"tools type: {type(template_iputs['tools'])}")
        # print(f"tools: {template_iputs['tools']}")

        prompt_and_completion = self.tokenizer.apply_chat_template(
            template_iputs['messages'],
            tools=template_iputs['tools'],
            tokenize=False,
            # add_generation_prompt is False since we don't need model output after all
            # messages.
            add_generation_prompt=False)

        # print("to get prompt template..")
        prompt = self.tokenizer.apply_chat_template(
            template_iputs['messages'][:-1],
            tools=template_iputs['tools'],
            tokenize=False,
            # add_generation_prompt is True since we would like to include
            # "<start_of_turn>model" in the prompt, if needed.
            add_generation_prompt=True)

        # print("to get completion..")

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