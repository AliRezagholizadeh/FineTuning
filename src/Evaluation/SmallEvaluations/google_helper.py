# used from https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb#scrollTo=pMY1P8Rtyopt

import pandas as pd
from random import randint
from pathlib import Path
import json
import re

def extract_function_call(model_output):
    """
    Parses a string containing specific function call markers and returns
    a list of function call objects. Here is an example of the obejct:

    ```
    <start_function_call>call:open_map{query:<escape>San Francisco<escape>}<end_function_call>
    ```

    Args:
        model_output (str): The model output string.

    Returns:
        list: A list of dictionaries representing the function calls.
    """
    results = []

    # Pattern to extract the full content of a single function call
    # Flags: DOTALL allows matching across newlines if necessary
    call_pattern = r"<start_function_call>(.*?)<end_function_call>"
    raw_calls = re.findall(call_pattern, model_output, re.DOTALL)

    for raw_call in raw_calls:
        # Check if the content starts with 'call:'
        if not raw_call.strip().startswith("call:"):
            continue

        # Extract function name
        # Expected format: call:func_name{...}
        try:
            # Split only on the first brace to separate name and args
            pre_brace, args_segment = raw_call.split("{", 1)

            function_name = pre_brace.replace("call:", "").strip()

            # Remove the trailing closing brace '}'
            args_content = args_segment.strip()
            if args_content.endswith("}"):
                args_content = args_content[:-1]

            arguments = {}

            # Pattern to extract arguments
            # Looks for: key:<escape>value<escape>
            # The key pattern [^:,]* ensures we don't accidentally eat previous commas
            arg_pattern = r"(?P<key>[^:,]*?):<escape>(?P<value>.*?)<escape>"

            arg_matches = re.finditer(arg_pattern, args_content, re.DOTALL)

            for match in arg_matches:
                key = match.group("key").strip()
                value = match.group("value")
                arguments[key] = value

            results.append({
                "function": {
                    "name": function_name,
                    "arguments": arguments
                }
            })

        except ValueError:
            # Handles cases where syntax might be malformed (e.g., missing '{')
            continue

    return results

def extract_text(model_output):
    """
    Extracts text content and removing the <end_of_turn> marker.

    Args:
        model_output (str): The model output string.

    Returns:
        str: The cleaned text.
    """
    if not model_output or model_output.startswith("<start_function_call>"):
        return None
    return model_output.replace("<end_of_turn>", "").strip()

from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

def get_eval_logs(dataset, pipe):
  batch_size = 1
  logs = []
  # Select a random sample from the test dataset
  for i, output in enumerate(pipe(KeyDataset(dataset, "prompt"), batch_size=batch_size)):
    orig_data = dataset[i]['text']
    messages = json.loads(orig_data)['messages']
    user_message = messages[1]
    assistant_first_message = messages[2]
    input_prompt = dataset[i]['prompt']
    # Generate the output
    model_output_only = output[0]['generated_text'][len(input_prompt):].strip()

    logs.append(
        {
            # The original user prompt/query.
            "user": user_message['content'],

            # List of ground truth function call objects.
            "target_fc": assistant_first_message.get('tool_calls', []),

            # Ground truth text response.
            "target_text": assistant_first_message.get('content'),

            # List of model-generated function call objects.
            "output_fc": extract_function_call(model_output_only),

            # Model-generated text response.
            "output_text": extract_text(model_output_only),
        }
    )

    if (i + 1) % batch_size == 0:
      print(f"Eval process: {(i + 1) * 100.0 / len(dataset):.2f}%")
  return logs

def get_scored_data_frame(dataset, pipe):
  logs = get_eval_logs(dataset, pipe)
  logs_df = pd.DataFrame.from_records(logs)

  scored = pd.DataFrame()
  scored['user'] = logs_df['user']
  scored['target_names'] = logs_df['target_fc'].apply(lambda x: [fc['function']['name'] for fc in x])
  scored['output_names'] = logs_df['output_fc'].apply(lambda x: [fc['function']['name'] for fc in x])
  scored["target_arguments"] = logs_df['target_fc'].apply(lambda x: [dict(sorted(fc['function']['arguments'].items())) for fc in x])
  scored["output_arguments"] = logs_df['output_fc'].apply(lambda x: [dict(sorted(fc['function']['arguments'].items())) for fc in x])
  scored['target_text'] = logs_df['target_text']
  scored['output_text'] = logs_df['output_text']
  scored["correct_names"] = scored["target_names"] == scored["output_names"]
  scored["correct_arguments"] = scored["target_arguments"] == scored["output_arguments"]
  scored["correct"] = scored["correct_names"] & scored["correct_arguments"]

  return scored

def review(scored):
  scored["incorrect_names"] = scored["target_names"] != scored["output_names"]
  scored["incorrect_arguments"] = scored["target_arguments"] != scored["output_arguments"]
  scored["incorrect"] = scored["incorrect_names"] | scored["incorrect_arguments"]

  for index, row in scored[scored["incorrect"]].iterrows():
    print(f"\033[1mSample #{index} prompt  \033[0m: {row['user']}")
    print(f"\033[1mSample #{index} expected\033[0m: {row['target_names']}, {row['target_arguments']}")
    print(f"\033[1mSample #{index} actual  \033[0m: {row['output_names']}, {row['output_arguments']}")
    print("---------------")
