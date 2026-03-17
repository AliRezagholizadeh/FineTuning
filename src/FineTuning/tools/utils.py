import yaml
from pathlib import Path
import copy


def get_model_dir_path(config, eval_fine_tuned = False, pre_finetuned_model = False, to_save: bool = False) -> Path:
    """
    
    :param config: 
    :param eval_fine_tuned: To access fined-tuned model for the evaluation. It will access Evaluate parameter in the config to access specific hyper_parameters.
    :param pre_trained_model: To access PreFineTuned model.
    :param to_save: 
    :return: 
    """
    assert "FROM" in config["FineTune"], "From which base dir name is required."
    fine_tune_from = config["FineTune"]["FROM"]

    if(to_save or eval_fine_tuned): # To access fine-tune dir to save or access it for evaluation
        if(not eval_fine_tuned):
            # store dir
            current_finetune_dir = Path(config["SolidConfig"]["FINETUNE_BASE_DIR_PATH"]) / fine_tune_from / \
                                   config["FineTune"]["Model"][fine_tune_from]["hug_base_model_name"] / \
                                   config["FineTune"]["Data"]["name"] / config["FineTune"]["FineTuneName"]
            # make dir
            if (not current_finetune_dir.is_dir()):
                current_finetune_dir.mkdir(parents=True, exist_ok=True)

            # searching for the same hyperparameters config
            not_from_base = [f"FineTune/Model/{base_type}" for base_type in config["FineTune"]["Model"].keys() if
                             base_type != fine_tune_from]

            except_keys = [
                "AutoModelForCausalLM",
                "SFT",
                "SolidConfig",
                "FineTune/FineTuneName",
                "FineTune/FROM",
                "FineTune/NumEpochs",
                "FineTune/Model/PreFineTuned",
            ] + not_from_base
            # call
            current_finetune_dir = find_model_conf_dir(current_finetune_dir, config, except_keys)
        else: # to access fine-tuned model for evaluation
            # store dir
            assert "PreFineTuned" in config["FineTune"]["Model"], "PreFineTuned not found."
            current_finetune_dir = Path(config["SolidConfig"]["FINETUNE_BASE_DIR_PATH"]) / fine_tune_from / \
                                   config["FineTune"]["Model"][fine_tune_from]["hug_base_model_name"] / \
                                   config["FineTune"]["Data"]["name"] / config["FineTune"]["FineTuneName"]

            current_finetune_dir = current_finetune_dir / config['Evaluate']['path']['hyperparameter_name']
            assert current_finetune_dir.is_dir(), f"PreFineTuned model is not available: {current_finetune_dir}"
        return current_finetune_dir
    else: # get base model dir
        
        if(pre_finetuned_model): # get per-fine-tuned model as a base model
            base_model_dir = Path(config["SolidConfig"]["MODEL_BASE_DIR_PATH"]) / "PreFineTuned" / \
                             config["FineTune"]["Model"]["PreFineTuned"]["hug_base_model_name"]
        else:
            # to base model
            base_model_dir = Path(config["SolidConfig"]["MODEL_BASE_DIR_PATH"]) / fine_tune_from / config["FineTune"]["Model"][fine_tune_from][
                "hug_base_model_name"]

        # make dir
        if (not base_model_dir.is_dir()):
            base_model_dir.mkdir(parents=True, exist_ok=True)


        return base_model_dir




def get_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.load(file, yaml.SafeLoader)


    return config



def del_dict_value(dict_: dict, key: str) -> None:
    """Delete value from a dictionary.

    Parameters:
        key: can be a nested variable, like: model/preload

    """
    nested_values = key.split("/")
    print(f"nested_values: {nested_values}")

    a = dict_
    for i in range(len(nested_values) - 1):
        print(f"- nested_values:{nested_values}")

        if(isinstance(a, dict)):
            a = a.get(nested_values[0], None)

        del nested_values[0]

    if(isinstance(a, dict) and a.get(nested_values[0])):
        del a[nested_values[0]]

def prune_dict(dictn: dict, base_keys, except_keys: list, parent_keys: str = "") -> None:
    """A recursive function to prune a copy of the dictionary to keep base_keys and delete except_keys.

    Parameters:
        base_keys, except_keys: a list of strings to store keys or nested keys (ex. key1/n_key1).
        parent_keys: a string to keep track of the nested keys
    """
    # if(parent_keys == ""): # make deep copy at the first call of the prune_dict.
    #     dict_ = deepcopy(dictn)
    # else:
    #     dict_ = dictn
    dict_ = copy.deepcopy(dictn)
    # iterate over the keys and call the function if nested dict found
    for key in dictn:
        if(isinstance(dict_[key], dict)): # nested key
            if((f"{parent_keys}{key}" in except_keys) or (f"{parent_keys}{key}" not in base_keys)):
                del dict_[key]
            else:
                dict_[key] = prune_dict(dict_[key], base_keys, except_keys, f"{parent_keys}{key}/")
        else:
            if ((f"{parent_keys}{key}" in except_keys) or (f"{parent_keys}{key}" not in base_keys)):
                print(f"'{parent_keys}{key}' is either in except ({except_keys}), or not in base_keys {base_keys} ---> So it will be deleted.")
                del dict_[key]


    return dict_


def find_dict_keys(dictn: dict, keys_found: list = [], parent_keys:str = ""):
    """
    A recursive function to find and store all keys and nested keys in keys_found.
    """
    for key in dictn:
        if(isinstance(dictn[key], dict)): # nested key
            keys_found.append(f"{parent_keys}{key}")
            keys_found = find_dict_keys(dictn[key], keys_found, f"{parent_keys}{key}/")
        else:
            keys_found.append(f"{parent_keys}{key}")

    return keys_found

def config_identical(base_config, model_config, except_keys):
    """
    Check if both configs are the same (considering base_config keys as pivot point, except some keys).

    Return:
        Boolean of whether two config is the same (with the mentioned considerations)
    """

    # find base keys given the base config
    base_keys = find_dict_keys(base_config)
    print(f"base_keys found: {base_keys}")
    model_config_pr = prune_dict(model_config, base_keys, except_keys)
    config_pr = prune_dict(base_config, base_keys, except_keys)

    print(f"model_config_pr: {model_config_pr}")
    print(f"config_pr: {config_pr}")

    return model_config_pr == config_pr

def write_yaml_config(config, file):
    with open(file, 'w') as conf_file:
        yaml.dump(config, conf_file, default_flow_style=False, sort_keys=False)




def find_model_conf_dir(base_dir, config, except_keys):
    """
    Find corresponding sub dir in the base_dir directory to access the model with the same hyperparameters
    (some parameters in the config file).
    If it does not exist in the base_dir directory, it will create a new sub dir (corresponding to a new hyperparameters)
     and copy the input config.

    Parameters:
        base_dir: the base directory which is going to be iterate over.
        config: This is the general config located in original project dir.
        except_keys: elements to be ignored while searching for the same config.
    Return:
        config: might need to be updated.
    """
    # epoch = None
    # except_keys = ["model/last_epoch", "model/model_dir", "config_dir", "experiment/name", "experiment/dir", "num_epochs"]
    # mask = []

    acceptable_suffix = ['.yaml', '.yml']

    # iterate over sub-directories
    model_config = None
    model_dir = None
    model_indx = 1
    if(subdirs:=base_dir.iterdir()):
        for subdir in subdirs:
            print(f"check if the sub dir ({str(subdir)}) has the identical config.")
            # check config
            if(subdir.is_dir() and (modelfiles:= subdir.iterdir())):
                for file_ in modelfiles:
                    print(f"check this sub dir/file {file_}")
                    # check the config exists (considering the main parameters).
                    # if(file_.is_file() and str(file_).split('/')[-1] == config["yml_config_file_name"]):
                    if(file_.is_file() and file_.suffix in acceptable_suffix):
                        model_config = get_config(file_)

                        # check if the config found is the same as the original config (considering the main parameters).
                        if(config_identical(config, model_config, except_keys)):
                            model_dir = file_.parent
                            print(f"⚠️ The same config to the current one found. So we use this path to write output: {model_dir}")

                            config = model_config
                            print(f"config updated by the model_config items. new config: {config}")

                            break
                        else:
                            print("-- The config here is not the same as the base config.")

            model_indx += 1


    # If a right model dir not found, create and copy the current config
    if(not model_dir):
        print("✅ New root made for storing the model. ")
        model_dir = base_dir / f"hyperP_setting_{model_indx}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # # set experiment dirs/sub dirs
        # exp_base_dir = Path('.') / config["experiment"]["dir"] / config["experiment"]["name"]
        #
        # # mkdir if required
        # if (not exp_base_dir.is_dir()):
        #     exp_base_dir.mkdir(parents=True, exist_ok=True)
        #
        # exp_dir = exp_base_dir /  f"hyperP_setting_{model_indx}"
        # exp_dir.mkdir(parents=True, exist_ok=True)

        # # add model/model_dir and config_dir
        # config["model"].update({"model_dir": str(model_dir)})
        # config.update({"config_dir": str(model_dir)})
        #
        # # update experiment
        # config["experiment"]["dir"] = str(exp_dir)

        # store the updated config file in the checkpoint dir
        conf_file = str(model_dir/"config.yaml")
        write_yaml_config(config, conf_file)
        # model_dir.write_text(model_dir/config["yml_config_file_name"], config, encoding="utf-8")
        print(f"✅ Updated config file stored in the path: {str(model_dir)}")
    else:
        print(f"---> model with the same config found: {model_dir}")



    return model_dir

