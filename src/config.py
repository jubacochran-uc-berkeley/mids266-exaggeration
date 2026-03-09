"""
Config loading and merging for experiment YAML files

Useage:
    from src.config import load_config
    config = load_config("configs/roberta_full.yaml")

"""

import yaml
from pyprojroot import here


def load_config(config_path: str) -> dict:
    """
    Load an experiment config, merging with base config if specified.

    If the experiment config contains a '_base_' key, that file is loaded
    first and experiment values override base values.

    :param config_path: path to experiment YAML (relative to project root)
    :type config_path: str
    :return: merged configuration dictionary
    :rtype: dict
    """

    config_file = here(config_path)
    with open(config_file, "r") as f:
        experiment = yaml.safe_load(f)

    
    # if no base exists return experiment config
    if "_base_" not in experiment:
        return experiment

    #Load base config from same directory as experiment config
    base_path = config_file.parent / experiment.pop("_base_") #removing _base_ key to reduse noise
    
    with open(base_path, "r") as f:
        base = yaml.safe_load(f)

    #Base then override with experiments
    base.update(experiment)
    return base