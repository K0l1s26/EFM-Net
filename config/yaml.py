import json
import os
import yaml
from argparse import Namespace
import argparse


def load_config(yaml_file):
    with open(yaml_file,"r",encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_config2(config_path):
    with open(config_path,"r",encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    return dict_to_namespace(cfg_dict)

def dict_to_namespace(cfg_dict):
    for key,value in cfg_dict.items():
        if isinstance(value,dict):
            cfg_dict[key] = dict_to_namespace(value)
    return Namespace(**cfg_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./experiments/base.yaml",
                        help="YAML config file path")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch_size from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning_rate from config")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model from config")

    args = parser.parse_args()


    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        config["batch_size"] = args.batch_size
        config["lr"] = args.lr
        config["model"] = args.model

        return config





