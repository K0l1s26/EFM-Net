import yaml
import argparse
import os
import torch
import json
from argparse import Namespace

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "experiments", "base.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="Training with YAML config")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Path to YAML config file")

    parser.add_argument("--BraTS2023", type=str, default=None, help="BraTS2023-GLI Dataset root directory")
    parser.add_argument("--BraTS2020", type=str, default=None, help="BraTS2020 Dataset root directory")

    parser.add_argument("--test_dir", type=str, default=None, help="Experiment name, output directory=outputs/{model}-{run-name}, isolate different experiments")
    parser.add_argument("--Dataset_name", type=str, default=None, help="Used to select different data reading and label conversion options")
    parser.add_argument("--run_name", type=str, default=None, help="Test Dataset root directory")
    parser.add_argument("--test_inference", type=str, default=None, help="BraTS2023-SSA Whether to reason")


    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--n_splits", type=int, default=None, help="Number of folds in cross-validation partitioning")
    parser.add_argument("--fold", type=int, default=None, help="Current number of cross-validation folds")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of DataLoader worker processes")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--input_size", nargs=3, type=int, default=None, help="Input image size")
    parser.add_argument("--in_channels", type=int, default=None, help="Number of input channels")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of output classes")
    parser.add_argument("--norm", type=str, default=None, help="Normalization type: BATCH / INSTANCE / GROUP")
    parser.add_argument("--act", type=str, default=None, help="Activation type: RELU / LEAKY_RELU / PRELU / GELU")
    parser.add_argument("--SEED", type=int, default=None, help="Random Seed")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--val_every", type=int, default=None,
                        help="Validation frequency, perform validation and save the model every val_every epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--resume", type=str2bool, default=None, help="Resume training from a breakpoint?")
    parser.add_argument("--roi_size", nargs=3, type=int, default=None, help="ROI size for windowed reasoning")
    parser.add_argument("--sw_batch_size", type=int, default=None, help="Batch size for windowed reasoning")
    parser.add_argument("--overlap", type=float, default=None, help="Overlap between windows")

    parser.add_argument("--device", type=str, default=None, help="Device uses cuda:0 or cpu")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key, val in vars(args).items():
        if key == "config":
            continue
        if val is not None:
            config[key] = val

    config["input_size"] = tuple(config["input_size"])
    config["roi_size"] = tuple(config["roi_size"])
    config["device"] = torch.device(config["device"])
    return dict_to_namespace(config)


def load_config(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return Namespace(**d)


def setup_outputs(config):
    _base_dir: str = os.path.dirname(os.path.abspath(__file__))
    name = f"{config.model}-{config.run_name}" if config.run_name else config.model
    config.outputs_dir = os.path.join(_base_dir, "..", "outputs", name, f"fold_{config.fold}")
    config.checkpoint_dir = os.path.join(config.outputs_dir, "checkpoints")
    config.test_output = os.path.join(config.outputs_dir, "BraTS-SSA")
    config.log_dir = os.path.join(config.outputs_dir, "logs")
    config.curves_dir = os.path.join(config.outputs_dir, "curves")
    config.visualize_dir = os.path.join(config.outputs_dir, "visualize")

    for path in [
        config.outputs_dir,
        config.checkpoint_dir,
        config.test_output,
        config.log_dir,
        config.curves_dir,
        config.visualize_dir
    ]:
        os.makedirs(path, exist_ok=True)
    save_dict = vars(config).copy()

    auto_generated_keys = [
        "outputs_dir",
        "checkpoint_dir",
        "log_dir",
        "curves_dir",
        "visualize_dir",
    ]
    for key in auto_generated_keys:
        save_dict.pop(key, None)

    save_dict["input_size"] = list(config.input_size)
    save_dict["roi_size"] = list(config.roi_size)
    save_dict["device"] = str(config.device)

    cfg_json_path = os.path.join(config.outputs_dir, "config.json")
    if not os.path.exists(cfg_json_path):
        with open(cfg_json_path, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=4, ensure_ascii=False)
        print(f"[Config] The final configuration has been saved to: {cfg_json_path}")
    else:
        print(f"[Config] The configuration file already exists, skipping save:{cfg_json_path}")


