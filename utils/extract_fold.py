import os
import numpy as np
from sklearn.model_selection import KFold
from datasets.data_augment import get_2023data_dicts, get_2020data_dicts
from config.config import parse_args


def get_case_name_from_dict(sample_dict, dataset_name):

    image_path = sample_dict["image"][0]   # 取 t1 模态的路径即可
    case_dir = os.path.dirname(image_path)
    return os.path.basename(case_dir)


def extract_and_save_folds(data_dir, dataset_name, n_splits=5,
                           random_state=42, save_dir="./fold_splits"):

    ds_name = dataset_name.lower()
    if ds_name == "brats2023":
        all_data_dict = get_2023data_dicts(data_dir)
    elif ds_name == "brats2020":
        all_data_dict = get_2020data_dicts(data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    all_data = np.array(all_data_dict)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(kfold.split(all_data))

    os.makedirs(save_dir, exist_ok=True)

    summary_lines = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        train_cases = [get_case_name_from_dict(all_data[i], dataset_name) for i in train_idx]
        val_cases   = [get_case_name_from_dict(all_data[i], dataset_name) for i in val_idx]

        # 写训练集
        train_txt = os.path.join(save_dir, f"fold{fold}_train.txt")
        with open(train_txt, "w") as f:
            f.write("\n".join(train_cases))

        # 写验证集
        val_txt = os.path.join(save_dir, f"fold{fold}_val.txt")
        with open(val_txt, "w") as f:
            f.write("\n".join(val_cases))

        summary_lines.append(
            f"Fold {fold}: train={len(train_cases)}, val={len(val_cases)} "
            f"| val_cases={val_cases[:5]} ... "
        )
        print(summary_lines[-1])

    # 写一个汇总
    with open(os.path.join(save_dir, "summary.txt"), "w") as f:
        f.write(f"dataset: {dataset_name}\n")
        f.write(f"data_dir: {data_dir}\n")
        f.write(f"n_splits: {n_splits}\n")
        f.write(f"random_state: {random_state}\n")
        f.write(f"total cases: {len(all_data_dict)}\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(summary_lines))

    print(f"\n Saved to {save_dir}/")
    print(f"   Per-fold file: fold0_train.txt, fold0_val.txt, fold1_train.txt, ...")
    print(f"  Summary file: summary.txt")

