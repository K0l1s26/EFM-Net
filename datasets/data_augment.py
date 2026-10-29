import os
import torch
import numpy as np
from monai import transforms
from monai.transforms import Compose, LoadImaged, MapTransform
from config.config import parse_args


def get_2023data_dicts(data_dir):
    case_ids = sorted(os.listdir(data_dir))
    data_dicts = []
    for i in case_ids:
        case_path = os.path.join(data_dir, i)
        if not os.path.isdir(case_path): continue
        data_dicts.append({
            "image": [
                os.path.join(case_path, f"{i}-t1n.nii.gz"),  # t1
                os.path.join(case_path, f"{i}-t1c.nii.gz"),  # t1ce
                os.path.join(case_path, f"{i}-t2w.nii.gz"),  # t2
                os.path.join(case_path, f"{i}-t2f.nii.gz"),  # flair
            ],
            "label": os.path.join(case_path, f"{i}-seg.nii.gz"),  # label
        })
    return data_dicts


def get_2020data_dicts(data_dir):
    case_ids = sorted(os.listdir(data_dir))
    data_dicts = []
    for i in case_ids:
        case_path = os.path.join(data_dir, i)
        if not os.path.isdir(case_path): continue

        data_dicts.append({
            "image": [
                os.path.join(case_path, f"{i}_t1.nii"),  # t1
                os.path.join(case_path, f"{i}_t1ce.nii"),  # t1ce
                os.path.join(case_path, f"{i}_t2.nii"),    # t2
                os.path.join(case_path, f"{i}_flair.nii"),  # flair
            ],
            "label": os.path.join(case_path, f"{i}_seg.nii"),  # label
        })
    return data_dicts


class ConvertBraTS2023Labelsd(MapTransform):
    """
        Channel 0 - TC (Tumor Core)  = label 1 + label 3
        Channel 1 - WT (Whole Tumor) = label 1 + label 2 + label 3
        Channel 2 - ET (Enhancing)   = label 3
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            if label.shape[0] == 1:
                label = label[0]
            result = [
                torch.logical_or(label == 1, label == 3),  # TC
                torch.logical_or(torch.logical_or(label == 1, label == 2), label == 3),  # WT
                label == 3,  # ET
            ]
            d[key] = torch.stack(result, dim=0).float()
        return d


# transforms
def data_transform(train_size, dataset_name, prob=0.5):
    ds_name = dataset_name.lower()
    if ds_name == "brats2023":
        print(">> Use BraTS2023 custom label conversion: ConvertBraTS2023Labelsd")
        label_transform = ConvertBraTS2023Labelsd(keys=["label"])

    elif ds_name == "brats2020":
        print(">> Use the official BraTS2020 labels for conversion: ConvertToMultiChannelBasedOnBratsClassesd")
        label_transform = transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label")

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}。Please enter 'BraTS2023' or 'BraTS2020'")

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            label_transform,
            # transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
            # ConvertBraTS2023Labelsd(keys=["label"]),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image",margin=10),

            transforms.SpatialPadd(
                keys=["image", "label"],
                # train = (128,128,128)
                spatial_size=train_size,
                mode="constant"),
            transforms.RandSpatialCropd(keys=["image","label"], roi_size=train_size, random_size=False),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandFlipd(keys=["image", "label"], prob=prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=prob, spatial_axis=2),

            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),

        ]
    )

    val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),

                label_transform,
                # transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                # ConvertBraTS2023Labelsd(keys=["label"]),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

    return train_transforms, val_transform


def test_transform():

    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        ConvertBraTS2023Labelsd(keys=["label"]),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])



