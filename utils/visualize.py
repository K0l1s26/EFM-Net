import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from monai.transforms import Activations,AsDiscrete
from monai.inferers import sliding_window_inference

from datasets.dataloader import BraTSData
from config.config import parse_args,setup_outputs
from models.model_selection import build_model


def infer_single_case(model, test_loader, case_idx, device, roi_size, sw_batch_size, overlap):
    model.eval()
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)

    # 取单个病例
    batch = None
    for i, b in enumerate(test_loader):
        if i == case_idx:
            batch = b
            break
    if batch is None:
        raise IndexError(f"case_idx={case_idx} beyond test_loader range")

    image = batch["image"].to(device)
    label = batch["label"].to(device)

    with torch.inference_mode():
        logits = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )
        pred = post_pred(post_sigmoid(logits))

        image = image[0].detach().cpu().numpy()
        label = label[0].detach().cpu().numpy()
        pred = pred[0].detach().cpu().numpy()

    return image, label, pred

def visualize_case(image, label, pred, save_dir, case_idx=0, modality_idx=0, slice_idx=None):
    modality_mapping = {0: "T1", 1: "T1ce", 2: "T2", 3: "FLAIR"}

    if slice_idx is None:
        slice_idx = find_best_wt_slice(label)

    img2d = image[modality_idx, :, :, slice_idx]
    img2d = normalize_for_display(img2d)

    gt_tc = label[0, :, :, slice_idx]
    gt_wt = label[1, :, :, slice_idx]
    gt_et = label[2, :, :, slice_idx]

    pr_tc = pred[0, :, :, slice_idx]
    pr_wt = pred[1, :, :, slice_idx]
    pr_et = pred[2, :, :, slice_idx]

    # 图像翻转
    img2d = np.rot90(img2d)
    gt_tc = np.rot90(gt_tc)
    gt_wt = np.rot90(gt_wt)
    gt_et = np.rot90(gt_et)
    pr_tc = np.rot90(pr_tc)
    pr_wt = np.rot90(pr_wt)
    pr_et = np.rot90(pr_et)


    cmap_wt = ListedColormap(['green'])
    cmap_tc = ListedColormap(['yellow'])
    cmap_et = ListedColormap(['red'])

    fig, axes = plt.subplots(2, 3, figsize=(13, 10))
    fig.suptitle(f"Case {case_idx} Visualization ({modality_mapping[modality_idx]})", fontsize=16)

    axes[0, 0].imshow(img2d, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img2d, cmap="gray")
    axes[0, 1].imshow(mask_tissue(gt_wt), cmap=cmap_wt, alpha=0.5)
    axes[0, 1].imshow(mask_tissue(gt_tc), cmap=cmap_tc, alpha=0.6)
    axes[0, 1].imshow(mask_tissue(gt_et), cmap=cmap_et, alpha=0.7)
    axes[0, 1].set_title("GT")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img2d, cmap="gray")
    axes[0, 2].imshow(mask_tissue(pr_wt), cmap=cmap_wt, alpha=0.5)
    axes[0, 2].imshow(mask_tissue(pr_tc), cmap=cmap_tc, alpha=0.6)
    axes[0, 2].imshow(mask_tissue(pr_et), cmap=cmap_et, alpha=0.7)
    axes[0, 2].set_title("Pred")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(img2d, cmap="gray")
    axes[1, 0].imshow(mask_tissue(pr_tc), cmap=cmap_tc, alpha=0.6)
    axes[1, 0].set_title("Pred TC")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(img2d, cmap="gray")
    axes[1, 1].imshow(mask_tissue(pr_wt), cmap=cmap_wt, alpha=0.6)
    axes[1, 1].set_title("Pred WT")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(img2d, cmap="gray")
    axes[1, 2].imshow(mask_tissue(pr_et), cmap=cmap_et, alpha=0.6)
    axes[1, 2].set_title("Pred ET")
    axes[1, 2].axis("off")

    plt.tight_layout()

    if save_dir is not None:
        save_path = os.path.join(save_dir, f"case_{case_idx}_{modality_mapping[modality_idx]}_slice_{slice_idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to: {save_path}")

    plt.close(fig)

def find_best_wt_slice(label):
    wt = label[1]
    voxel_nums = wt.sum(axis=(0, 1))
    best_slice = np.argmax(voxel_nums)
    return int(best_slice)


def normalize_for_display(img2d):
    img2d = img2d.astype(np.float32)
    mask = img2d != 0

    if not np.any(mask):
        return np.zeros_like(img2d, dtype=np.float32)

    brain_voxels = img2d[mask]
    q_min = np.percentile(brain_voxels, 1)
    q_max = np.percentile(brain_voxels, 99)
    img_show = np.clip(img2d, q_min, q_max)
    img_show = (img_show - q_min) / (q_max - q_min + 1e-8)
    img_show[~mask] = 0
    return img_show


def mask_tissue(mask):
    return np.ma.masked_where(mask == 0, mask)


