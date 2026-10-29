import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random
from datasets.dataloader import BraTSData

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_batch(train_loader, slice_idx=None, sample_idx=0):

    batch = next(iter(train_loader))
    image = batch["image"][sample_idx]
    label = batch["label"][sample_idx]

    filenames = batch["image"].meta["filename_or_obj"]
    case_id = filenames[sample_idx].split("/")[-2]

    if slice_idx is None:
        wt_area = label[1].sum(dim=(0, 1))
        slice_idx = wt_area.argmax().item()
        print(f"WT area per slice max at: {slice_idx}, area: {wt_area[slice_idx].item():.0f} voxels")


    modalities = ["T1n", "T1c", "T2w", "T2f(FLAIR)"]

    label_names = ["TC(Tumor Core)", "WT(Whole Tumor)", "ET(Enhancing Tumor)"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))


    for i in range(4):
        axes[0, i].imshow(image[i, :, :, slice_idx].numpy(), cmap="gray")
        axes[0, i].set_title(f"{modalities[i]}\nshape: {tuple(image[i].shape)}")
        axes[0, i].axis("off")


    for i in range(3):
        axes[1, i].imshow(label[i, :, :, slice_idx].float().numpy(), cmap="hot")
        axes[1, i].set_title(label_names[i])
        axes[1, i].axis("off")

    flair_slice = image[3, :, :, slice_idx].numpy()
    label_overlay = label[:, :, :, slice_idx].float().sum(0).numpy()
    axes[1, 3].imshow(flair_slice, cmap="gray")
    axes[1, 3].imshow(label_overlay, cmap="hot", alpha=0.5)
    axes[1, 3].set_title("FLAIR + Label Overlay")
    axes[1, 3].axis("off")

    plt.suptitle(f"Case: {case_id} | Slice {slice_idx}", fontsize=14)
    plt.tight_layout()
    # plt.show()
    save_dir = "../test_picture"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"Case_{case_id}_slice_{slice_idx}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    print(f"The visualization image has been saved to: {save_path}")

    print(f"image shape: {tuple(image.shape)}")
    print(f"label shape: {tuple(label.shape)}")
    print(f"image range: [{image.min():.4f}, {image.max():.4f}]")
    print(f"label unique: {torch.unique(label.float()).tolist()}")


