import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from monai.transforms import Activations, AsDiscrete
from monai.inferers import sliding_window_inference

from models.model_selection import build_model
from datasets.dataloader import BraTSData
from config.config import parse_args, setup_outputs


CHANNEL_NAMES = ["TC", "WT", "ET"]


def load_model(ckpt_path, model_name, in_channels, num_classes, device):
    model = build_model(model_name, in_channels, num_classes).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded: {ckpt_path}")
    print(f"Best Epoch: {checkpoint.get('best_epoch', 'N/A')}, "
          f"Best Mean Dice: {checkpoint.get('best_metric', 'N/A')}")
    return model


def per_case_dice(pred, label, eps=1e-8):
    dims = tuple(range(2, pred.ndim))
    p = pred.float()
    g = label.float()

    inter = (p * g).sum(dim=dims)
    denom = p.sum(dim=dims) + g.sum(dim=dims)

    dice = torch.where(
        denom > 0,
        2.0 * inter / (denom + eps),
        torch.full_like(denom, float("nan")),
    )

    gt_empty = g.sum(dim=dims) == 0
    pred_empty = p.sum(dim=dims) == 0

    return dice.squeeze(0), gt_empty.squeeze(0), pred_empty.squeeze(0)


def get_case_id(batch, idx):
    for key in ("case_id", "case_name", "id", "filename", "name"):
        if key in batch:
            v = batch[key]
            if isinstance(v, (list, tuple)):
                return str(v[0])
            if torch.is_tensor(v):
                if v.numel() == 1:
                    return str(v.item())
                return str(v.flatten()[0].item())
            return str(v)
    return f"case_{idx:03d}"


def infer_ssa_per_case(cfg, ckpt_path=None, method_name="Ours", out_csv=None,
                       device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ckpt_path or os.path.join(cfg.checkpoint_dir, "best_model.pth")

    ds_name = cfg.Dataset_name.lower()
    if ds_name == "brats2023":
        data_dir = cfg.BraTS2023
    elif ds_name == "brats2020":
        data_dir = cfg.BraTS2020
    else:
        data_dir = cfg.test_dir

    data = BraTSData(
        cfg.Dataset_name,
        batch_size=1,
        num_workers=cfg.num_workers,
        train_size=cfg.input_size,
        n_splits=cfg.n_splits,
        random_state=42,
        cache_data=False,
        data_dir=data_dir,
        test_dir=cfg.test_dir,
    )
    test_loader = data.get_test_loader()

    print(f"SSA test cases: {len(test_loader)} (expected 60)")

    model = load_model(
        ckpt_path, cfg.model, cfg.in_channels, cfg.num_classes, device
    )

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)

    rows = []
    with torch.inference_mode():
        for idx, batch in enumerate(tqdm(test_loader, desc="SSA per-case")):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            case_id = get_case_id(batch, idx)

            logits = sliding_window_inference(
                inputs=images,
                roi_size=cfg.roi_size,
                sw_batch_size=cfg.sw_batch_size,
                predictor=model,
                overlap=cfg.overlap,
            )

            preds = post_pred(post_sigmoid(logits))  # [1, 3, H, W, D]

            dice, gt_empty, pred_empty = per_case_dice(preds, labels)

            dice_np = dice.detach().cpu().numpy()
            gt_empty_np = gt_empty.detach().cpu().numpy().astype(int)
            pred_empty_np = pred_empty.detach().cpu().numpy().astype(int)

            row = {
                "method": method_name,
                "case": case_id,
            }

            for c, name in enumerate(CHANNEL_NAMES):
                d = dice_np[c]
                row[f"{name}_Dice"] = float(d) * 100.0 if not np.isnan(d) else np.nan
                row[f"{name}_GT_empty"] = int(gt_empty_np[c])
                row[f"{name}_Pred_empty"] = int(pred_empty_np[c])

            valid = dice_np[~np.isnan(dice_np)]
            row["Avg_Dice"] = float(np.mean(valid)) * 100.0 if len(valid) else np.nan

            rows.append(row)

    df = pd.DataFrame(rows)

    if out_csv is None:
        out_dir = getattr(cfg, "test_output",
                          getattr(cfg, "outputs_dir", "."))
        out_csv = os.path.join(out_dir, f"ssa_per_case_{method_name}.csv")

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("\nPer-region Dice (%)")
    print("-" * 80)
    for name in CHANNEL_NAMES:
        s = df[f"{name}_Dice"].dropna()
        if len(s) > 0:
            print(
                f"{name}: n={len(s):3d}, "
                f"mean={s.mean():6.2f}, std={s.std():5.2f}, "
                f"median={s.median():6.2f}, "
                f"IQR=({s.quantile(0.25):.2f}, {s.quantile(0.75):.2f})"
            )
    print("-" * 80)
    print(f"Saved: {out_csv}")

    return df
