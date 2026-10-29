import torch
import numpy as np
from thop import profile
import time
from tqdm import tqdm
import os

from monai.transforms import Activations, AsDiscrete
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.utils.enums import MetricReduction
from monai.inferers import sliding_window_inference

from models.model_selection import build_model
from datasets.dataloader import BraTSData
from utils.logger import save_inference_results
from config.config import parse_args, setup_outputs


def load_model(ckpt_path, model_name, in_channels, num_classes, device):
    model = build_model(model_name, in_channels, num_classes).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded: {ckpt_path}")
    print(f"Best Epoch: {checkpoint.get('best_epoch', 'N/A')}, "
          f"Best Mean Dice: {checkpoint.get('best_metric', 'N/A')}")
    return model


def compute_params_gflops(model, device):
    params = sum(p.numel() for p in model.parameters()) / 1e6
    x = torch.randn(1, 4, 128, 128, 128).to(device)
    flops, _ = profile(model, inputs=(x,), verbose=False)

    del x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return params, flops / 1e9


def test(model,test_loader,device,roi_size,sw_batch_size,overlap):

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_metric = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )
    hd95_metric = HausdorffDistanceMetric(
        include_background=True,
        percentile=95,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )

    warmup_batch = next(iter(test_loader))
    warmup_images = warmup_batch["image"].to(device)
    print("Warming up ...")
    with torch.inference_mode():
        for _ in range(3):
            sliding_window_inference(warmup_images, roi_size, sw_batch_size,
                                         model, overlap)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    pbar = tqdm(test_loader, desc='Testing')

    with torch.inference_mode():
        for batch in pbar:
            images = batch["image"].to(device)  # [1, 4, H, W, D]
            labels = batch["label"].to(device) # [1, 3, H, W, D]

            t0 = time.time()
            logits = sliding_window_inference(
                inputs=images,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            infer_time = time.time() - t0
            # 后处理
            preds = post_pred(post_sigmoid(logits))  # [1, 3, H, W, D]
            if device.type == 'cuda':
                torch.cuda.synchronize()

            times.append(infer_time)
            dice_metric(y_pred=preds, y=labels)
            hd95_metric(y_pred=preds, y=labels)

    dice,_ = dice_metric.aggregate()    # [TC, WT, ET]
    hd95,_ = hd95_metric.aggregate()    # [TC, WT, ET]

    dice_metric.reset()
    hd95_metric.reset()

    total_time = sum(times)
    num_cases = len(times)
    avg_time = total_time / num_cases
    fps = 1.0 / avg_time
    times_arr = np.array(times)
    std_time = times_arr.std()

    efficiency = {
        'num_cases': num_cases,
        'total_time': total_time,
        'avg_time': avg_time,
        'std_time': std_time,
        'fps': fps,
        'min_time': times_arr.min(),
        'max_time': times_arr.max(),  #
    }

    return dice, hd95, efficiency


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ckpt = os.path.join(cfg.checkpoint_dir, 'best_model.pth')

    dataset_name = cfg.Dataset_name
    ds_name = dataset_name.lower()
    if ds_name == "brats2023":
        data_dir = cfg.BraTS2023
    elif ds_name == "brats2020":
        data_dir = cfg.BraTS2020
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    data = BraTSData(
        dataset_name,
        batch_size=1,
        num_workers=cfg.num_workers,
        train_size=cfg.input_size,
        n_splits=cfg.n_splits,
        random_state=42,
        cache_data=False,
        data_dir= data_dir,
        test_dir = cfg.test_dir,
    )

    test_output_dir = cfg.test_output
    if cfg.test_inference :
        test_loader = data.get_test_loader()
        outputs_dir = test_output_dir
        print(f"Test samples: {len(test_loader)}")
    else:
        _, test_loader = data.get_fold_loader(cfg.fold)
        print(f"val samples: {len(test_loader)}")
        outputs_dir = cfg.outputs_dir


    model = load_model(ckpt, cfg.model, cfg.in_channels,
                       cfg.num_classes, device)
    params, gflops = compute_params_gflops(model, device)

    dice, hd95, eff = test(model, test_loader, device,
                           roi_size=cfg.roi_size,
                           sw_batch_size=cfg.sw_batch_size,
                           overlap=cfg.overlap,
                           )

    tc_d, wt_d, et_d = dice[0].item() * 100, dice[1].item() * 100, dice[2].item() * 100
    mean_d = (tc_d + wt_d + et_d) / 3

    tc_h, wt_h, et_h = hd95[0].item(), hd95[1].item(), hd95[2].item()
    mean_h = (tc_h + wt_h + et_h) / 3

    print(f"\n{'=' * 70}")
    print(f"Model: {cfg.model}  |  Params: {params:.2f} M  |  GFLOPs: {gflops:.2f} G")
    print(f"{'=' * 70}")
    print(f"{'Region':<8} {'Dice (%)':<12} {'HD95 (voxel)':<16}")
    print(f"{'-' * 36}")
    print(f"{'ET':<8} {et_d:<12.2f} {et_h:<16.4f}")
    print(f"{'WT':<8} {wt_d:<12.2f} {wt_h:<16.4f}")
    print(f"{'TC':<8} {tc_d:<12.2f} {tc_h:<16.4f}")
    print(f"{'-' * 36}")
    print(f"{'Mean':<8} {mean_d:<12.2f} {mean_h:<16.4f}")
    print(f"{'=' * 70}")

    print(f"\n{'=' * 70}")
    print(f"Inference Efficiency")
    print(f"{'=' * 70}")
    print(f"  Total cases:       {eff['num_cases']}")
    print(f"  Total time:        {eff['total_time']:.2f} s")
    print(f"  Avg time/case:     {eff['avg_time']:.4f} s  +/- {eff['std_time']:.4f} s")
    print(f"  Min / Max:         {eff['min_time']:.4f} s / {eff['max_time']:.4f} s")
    print(f"  FPS (volumes/s):   {eff['fps']:.4f}")
    print(f"{'=' * 70}")

    save_inference_results(
        output_dir=outputs_dir,
        cfg=cfg,
        params=params,
        gflops=gflops,
        dice=dice,
        hd95=hd95,
        eff=eff,
    )




