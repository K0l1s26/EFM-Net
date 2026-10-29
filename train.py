import torch
import os
import gc
import random
import time
import numpy as np

from tqdm import tqdm
from torch.amp import autocast, GradScaler

from config.config import parse_args, setup_outputs
from datasets.dataloader import BraTSData
from utils.logger import save_history

from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete
from monai.utils.enums import MetricReduction
from monai.inferers import sliding_window_inference




def set_seeds(seed=42):
    from monai.utils import set_determinism as monai_set_determinism
    monai_set_determinism(seed=seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    worker_seed = worker_info.seed % 2**32
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_one_epoch(model, loader, optimizer, criterion, dice_metric,
                    post_sigmoid, post_pred, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    dice_metric.reset()
    start_time = time.time()

    pbar = tqdm(loader, desc=f'Train Epoch {epoch}/{total_epochs}')
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(images)
        logits = logits.float()
        loss = criterion(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f' Epoch {epoch}: NaN/Inf loss detected, skipping batch!')
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1
        with torch.no_grad():
            final_outputs = post_pred(post_sigmoid(logits))
            dice_metric(y_pred=final_outputs, y=labels)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
        })

    elapsed = time.time() - start_time

    if valid_batches > 0:
        epoch_loss = total_loss / valid_batches
        metric, _ = dice_metric.aggregate()
        tc, wt, et = metric[0].item(), metric[1].item(), metric[2].item()
    else:
        epoch_loss = 0.0
        tc, wt, et = 0.0, 0.0, 0.0

    # print(metric.shape)
    # print(metric)

    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    skipped_batches = len(loader) - valid_batches
    skip_warning = f' | Skipped: {skipped_batches}' if skipped_batches > 0 else ''

    print(f'Train Epoch {epoch}/{total_epochs} | '
          f'Loss: {epoch_loss:.4f} | '
          f'TC: {tc:.4f} WT: {wt:.4f} ET: {et:.4f} | '
          f'Time: {minutes}m {seconds:.1f}s'
          f'{skip_warning}')

    return epoch_loss ,(tc, wt, et)

@torch.no_grad()
def val_one_epoch(model, loader, dice_metric, post_sigmoid, post_pred,
                  roi_size, sw_batch_size, infer_overlap, device, epoch, total_epochs):
    model.eval()
    dice_metric.reset()
    start_time = time.time()

    pbar = tqdm(loader, desc=f'Val   Epoch {epoch}/{total_epochs}')
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = sliding_window_inference(
                inputs=images,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=infer_overlap,
            )

        logits = logits.float()


        final_outputs = post_pred(post_sigmoid(logits))
        dice_metric(y_pred=final_outputs, y=labels)

    elapsed = time.time() - start_time
    metric, _ = dice_metric.aggregate()
    tc, wt, et = metric[0].item(), metric[1].item(), metric[2].item()
    mean_dice = (tc + wt + et) / 3

    # print(metric.shape)
    # print(metric)

    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    print(f'Val   Epoch {epoch}/{total_epochs} | '
          f'Mean: {mean_dice:.4f} | '
          f'TC: {tc:.4f} WT: {wt:.4f} ET: {et:.4f} | '
          f'Time: {minutes}m {seconds:.1f}s')

    return mean_dice, (tc, wt, et)


def save_checkpoint(ckpt_path, epoch, best_epoch, model, optimizer, scheduler,
                    best_metric,best_tc,best_wt,best_et, patience_counter, history):
    torch.save({
        'epoch': epoch,
        'best_epoch': best_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric,
        'best_tc': best_tc,
        'best_wt': best_wt,
        'best_et': best_et,

        'patience_counter': patience_counter,
        'history': history,
    }, ckpt_path)


def load_checkpoint(ckpt_path, model, optimizer, scheduler, device,
                    history_template):

    checkpoint = torch.load(ckpt_path, map_location=device,weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_epoch = checkpoint.get('best_epoch', 0)
    best_metric = checkpoint['best_metric']
    patience_counter = checkpoint.get('patience_counter', 0)
    history = checkpoint.get('history', history_template)

    best_tc = checkpoint.get('best_tc', None)
    best_wt = checkpoint.get('best_wt', None)
    best_et = checkpoint.get('best_et', None)

    return start_epoch, best_epoch, best_metric,best_tc,best_wt,best_et, patience_counter, history


class DiceBCELoss(torch.nn.Module):
    def __init__(self, lambda_dice=1.0, lambda_bce=1.0):
        super().__init__()

        self.dice = DiceLoss(
            include_background=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=False,
            sigmoid=True,
            softmax=False,
        )

        self.bce = torch.nn.BCEWithLogitsLoss()

        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce

    def forward(self, logits, targets):
        targets = targets.float()

        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)

        return (
            self.lambda_dice * dice_loss
            + self.lambda_bce * bce_loss
        )



def main(cfg):

    SEED = cfg.SEED
    device = cfg.device

    set_seeds(seed=SEED)
    g = torch.Generator()
    g.manual_seed(SEED)
    dataset_name = cfg.Dataset_name
    ds_name = dataset_name.lower()
    if ds_name == "brats2023":
        data_dir = cfg.BraTS2023
    elif ds_name == "brats2020":
        data_dir = cfg.BraTS2020
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")


    data = BraTSData(data_dir=data_dir,
                     dataset_name=dataset_name,
                     batch_size=cfg.batch_size,
                     num_workers=cfg.num_workers,
                     train_size=cfg.input_size,
                     n_splits=cfg.n_splits,
                     cache_data=False,
                     worker_init_fn=seed_worker,
                     generator=g
                     )

    train_loader, val_loader = data.get_fold_loader(cfg.fold)

    from models.model_selection import build_model

    model = build_model(cfg.model, cfg.in_channels, cfg.num_classes).to(device)
    print(f'Model parameter count: {sum(p.numel() for p in model.parameters()):,}')

    criterion = DiceBCELoss(
        lambda_dice=1.0,
        lambda_bce=1.0,
    )

    dice_metric = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs,eta_min=1e-7)

    best_metric = 0.0
    best_epoch = 0
    best_tc, best_wt, best_et = 0.0, 0.0, 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_tc': [], 'train_wt': [], 'train_et': [],
        'val_mean': [], 'val_tc': [], 'val_wt': [], 'val_et': [],
    }


    start_epoch = 0

    latest_path = os.path.join(cfg.checkpoint_dir, 'latest_checkpoint.pth')
    best_path = os.path.join(cfg.checkpoint_dir, 'best_model.pth')
    resume_path = latest_path if os.path.exists(latest_path) else best_path

    if cfg.resume and os.path.exists(resume_path):
        start_epoch, best_epoch, best_metric, best_tc,best_wt,best_et,patience_counter, history = load_checkpoint(
            ckpt_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            history_template=history,
        )


        patience_counter = 0
        print(f'>>> 从 {os.path.basename(resume_path)} Resume training '
              f'(Epoch {start_epoch}), Current best Mean Dice: {best_metric:.4f}\n'
              f'Best Epoch: {best_epoch}\n'
              f'Mean: {best_metric:.4f}\n'
              f'TC: {best_tc:.4f} WT: {best_wt:.4f} ET: {best_et:.4f}')

    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        train_loss, (tr_tc, tr_wt, tr_et) = train_one_epoch(
            model, train_loader, optimizer, criterion, dice_metric,
            post_sigmoid, post_pred, device, epoch, cfg.epochs
        )
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_tc'].append(tr_tc)
        history['train_wt'].append(tr_wt)
        history['train_et'].append(tr_et)

        val_interval = cfg.val_every
        if epoch % val_interval == 0 or epoch == 1:
            mean_dice, (val_tc, val_wt, val_et) = val_one_epoch(
                model, val_loader, dice_metric, post_sigmoid, post_pred,
                roi_size= cfg.roi_size, sw_batch_size=cfg.sw_batch_size,
                infer_overlap=cfg.overlap,device=device, epoch=epoch, total_epochs=cfg.epochs
            )

            history['val_mean'].append(mean_dice)
            history['val_tc'].append(val_tc)
            history['val_wt'].append(val_wt)
            history['val_et'].append(val_et)

            if mean_dice > best_metric:
                best_metric = mean_dice
                best_epoch = epoch
                best_tc, best_wt, best_et = val_tc, val_wt, val_et
                patience_counter = 0
                ckpt_path = os.path.join(cfg.checkpoint_dir, 'best_model.pth')
                save_checkpoint(
                    ckpt_path=ckpt_path,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric=best_metric,
                    best_tc=best_tc,
                    best_wt=best_wt,
                    best_et=best_et,
                    patience_counter=patience_counter,
                    history=history,
                )
                print(f'  >>> Save the best model → {ckpt_path} (Mean Dice: {best_metric:.4f})')
            else:
                patience_counter += 1
                print(f'  >>>Verification not elevated ({patience_counter}/{cfg.patience})，Current best: {best_metric:.4f}',
                      f'| TC:{best_tc:.4f} WT:{best_wt:.4f} ET:{best_et:.4f}')

                if patience_counter >= cfg.patience:
                    print(f'\nEarly stopping triggered! Consecutive {cfg.patience} epochs without improvement, training stopped at epoch {epoch}.')
                    break
        else:
            for key in ['val_mean', 'val_tc', 'val_wt', 'val_et']:
                history[key].append(None)

        save_checkpoint(
            ckpt_path=os.path.join(cfg.checkpoint_dir, 'latest_checkpoint.pth'),
            epoch=epoch,
            best_epoch=best_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_metric,
            best_tc=best_tc,
            best_wt=best_wt,
            best_et=best_et,
            patience_counter=patience_counter,
            history=history,
        )

        torch.cuda.empty_cache()
        gc.collect()

    print(f'\n{"=" * 50}')
    print(f'Training complete! Best Mean Dice: {best_metric:.4f} (Epoch {best_epoch}) '
          f'| TC:{best_tc:.4f} WT:{best_wt:.4f} ET:{best_et:.4f}')
    print(f'{"=" * 50}')
    return history, best_metric


if __name__ == '__main__':
    cfg = parse_args()
    cfg.model = "EFMNet"
    cfg.Dateset_name = "BraTS2023"
    # cfg.num_workers = 8
    cfg.run_name = "BraTS2023"
    setup_outputs(cfg)

    history, best = main(cfg)
    save_history(history, os.path.join(cfg.log_dir, 'history.json'))