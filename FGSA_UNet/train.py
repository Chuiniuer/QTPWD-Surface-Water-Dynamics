# -*- coding: utf-8 -*-
r"""
Train / resume FGSA-UNet for 17-channel water segmentation on full dataset.

Dataset:
K:\Monthly surface water dynamics TIB\supervised classification
\step1_machine_learning_output\generalisation\tiles_train_test_val_splitted

Expected structure:
tiles_train_test_val_splitted
├── train
│   ├── images_norm
│   ├── labels_npy
│   └── valid_masks
├── val
│   ├── images_norm
│   ├── labels_npy
│   └── valid_masks
└── test
    ├── images_norm
    ├── labels_npy
    └── valid_masks

Model:
fgsa_unet_17ch.py -> FGSAUNet
"""

import csv
import time
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if __package__:
    from .model import FGSAUNet
else:
    from model import FGSAUNet


# 1. Paths
DATA_ROOT = Path(
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step1_machine_learning_output\generalisation\tiles_train_test_val_splitted"
)

OUT_DIR = Path(
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step2_semantic_segmentation\model_comparison\model_outputs"
    r"\FULL_DATA_FGSA_UNET_200EPOCHS_SAFE_FP32"
)


# 2. Settings
EPOCHS = 200
BATCH_SIZE = 4
NUM_WORKERS = 2

LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

THRESHOLD = 0.5

USE_AMP = False

GRAD_CLIP_NORM = 1.0

IN_CHANNELS = 17
NUM_CLASSES = 1
FEATURES = (32, 64, 128, 256, 512)
NORM_TYPE = "group"
USE_SPATIAL_GATE = True

LOG_EVERY_N_STEPS = 20
SAVE_CHECKPOINT_EVERY = 20


# 2.1 Resume settings
RESUME = True

PREFERRED_RESUME_PATH = OUT_DIR / "last_model.pth"

FALLBACK_RESUME_PATH = OUT_DIR / "checkpoint_epoch_040.pth"


def get_resume_path():
    if PREFERRED_RESUME_PATH.exists():
        return PREFERRED_RESUME_PATH
    if FALLBACK_RESUME_PATH.exists():
        return FALLBACK_RESUME_PATH
    return None

# 3. Seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# 4. Dataset
class WaterTileDataset(Dataset):
    def __init__(self, root, split):
        root = Path(root)
        self.split = split

        self.img_dir = root / split / "images_norm"
        self.lbl_dir = root / split / "labels_npy"
        self.mask_dir = root / split / "valid_masks"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"images_norm directory not found: {self.img_dir}")
        if not self.lbl_dir.exists():
            raise FileNotFoundError(f"labels_npy directory not found: {self.lbl_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"valid_masks directory not found: {self.mask_dir}")

        self.imgs = sorted(self.img_dir.glob("*.npy"))

        if len(self.imgs) == 0:
            raise RuntimeError(f"{split} no images_norm/*.npy files found")

        print("=" * 80)
        print(f"{split}: {len(self.imgs)} samples")
        print("Image dir:", self.img_dir)
        print("Label dir:", self.lbl_dir)
        print("Mask dir :", self.mask_dir)
        print("Example  :", self.imgs[0].name)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        stem = img_path.stem

        label_path = self.lbl_dir / f"{stem}.npy"
        mask_path = self.mask_dir / f"{stem}.npy"

        if not label_path.exists():
            raise FileNotFoundError(f"Label not found: {label_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Valid mask not found: {mask_path}")

        x = np.load(img_path).astype(np.float32, copy=False)
        y = np.load(label_path).astype(np.float32, copy=False)
        m = np.load(mask_path).astype(np.float32, copy=False)

        if x.ndim != 3:
            raise ValueError(f"Invalid image dimensions: {img_path}, shape={x.shape}")

        if x.shape[0] != IN_CHANNELS and x.shape[-1] == IN_CHANNELS:
            x = np.transpose(x, (2, 0, 1))

        if x.shape[0] != IN_CHANNELS:
            raise ValueError(f"Unexpected channel count; expected {IN_CHANNELS}: {img_path}, shape={x.shape}")

        if y.ndim == 2:
            y = y[None, :, :]
        elif y.ndim == 3 and y.shape[0] != 1 and y.shape[-1] == 1:
            y = np.transpose(y, (2, 0, 1))

        if m.ndim == 2:
            m = m[None, :, :]
        elif m.ndim == 3 and m.shape[0] != 1 and m.shape[-1] == 1:
            m = np.transpose(m, (2, 0, 1))

        y = (y > 0.5).astype(np.float32, copy=False)
        m = (m > 0.5).astype(np.float32, copy=False)

        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)


def make_loader(dataset, split):
    shuffle = split == "train"

    if NUM_WORKERS > 0:
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=1,
            drop_last=False,
        )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


# 5. Loss
def masked_bce_loss(logits, y, mask, criterion):
    valid_sum = mask.sum()

    if valid_sum <= 0:
        return logits.sum() * 0.0

    loss_map = criterion(logits, y)
    loss = (loss_map * mask).sum() / (valid_sum + 1e-7)
    return loss


# 6. Metrics
@torch.no_grad()
def calc_counts_from_logits(logits, y, mask):
    pred = (torch.sigmoid(logits) > THRESHOLD).float()
    y = (y > 0.5).float()
    valid = mask > 0.5

    pred = pred[valid]
    y = y[valid]

    if pred.numel() == 0:
        return 0, 0, 0, 0

    tp = (pred * y).sum().item()
    fp = (pred * (1.0 - y)).sum().item()
    fn = ((1.0 - pred) * y).sum().item()
    tn = ((1.0 - pred) * (1.0 - y)).sum().item()

    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    eps = 1e-7

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    ua = tp / (tp + fp + eps)
    pa = tp / (tp + fn + eps)

    f1 = 2.0 * ua * pa / (ua + pa + eps)
    iou = tp / (tp + fp + fn + eps)

    iou_nonwater = tn / (tn + fp + fn + eps)
    miou = 0.5 * (iou + iou_nonwater)

    return {
        "Accuracy": accuracy,
        "PA": pa,
        "UA": ua,
        "F1": f1,
        "IoU": iou,
        "mIoU": miou,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


# 7. Train and evaluate
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    epoch=None,
    step_log_path=None,
    global_step=0,
):
    model.train()

    total_loss = 0.0
    total_valid_pixels = 0.0

    skipped_empty_mask_batches = 0
    skipped_nonfinite_batches = 0

    pbar = tqdm(loader, desc="Train")

    for batch_idx, (x, y, mask) in enumerate(pbar, start=1):
        step_t0 = time.time()

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        valid_pixels = mask.sum().item()

        if valid_pixels <= 0:
            skipped_empty_mask_batches += 1
            global_step += 1

            if step_log_path is not None:
                append_step_log(
                    step_log_path,
                    {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "num_batches": len(loader),
                        "global_step": global_step,
                        "train_loss_step": "",
                        "valid_pixels": valid_pixels,
                        "lr": optimizer.param_groups[0]["lr"],
                        "step_time_sec": time.time() - step_t0,
                        "skipped": "empty_mask",
                    },
                )
            continue

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = masked_bce_loss(logits, y, mask, criterion)

            if not torch.isfinite(loss):
                skipped_nonfinite_batches += 1
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                print(
                    f"[Warning] skip non-finite loss | "
                    f"epoch={epoch}, batch={batch_idx}, "
                    f"loss={loss.item()}, valid_pixels={valid_pixels}"
                )

                if step_log_path is not None:
                    append_step_log(
                        step_log_path,
                        {
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "num_batches": len(loader),
                            "global_step": global_step,
                            "train_loss_step": "",
                            "valid_pixels": valid_pixels,
                            "lr": optimizer.param_groups[0]["lr"],
                            "step_time_sec": time.time() - step_t0,
                            "skipped": "nonfinite_loss",
                        },
                    )
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(x)
            loss = masked_bce_loss(logits, y, mask, criterion)

            if not torch.isfinite(loss):
                skipped_nonfinite_batches += 1
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                print(
                    f"[Warning] skip non-finite loss | "
                    f"epoch={epoch}, batch={batch_idx}, "
                    f"loss={loss.item()}, valid_pixels={valid_pixels}"
                )

                if step_log_path is not None:
                    append_step_log(
                        step_log_path,
                        {
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "num_batches": len(loader),
                            "global_step": global_step,
                            "train_loss_step": "",
                            "valid_pixels": valid_pixels,
                            "lr": optimizer.param_groups[0]["lr"],
                            "step_time_sec": time.time() - step_t0,
                            "skipped": "nonfinite_loss",
                        },
                    )
                continue

            loss.backward()

            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

            optimizer.step()

        loss_value = float(loss.item())

        total_loss += loss_value * valid_pixels
        total_valid_pixels += valid_pixels

        global_step += 1
        step_time = time.time() - step_t0

        if step_log_path is not None and (
            global_step % LOG_EVERY_N_STEPS == 0
            or batch_idx == 1
            or batch_idx == len(loader)
        ):
            append_step_log(
                step_log_path,
                {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "num_batches": len(loader),
                    "global_step": global_step,
                    "train_loss_step": loss_value,
                    "valid_pixels": valid_pixels,
                    "lr": optimizer.param_groups[0]["lr"],
                    "step_time_sec": step_time,
                    "skipped": "",
                },
            )

        pbar.set_postfix(
            loss=f"{loss_value:.4f}",
            skip_nan=skipped_nonfinite_batches,
            skip_empty=skipped_empty_mask_batches,
        )

    avg_loss = total_loss / max(total_valid_pixels, 1.0)

    print(
        f"[Train epoch check] "
        f"skipped_nonfinite_batches={skipped_nonfinite_batches}, "
        f"skipped_empty_mask_batches={skipped_empty_mask_batches}, "
        f"used_valid_pixels={total_valid_pixels:.0f}"
    )

    return avg_loss, global_step, skipped_nonfinite_batches, skipped_empty_mask_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()

    total_loss = 0.0
    total_valid_pixels = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    skipped_empty_mask_batches = 0
    skipped_nonfinite_loss_batches = 0

    pbar = tqdm(loader, desc=desc)

    for x, y, mask in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        valid_pixels = mask.sum().item()

        if valid_pixels <= 0:
            skipped_empty_mask_batches += 1
            continue

        if USE_AMP:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = masked_bce_loss(logits, y, mask, criterion)
        else:
            logits = model(x)
            loss = masked_bce_loss(logits, y, mask, criterion)

        if torch.isfinite(loss):
            total_loss += loss.item() * valid_pixels
            total_valid_pixels += valid_pixels
        else:
            skipped_nonfinite_loss_batches += 1

        tp, fp, fn, tn = calc_counts_from_logits(logits, y, mask)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        running_metrics = metrics_from_counts(total_tp, total_fp, total_fn, total_tn)

        pbar.set_postfix(
            loss=f"{float(loss.item()):.4f}" if torch.isfinite(loss) else "nan",
            iou=f"{running_metrics['IoU']:.4f}",
            f1=f"{running_metrics['F1']:.4f}",
        )

    avg_loss = total_loss / max(total_valid_pixels, 1.0)

    metrics = metrics_from_counts(total_tp, total_fp, total_fn, total_tn)
    metrics["loss"] = avg_loss
    metrics["skipped_empty_mask_batches"] = skipped_empty_mask_batches
    metrics["skipped_nonfinite_loss_batches"] = skipped_nonfinite_loss_batches

    return metrics


# 8. Group weights inspection
@torch.no_grad()
def inspect_group_weights(model, loader, device, max_batches=10):
    if not hasattr(model, "get_group_weights"):
        return None

    model.eval()
    all_weights = []

    for i, (x, y, mask) in enumerate(loader):
        if i >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        w = model.get_group_weights(x)
        all_weights.append(w.detach().cpu().numpy())

    if len(all_weights) == 0:
        return None

    all_weights = np.concatenate(all_weights, axis=0)

    mean_weights = all_weights.mean(axis=0)
    std_weights = all_weights.std(axis=0)

    return {
        "mean_spectral": float(mean_weights[0]),
        "mean_water_index": float(mean_weights[1]),
        "mean_terrain": float(mean_weights[2]),
        "mean_texture": float(mean_weights[3]),

        "std_spectral": float(std_weights[0]),
        "std_water_index": float(std_weights[1]),
        "std_terrain": float(std_weights[2]),
        "std_texture": float(std_weights[3]),
    }


# 9. Log utilities
def append_log(log_path, row):
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "Accuracy",
        "PA",
        "UA",
        "F1",
        "IoU",
        "mIoU",
        "epoch_time_sec",
        "total_params_m",
        "trainable_params_m",
        "batch_size",
        "num_workers",
        "use_amp",
        "skipped_train_nonfinite_batches",
        "skipped_train_empty_mask_batches",
        "skipped_val_nonfinite_loss_batches",
        "skipped_val_empty_mask_batches",
        "mean_spectral",
        "mean_water_index",
        "mean_terrain",
        "mean_texture",
        "std_spectral",
        "std_water_index",
        "std_terrain",
        "std_texture",
    ]

    file_exists = log_path.exists()

    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def append_step_log(log_path, row):
    fieldnames = [
        "epoch",
        "batch_idx",
        "num_batches",
        "global_step",
        "train_loss_step",
        "valid_pixels",
        "lr",
        "step_time_sec",
        "skipped",
    ]

    file_exists = log_path.exists()

    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_test_metrics(path, row):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# 9.1 Resume helper
def load_resume_checkpoint(model, optimizer, device):
    resume_path = get_resume_path()

    if resume_path is None:
        print("=" * 100)
        print("RESUME=True, but no checkpoint found. Start from epoch 1.")
        print("Tried:")
        print("  ", PREFERRED_RESUME_PATH)
        print("  ", FALLBACK_RESUME_PATH)
        print("=" * 100)
        return 1, 0, {"epoch": -1, "val_iou": -1.0, "metrics": None}

    print("=" * 100)
    print(f"Resume from checkpoint: {resume_path}")
    print("=" * 100)

    checkpoint = torch.load(resume_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    last_epoch = int(checkpoint.get("epoch", 0))
    start_epoch = last_epoch + 1
    global_step = int(checkpoint.get("global_step", 0))

    best = None

    if "best" in checkpoint and checkpoint["best"] is not None:
        best = checkpoint["best"]

    if best is None:
        best_model_path = OUT_DIR / "best_model.pth"

        if best_model_path.exists():
            best_ckpt = torch.load(best_model_path, map_location=device)

            if "best" in best_ckpt and best_ckpt["best"] is not None:
                best = best_ckpt["best"]
            else:
                best = {
                    "epoch": int(best_ckpt.get("epoch", -1)),
                    "val_iou": -1.0,
                    "metrics": None,
                }
        else:
            best = {
                "epoch": -1,
                "val_iou": -1.0,
                "metrics": None,
            }

    print(f"Checkpoint epoch: {last_epoch}")
    print(f"Start from epoch: {start_epoch}")
    print(f"Global step     : {global_step}")
    print(f"Best epoch      : {best.get('epoch', -1)}")
    print(f"Best val IoU    : {best.get('val_iou', -1.0)}")
    print("=" * 100)

    return start_epoch, global_step, best


# 10. Main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print("=" * 100)
    print("Train FGSA-UNet on full dataset")
    print("=" * 100)
    print("GPU:", torch.cuda.get_device_name(0))
    print("DATA_ROOT:", DATA_ROOT)
    print("OUT_DIR:", OUT_DIR)
    print("EPOCHS:", EPOCHS)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("NUM_WORKERS:", NUM_WORKERS)
    print("USE_AMP:", USE_AMP)
    print("GRAD_CLIP_NORM:", GRAD_CLIP_NORM)
    print("USE_SPATIAL_GATE:", USE_SPATIAL_GATE)
    print("RESUME:", RESUME)
    print("=" * 100)

    train_ds = WaterTileDataset(DATA_ROOT, "train")
    val_ds = WaterTileDataset(DATA_ROOT, "val")
    test_ds = WaterTileDataset(DATA_ROOT, "test")

    train_loader = make_loader(train_ds, "train")
    val_loader = make_loader(val_ds, "val")
    test_loader = make_loader(test_ds, "test")

    model = FGSAUNet(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        features=FEATURES,
        norm_type=NORM_TYPE,
        use_spatial_gate=USE_SPATIAL_GATE,
    ).to(device)

    total_params, trainable_params = count_params(model)
    total_params_m = total_params / 1e6
    trainable_params_m = trainable_params / 1e6

    print(f"Total params    : {total_params_m:.6f} M")
    print(f"Trainable params: {trainable_params_m:.6f} M")

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    log_path = OUT_DIR / "training_log.csv"
    step_log_path = OUT_DIR / "training_step_log.csv"
    best_model_path = OUT_DIR / "best_model.pth"
    last_model_path = OUT_DIR / "last_model.pth"
    test_metrics_path = OUT_DIR / "test_metrics.csv"

    best = {
        "epoch": -1,
        "val_iou": -1.0,
        "metrics": None,
    }

    global_step = 0
    start_epoch = 1

    if RESUME:
        start_epoch, global_step, best = load_resume_checkpoint(
            model=model,
            optimizer=optimizer,
            device=device,
        )

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        print("\n" + "=" * 80)
        print(f"FGSA-UNet | Epoch {epoch}/{EPOCHS}")

        train_loss, global_step, skipped_train_nonfinite, skipped_train_empty = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
            step_log_path=step_log_path,
            global_step=global_step,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            desc="Val",
        )

        group_stats = inspect_group_weights(
            model=model,
            loader=val_loader,
            device=device,
            max_batches=10,
        )

        if group_stats is None:
            group_stats = {
                "mean_spectral": "",
                "mean_water_index": "",
                "mean_terrain": "",
                "mean_texture": "",
                "std_spectral": "",
                "std_water_index": "",
                "std_terrain": "",
                "std_texture": "",
            }

        epoch_time = time.time() - t0

        print(f"Train loss: {train_loss:.4f}")
        print(
            f"Val: loss={val_metrics['loss']:.4f}, "
            f"Accuracy={val_metrics['Accuracy']:.4f}, "
            f"PA={val_metrics['PA']:.4f}, "
            f"UA={val_metrics['UA']:.4f}, "
            f"F1={val_metrics['F1']:.4f}, "
            f"IoU={val_metrics['IoU']:.4f}, "
            f"mIoU={val_metrics['mIoU']:.4f}"
        )

        print(
            f"Skipped train nonfinite={skipped_train_nonfinite}, "
            f"train empty mask={skipped_train_empty}, "
            f"val nonfinite={val_metrics['skipped_nonfinite_loss_batches']}, "
            f"val empty mask={val_metrics['skipped_empty_mask_batches']}"
        )

        print(
            "Group weights mean: "
            f"spectral={group_stats['mean_spectral']}, "
            f"water_index={group_stats['mean_water_index']}, "
            f"terrain={group_stats['mean_terrain']}, "
            f"texture={group_stats['mean_texture']}"
        )

        print(f"Time: {epoch_time:.1f}s")

        append_log(
            log_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "Accuracy": val_metrics["Accuracy"],
                "PA": val_metrics["PA"],
                "UA": val_metrics["UA"],
                "F1": val_metrics["F1"],
                "IoU": val_metrics["IoU"],
                "mIoU": val_metrics["mIoU"],
                "epoch_time_sec": epoch_time,
                "total_params_m": total_params_m,
                "trainable_params_m": trainable_params_m,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "use_amp": USE_AMP,
                "skipped_train_nonfinite_batches": skipped_train_nonfinite,
                "skipped_train_empty_mask_batches": skipped_train_empty,
                "skipped_val_nonfinite_loss_batches": val_metrics["skipped_nonfinite_loss_batches"],
                "skipped_val_empty_mask_batches": val_metrics["skipped_empty_mask_batches"],
                **group_stats,
            },
        )

        torch.save(
            {
                "epoch": epoch,
                "model_name": "FGSA-UNet",
                "model_key": "fgsa_unet",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "total_params": total_params,
                "trainable_params": trainable_params,
                "use_spatial_gate": USE_SPATIAL_GATE,
                "global_step": global_step,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "use_amp": USE_AMP,
                "best": best,
            },
            last_model_path,
        )

        if epoch % SAVE_CHECKPOINT_EVERY == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": "FGSA-UNet",
                    "model_key": "fgsa_unet",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "use_spatial_gate": USE_SPATIAL_GATE,
                    "global_step": global_step,
                    "batch_size": BATCH_SIZE,
                    "num_workers": NUM_WORKERS,
                    "use_amp": USE_AMP,
                    "best": best,
                },
                OUT_DIR / f"checkpoint_epoch_{epoch:03d}.pth",
            )

        if val_metrics["IoU"] > best["val_iou"]:
            best = {
                "epoch": epoch,
                "val_iou": val_metrics["IoU"],
                "metrics": val_metrics,
            }

            torch.save(
                {
                    "epoch": epoch,
                    "model_name": "FGSA-UNet",
                    "model_key": "fgsa_unet",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best": best,
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "use_spatial_gate": USE_SPATIAL_GATE,
                    "global_step": global_step,
                    "batch_size": BATCH_SIZE,
                    "num_workers": NUM_WORKERS,
                    "use_amp": USE_AMP,
                },
                best_model_path,
            )

            print(f"Saved best model: best_val_iou={best['val_iou']:.4f}")

    print("\nLoading best model for test...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        desc="Test",
    )

    test_group_stats = inspect_group_weights(
        model=model,
        loader=test_loader,
        device=device,
        max_batches=20,
    )

    if test_group_stats is None:
        test_group_stats = {
            "mean_spectral": "",
            "mean_water_index": "",
            "mean_terrain": "",
            "mean_texture": "",
            "std_spectral": "",
            "std_water_index": "",
            "std_terrain": "",
            "std_texture": "",
        }

    row = {
        "model_name": "FGSA-UNet",
        "model_key": "fgsa_unet",
        "best_epoch": best["epoch"],
        "best_val_iou": best["val_iou"],
        "test_loss": test_metrics["loss"],
        "Accuracy": test_metrics["Accuracy"],
        "PA": test_metrics["PA"],
        "UA": test_metrics["UA"],
        "F1": test_metrics["F1"],
        "IoU": test_metrics["IoU"],
        "mIoU": test_metrics["mIoU"],
        "total_params_m": total_params_m,
        "trainable_params_m": trainable_params_m,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "use_amp": USE_AMP,
        "use_spatial_gate": USE_SPATIAL_GATE,
        "TP": test_metrics["TP"],
        "FP": test_metrics["FP"],
        "FN": test_metrics["FN"],
        "TN": test_metrics["TN"],
        "skipped_test_nonfinite_loss_batches": test_metrics["skipped_nonfinite_loss_batches"],
        "skipped_test_empty_mask_batches": test_metrics["skipped_empty_mask_batches"],
        **test_group_stats,
    }

    save_test_metrics(test_metrics_path, row)

    print("\nFinal Test Result")
    print("=" * 100)
    for k, v in row.items():
        print(f"{k}: {v}")

    print("\nSaved files:")
    print("Log:", log_path)
    print("Step log:", step_log_path)
    print("Best model:", best_model_path)
    print("Last model:", last_model_path)
    print("Test metrics:", test_metrics_path)


if __name__ == "__main__":
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(description="Train or resume FGSA U-Net.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    DATA_ROOT = args.data_root
    OUT_DIR = args.output_dir
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    RESUME = not args.no_resume
    PREFERRED_RESUME_PATH = OUT_DIR / "last_model.pth"
    FALLBACK_RESUME_PATH = OUT_DIR / "checkpoint_epoch_040.pth"
    multiprocessing.freeze_support()
    main()
